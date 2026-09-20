"""Stage-one Jetson acquisition service. No identity or business decisions."""
import hmac
import json
import math
import os
import re
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

MAX_BYTES = 32 * 1024 * 1024
MAX_PREVIEW_BYTES = 16 * 1024 * 1024
SUPPORTED_FORMATS = {"BGR", "NV12"}


def read_secret(path):
    value = Path(path).read_text().strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("invalid secret file")
    return value


def packed_bgr(data, width, height, stride, offset=0):
    size = width * height * 3
    if width <= 0 or height <= 0 or size > MAX_BYTES or stride < width * 3 or offset < 0:
        raise ValueError("invalid frame layout")
    if len(data) < offset + (height - 1) * stride + width * 3:
        raise ValueError("truncated frame")
    return b"".join(bytes(data[offset + row * stride:offset + row * stride + width * 3])
                    for row in range(height))


def packed_nv12(data, width, height, y_stride, uv_stride, y_offset=0, uv_offset=None):
    """Pack padded NV12 rows into a compact Y plane followed by UV plane."""
    if (width <= 0 or height <= 0 or width % 2 or height % 2 or
            width * height * 3 // 2 > MAX_BYTES or y_stride < width or
            uv_stride < width or y_offset < 0):
        raise ValueError("invalid NV12 frame layout")
    if uv_offset is None:
        uv_offset = y_offset + y_stride * height
    if uv_offset < 0:
        raise ValueError("invalid NV12 frame layout")
    y_end = y_offset + (height - 1) * y_stride + width
    uv_end = uv_offset + (height // 2 - 1) * uv_stride + width
    if len(data) < max(y_end, uv_end):
        raise ValueError("truncated NV12 frame")

    packed = bytearray(width * height * 3 // 2)
    cursor = 0
    for row in range(height):
        start = y_offset + row * y_stride
        packed[cursor:cursor + width] = data[start:start + width]
        cursor += width
    for row in range(height // 2):
        start = uv_offset + row * uv_stride
        packed[cursor:cursor + width] = data[start:start + width]
        cursor += width
    return bytes(packed)


class LatestFrame:
    def __init__(self, camera_id, max_age=2.0, frame_format="BGR", output_fps=1):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
            raise ValueError("invalid camera id")
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("invalid max age")
        frame_format = frame_format.upper()
        if frame_format not in SUPPORTED_FORMATS:
            raise ValueError("unsupported frame format")
        if type(output_fps) is not int or not 1 <= output_fps <= 30:
            raise ValueError("invalid output fps")
        self.camera_id, self.max_age, self.frame_format = camera_id, max_age, frame_format
        self.output_fps = output_fps
        self.contract_version = 2 if frame_format == "NV12" else 1
        self.lock = threading.Lock()
        self.epoch, self.sequence = str(uuid.uuid4()), 0
        self.frame = None
        self.state = "starting"

    def reset(self, state="reconnecting"):
        with self.lock:
            self.epoch, self.sequence = str(uuid.uuid4()), 0
            self.frame, self.state = None, state

    def publish(self, payload, width, height, frame_format=None):
        frame_format = (frame_format or self.frame_format).upper()
        if frame_format == "NV12" and (width % 2 or height % 2):
            raise ValueError("invalid NV12 dimensions")
        expected = width * height * 3 if frame_format == "BGR" else width * height * 3 // 2
        if frame_format != self.frame_format or len(payload) != expected or not 0 < len(payload) <= MAX_BYTES:
            raise ValueError("invalid frame")
        with self.lock:
            self.sequence += 1
            meta = dict(version=self.contract_version, camera_id=self.camera_id, stream_epoch=self.epoch,
                        sequence=self.sequence, width=width, height=height, format=frame_format,
                        decoded_monotonic_ns=time.monotonic_ns(), captured_at=None)
            self.frame = (meta, payload)
            self.state = "streaming"

    def snapshot(self):
        with self.lock:
            if self.frame and (time.monotonic_ns() - self.frame[0]["decoded_monotonic_ns"]) / 1e9 <= self.max_age:
                return self.frame
        return None


class LatestPreview:
    """Latest hardware-encoded JPEG for browser observation only."""

    def __init__(self, max_age=2.0, output_fps=5, quality=85):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("invalid preview max age")
        if type(output_fps) is not int or not 1 <= output_fps <= 15:
            raise ValueError("invalid preview fps")
        if type(quality) is not int or not 1 <= quality <= 100:
            raise ValueError("invalid preview quality")
        self.max_age, self.output_fps, self.quality = max_age, output_fps, quality
        self.lock = threading.Lock()
        self.sequence = 0
        self.frame = None

    def reset(self):
        with self.lock:
            self.frame = None

    def publish(self, payload):
        if not 4 <= len(payload) <= MAX_PREVIEW_BYTES or payload[:2] != b"\xff\xd8" or payload[-2:] != b"\xff\xd9":
            raise ValueError("invalid preview jpeg")
        with self.lock:
            self.sequence += 1
            self.frame = ({"sequence": self.sequence, "encoded_monotonic_ns": time.monotonic_ns()}, payload)

    def snapshot(self):
        with self.lock:
            if self.frame and (time.monotonic_ns() - self.frame[0]["encoded_monotonic_ns"]) / 1e9 <= self.max_age:
                return self.frame
        return None


def handler_for(store, token, preview=None, max_preview_clients=4):
    # The preview is a long-lived response.  Bounding it keeps a browser tab
    # (or a hostile LAN client) from pinning every server thread.
    if type(max_preview_clients) is not int or not 1 <= max_preview_clients <= 32:
        raise ValueError("invalid preview client limit")
    preview_slots = threading.Semaphore(max_preview_clients)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *_args):
            pass  # No URLs, tokens or frame data in request logs.

        def do_GET(self):
            if self.path == "/health":
                ready = store.snapshot() is not None
                body = json.dumps({"ready": ready, "camera_id": store.camera_id,
                                   "contract_version": store.contract_version}).encode()
                return self.respond(200 if ready else 503, body, "application/json")
            if self.path == "/v1/preview.jpg":
                image = preview.snapshot() if preview else None
                if image is None:
                    return self.respond(503)
                return self.respond(200, image[1], "image/jpeg")
            if self.path == "/v1/preview.mjpeg":
                return self.stream_preview()
            if self.path != "/v1/frame":
                return self.respond(404)
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                return self.respond(401)
            frame = store.snapshot()
            if frame is None:
                return self.respond(503)
            meta, payload = frame
            return self.respond(200, payload, "application/octet-stream", meta)

        def stream_preview(self):
            if preview is None:
                return self.respond(404)
            if not preview_slots.acquire(blocking=False):
                return self.respond(503)
            try:
                self.write_preview_stream()
            finally:
                preview_slots.release()

        def write_preview_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=iris-preview")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_sequence = 0
            try:
                while True:
                    image = preview.snapshot()
                    if image is not None:
                        meta, payload = image
                        if meta["sequence"] != last_sequence:
                            self.wfile.write(
                                b"--iris-preview\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n"
                            )
                            self.wfile.flush()
                            last_sequence = meta["sequence"]
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError, OSError, TimeoutError):
                return

        def respond(self, status, payload=b"", content_type="text/plain", meta=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            if meta:
                self.send_header("X-Iris-Frame", json.dumps(meta, separators=(",", ":")))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (OSError, TimeoutError):
                pass
    return Handler


def capture_loop(store, preview, uri, stop):
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo
    Gst.init(None)
    while not stop.is_set():
        pipeline = None
        store.reset()
        preview.reset()
        stage = "pipeline-create"
        try:
            # URI is assigned as a property, never interpolated in pipeline syntax.
            if store.frame_format == "NV12":
                output = "! nvvidconv ! video/x-raw,format=NV12 "
            else:
                output = "! nvvidconv ! video/x-raw,format=BGRx " \
                         "! videoconvert ! video/x-raw,format=BGR "
            pipeline = Gst.parse_launch(
                "rtspsrc name=source protocols=tcp latency=150 drop-on-latency=true "
                "! application/x-rtp,media=video,encoding-name=H265 "
                f"! rtph265depay ! h265parse ! nvv4l2decoder ! tee name=decoded "
                "decoded. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
                f"! videorate drop-only=true max-rate={store.output_fps} "
                + output +
                "! appsink name=frames max-buffers=1 drop=true sync=false enable-last-sample=false "
                "decoded. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
                f"! videorate drop-only=true max-rate={preview.output_fps} "
                f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvjpegenc quality={preview.quality} "
                "! appsink name=preview max-buffers=1 drop=true sync=false enable-last-sample=false"
            )
            pipeline.get_by_name("source").set_property("location", uri)
            sink, preview_sink, bus = pipeline.get_by_name("frames"), pipeline.get_by_name("preview"), pipeline.get_bus()
            stage = "pipeline-start"
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("pipeline unavailable")
            last_frame = time.monotonic()
            while not stop.is_set():
                stage = "stream-read"
                if bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS):
                    raise RuntimeError("stream interrupted")
                sample = sink.emit("try-pull-sample", Gst.SECOND)
                if sample is None:
                    if time.monotonic() - last_frame > 5:
                        raise RuntimeError("stream stalled")
                    continue
                stage = "frame-layout"
                info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
                buffer = sample.get_buffer()
                stage = "frame-map"
                success, mapping = buffer.map(Gst.MapFlags.READ)
                if not success:
                    raise RuntimeError("frame mapping failed")
                try:
                    stage = "frame-pack"
                    if store.frame_format == "NV12":
                        payload = packed_nv12(
                            mapping.data, info.width, info.height,
                            info.stride[0], info.stride[1], info.offset[0], info.offset[1],
                        )
                    else:
                        payload = packed_bgr(mapping.data, info.width, info.height, info.stride[0], info.offset[0])
                finally:
                    buffer.unmap(mapping)
                store.publish(payload, info.width, info.height)
                preview_sample = preview_sink.emit("try-pull-sample", 0)
                if preview_sample is not None:
                    stage = "preview-map"
                    preview_buffer = preview_sample.get_buffer()
                    success, preview_mapping = preview_buffer.map(Gst.MapFlags.READ)
                    if not success:
                        raise RuntimeError("preview mapping failed")
                    try:
                        preview.publish(bytes(preview_mapping.data))
                    finally:
                        preview_buffer.unmap(preview_mapping)
                last_frame = time.monotonic()
        except Exception as exc:
            # Gst errors can contain credentials; only expose a bounded state.
            store.reset("unavailable")
            print(json.dumps({"stage": stage, "error_type": type(exc).__name__}), flush=True)
        finally:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
        stop.wait(3)


def main():
    token = read_secret(os.environ.get("IRIS_IP_TOKEN_FILE", "/run/secrets/engine_token"))
    uri = read_secret(os.environ.get("IRIS_IP_RTSP_FILE", "/run/secrets/camera_rtsp"))
    parsed = urlsplit(uri)
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token) or parsed.scheme != "rtsp" or not parsed.hostname:
        raise ValueError("invalid engine configuration")
    frame_format = os.environ.get("IRIS_IP_FRAME_FORMAT", "BGR").strip().upper()
    output_fps = int(os.environ.get("IRIS_IP_OUTPUT_FPS", "10"))
    preview_fps = int(os.environ.get("IRIS_IP_PREVIEW_FPS", "5"))
    preview_quality = int(os.environ.get("IRIS_IP_PREVIEW_JPEG_QUALITY", "85"))
    store = LatestFrame(
        os.environ.get("IRIS_IP_CAMERA_ID", "entrada"),
        frame_format=frame_format,
        output_fps=output_fps,
    )
    preview = LatestPreview(output_fps=preview_fps, quality=preview_quality)
    stop = threading.Event()
    worker = threading.Thread(target=capture_loop, args=(store, preview, uri, stop), daemon=True)
    # Threaded on purpose: /v1/preview.mjpeg is an open-ended response and a
    # serial server would stop answering /v1/frame for as long as a browser
    # keeps the preview open, silently halting recognition in iris-app.
    server = ThreadingHTTPServer(("0.0.0.0", 8090), handler_for(store, token, preview))
    server.daemon_threads = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_args: stop.set())
    worker.start()
    serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
    serving.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()

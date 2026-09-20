"""Short-lived real-RTSP DeepStream detector shadow probe.

The URI is read from a secret and assigned as a GStreamer property, so it is
not interpolated into a command line or emitted in logs. This probe does not
connect to iris-app and does not publish detection results.
"""
import os
import sys
import time
from pathlib import Path


def read_secret(path: str) -> str:
    value = Path(path).read_text().strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("invalid secret")
    return value


def main() -> int:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    pipeline = Gst.parse_launch(
        "rtspsrc name=source protocols=tcp latency=150 drop-on-latency=true "
        "! application/x-rtp,media=video,encoding-name=H265 "
        "! rtph265depay ! h265parse ! nvv4l2decoder "
        "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
        "! videorate drop-only=true max-rate=1 "
        "! nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080 "
        "! mux.sink_0 "
        "nvstreammux name=mux batch-size=1 width=1920 height=1080 live-source=true "
        "! nvinfer config-file-path=/tmp/retinaface_nvinfer.txt "
        "! fakesink sync=false"
    )
    pipeline.get_by_name("source").set_property(
        "location", read_secret(os.environ.get("IRIS_IP_RTSP_FILE", "/run/secrets/camera_rtsp"))
    )
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("shadow pipeline unavailable")

    bus = pipeline.get_bus()
    deadline = time.monotonic() + float(os.environ.get("IRIS_SHADOW_SECONDS", "20"))
    try:
        while time.monotonic() < deadline:
            message = bus.timed_pop_filtered(
                500 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if message is None:
                continue
            if message.type == Gst.MessageType.ERROR:
                error, _debug = message.parse_error()
                print(json_safe_error(error), file=sys.stderr, flush=True)
                return 1
            return 0
    finally:
        pipeline.set_state(Gst.State.NULL)
    print("shadow-ok", flush=True)
    return 0


def json_safe_error(error) -> str:
    return "shadow-error:" + type(error).__name__


if __name__ == "__main__":
    raise SystemExit(main())

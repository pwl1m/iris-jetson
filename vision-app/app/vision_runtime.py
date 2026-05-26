import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .recognizer import InsightFaceRecognizer, cosine_similarity, decode_image
from .settings import Settings
from .storage import FaceStore


class _GstUsbCapture:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._gst = None
        self._pipeline = None
        self._sink = None
        self._opened = False

    def open(self) -> bool:
        try:
            import gi
        except ImportError:
            return False

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)

        pipeline = self.settings.stream_gst_pipeline.strip() or self._default_pipeline()
        self._gst = Gst
        self._pipeline = Gst.parse_launch(pipeline)
        self._sink = self._pipeline.get_by_name("visionappsink")
        if self._sink is None:
            raise RuntimeError("appsink ausente no pipeline GStreamer")

        change = self._pipeline.set_state(Gst.State.PLAYING)
        if change == Gst.StateChangeReturn.FAILURE:
            self.release()
            return False

        self._opened = True
        return True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened or self._sink is None or self._gst is None:
            return False, None

        sample = self._sink.emit("try-pull-sample", int(1e9))
        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))

        ok, map_info = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            return False, None

        try:
            frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()
        finally:
            buffer.unmap(map_info)

        return True, frame

    def release(self) -> None:
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = None
        self._sink = None
        self._opened = False

    def _default_pipeline(self) -> str:
        device = self.settings.usb_camera_device
        width = self.settings.usb_camera_width
        height = self.settings.usb_camera_height
        fps = max(1, self.settings.usb_camera_fps)
        input_format = self.settings.usb_camera_input_format.strip().lower()

        if input_format == "mjpeg":
            return (
                f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
                f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                "jpegparse ! nvjpegdec ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=visionappsink drop=true max-buffers=1 sync=false"
            )

        return (
            f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=visionappsink drop=true max-buffers=1 sync=false"
        )


class VisionRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("vision-app")
        self.store = FaceStore(settings.face_db_path)
        self.recognizer = InsightFaceRecognizer(
            model_name=settings.face_model_name,
            model_root=settings.face_model_root,
            det_size=settings.det_size_tuple,
            providers=settings.providers_list,
            ctx_id=settings.face_ctx_id,
        )
        self._stop_event = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._recognizer_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._stats = {
            "running": False,
            "captures_seen": 0,
            "events_written": 0,
            "frames_read": 0,
            "recognition_errors": 0,
            "last_capture_at": None,
            "last_event_at": None,
            "last_error": None,
            "stream_url": settings.stream_url,
            "stream_source_kind": settings.stream_source_kind_normalized,
            "camera": settings.camera_name,
        }
        self._latest_event: dict | None = None
        self._latest_preview_jpeg: bytes | None = None
        self._latest_preview_at: str | None = None
        self._last_preview_update_monotonic = 0.0

    def start(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.settings.worker_log_level),
            format="%(asctime)s %(levelname)s %(message)s",
        )
        if self.settings.stream_worker_enabled:
            self.start_stream()

    def stop(self) -> None:
        self.stop_stream()

    def start_stream(self) -> dict:
        if self._stream_thread and self._stream_thread.is_alive():
            return self.stream_status()

        self._stop_event.clear()
        self._stream_thread = threading.Thread(target=self._stream_loop, name="stream-worker", daemon=True)
        self._stream_thread.start()
        self._set_stats(running=True, last_error=None)
        return self.stream_status()

    def stop_stream(self) -> dict:
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=5)
        self._set_stats(running=False)
        return self.stream_status()

    def enroll(self, subject: str, filename: str | None, payload: bytes) -> dict:
        image = decode_image(payload)
        with self._recognizer_lock:
            embedding, metadata = self.recognizer.extract_best(image)
        source = f"{self.settings.face_model_name}:{filename}" if filename else self.settings.face_model_name
        embedding_id = self.store.add_embedding(subject=subject, embedding=embedding, source=source)
        return {"status": "enrolled", "subject": subject, "embedding_id": embedding_id, "face": metadata}

    def enroll_capture(self, subject: str, capture_id: str) -> dict:
        path = self.capture_image_path(capture_id)
        return self.enroll(subject=subject, filename=path.name, payload=path.read_bytes())

    def delete_subject(self, subject: str) -> dict:
        deleted = self.store.delete_subject(subject)
        return {"status": "deleted", "subject": subject, "deleted": deleted}

    def subject_samples(self, subject: str) -> dict:
        samples = []
        for sample in self.store.samples(subject):
            source = sample.get("source") or ""
            filename = source.split(":", 1)[-1]
            capture_id = filename[:-4] if filename.endswith(".jpg") else None
            sample["capture_id"] = capture_id
            sample["image_url"] = f"/captures/{capture_id}/image" if capture_id else None
            samples.append(sample)
        return {"subject": subject, "samples": samples}

    def delete_sample(self, sample_id: int) -> dict:
        deleted = self.store.delete_sample(sample_id)
        return {"status": "deleted", "sample_id": sample_id, "deleted": deleted}

    def recognize_bytes(self, payload: bytes) -> dict:
        return self.recognize_image(decode_image(payload))

    def recognize_image(self, image: np.ndarray) -> dict:
        with self._recognizer_lock:
            query_embedding, metadata = self.recognizer.extract_best(image)

        matches = []
        for subject, stored_embedding, source in self.store.embeddings():
            matches.append(
                {
                    "subject": subject,
                    "similarity": cosine_similarity(query_embedding, stored_embedding),
                    "source": source,
                }
            )

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        matches = matches[: self.settings.face_max_results]
        best = matches[0] if matches else None
        accepted = bool(best and best["similarity"] >= self.settings.face_similarity_threshold)
        return {
            "status": "matched" if accepted else "no_match",
            "subject": best["subject"] if accepted else None,
            "similarity": best["similarity"] if best else None,
            "face": metadata,
            "candidates": matches,
        }

    def subjects(self) -> dict:
        return {"subjects": self.store.list_subjects()}

    def health(self) -> dict:
        engine = self.engine_status()
        return {
            "status": "ok",
            "model": self.settings.face_model_name,
            "threshold": self.settings.face_similarity_threshold,
            "providers": self.settings.providers_list,
            "ctx_id": self.settings.face_ctx_id,
            "engine": engine,
            "stream": self.stream_status(),
        }

    def engine_status(self) -> dict:
        report = self.recognizer.provider_report()
        mode = "accelerated" if any(
            item in report.get("active", [])
            for item in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        ) else "cpu_only"
        report["mode"] = mode
        return report

    def stream_status(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["thread_alive"] = bool(self._stream_thread and self._stream_thread.is_alive())
        stats["enabled"] = self.settings.stream_worker_enabled
        stats["interval_seconds"] = self.settings.stream_capture_interval_seconds
        with self._preview_lock:
            stats["preview_at"] = self._latest_preview_at
        return stats

    def latest_preview_jpeg(self) -> bytes | None:
        with self._preview_lock:
            return self._latest_preview_jpeg

    def recent_events(self, limit: int = 20) -> dict:
        limit = max(1, min(limit, 500))
        path = self._event_log()
        if not path.exists():
            return {"events": []}

        rows = path.read_text(encoding="utf-8").splitlines()[-limit:]
        events = [json.loads(row) for row in rows if row.strip()]
        events.reverse()
        return {"events": events}

    def latest_event(self) -> dict:
        if self._latest_event:
            return {"event": self._latest_event}
        events = self.recent_events(limit=1)["events"]
        return {"event": events[0] if events else None}

    def capture_image_path(self, capture_id: str) -> Path:
        candidate = self._capture_dir() / f"{capture_id}.jpg"
        if not candidate.exists():
            raise FileNotFoundError(capture_id)
        return candidate

    def _capture_dir(self) -> Path:
        path = Path(self.settings.capture_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _event_log(self) -> Path:
        path = Path(self.settings.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _set_stats(self, **updates) -> None:
        with self._stats_lock:
            self._stats.update(updates)

    def _increment_stat(self, name: str) -> int:
        with self._stats_lock:
            self._stats[name] = int(self._stats.get(name) or 0) + 1
            return int(self._stats[name])

    def _write_event(self, event: dict) -> None:
        with self._event_log().open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=True) + "\n")

    def _save_capture(self, capture_id: str, frame: np.ndarray) -> Path:
        path = self._capture_dir() / f"{capture_id}.jpg"
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality])
        if not ok:
            raise RuntimeError(f"falha ao salvar captura: {path}")
        return path

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _stream_loop(self) -> None:
        self.logger.info(
            "stream worker starting source_kind=%s url=%s",
            self.settings.stream_source_kind_normalized,
            self.settings.stream_url,
        )
        while not self._stop_event.is_set():
            self._wait_for_stream_source()
            if self._stop_event.is_set():
                break
            capture = self._open_capture()
            if not capture.isOpened():
                self._set_stats(last_error=f"source indisponivel ({self.settings.stream_source_kind_normalized})")
                self.logger.warning("stream unavailable, retrying in %.1fs", self.settings.stream_reconnect_delay_seconds)
                time.sleep(self.settings.stream_reconnect_delay_seconds)
                continue

            if self.settings.stream_source_kind_normalized == "rtsp" and self.settings.stream_reader_buffer_size > 0:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self.settings.stream_reader_buffer_size))

            self._set_stats(running=True, last_error=None)
            last_processed = 0.0
            try:
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._set_stats(last_error="falha ao ler frame")
                        break

                    self._increment_stat("frames_read")
                    now = time.monotonic()
                    self._update_preview(frame, now)
                    if now - last_processed < self.settings.stream_capture_interval_seconds:
                        continue
                    last_processed = now

                    self._process_frame(frame)
            finally:
                capture.release()

            if not self._stop_event.is_set():
                time.sleep(self.settings.stream_reconnect_delay_seconds)

        self._set_stats(running=False)
        self.logger.info("stream worker stopped")

    def _wait_for_stream_source(self) -> None:
        if self.settings.stream_source_kind_normalized == "jetson_gst_usb":
            return

        parsed = urllib.parse.urlparse(self.settings.stream_url)
        host = parsed.hostname
        port = parsed.port or (554 if parsed.scheme == "rtsp" else None)
        probe_url = (self.settings.stream_source_probe_url or "").strip()
        deadline = time.monotonic() + max(0.0, self.settings.stream_source_ready_timeout_seconds)

        while not self._stop_event.is_set():
            host_ready = True
            if host and port:
                try:
                    socket.gethostbyname(host)
                    with socket.create_connection((host, port), timeout=2):
                        pass
                except OSError as exc:
                    host_ready = False
                    self._set_stats(last_error=f"aguardando stream source: {exc}")

            probe_ready = True
            if host_ready and probe_url:
                try:
                    with urllib.request.urlopen(probe_url, timeout=2) as response:
                        if int(getattr(response, "status", 200)) >= 400:
                            raise urllib.error.HTTPError(
                                probe_url,
                                int(response.status),
                                "probe failure",
                                hdrs=response.headers,
                                fp=None,
                            )
                except Exception as exc:
                    probe_ready = False
                    self._set_stats(last_error=f"aguardando stream probe: {exc}")

            if host_ready and probe_ready:
                return

            if time.monotonic() >= deadline:
                self.logger.warning(
                    "stream source not ready yet url=%s probe=%s, continuing with open attempts",
                    self.settings.stream_url,
                    probe_url or "-",
                )
                return

            time.sleep(min(1.0, self.settings.stream_reconnect_delay_seconds))

    def _open_capture(self):
        if self.settings.stream_source_kind_normalized == "jetson_gst_usb":
            capture = _GstUsbCapture(self.settings)
            try:
                opened = capture.open()
            except Exception as exc:
                self.logger.exception("failed to open gst usb capture: %s", exc)
                self._set_stats(last_error=f"falha ao abrir gst usb: {exc}")
                capture.release()
                return capture
            if not opened:
                self._set_stats(last_error="falha ao abrir gst usb")
            return capture

        return cv2.VideoCapture(self.settings.stream_url, cv2.CAP_FFMPEG)

    def _update_preview(self, frame: np.ndarray, now_monotonic: float) -> None:
        if now_monotonic - self._last_preview_update_monotonic < self.settings.stream_preview_update_interval_seconds:
            return
        self._last_preview_update_monotonic = now_monotonic

        preview = frame
        max_width = max(0, self.settings.stream_preview_max_width)
        if max_width and preview.shape[1] > max_width:
            scale = max_width / float(preview.shape[1])
            target_height = max(1, int(preview.shape[0] * scale))
            preview = cv2.resize(preview, (max_width, target_height), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.stream_preview_jpeg_quality)],
        )
        if not ok:
            return

        with self._preview_lock:
            self._latest_preview_jpeg = encoded.tobytes()
            self._latest_preview_at = self._now()

    def _process_frame(self, frame: np.ndarray) -> None:
        capture_number = self._increment_stat("captures_seen")
        captured_at = self._now()
        self._set_stats(last_capture_at=captured_at)

        try:
            result = self.recognize_image(frame)
            face_score = float(result["face"].get("det_score") or 0.0)
            if face_score < self.settings.stream_min_face_score:
                return

            capture_id = f"{captured_at.replace(':', '').replace('+', 'Z')}_{capture_number:08d}"
            image_path = self._save_capture(capture_id, frame)
            if capture_number % 100 == 0:
                self._prune_captures()
            event = {
                "capture_id": capture_id,
                "capture_number": capture_number,
                "camera": self.settings.camera_name,
                "captured_at": captured_at,
                "image_path": str(image_path),
                "image_url": f"/captures/{capture_id}/image",
                "recognition": result,
            }
            self._write_event(event)
            self._latest_event = event
            self._increment_stat("events_written")
            self._set_stats(last_event_at=captured_at, last_error=None)
            self.logger.info(
                "capture=%s status=%s subject=%s similarity=%s",
                capture_id,
                result.get("status"),
                result.get("subject"),
                result.get("similarity"),
            )
        except ValueError as exc:
            self._increment_stat("recognition_errors")
            self._set_stats(last_error=str(exc))
        except Exception as exc:
            self._increment_stat("recognition_errors")
            self._set_stats(last_error=str(exc))
            self.logger.exception("failed to process stream frame: %s", exc)

    def _prune_captures(self) -> None:
        max_files = self.settings.stream_max_capture_files
        if max_files <= 0:
            return

        files = sorted(self._capture_dir().glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
        for old_file in files[max_files:]:
            try:
                old_file.unlink()
            except OSError:
                self.logger.warning("failed to remove old capture %s", old_file)

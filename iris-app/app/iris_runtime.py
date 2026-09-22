import json
import logging
import os
import queue
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
import paho.mqtt.client as mqtt

from .cameras import stream_render_mode, configured_cameras, inventory_stream_url
from .detector import InsightFaceDetector
from .face_tracking import FaceTrack, FaceTrackManager, TrackCandidate, select_within_budget
from .ip_capture import IpEngineCapture
from .pipeline import IrisPipeline
from .recognizer import NoFaceDetectedError, InsightFaceRecognizer, cosine_similarity, decode_image
from .roi import bbox_center_inside
from .schemas import CameraConfig
from .settings import Settings
from .storage import FaceStore, FrameCensusStore, find_jsonl_event, read_jsonl_events


class _GstUsbCapture:
    def __init__(self, settings: Settings, camera: CameraConfig, sampled: bool = False):
        self.settings = settings
        self.camera = camera
        self.sampled = sampled
        self._gst = None
        self._pipeline = None
        self._sink = None
        self._opened = False

    def open(self) -> bool:
        if self.sampled:
            # The container image already carries the software elements used by
            # this pipeline. Do not scan the host JetPack plugin directory here:
            # it is mounted for NVIDIA elements but has a different GLib ABI.
            os.environ.pop("GST_PLUGIN_PATH", None)
            os.environ.pop("GST_PLUGIN_SYSTEM_PATH", None)
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
        self._sink = self._pipeline.get_by_name("irisappsink")
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

        sample = self._sink.emit("pull-sample")
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
        device = self.camera.device or self.settings.usb_camera_device
        width = self.camera.width or self.settings.usb_camera_width
        height = self.camera.height or self.settings.usb_camera_height
        fps = max(1, self.camera.fps or self.settings.usb_camera_fps)
        input_format = self.settings.usb_camera_input_format.strip().lower()

        if input_format in {"mjpeg", "mjpg"}:
            sample_branch = ""
            if self.sampled:
                # videorate is after jpegdec because it operates on raw video.
                # This still decodes upstream MJPEG, but drops before appsink so
                # Python only maps/copies frames selected for inference.
                interval = max(1, round(self.settings.stream_capture_interval_seconds))
                sample_branch = f"videorate drop-only=true ! video/x-raw,framerate=1/{interval} ! "
            return (
                f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
                f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                "jpegparse ! jpegdec ! "
                f"{sample_branch}"
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=irisappsink drop=true max-buffers=1 sync=false"
            )

        return (
            f"v4l2src device={device} io-mode=2 do-timestamp=true ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=irisappsink drop=true max-buffers=1 sync=false"
        )


class _CameraWorkerState:
    def __init__(self, camera: CameraConfig, tracker: FaceTrackManager):
        self.camera = camera
        self.tracker = tracker
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.stats_lock = threading.Lock()
        self.preview_lock = threading.Lock()
        self.stats = {
            "running": False,
            "captures_seen": 0,
            "events_written": 0,
            "occlusions_written": 0,
            "frames_read": 0,
            "faces_detected": 0,
            # Censo: tudo que o detector devolveu, inclusive quem esta longe
            # demais ou torto demais para virar track.  Detectar e plano (~43 ms
            # pedindo 3 ou 50), entao contar quem passa nao custa orcamento.
            "faces_seen_total": 0,
            "faces_below_track_size": 0,
            "faces_in_last_frame": 0,
            "faces_in_frame_peak": 0,
            "faces_materialized": 0,
            "events_suppressed_debounce": 0,
            "events_suppressed_quality": 0,
            "materialize_deferred": 0,
            "frames_over_budget": 0,
            "tracks_started": 0,
            "recognition_attempts": 0,
            "active_tracks": 0,
            "no_face_detected": 0,
            "recognition_errors": 0,
            "last_occlusion_at": None,
            "last_no_face_at": None,
            "last_capture_at": None,
            "last_event_at": None,
            "last_error": None,
            "push_sent": 0,
            "push_errors": 0,
            "last_push_error": None,
            "mqtt_sent": 0,
            "mqtt_errors": 0,
            "last_mqtt_error": None,
            "stream_url": camera.stream_url,
            "stream_source_kind": camera.source_kind,
            "camera": camera.camera_id,
        }
        self.latest_event: dict | None = None
        self.latest_preview_jpeg: bytes | None = None
        self.latest_preview_at: str | None = None
        self.latest_enrollment_jpeg: bytes | None = None
        self.latest_enrollment_at: str | None = None
        self.last_preview_update_monotonic = 0.0
        self.last_landmarks = 0


class IrisRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("iris-app")
        self.store = FaceStore(settings.face_db_path)
        self.census_store = FrameCensusStore(settings.frame_census_db_path)
        self.recognizer = InsightFaceRecognizer(
            model_name=settings.face_model_name,
            model_root=settings.face_model_root,
            det_size=settings.det_size_tuple,
            providers=settings.providers_list,
            ctx_id=settings.face_ctx_id,
            trt_fp16=settings.face_trt_fp16,
            trt_engine_cache_path=settings.face_trt_engine_cache_path,
        )
        self.detector = InsightFaceDetector(self.recognizer, crop_padding=settings.pipeline_face_crop_padding)
        self.pipeline = IrisPipeline(settings, self.detector, self.store)
        self.cameras = configured_cameras(settings)
        self._camera_workers = {
            camera.camera_id: _CameraWorkerState(
                camera,
                FaceTrackManager(
                    max_tracks=settings.pipeline_max_faces,
                    iou_threshold=settings.pipeline_track_iou_threshold,
                    ttl_seconds=settings.pipeline_track_ttl_seconds,
                    min_frames=settings.pipeline_track_min_frames,
                    retry_seconds=settings.pipeline_track_retry_seconds,
                ),
            )
            for camera in self.cameras
        }
        self._recognizer_lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._occlusion_lock = threading.Lock()
        self._retention_lock = threading.Lock()
        self._push_stop_event = threading.Event()
        self._push_threads: list[threading.Thread] = []
        self._push_queue: queue.Queue[dict] = queue.Queue(maxsize=max(1, settings.onix_push_queue_size))
        self._mqtt_stop_event = threading.Event()
        self._mqtt_thread: threading.Thread | None = None
        self._mqtt_queue: queue.Queue[dict] = queue.Queue(maxsize=max(1, settings.mqtt_publish_queue_size))
        self._mqtt_client: mqtt.Client | None = None
        self._mqtt_debounce: dict[tuple[str, str, str], float] = {}
        self._mqtt_debounce_lock = threading.Lock()
        # Debounce de PERSISTENCIA, distinto do de MQTT: aquele so evita
        # republicar, este evita gravar linha e dois JPEGs. Limitado pelo numero
        # de sujeitos cadastrados, porque no_match nunca entra aqui.
        self._event_debounce: dict[tuple[str, str], float] = {}
        self._event_debounce_lock = threading.Lock()
        self.warm_up_report: dict | None = None

    def start(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.settings.worker_log_level),
            format="%(asctime)s %(levelname)s %(message)s",
        )
        self._start_push_worker()
        self._start_mqtt_worker()
        self._warm_up_models()
        if self.settings.stream_worker_enabled:
            self.start_stream()

    def stop(self) -> None:
        self.stop_stream()
        self._stop_push_worker()
        self._stop_mqtt_worker()

    def start_stream(self) -> dict:
        for camera in self.cameras:
            if camera.enabled:
                self._start_camera_worker(camera.camera_id)
        return self.stream_status()

    def stop_stream(self) -> dict:
        for camera in self.cameras:
            self._stop_camera_worker(camera.camera_id)
        return self.stream_status()

    def _start_camera_worker(self, camera_id: str) -> None:
        state = self._camera_workers[camera_id]
        if not state.camera.enabled:
            raise ValueError("camera desabilitada na configuracao")
        if state.thread and state.thread.is_alive():
            return
        state.stop_event.clear()
        state.thread = threading.Thread(
            target=self._stream_loop,
            args=(state.camera,),
            name=f"stream-worker-{camera_id}",
            daemon=True,
        )
        state.thread.start()
        self._set_stats(state, running=True, last_error=None)

    def _stop_camera_worker(self, camera_id: str) -> None:
        state = self._camera_workers[camera_id]
        state.stop_event.set()
        if state.thread:
            state.thread.join(timeout=5)
        state.tracker.clear()
        self._set_stats(state, active_tracks=0)
        self._set_stats(state, running=False)

    def _warm_up_models(self) -> None:
        """Pay the TensorRT engine build at boot instead of on the first face."""
        if not self.settings.pipeline_warm_up_enabled:
            self.warm_up_report = {"skipped": True}
            return
        started = time.monotonic()
        try:
            with self._recognizer_lock:
                report = self.recognizer.warm_up()
        except Exception as exc:
            # Never block startup: a failed warm-up only means the first real
            # face pays the build, which is the behaviour we had before.
            self.warm_up_report = {"error": str(exc)}
            self.logger.warning("model warm-up failed, first face pays the engine build: %s", exc)
            return
        report["total_seconds"] = round(time.monotonic() - started, 3)
        self.warm_up_report = report
        self.logger.info("model warm-up finished %s", report)

    def enroll(self, subject: str, filename: str | None, payload: bytes) -> dict:
        image = decode_image(payload)
        with self._recognizer_lock:
            candidate = self.recognizer.inspect_best(image, padding=self.settings.pipeline_face_crop_padding)
            embedding = candidate["embedding"]
            metadata = candidate["metadata"]
            reference_image = candidate["crop"]
        source = f"{self.settings.face_model_name}:{filename}" if filename else self.settings.face_model_name
        embedding_id = self.store.add_embedding(subject=subject, embedding=embedding, source=source)
        reference_image_url = self._save_reference_image(embedding_id, reference_image)
        self._publish_subjects_snapshot()
        return {
            "status": "enrolled",
            "subject": subject,
            "embedding_id": embedding_id,
            "reference_image_url": reference_image_url,
            "face": metadata,
        }

    def enroll_capture(self, subject: str, capture_id: str) -> dict:
        path = self.capture_image_path(capture_id)
        return self.enroll(subject=subject, filename=path.name, payload=path.read_bytes())

    def delete_subject(self, subject: str) -> dict:
        samples = list(self.store.samples(subject))
        deleted = self.store.delete_subject(subject)
        for sample in samples:
            self._delete_reference_image(int(sample["id"]))
        self._publish_subjects_snapshot()
        return {"status": "deleted", "subject": subject, "deleted": deleted}

    def rename_subject(self, subject: str, new_subject: str) -> dict:
        subject = subject.strip()
        new_subject = new_subject.strip()
        if not subject or not new_subject:
            raise ValueError("nome atual e novo nome sao obrigatorios")
        if subject == new_subject:
            return {"status": "unchanged", "subject": subject, "renamed": 0}

        renamed = self.store.rename_subject(subject, new_subject)
        if not renamed:
            raise FileNotFoundError(subject)
        self._publish_subjects_snapshot()
        return {"status": "renamed", "subject": new_subject, "previous_subject": subject, "renamed": renamed}

    def subject_samples(self, subject: str) -> dict:
        samples = []
        for sample in self.store.samples(subject):
            source = sample.get("source") or ""
            filename = source.split(":", 1)[-1]
            capture_id = filename[:-4] if filename.endswith(".jpg") else None
            sample_id = int(sample["id"])
            sample["capture_id"] = capture_id
            sample["image_url"] = self._sample_image_url(sample_id, capture_id)
            sample["reference_image_url"] = f"/samples/{sample_id}/image" if self._reference_image_path(sample_id).exists() else None
            samples.append(sample)
        return {"subject": subject, "samples": samples}

    def delete_sample(self, sample_id: int) -> dict:
        deleted = self.store.delete_sample(sample_id)
        if deleted:
            self._delete_reference_image(sample_id)
            self._publish_subjects_snapshot()
        return {"status": "deleted", "sample_id": sample_id, "deleted": deleted}

    def recognize_bytes(self, payload: bytes) -> dict:
        return self.recognize_image(decode_image(payload))

    def recognize_image(self, image: np.ndarray) -> dict:
        detection, quality, recognition = self.pipeline.analyze_frame(image)
        return {
            **recognition,
            "quality": quality,
            "detector": self.detector_status(),
            "face": detection.metadata,
        }

    def compare_bytes(self, payload: bytes) -> dict:
        image = decode_image(payload)
        detection, quality, recognition = self.pipeline.analyze_frame(image)
        return {
            "camera_id": self.settings.camera_1_id,
            "detector": {
                **self.detector_status(),
                "score": detection.det_score,
                "bbox": detection.bbox,
            },
            "quality": quality,
            "recognition": recognition,
        }

    def _camera_by_id(self, camera_id: str | None) -> CameraConfig | None:
        if not camera_id:
            return None
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None

    def people_count_bytes(self, payload: bytes, camera_id: str | None = None) -> dict:
        return self.people_count_image(decode_image(payload), camera_id=camera_id)

    def people_count_image(self, image: np.ndarray, camera_id: str | None = None) -> dict:
        """Conta rostos distintos numa foto avulsa e persiste o resultado.

        Reaproveita so a deteccao (SCRFD), sem landmarks/embedding -- o mesmo
        estagio barato que ja roda no censo do worker (~43 ms, plano). Nao
        reconhece nem cadastra ninguem; e so contagem.
        """
        camera = self._camera_by_id(camera_id)
        roi = camera.roi if camera else None
        with self._recognizer_lock:
            candidates = self.pipeline.detect_candidates(
                image, max_faces=self.settings.effective_detect_max_faces
            )
        height, width = image.shape[:2]
        boxes = []
        face_count = 0
        for candidate in candidates:
            bbox = [float(value) for value in candidate["bbox"]]
            inside = roi is None or bbox_center_inside(bbox, roi, width, height)
            if inside:
                face_count += 1
            boxes.append({
                "bbox": bbox,
                "det_score": float(candidate["det_score"]),
                "inside_roi": inside if roi is not None else None,
            })
        captured_at = self._now()
        census_id = self.census_store.record(
            camera_id=camera_id or "unspecified",
            captured_at=captured_at,
            face_count=face_count,
            roi_applied=roi is not None,
            source="upload",
        )
        return {
            "census_id": census_id,
            "camera_id": camera_id,
            "captured_at": captured_at,
            "face_count": face_count,
            "faces_detected_total": len(candidates),
            "roi_applied": roi is not None,
            "boxes": boxes,
        }

    def frame_census(self, limit: int = 20, camera_id: str | None = None, since: int | None = None) -> dict:
        return {"census": self.census_store.list_recent(limit=limit, camera_id=camera_id, since=since)}

    def detector_status(self) -> dict:
        return {
            "name": "insightface",
            "model": self.settings.face_model_name,
            "det_size": self.settings.det_size_tuple,
            "single_face": self.settings.pipeline_single_face,
            "max_faces": self.settings.pipeline_max_faces,
            "detect_max_faces": self.settings.effective_detect_max_faces,
            "materialize_budget": self.settings.effective_materialize_budget,
            "tracking": {
                "iou_threshold": self.settings.pipeline_track_iou_threshold,
                "ttl_seconds": self.settings.pipeline_track_ttl_seconds,
                "min_frames": self.settings.pipeline_track_min_frames,
            },
            "crop_padding": self.settings.pipeline_face_crop_padding,
        }

    def cameras_status(self) -> dict:
        items = []
        primary_id = self.settings.camera_1_id
        for camera in self.cameras:
            stream = self.stream_status(camera.camera_id)
            healthy = self._stream_is_healthy(stream)
            render_mode, render_content_type = stream_render_mode(
                self.settings.camera_stream_render_mode
            )
            items.append(
                {
                    "camera_id": camera.camera_id,
                    "enabled": camera.enabled,
                    "primary": camera.primary,
                    "source_kind": camera.source_kind,
                    "device": camera.device,
                    "stream_url": inventory_stream_url(
                        camera.source_kind, camera.public_stream_url, camera.stream_url
                    ),
                    # Consumed internally by the Onix resolver. The browser
                    # still receives only the selected stream_url from /stream.
                    "stream_urls": camera.public_stream_urls,
                    # Aditivo: o Onix hoje devolve 'mjpeg' fixo e ignora estes
                    # campos. Quando o patch de render_mode for aplicado la, o
                    # front passa a escolher o player pelo que o device diz.
                    "stream_render_mode": render_mode,
                    "stream_content_type": render_content_type,
                    "input_format": camera.input_format,
                    "width": camera.width,
                    "height": camera.height,
                    "fps": camera.fps,
                    "camera_serial_number": camera.serial_number,
                    "camera_model_name": camera.model_name,
                    "capture_stable_path": camera.stable_path,
                    "active": healthy,
                    "worker_attached": bool(stream.get("thread_alive")),
                    "checks_per_second": self.settings.effective_checks_per_second,
                    "last_frame_at": stream.get("last_capture_at"),
                    "last_error": stream.get("last_error"),
                }
            )
        return {"cameras": items, "primary_camera_id": primary_id}

    def camera_status(self, camera_id: str) -> dict:
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                stream = self.stream_status(camera_id)
                return {
                    "camera": {
                        **camera.__dict__,
                        "active": self._stream_is_healthy(stream),
                        "worker_attached": bool(stream.get("thread_alive")),
                        "checks_per_second": self.settings.effective_checks_per_second,
                        "last_frame_at": stream.get("last_capture_at"),
                        "last_error": stream.get("last_error"),
                    }
                }
        raise KeyError(camera_id)

    def start_camera(self, camera_id: str) -> dict:
        if camera_id not in self._camera_workers:
            raise KeyError(camera_id)
        self._start_camera_worker(camera_id)
        return self.stream_status(camera_id)

    def stop_camera(self, camera_id: str) -> dict:
        if camera_id not in self._camera_workers:
            raise KeyError(camera_id)
        self._stop_camera_worker(camera_id)
        return self.stream_status(camera_id)

    def debug_pipeline(self) -> dict:
        return {
            "detector": self.detector_status(),
            "quality": {
                "min_det_score": self.settings.face_min_det_score,
                "min_width": self.settings.face_min_width,
                "min_height": self.settings.face_min_height,
                "min_blur_score": self.settings.face_min_blur_score,
            },
            "checks_per_second": self.settings.effective_checks_per_second,
            "interval_seconds": self.settings.stream_capture_interval_seconds,
            "cameras": self.cameras_status()["cameras"],
            "engine": self.engine_status(),
        }

    def _recent_events(self, limit: int = 20, camera_id: str | None = None, since: str | None = None) -> dict:
        limit = max(1, min(limit, 500))

        def _matches(event: dict) -> bool:
            if camera_id and event.get("camera_id", event.get("camera")) != camera_id:
                return False
            return True

        return {
            "events": read_jsonl_events(self._event_log(), limit=limit, since=since, predicate=_matches)
        }

    def event_by_id(self, event_id: str) -> dict:
        event = find_jsonl_event(self._event_log(), event_id)
        if not event:
            raise FileNotFoundError(event_id)
        return {"event": event}

    def face_image_path(self, event_id: str) -> Path:
        event = find_jsonl_event(self._event_log(), event_id)
        if not event or not event.get("face_image_path"):
            raise FileNotFoundError(event_id)
        path = Path(event["face_image_path"])
        if not path.exists():
            raise FileNotFoundError(event_id)
        return path

    def frame_image_path(self, event_id: str) -> Path:
        event = find_jsonl_event(self._event_log(), event_id)
        if not event or not event.get("image_path"):
            raise FileNotFoundError(event_id)
        path = Path(event["image_path"])
        if not path.exists():
            raise FileNotFoundError(event_id)
        return path

    def enroll_event(self, subject: str, event_id: str) -> dict:
        path = self.face_image_path(event_id)
        return self.enroll(subject=subject, filename=path.name, payload=path.read_bytes())

    def recognize_image_legacy(self, image: np.ndarray) -> dict:
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
        return {"subjects": self._subjects_with_primary_images()}

    def health(self) -> dict:
        engine = self.engine_status()
        streams = [self.stream_status(camera.camera_id) for camera in self.cameras]
        enabled_streams = [stream for stream in streams if stream.get("enabled")]
        status = "ok" if enabled_streams and all(self._stream_is_healthy(stream) for stream in enabled_streams) else "degraded"
        return {
            "status": status,
            "controller_status": "online",
            "model": self.settings.face_model_name,
            "threshold": self.settings.face_similarity_threshold,
            "providers": self.settings.providers_list,
            "ctx_id": self.settings.face_ctx_id,
            "engine": engine,
            "stream": self.stream_status(),
            "streams": streams,
        }

    def _stream_is_healthy(self, stream: dict) -> bool:
        if not stream.get("enabled"):
            return False
        if not stream.get("running") or not stream.get("thread_alive"):
            return False
        last_capture_at = stream.get("last_capture_at")
        if not last_capture_at:
            return False
        try:
            captured_at = datetime.fromisoformat(str(last_capture_at).replace("Z", "+00:00"))
            if captured_at.tzinfo is None:
                captured_at = captured_at.replace(tzinfo=timezone.utc)
            max_age = max(
                15.0,
                float(self.settings.stream_capture_interval_seconds) * 4.0,
                float(self.settings.stream_reconnect_delay_seconds) * 2.0,
            )
            return (datetime.now(timezone.utc) - captured_at).total_seconds() <= max_age
        except (TypeError, ValueError):
            return False

    def engine_status(self) -> dict:
        report = self.recognizer.provider_report()
        mode = "accelerated" if any(
            item in report.get("active", [])
            for item in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        ) else "cpu_only"
        report["mode"] = mode
        report["warm_up"] = self.warm_up_report
        return report

    def stream_status(self, camera_id: str | None = None) -> dict:
        if camera_id is None:
            # The legacy top-level `stream` field is consumed by Onix health
            # checks. Point it at an enabled camera instead of a disabled
            # primary slot; per-camera status remains available in `streams`.
            camera_id = next(
                (camera.camera_id for camera in self.cameras if camera.enabled),
                self.settings.camera_1_id,
            )
        state = self._camera_workers[camera_id]
        with state.stats_lock:
            stats = dict(state.stats)
        stats["camera_id"] = camera_id
        stats["thread_alive"] = bool(state.thread and state.thread.is_alive())
        stats["enabled"] = bool(state.camera.enabled and self.settings.stream_worker_enabled)
        stats["interval_seconds"] = self.settings.stream_capture_interval_seconds
        with state.preview_lock:
            stats["preview_at"] = state.latest_preview_at
        return stats

    def crowd_status(self, camera_id: str | None = None) -> dict:
        """Quantas pessoas a camera VE, separado de quantas ela RECONHECE.

        Sao dois numeros com potencias diferentes e o produto precisa dos dois:
        num evento, a maioria de quem atravessa o quadro esta longe demais ou
        de angulo ruim para reconhecimento, mas ainda assim passou. Detectar e
        plano (~43 ms por frame pedindo 3 ou 50 rostos), entao o censo nao
        compete com o reconhecimento pelo orcamento.

        Ressalva importante: isto conta ROSTOS DETECTADOS, nao pessoas. Quem
        estiver de costas, de perfil fechado ou pequeno demais para o detector
        nao entra em numero nenhum. Contagem de pessoas de verdade exigiria um
        detector de corpo, que nao faz parte deste pipeline.
        """
        cameras = (
            [camera_id] if camera_id is not None
            else [camera.camera_id for camera in self.cameras if camera.enabled]
        )
        saida = []
        for item in cameras:
            state = self._camera_workers[item]
            with state.stats_lock:
                stats = dict(state.stats)
            vistos = int(stats.get("faces_seen_total") or 0)
            materializados = int(stats.get("faces_materialized") or 0)
            saida.append({
                "camera_id": item,
                "seen": {
                    "faces_in_last_frame": stats.get("faces_in_last_frame", 0),
                    "faces_in_frame_peak": stats.get("faces_in_frame_peak", 0),
                    "faces_seen_total": vistos,
                    "faces_below_track_size": stats.get("faces_below_track_size", 0),
                    "detect_max_faces": self.settings.effective_detect_max_faces,
                },
                "recognized": {
                    "active_tracks": stats.get("active_tracks", 0),
                    "tracks_started": stats.get("tracks_started", 0),
                    "faces_materialized": materializados,
                    "events_written": stats.get("events_written", 0),
                    "max_tracks": self.settings.pipeline_max_faces,
                },
                "suppressed": {
                    "by_debounce": stats.get("events_suppressed_debounce", 0),
                    "by_quality": stats.get("events_suppressed_quality", 0),
                    "debounce_seconds": self.settings.event_debounce_seconds,
                    "unmatched_min_width": self.settings.event_unmatched_min_width,
                    "unmatched_min_det_score": self.settings.event_unmatched_min_det_score,
                },
                "budget": {
                    "materialize_per_frame": self.settings.effective_materialize_budget,
                    "materialize_deferred": stats.get("materialize_deferred", 0),
                    "frames_over_budget": stats.get("frames_over_budget", 0),
                    "checks_per_second": self.settings.effective_checks_per_second,
                },
                # Fracao do que foi visto que chegou a virar reconhecimento.
                # Cair muito significa que o censo enxerga gente que o orcamento
                # nao alcanca: subir materialize_per_frame ou aceitar a perda.
                "materialized_ratio": round(materializados / vistos, 4) if vistos else None,
            })
        return {"cameras": saida}

    def latest_preview_jpeg(self, camera_id: str | None = None) -> bytes | None:
        camera_id = camera_id or self.settings.camera_1_id
        state = self._camera_workers[camera_id]
        with state.preview_lock:
            return state.latest_preview_jpeg

    def latest_enrollment_jpeg(self, camera_id: str) -> bytes | None:
        state = self._camera_workers[camera_id]
        with state.preview_lock:
            payload = state.latest_enrollment_jpeg
        if not payload or not self._stream_is_healthy(self.stream_status(camera_id)):
            return None
        return payload

    def enroll_latest_camera_frame(self, camera_id: str, subject: str) -> dict:
        payload = self.latest_enrollment_jpeg(camera_id)
        if not payload:
            raise FileNotFoundError(camera_id)
        return self.enroll(
            subject=subject,
            filename=f"camera_{camera_id}_{self._now().replace(':', '').replace('+', 'Z')}.jpg",
            payload=payload,
        )

    def recent_events(self, limit: int = 20, camera_id: str | None = None, since: str | None = None) -> dict:
        return self._recent_events(limit=limit, camera_id=camera_id, since=since)

    def latest_event(self, camera_id: str | None = None) -> dict:
        camera_id = camera_id or self.settings.camera_1_id
        state = self._camera_workers[camera_id]
        if state.latest_event:
            return {"event": state.latest_event}
        events = self._recent_events(limit=1, camera_id=camera_id)["events"]
        return {"event": events[0] if events else None}

    def recent_occlusions(
        self,
        limit: int = 50,
        since: str | None = None,
        camera_id: str | None = None,
        recognition_status: str | None = None,
        occlusion_class: str | None = None,
    ) -> dict:
        limit = max(1, min(limit, 500))

        def _matches(event: dict) -> bool:
            if camera_id and event.get("camera_id", event.get("camera")) != camera_id:
                return False
            if recognition_status and (event.get("recognition") or {}).get("status") != recognition_status:
                return False
            if occlusion_class and event.get("occlusion_class") != occlusion_class:
                return False
            return True

        return {
            "events": read_jsonl_events(self._occlusion_log(), limit=limit, since=since, predicate=_matches)
        }

    def occlusion_by_id(self, event_id: str) -> dict:
        event = find_jsonl_event(self._occlusion_log(), event_id)
        if not event:
            raise FileNotFoundError(event_id)
        return {"event": event}

    def occlusion_frame_image_path(self, event_id: str) -> Path:
        event = find_jsonl_event(self._occlusion_log(), event_id)
        if not event or not event.get("image_path"):
            raise FileNotFoundError(event_id)
        path = Path(event["image_path"])
        if not path.exists():
            raise FileNotFoundError(event_id)
        return path

    def occlusion_face_image_path(self, event_id: str) -> Path:
        event = find_jsonl_event(self._occlusion_log(), event_id)
        if not event or not event.get("face_image_path"):
            raise FileNotFoundError(event_id)
        path = Path(event["face_image_path"])
        if not path.exists():
            raise FileNotFoundError(event_id)
        return path

    def capture_image_path(self, capture_id: str) -> Path:
        candidate = self._capture_dir() / f"{capture_id}.jpg"
        if not candidate.exists():
            raise FileNotFoundError(capture_id)
        return candidate

    def sample_image_path(self, sample_id: int) -> Path:
        sample = self.store.sample(sample_id)
        if not sample:
            raise FileNotFoundError(str(sample_id))

        reference = self._reference_image_path(sample_id)
        if reference.exists():
            return reference

        source = sample.get("source") or ""
        filename = source.split(":", 1)[-1] if ":" in source else source
        capture_id = filename[:-4] if filename.endswith(".jpg") else None
        if capture_id:
            return self.capture_image_path(capture_id)

        raise FileNotFoundError(str(sample_id))

    def _subjects_with_primary_images(self) -> list[dict]:
        subjects = []
        for entry in self.store.list_subjects():
            subject = dict(entry)
            samples = list(self.store.samples(str(subject["subject"])))
            primary = samples[0] if samples else None
            if primary:
                source = primary.get("source", "")
                filename = source.split(":", 1)[-1] if ":" in source else source
                capture_id = filename[:-4] if filename.endswith(".jpg") else None
                subject["primary_image_url"] = self._sample_image_url(int(primary["id"]), capture_id)
            else:
                subject["primary_image_url"] = None
            subjects.append(subject)
        return subjects

    def _reference_image_dir(self) -> Path:
        path = Path(self.settings.reference_image_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _reference_image_path(self, sample_id: int) -> Path:
        return self._reference_image_dir() / f"{int(sample_id)}.jpg"

    def _save_reference_image(self, sample_id: int, image: np.ndarray) -> str | None:
        path = self._reference_image_path(sample_id)
        try:
            ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise RuntimeError("cv2.imwrite retornou falso")
            return f"/samples/{int(sample_id)}/image"
        except Exception as exc:
            self.logger.warning("failed to save reference image sample=%s: %s", sample_id, exc)
            return None

    def _delete_reference_image(self, sample_id: int) -> None:
        path = self._reference_image_path(sample_id)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            self.logger.warning("failed to delete reference image sample=%s", sample_id)

    def _sample_image_url(self, sample_id: int, capture_id: str | None = None) -> str | None:
        if self._reference_image_path(sample_id).exists():
            return f"/samples/{int(sample_id)}/image"
        if capture_id:
            return f"/captures/{capture_id}/image"
        return None

    def _capture_dir(self) -> Path:
        path = Path(self.settings.capture_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _face_dir(self) -> Path:
        path = Path(self.settings.face_crop_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _event_log(self) -> Path:
        path = Path(self.settings.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _set_stats(self, state: _CameraWorkerState, **updates) -> None:
        with state.stats_lock:
            state.stats.update(updates)

    def _increment_stat(self, state: _CameraWorkerState, name: str) -> int:
        with state.stats_lock:
            state.stats[name] = int(state.stats.get(name) or 0) + 1
            return int(state.stats[name])

    def _state_for_event(self, event: dict) -> _CameraWorkerState:
        camera_id = str(event.get("camera_id") or self.settings.camera_1_id)
        return self._camera_workers.get(camera_id) or self._camera_workers[self.settings.camera_1_id]

    def _write_event(self, event: dict) -> None:
        with self._event_lock:
            with self._event_log().open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, ensure_ascii=True) + "\n")

    def _save_capture(self, capture_id: str, frame: np.ndarray) -> Path:
        path = self._capture_dir() / f"{capture_id}.jpg"
        ok = cv2.imwrite(str(path), frame, [
            int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0,
        ])
        if not ok:
            raise RuntimeError(f"falha ao salvar captura: {path}")
        return path

    def _save_face_crop(self, event_id: str, crop: np.ndarray) -> Path:
        path = self._face_dir() / f"{event_id}.jpg"
        ok = cv2.imwrite(str(path), crop, [
            int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0,
        ])
        if not ok:
            raise RuntimeError(f"falha ao salvar crop facial: {path}")
        return path

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _stream_loop(self, camera: CameraConfig) -> None:
        state = self._camera_workers[camera.camera_id]
        self.logger.info(
            "stream worker starting camera=%s source_kind=%s device=%s url=%s",
            camera.camera_id,
            camera.source_kind,
            camera.device or "-",
            camera.stream_url,
        )
        while not state.stop_event.is_set():
            self._wait_for_stream_source(camera, state)
            if state.stop_event.is_set():
                break
            capture = self._open_capture(camera)
            if not capture.isOpened():
                self._set_stats(state, running=False, last_error=f"source indisponivel ({camera.source_kind})")
                self.logger.warning("camera=%s unavailable, retrying in %.1fs", camera.camera_id, self.settings.stream_reconnect_delay_seconds)
                time.sleep(self.settings.stream_reconnect_delay_seconds)
                continue

            if camera.source_kind == "rtsp" and self.settings.stream_reader_buffer_size > 0:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self.settings.stream_reader_buffer_size))

            self._set_stats(state, running=True, last_error=None)
            last_processed = 0.0
            try:
                while not state.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if camera.source_kind == "ip_engine" and getattr(capture, "transient_failure", False):
                            continue
                        self._set_stats(state, running=False, last_error="falha ao ler frame")
                        break

                    self._increment_stat(state, "frames_read")
                    now = time.monotonic()
                    self._update_preview(state, frame, now)
                    if camera.source_kind not in {"gst_usb_sampled", "ip_engine"} and now - last_processed < self.settings.stream_capture_interval_seconds:
                        continue
                    last_processed = now

                    self._process_frame(frame, camera, state)
            finally:
                capture.release()
                state.tracker.clear()
                self._set_stats(state, active_tracks=0)

            if not state.stop_event.is_set():
                time.sleep(self.settings.stream_reconnect_delay_seconds)

        self._set_stats(state, running=False)
        self.logger.info("stream worker stopped camera=%s", camera.camera_id)

    def _wait_for_stream_source(self, camera: CameraConfig, state: _CameraWorkerState) -> None:
        if camera.source_kind in {"jetson_gst_usb", "usb", "gst_usb_sampled", "ip_engine"}:
            return

        parsed = urllib.parse.urlparse(camera.stream_url)
        host = parsed.hostname
        port = parsed.port or (554 if parsed.scheme == "rtsp" else None)
        probe_url = (self.settings.stream_source_probe_url or "").strip()
        deadline = time.monotonic() + max(0.0, self.settings.stream_source_ready_timeout_seconds)

        while not state.stop_event.is_set():
            host_ready = True
            if host and port:
                try:
                    socket.gethostbyname(host)
                    with socket.create_connection((host, port), timeout=2):
                        pass
                except OSError as exc:
                    host_ready = False
                    self._set_stats(state, last_error=f"aguardando stream source: {exc}")

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
                    self._set_stats(state, last_error=f"aguardando stream probe: {exc}")

            if host_ready and probe_ready:
                return

            if time.monotonic() >= deadline:
                self.logger.warning(
                    "stream source not ready yet url=%s probe=%s, continuing with open attempts",
                    camera.stream_url,
                    probe_url or "-",
                )
                return

            time.sleep(min(1.0, self.settings.stream_reconnect_delay_seconds))

    def _open_capture(self, camera: CameraConfig):
        if camera.source_kind == "ip_engine":
            return IpEngineCapture(
                camera.stream_url, camera.camera_id, self.settings.ip_engine_token_file,
                interval=self.settings.stream_capture_interval_seconds,
                max_age=self.settings.ip_engine_max_frame_age_seconds,
            )
        if camera.source_kind in {"jetson_gst_usb", "usb"}:
            capture = cv2.VideoCapture(camera.device, cv2.CAP_V4L2)
            if self.settings.usb_camera_input_format.strip().lower() in {"mjpeg", "mjpg"}:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.usb_camera_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.usb_camera_height))
            capture.set(cv2.CAP_PROP_FPS, float(self.settings.usb_camera_fps))
            return capture

        if camera.source_kind == "gst_usb_sampled":
            capture = _GstUsbCapture(self.settings, camera, sampled=True)
            capture.open()
            return capture

        return cv2.VideoCapture(camera.stream_url, cv2.CAP_FFMPEG)

    def _update_preview(self, state: _CameraWorkerState, frame: np.ndarray, now_monotonic: float) -> None:
        if not self.settings.stream_preview_enabled:
            return
        if now_monotonic - state.last_preview_update_monotonic < self.settings.stream_preview_update_interval_seconds:
            return
        state.last_preview_update_monotonic = now_monotonic

        preview = frame
        max_width = max(0, self.settings.stream_preview_max_width)
        if max_width and preview.shape[1] > max_width:
            scale = max_width / float(preview.shape[1])
            target_height = max(1, int(preview.shape[0] * scale))
            preview = cv2.resize(preview, (max_width, target_height), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [
                int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.stream_preview_jpeg_quality),
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0,
            ],
        )
        if not ok:
            return

        with state.preview_lock:
            state.latest_preview_jpeg = encoded.tobytes()
            state.latest_preview_at = self._now()

    def _update_enrollment_frame(self, state: _CameraWorkerState, frame: np.ndarray, captured_at: str) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0],
        )
        if not ok:
            return
        with state.preview_lock:
            state.latest_enrollment_jpeg = encoded.tobytes()
            state.latest_enrollment_at = captured_at

    def _occlusion_log(self) -> Path:
        path = Path(self.settings.occlusion_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _classify_occlusion(self, recognition: dict) -> str:
        if recognition.get("status") == "matched" and recognition.get("subject"):
            return "known_subject_occluded"
        similarity = recognition.get("similarity")
        if similarity is not None and float(similarity) >= max(0.0, self.settings.face_similarity_threshold - 0.05):
            return "low_confidence_occlusion"
        return "unknown_subject_occluded"

    def _write_occlusion(self, entry: dict) -> None:
        with self._occlusion_lock:
            with self._occlusion_log().open("a", encoding="utf-8") as output:
                output.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _start_push_worker(self) -> None:
        if not self.settings.onix_push_enabled:
            return
        if not self.settings.onix_push_url.strip() or not self.settings.onix_push_token.strip():
            self.logger.warning("ONIX push enabled without ONIX_PUSH_URL or ONIX_PUSH_TOKEN")
            return
        if any(thread.is_alive() for thread in self._push_threads):
            return

        self._push_stop_event.clear()
        worker_count = max(1, min(16, int(self.settings.onix_push_workers)))
        self._push_threads = []
        for index in range(worker_count):
            thread = threading.Thread(target=self._push_loop, name=f"onix-push-worker-{index + 1}", daemon=True)
            thread.start()
            self._push_threads.append(thread)

    def _stop_push_worker(self) -> None:
        self._push_stop_event.set()
        for thread in self._push_threads:
            thread.join(timeout=3)

    def _enqueue_onix_push(self, kind: str, event: dict) -> None:
        if not self.settings.onix_push_enabled:
            return

        state = self._state_for_event(event)
        device_uid = self.settings.onix_push_device_uid.strip()
        if not device_uid:
            self._set_stats(state, last_push_error="ONIX_PUSH_DEVICE_UID ausente")
            return

        payload = {
            "device_uid": device_uid,
            "kind": kind,
            "event": event,
            "sent_at": self._now(),
        }

        try:
            self._push_queue.put_nowait(payload)
        except queue.Full:
            self._increment_stat(state, "push_errors")
            self._set_stats(state, last_push_error="fila de push Onix cheia")

    def _start_mqtt_worker(self) -> None:
        if not self.settings.mqtt_publish_enabled:
            return
        if not self.settings.mqtt_publish_host.strip():
            self.logger.warning("MQTT publish enabled without MQTT_PUBLISH_HOST")
            return
        if self._mqtt_thread and self._mqtt_thread.is_alive():
            return

        self._mqtt_stop_event.clear()
        self._mqtt_thread = threading.Thread(target=self._mqtt_loop, name="mqtt-publish-worker", daemon=True)
        self._mqtt_thread.start()

    def _stop_mqtt_worker(self) -> None:
        self._mqtt_stop_event.set()
        if self._mqtt_thread:
            self._mqtt_thread.join(timeout=3)
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.disconnect()
            except Exception:
                pass

    def _mqtt_debounce_key(self, kind: str, event: dict) -> tuple[str, str, str]:
        camera_id = str(event.get("camera_id") or event.get("camera") or "default")
        subject = str((event.get("recognition") or {}).get("subject") or "unknown")
        return (kind, camera_id, subject)

    def _is_mqtt_debounced(self, kind: str, event: dict) -> bool:
        window = self.settings.mqtt_debounce_seconds
        if window <= 0:
            return False
        key = self._mqtt_debounce_key(kind, event)
        now = time.monotonic()
        with self._mqtt_debounce_lock:
            last = self._mqtt_debounce.get(key)
            if last is not None and (now - last) < window:
                return True
            self._mqtt_debounce[key] = now
        return False

    def _enqueue_mqtt_publish(self, kind: str, event: dict) -> None:
        if not self.settings.mqtt_publish_enabled:
            return

        state = self._state_for_event(event)
        if self._is_mqtt_debounced(kind, event):
            self.logger.debug("mqtt debounce suprimiu kind=%s camera=%s subject=%s", kind, event.get("camera_id"), (event.get("recognition") or {}).get("subject"))
            return

        device_uid = self._device_uid_for_publish()
        if not device_uid:
            self._set_stats(state, last_mqtt_error="device_uid MQTT ausente")
            return

        payload = {
            "device_uid": device_uid,
            "kind": kind,
            "event": event,
            "sent_at": self._now(),
        }

        try:
            self._mqtt_queue.put_nowait(payload)
        except queue.Full:
            self._increment_stat(state, "mqtt_errors")
            self._set_stats(state, last_mqtt_error="fila MQTT cheia")

    def _mqtt_loop(self) -> None:
        while not self._mqtt_stop_event.is_set():
            try:
                payload = self._mqtt_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._publish_mqtt_payload(payload)
            finally:
                self._mqtt_queue.task_done()

    def _publish_mqtt_payload(self, payload: dict) -> None:
        state = self._state_for_event(payload.get("event") or {})
        try:
            client = self._ensure_mqtt_client()
            topic = self._mqtt_topic(str(payload.get("kind") or "event"), str(payload["device_uid"]))
            body = json.dumps(payload, ensure_ascii=True)
            result = client.publish(topic, body, qos=max(0, min(2, int(self.settings.mqtt_publish_qos))))
            result.wait_for_publish(timeout=5)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(mqtt.error_string(result.rc))
            self._increment_stat(state, "mqtt_sent")
            self._set_stats(state, last_mqtt_error=None)
        except Exception as exc:
            self._increment_stat(state, "mqtt_errors")
            self._set_stats(state, last_mqtt_error=str(exc))
            self.logger.warning("failed to publish Iris event to MQTT: %s", exc)
            try:
                if self._mqtt_client is not None:
                    self._mqtt_client.disconnect()
            except Exception:
                pass
            self._mqtt_client = None

    def _ensure_mqtt_client(self) -> mqtt.Client:
        if self._mqtt_client is not None:
            return self._mqtt_client

        client_id = self.settings.mqtt_publish_client_id.strip() or f"iris-{socket.gethostname()}"
        client = mqtt.Client(client_id=client_id, clean_session=True)
        username = self.settings.mqtt_publish_username.strip()
        if username:
            client.username_pw_set(username, self.settings.mqtt_publish_password)
        client.connect(
            self.settings.mqtt_publish_host,
            max(1, int(self.settings.mqtt_publish_port)),
            keepalive=max(5, int(self.settings.mqtt_publish_keepalive_seconds)),
        )
        client.loop_start()
        self._mqtt_client = client
        return client

    def _mqtt_topic(self, kind: str, device_uid: str) -> str:
        if kind == "occlusion":
            suffix = "occlusions"
        elif kind == "subjects":
            suffix = "subjects"
        else:
            suffix = "events"
        prefix = self.settings.mqtt_publish_topic_prefix.strip().strip("/") or "iris"
        return f"{prefix}/{device_uid}/{suffix}"

    def _device_uid_for_publish(self) -> str:
        return self.settings.onix_push_device_uid.strip() or self.settings.mqtt_publish_client_id.strip()

    def _publish_subjects_snapshot(self) -> None:
        if not self.settings.mqtt_publish_enabled:
            return
        device_uid = self._device_uid_for_publish()
        if not device_uid:
            return
        try:
            subjects_data = []
            for entry in self._subjects_with_primary_images():
                name = entry["subject"] if isinstance(entry, dict) else str(entry)
                image_url = entry.get("primary_image_url") if isinstance(entry, dict) else None
                subjects_data.append({"subject": name, "primary_image_url": image_url})
            payload = {
                "device_uid": device_uid,
                "kind": "subjects",
                "subjects": subjects_data,
                "sent_at": self._now(),
            }
            self._mqtt_queue.put_nowait(payload)
        except Exception as exc:
            self.logger.warning("failed to publish subjects snapshot to MQTT: %s", exc)

    def _push_loop(self) -> None:
        while not self._push_stop_event.is_set():
            try:
                payload = self._push_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._post_onix_payload(payload)
            finally:
                self._push_queue.task_done()

    def _post_onix_payload(self, payload: dict) -> None:
        state = self._state_for_event(payload.get("event") or {})
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Iris-Ingest-Token": self.settings.onix_push_token,
            "User-Agent": "iris-jetson-push/1.0",
        }
        attempts = max(1, int(self.settings.onix_push_max_retries))

        for attempt in range(1, attempts + 1):
            try:
                request = urllib.request.Request(
                    self.settings.onix_push_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=max(0.2, self.settings.onix_push_timeout_seconds)) as response:
                    status = int(response.status)
                    if 200 <= status < 300:
                        self._increment_stat(state, "push_sent")
                        self._set_stats(state, last_push_error=None)
                        return
                    raise RuntimeError(f"HTTP {status}")
            except Exception as exc:
                if attempt >= attempts:
                    self._increment_stat(state, "push_errors")
                    self._set_stats(state, last_push_error=str(exc))
                    self.logger.warning("failed to push Iris event to Onix: %s", exc)
                    return
                time.sleep(max(0.1, self.settings.onix_push_retry_delay_seconds))

    def _process_frame(self, frame: np.ndarray, camera: CameraConfig, state: _CameraWorkerState) -> None:
        capture_number = self._increment_stat(state, "captures_seen")
        captured_at = self._now()
        self._set_stats(state, last_capture_at=captured_at)
        self._update_enrollment_frame(state, frame, captured_at)

        try:
            with self._recognizer_lock:
                candidates = self.pipeline.detect_candidates(
                    frame, max_faces=self.settings.effective_detect_max_faces
                )
            self._observe_census(state, len(candidates))
            self._record_frame_census(camera, candidates, frame, captured_at, capture_number)
            if not candidates:
                state.tracker.expire(time.monotonic())
                self._increment_stat(state, "no_face_detected")
                self._set_stats(state, active_tracks=state.tracker.active_count, last_no_face_at=captured_at, last_error=None)
                return

            track_candidates = []
            for candidate in candidates:
                left, top, right, bottom = candidate["bbox"]
                if min(right - left, bottom - top) < self.settings.pipeline_track_min_face_size:
                    # Visto e contado, mas pequeno demais para reconhecer: e a
                    # pessoa distante que atravessa o quadro sem angulo util.
                    self._increment_stat(state, "faces_below_track_size")
                    continue
                track_candidates.append(TrackCandidate(
                    bbox=list(candidate["bbox"]),
                    rank=self.pipeline.candidate_rank(candidate),
                    payload=candidate,
                    frame=frame,
                    captured_at=captured_at,
                    capture_number=capture_number,
                ))
            if not track_candidates:
                state.tracker.expire(time.monotonic())
                self._set_stats(state, active_tracks=state.tracker.active_count, last_error=None)
                return

            for _ in track_candidates:
                self._increment_stat(state, "faces_detected")
            ready_tracks = state.tracker.update(track_candidates, time.monotonic())
            started = state.tracker.last_created_count
            if started:
                for _ in range(started):
                    self._increment_stat(state, "tracks_started")
            self._set_stats(state, active_tracks=state.tracker.active_count, last_error=None)

            for track in self._within_budget(ready_tracks, state):
                self._increment_stat(state, "recognition_attempts")
                self._increment_stat(state, "faces_materialized")
                completed = self._process_track(track, camera, state)
                state.tracker.mark_attempt(track.track_id, time.monotonic(), completed=completed)
        except NoFaceDetectedError:
            self._increment_stat(state, "no_face_detected")
            self._set_stats(state, last_no_face_at=captured_at, last_error=None)
        except ValueError as exc:
            self._increment_stat(state, "recognition_errors")
            self._set_stats(state, last_error=str(exc))
        except Exception as exc:
            self._increment_stat(state, "recognition_errors")
            self._set_stats(state, last_error=str(exc))
            self.logger.exception("failed to process stream frame camera=%s: %s", camera.camera_id, exc)

    def _event_persistence_gate(
        self,
        recognition: dict,
        detection,
        camera: CameraConfig,
        state: _CameraWorkerState,
    ) -> str | None:
        """Decide se vale gravar este evento. `None` grava.

        Cada evento persistido custa uma linha de JSONL e dois JPEGs, entao o
        portao vem ANTES de `_save_capture`, nao depois. Eventos de oclusao nao
        chegam aqui: o ramo deles retorna antes.

        Dois criterios, deliberadamente diferentes:

        `matched` sai por **debounce** de (camera, sujeito). Uma pessoa parada
        na frente da camera gerava um evento por track fechado: medido em 3,5
        meses, 101.489 eventos de uma pessoa so. O centesimo milesimo nao diz
        nada que o primeiro nao dissesse.

        `no_match` **nunca** sai por debounce, porque o `subject` e o mesmo
        `None` para todo mundo e uma pessoa suprimiria outra. Sai por qualidade
        do rosto: 84,8% dos no_match tem 48-64 px e det_score medíocre, nao
        servem para reconhecer ninguem depois, e na fase de recorrencia seriam a
        origem das fusoes erradas de identidade.

        Devolve `"debounce"` ou `"quality"` quando suprime, para que o chamador
        escolha se o track esta encerrado ou se vale tentar de novo.
        """
        status = recognition.get("status")

        if status == "matched":
            window = float(self.settings.event_debounce_seconds)
            subject = str(recognition.get("subject") or "")
            if window <= 0 or subject == "":
                return None
            key = (camera.camera_id, subject)
            now = time.monotonic()
            with self._event_debounce_lock:
                last = self._event_debounce.get(key)
                if last is not None and (now - last) < window:
                    self._increment_stat(state, "events_suppressed_debounce")
                    return "debounce"
                self._event_debounce[key] = now
            return None

        min_width = int(self.settings.event_unmatched_min_width)
        min_score = float(self.settings.event_unmatched_min_det_score)
        if min_width <= 0 and min_score <= 0:
            return None

        width = float(detection.bbox[2]) - float(detection.bbox[0])
        if width < min_width or float(detection.det_score) < min_score:
            self._increment_stat(state, "events_suppressed_quality")
            return "quality"
        return None

    def _observe_census(self, state: _CameraWorkerState, detected: int) -> None:
        """Registra quantos rostos o detector viu neste frame, materializados ou nao."""
        with state.stats_lock:
            state.stats["faces_seen_total"] += detected
            state.stats["faces_in_last_frame"] = detected
            if detected > state.stats["faces_in_frame_peak"]:
                state.stats["faces_in_frame_peak"] = detected

    def _record_frame_census(
        self,
        camera: CameraConfig,
        candidates: list[dict],
        frame: np.ndarray,
        captured_at: str,
        capture_number: int,
    ) -> None:
        """Persiste o censo deste frame em `frame_census`.

        Roda a ROI da camera (se houver) so aqui -- o tracking, o
        reconhecimento e a oclusao logo abaixo continuam recebendo TODOS os
        `candidates`, sem filtro nenhum. Ligar uma ROI muda o que o contador
        novo reporta, nunca o que o pipeline ja calibrado faz.
        """
        roi = camera.roi
        if roi is not None:
            height, width = frame.shape[:2]
            face_count = sum(
                1 for candidate in candidates
                if bbox_center_inside(candidate["bbox"], roi, width, height)
            )
        else:
            face_count = len(candidates)
        try:
            self.census_store.record(
                camera_id=camera.camera_id,
                captured_at=captured_at,
                face_count=face_count,
                capture_number=capture_number,
                roi_applied=roi is not None,
                source="worker",
            )
        except Exception as exc:
            # O censo e observabilidade, nao contrato: uma falha de escrita
            # aqui nao pode derrubar o frame que ja processou reconhecimento
            # e oclusao normalmente.
            self.logger.warning("failed to persist frame census camera=%s: %s", camera.camera_id, exc)

    def _within_budget(self, ready_tracks: list[FaceTrack], state: _CameraWorkerState) -> list[FaceTrack]:
        """Materializa ate `pipeline_materialize_budget` tracks neste frame.

        Landmarks e embedding custam ~21 ms por rosto e sao o unico item do
        pipeline que escala com a quantidade de gente; a deteccao e plana.  Sem
        teto, um frame com muitos tracks prontos estoura o orcamento de frame e
        a perda vira nao deterministica.  Com teto, o excedente simplesmente nao
        recebe `mark_attempt` e continua pronto no frame seguinte: como o track
        vive `ttl_seconds`, ele tem varias chances antes de expirar.

        Ordem: quem nunca foi tentado primeiro, depois o de maior `rank` (rosto
        maior, mais nitido e com score melhor).  Isso evita que um track novo
        fique preso atras de retentativas.
        """
        selecionados, adiados = select_within_budget(
            ready_tracks, self.settings.effective_materialize_budget
        )
        if adiados:
            with state.stats_lock:
                state.stats["materialize_deferred"] += adiados
                state.stats["frames_over_budget"] += 1
        return selecionados

    def _process_track(self, track: FaceTrack, camera: CameraConfig, state: _CameraWorkerState) -> bool:
        """Analyze the best retained candidate once; return whether the track is final."""
        # Identity of the evidence frame, not of the frame that closed the
        # track: the two are up to ttl_seconds apart and the JPEG that reaches
        # ViewCare must match the timestamp reported with it.
        captured_at = track.best.captured_at
        capture_number = track.best.capture_number
        with self._recognizer_lock:
            detection, quality, recognition = self.pipeline.analyze_candidate(track.best.frame, track.best.payload)
        face_score = float(detection.metadata.get("det_score") or 0.0)
        current_landmarks = int(detection.metadata.get("landmarks_detected") or 0)
        visual_occlusion = (recognition.get("face") or {}).get("visual_occlusion") or {}
        event_id = (
            f"{camera.camera_id}_{captured_at.replace(':', '').replace('+', 'Z')}_"
            f"{capture_number:08d}_t{track.track_id:03d}"
        )
        quality_reason = quality.get("reason")
        visual_occlusion_attempt = bool(
            visual_occlusion.get("suspected") and quality_reason in {None, "det_score_baixo"}
        )
        track_info = {
            "id": track.track_id,
            "frames_seen": track.frames_seen,
            "best_rank": round(track.best.rank, 4),
            "first_seen_monotonic": round(track.first_seen, 4),
        }

        if quality_reason == "face_ocluida" or visual_occlusion_attempt:
            sudden = track.last_landmarks >= 80 and current_landmarks < 40
            image_path = self._save_capture(event_id, track.best.frame)
            face_path = self._save_face_crop(event_id, detection.crop) if self.settings.pipeline_save_face_crop else None
            occlusion_reason = quality_reason or "suspected_visual_occlusion"
            occlusion_event = {
                "event_id": event_id,
                "capture_id": event_id,
                "capture_number": capture_number,
                "event_type": "occlusion",
                "occlusion_reason": occlusion_reason,
                "occlusion_class": self._classify_occlusion(recognition),
                "camera": camera.camera_id,
                "captured_at": captured_at,
                "camera_id": camera.camera_id,
                "track": track_info,
                "image_path": str(image_path),
                "image_url": f"/captures/{event_id}/image",
                "frame_image_url": f"/occlusions/{event_id}/frame.jpg",
                "face_image_path": str(face_path) if face_path else None,
                "face_image_url": f"/occlusions/{event_id}/face.jpg" if face_path else None,
                "detector": {**self.detector_status(), "score": detection.det_score, "bbox": detection.bbox},
                "quality": quality,
                "recognition": recognition,
                "occlusion": {
                    "landmarks_detected": current_landmarks,
                    "total_landmarks": detection.metadata.get("total_landmarks", 0),
                    "occlusion_ratio": detection.metadata.get("occlusion_ratio", 1.0),
                    "det_score": detection.det_score,
                    "bbox": detection.bbox,
                    "sudden": sudden,
                    "previous_landmarks": track.last_landmarks,
                    "visual": visual_occlusion,
                },
            }
            self._write_occlusion(occlusion_event)
            self._enqueue_onix_push("occlusion", occlusion_event)
            self._enqueue_mqtt_publish("occlusion", occlusion_event)
            self._increment_stat(state, "occlusions_written")
            self._set_stats(state, last_error=occlusion_reason, last_occlusion_at=captured_at, last_event_at=captured_at)
            track.last_landmarks = 0
            return True

        if face_score < self.settings.stream_min_face_score or not quality.get("accepted"):
            track.last_landmarks = 0
            return False

        track.last_landmarks = current_landmarks

        suppressed = self._event_persistence_gate(recognition, detection, camera, state)
        if suppressed == "debounce":
            # Ja sabemos quem e e acabamos de registrar: o track esta resolvido.
            return True
        if suppressed == "quality":
            # Pode melhorar se a pessoa se aproximar, entao o track continua
            # elegivel em vez de ser encerrado com um rosto ruim.
            return False

        image_path = self._save_capture(event_id, track.best.frame)
        face_path = self._save_face_crop(event_id, detection.crop) if self.settings.pipeline_save_face_crop else None
        if capture_number % 100 == 0:
            self._prune_captures()
        event = {
            "event_id": event_id,
            "capture_id": event_id,
            "capture_number": capture_number,
            "camera": camera.camera_id,
            "camera_id": camera.camera_id,
            "captured_at": captured_at,
            "track": track_info,
            "image_path": str(image_path),
            "image_url": f"/captures/{event_id}/image",
            "frame_image_url": f"/events/{event_id}/frame.jpg",
            "face_image_path": str(face_path) if face_path else None,
            "face_image_url": f"/events/{event_id}/face.jpg" if face_path else None,
            "detector": {**self.detector_status(), "score": detection.det_score, "bbox": detection.bbox},
            "quality": quality,
            "recognition": recognition,
        }
        self._write_event(event)
        self._enqueue_onix_push("event", event)
        self._enqueue_mqtt_publish("event", event)
        state.latest_event = event
        self._increment_stat(state, "events_written")
        self._set_stats(state, last_event_at=captured_at, last_error=None)
        self.logger.info(
            "event=%s track=%s status=%s subject=%s similarity=%s",
            event_id,
            track.track_id,
            recognition.get("status"),
            recognition.get("subject"),
            recognition.get("similarity"),
        )
        return True

    def _prune_captures(self) -> None:
        with self._retention_lock:
            cutoff = None
            retention_hours = self.settings.stream_capture_retention_hours
            if retention_hours and retention_hours > 0:
                cutoff = time.time() - retention_hours * 3600.0

            max_files = self.settings.stream_max_capture_files
            if cutoff is None and max_files <= 0:
                return

            for directory, label in ((self._capture_dir(), "capture"), (self._face_dir(), "face crop")):
                files = []
                for item in directory.glob("*.jpg"):
                    try:
                        files.append((item, item.stat().st_mtime))
                    except FileNotFoundError:
                        continue
                files.sort(key=lambda entry: entry[1], reverse=True)
                for index, (old_file, modified_at) in enumerate(files):
                    too_old = cutoff is not None and modified_at < cutoff
                    over_limit = max_files > 0 and index >= max_files
                    if not too_old and not over_limit:
                        continue
                    try:
                        old_file.unlink(missing_ok=True)
                    except OSError:
                        self.logger.warning("failed to remove old %s %s", label, old_file)

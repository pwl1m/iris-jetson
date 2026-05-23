import json
import logging
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import requests

from .recognizer import InsightFaceRecognizer, cosine_similarity, decode_image
from .settings import Settings
from .storage import FaceStore


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
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._mqtt_loop, name="mqtt-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def enroll(self, subject: str, filename: str | None, payload: bytes) -> dict:
        image = decode_image(payload)
        embedding, metadata = self.recognizer.extract_best(image)
        embedding_id = self.store.add_embedding(subject=subject, embedding=embedding, source=filename)
        return {"status": "enrolled", "subject": subject, "embedding_id": embedding_id, "face": metadata}

    def recognize_bytes(self, payload: bytes) -> dict:
        image = decode_image(payload)
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
        return {
            "status": "ok",
            "model": self.settings.face_model_name,
            "threshold": self.settings.face_similarity_threshold,
            "providers": self.settings.providers_list,
            "mqtt_topic": self.settings.mqtt_topic,
        }

    def _event_log(self) -> Path:
        path = Path(self.settings.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_event(self, event: dict) -> None:
        with self._event_log().open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=True) + "\n")

    def _should_process(self, payload: dict) -> bool:
        if payload.get("type") not in self.settings.recognition_event_types_set:
            return False
        after = payload.get("after") or {}
        return after.get("label") == "person"

    def _recognize_snapshot(self, event_id: str) -> dict:
        snapshot_url = f"{self.settings.frigate_url}/api/events/{event_id}/snapshot.jpg"
        snapshot = requests.get(snapshot_url, timeout=10)
        snapshot.raise_for_status()
        return self.recognize_bytes(snapshot.content)

    def _on_message(self, client: mqtt.Client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if not self._should_process(payload):
                return
            after = payload.get("after") or {}
            event_id = after.get("id")
            if not event_id:
                return

            result = self._recognize_snapshot(event_id)
            event = {
                "event_id": event_id,
                "camera": after.get("camera"),
                "timestamp": after.get("start_time"),
                "recognition": result,
            }
            self._write_event(event)
            client.publish("facial/recognitions", json.dumps(event, ensure_ascii=True), retain=False)
            self.logger.info(
                "recognition event_id=%s status=%s subject=%s",
                event_id,
                result.get("status"),
                result.get("subject"),
            )
        except Exception as exc:
            self.logger.exception("failed to process mqtt event: %s", exc)

    def _mqtt_loop(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.settings.worker_log_level),
            format="%(asctime)s %(levelname)s %(message)s",
        )
        while not self._stop_event.is_set():
            try:
                client = mqtt.Client()
                client.on_message = self._on_message
                client.connect(self.settings.mqtt_host, self.settings.mqtt_port, 60)
                client.subscribe(self.settings.mqtt_topic)
                self.logger.info(
                    "listening mqtt://%s:%s topic=%s",
                    self.settings.mqtt_host,
                    self.settings.mqtt_port,
                    self.settings.mqtt_topic,
                )
                while not self._stop_event.is_set():
                    client.loop(timeout=1.0)
                client.disconnect()
            except Exception as exc:
                self.logger.exception("mqtt loop failed: %s", exc)
                time.sleep(5)


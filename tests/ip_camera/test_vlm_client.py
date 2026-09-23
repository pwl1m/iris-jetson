import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vlm_client", ROOT / "iris-app/app/vlm_client.py")
vlm_client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_client)


class VlmClientTests(unittest.TestCase):
    def settings(self):
        return SimpleNamespace(
            event_log_path="/data/events/recognitions.jsonl",
            vlm_shared_event_root="/sources/iris-events",
            vlm_prompt_id="iris-scene-v1",
            vlm_priority=100,
            vlm_job_max_attempts=3,
            vlm_callback_url="",
        )

    def test_builds_deduplicated_job_for_shared_capture(self):
        payload = vlm_client.build_vlm_job(
            self.settings(),
            "event",
            {
                "event_id": "entrada_001",
                "camera_id": "entrada",
                "captured_at": "2026-09-22T12:00:00+00:00",
                "image_path": "/data/events/captures/entrada_001.jpg",
                "recognition": {"status": "matched", "subject": "not-forwarded"},
            },
        )
        self.assertEqual(payload["image_path"], "/sources/iris-events/captures/entrada_001.jpg")
        self.assertEqual(payload["idempotency_key"], "iris:event:entrada_001:iris-scene-v1")
        self.assertEqual(payload["metadata"]["recognition_status"], "matched")
        self.assertEqual(payload["metadata"]["subject"], "not-forwarded")

    def test_rejects_path_outside_event_volume(self):
        with self.assertRaises(ValueError):
            vlm_client.build_vlm_job(
                self.settings(),
                "event",
                {"event_id": "bad", "image_path": "/tmp/outside.jpg"},
            )


if __name__ == "__main__":
    unittest.main()

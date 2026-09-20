import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_tracker():
    spec = importlib.util.spec_from_file_location("face_tracking", ROOT / "iris-app/app/face_tracking.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tracking = load_tracker()


def candidate(box, rank=1.0, captured_at="2026-09-19T00:00:00+00:00", capture_number=1):
    return tracking.TrackCandidate(box, rank, {"box": box}, object(), captured_at, capture_number)


class FaceTrackTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(tracking.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(tracking.iou([0, 0, 0, 0], [0, 0, 1, 1]), 0.0)

    def test_best_frame_is_retained_and_single_track_becomes_ready(self):
        manager = tracking.FaceTrackManager(max_tracks=3, min_frames=2, retry_seconds=1.0)
        self.assertEqual(manager.update([candidate([0, 0, 20, 20], 0.7)], 0.0), [])
        ready = manager.update([candidate([1, 0, 21, 20], 0.9)], 0.1)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].track_id, 1)
        self.assertEqual(ready[0].best.rank, 0.9)
        manager.mark_attempt(1, 0.1, completed=True)
        self.assertEqual(manager.ready(1.2), [])

    def test_multiple_faces_are_independent_and_bounded(self):
        manager = tracking.FaceTrackManager(max_tracks=2, min_frames=1)
        ready = manager.update([
            candidate([0, 0, 20, 20]),
            candidate([50, 0, 70, 20]),
            candidate([100, 0, 120, 20]),
        ], 0.0)
        self.assertEqual({track.track_id for track in ready}, {1, 2})
        self.assertEqual(manager.active_count, 2)

    def test_best_candidate_carries_its_own_capture_identity(self):
        # The event must be stamped with the evidence frame, not with the later
        # frame that happened to close the track.
        manager = tracking.FaceTrackManager(max_tracks=1, min_frames=2, retry_seconds=1.0)
        manager.update([candidate([0, 0, 20, 20], 0.9, "2026-09-19T00:00:00+00:00", 10)], 0.0)
        ready = manager.update([candidate([1, 0, 21, 20], 0.4, "2026-09-19T00:00:01+00:00", 20)], 0.5)
        self.assertEqual(ready[0].best.captured_at, "2026-09-19T00:00:00+00:00")
        self.assertEqual(ready[0].best.capture_number, 10)

    def test_expiry_allows_new_person(self):
        manager = tracking.FaceTrackManager(max_tracks=1, min_frames=1, ttl_seconds=1.0)
        manager.update([candidate([0, 0, 20, 20])], 0.0)
        manager.expire(1.1)
        ready = manager.update([candidate([50, 0, 70, 20])], 1.1)
        self.assertEqual(ready[0].track_id, 2)


if __name__ == "__main__":
    unittest.main()

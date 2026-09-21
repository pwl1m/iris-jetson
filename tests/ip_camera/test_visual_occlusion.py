"""Occlusion heuristic: coverage floor, scale invariance and exposure context.

The production history of the USB line (1997 occlusion events, 372230 analyzed
faces) showed the heuristic never ran on 99.4% of faces because its size floor
was 96 px while 92.6% of real faces are 48-64 px wide.  These tests pin the
properties that made lowering the floor safe.
"""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def settings_defaults():
    """Read Settings defaults straight from the source.

    pydantic-settings is not installed on the host, and the point here is to
    assert the values the deploy actually ships, not a copy of them.
    """
    tree = ast.parse((ROOT / "iris-app/app/settings.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Settings")
    values = {}
    for node in klass.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return types.SimpleNamespace(**values)


def load_module():
    # Stub the package so the relative imports resolve without pulling in
    # insightface, fastapi, pydantic or the runtime.
    pkg = types.ModuleType("irisapp")
    pkg.__path__ = [str(ROOT / "iris-app/app")]
    sys.modules["irisapp"] = pkg
    spec = importlib.util.spec_from_file_location("irisapp.schemas", ROOT / "iris-app/app/schemas.py")
    schemas = importlib.util.module_from_spec(spec)
    sys.modules["irisapp.schemas"] = schemas
    spec.loader.exec_module(schemas)
    sys.modules["irisapp.settings"] = types.SimpleNamespace(Settings=settings_defaults)
    spec = importlib.util.spec_from_file_location("irisapp.visual_occlusion", ROOT / "iris-app/app/visual_occlusion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, settings_defaults, schemas


occlusion, Settings, schemas = load_module()


def detection(crop, det_score=0.80):
    height, width = crop.shape[:2]
    return schemas.DetectionResult(
        bbox=[0.0, 0.0, float(width), float(height)],
        det_score=det_score,
        crop=crop,
        embedding=np.zeros(512, dtype=np.float32),
        metadata={"det_score": det_score},
    )


def face_like(side, skin=(150, 170, 210), lower=None, cover_from=0.45):
    """A flat skin-toned square, optionally covered from `cover_from` down."""
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    crop[:, :] = skin
    if lower is not None:
        crop[int(side * cover_from):, :] = lower
    return crop


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def test_floor_follows_the_recognition_minimum(self):
        # It is incoherent to accept a face for identity but refuse to check it
        # for occlusion; 92.6% of production faces are 48-64 px wide.
        self.assertEqual(self.settings.visual_occlusion_min_width, self.settings.face_min_width)
        self.assertEqual(self.settings.visual_occlusion_min_height, self.settings.face_min_height)

    def test_a_48px_face_is_evaluated_not_skipped(self):
        result = occlusion.evaluate_visual_occlusion(
            self.settings, detection(face_like(48)), {}, {"status": "matched", "similarity": 0.9}
        )
        self.assertNotIn("face_too_small_for_visual_occlusion", result["signals"])
        self.assertIn("skin_ratio", result["metrics"])

    def test_small_faces_are_measured_but_never_convicted(self):
        # The floor dropped from 96 to 64 on 2026-09-21, measured in the real IP
        # scene: 0% of faces below 48 px, 51.1% in 48-64, 23.4% in 64-96, 25.5%
        # above 96.  The 48-64 band is still half the traffic and has no labelled
        # occlusion case yet, so it keeps being measured without being convicted.
        covered = face_like(56, lower=(10, 10, 10), cover_from=0.2)
        result = occlusion.evaluate_visual_occlusion(
            self.settings, detection(covered, det_score=0.60), {}, {"status": "no_match", "similarity": None}
        )
        self.assertGreaterEqual(result["score"], self.settings.visual_occlusion_score_threshold)
        self.assertFalse(result["suspected"])
        self.assertIn("below_verdict_floor", result["signals"])

    def test_the_floor_follows_what_the_scene_supports(self):
        # 64, nao 96: ver o comentario em settings.py.
        self.assertEqual(self.settings.visual_occlusion_verdict_min_size, 64)

    def test_a_face_just_above_the_new_floor_can_be_convicted(self):
        # Esta e a faixa que a mudanca de 96 para 64 passou a cobrir.
        covered = face_like(72, lower=(10, 10, 10), cover_from=0.2)
        result = occlusion.evaluate_visual_occlusion(
            self.settings, detection(covered, det_score=0.60), {}, {"status": "no_match", "similarity": None}
        )
        self.assertNotIn("below_verdict_floor", result["signals"])
        self.assertTrue(result["suspected"])

    def test_above_the_verdict_floor_the_score_still_decides(self):
        covered = face_like(128, lower=(10, 10, 10), cover_from=0.2)
        result = occlusion.evaluate_visual_occlusion(
            self.settings, detection(covered, det_score=0.60), {}, {"status": "no_match", "similarity": None}
        )
        self.assertNotIn("below_verdict_floor", result["signals"])
        self.assertTrue(result["suspected"])

    def test_below_the_floor_still_reports_why(self):
        result = occlusion.evaluate_visual_occlusion(
            self.settings, detection(face_like(32)), {}, {"status": "matched", "similarity": 0.9}
        )
        self.assertEqual(result["signals"], ["face_too_small_for_visual_occlusion"])
        self.assertFalse(result["suspected"])


class ScaleInvarianceTests(unittest.TestCase):
    """Measured on production crops: metrics hold from 48 px up."""

    def setUp(self):
        self.settings = Settings()

    def metrics_at(self, crop, side):
        resized = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
        return occlusion.evaluate_visual_occlusion(
            self.settings, detection(resized), {}, {"status": "no_match", "similarity": None}
        )["metrics"]

    def test_covered_lower_face_scores_the_same_at_every_size(self):
        crop = face_like(256, lower=(20, 20, 20))
        values = [self.metrics_at(crop, side) for side in (48, 64, 96, 128)]
        for other in values[1:]:
            self.assertAlmostEqual(values[0]["skin_ratio"], other["skin_ratio"], delta=0.05)
            self.assertAlmostEqual(values[0]["dark_lower_ratio"], other["dark_lower_ratio"], delta=0.05)

    def test_clear_and_covered_faces_stay_separated_at_48px(self):
        clear = self.metrics_at(face_like(256), 48)
        covered = self.metrics_at(face_like(256, lower=(20, 20, 20)), 48)
        self.assertGreater(clear["skin_ratio"], covered["skin_ratio"])
        self.assertGreater(covered["dark_lower_ratio"], clear["dark_lower_ratio"])


class ExposureContextTests(unittest.TestCase):
    """The dark threshold is absolute, so backlight scores like a cover."""

    def setUp(self):
        self.settings = Settings()

    def evaluate(self, crop):
        return occlusion.evaluate_visual_occlusion(
            self.settings, detection(crop), {}, {"status": "no_match", "similarity": None}
        )

    def test_asymmetry_separates_a_cover_from_a_dark_frame(self):
        covered = self.evaluate(face_like(128, lower=(20, 20, 20)))["metrics"]
        underexposed = self.evaluate(face_like(128, skin=(20, 22, 26)))["metrics"]
        # A cover darkens only the lower face; backlight darkens both equally.
        self.assertGreater(covered["dark_asymmetry"], 0.3)
        self.assertAlmostEqual(underexposed["dark_asymmetry"], 0.0, delta=0.05)

    def test_luminance_is_reported_so_the_two_can_be_told_apart(self):
        bright = self.evaluate(face_like(128))["metrics"]
        dark = self.evaluate(face_like(128, skin=(20, 22, 26)))["metrics"]
        self.assertGreater(bright["crop_luminance"], dark["crop_luminance"])
        self.assertLess(dark["crop_luminance"], 55)

    def test_landmark_channel_is_inert_with_a_regressor(self):
        # 1997 of 1997 production events reported 106/106 landmarks; the +1.0
        # landmark_occlusion signal is unreachable with landmark_2d_106.
        result = occlusion.evaluate_visual_occlusion(
            self.settings,
            detection(face_like(128, lower=(20, 20, 20))),
            {"reason": None},
            {"status": "no_match", "face": {"landmarks_detected": 106, "total_landmarks": 106, "occluded": False}},
        )
        self.assertNotIn("landmark_occlusion", result["signals"])
        self.assertEqual(result["metrics"]["landmark_ratio"], 1.0)


class PoseAndBlurObservabilityTests(unittest.TestCase):
    """Pose and blur ride along as metrics, never as score.

    Measured on the labelled set in the operating range (bbox >= 96 px):
    |yaw| median was 1.457 on the 23 production-labelled events against 0.512 on
    the 10 real negatives, and 0.274 on the 35 curated garment-occlusion frames.
    So the production "occlusion" log is mostly profile pose, while a real cover
    is frontal.  Blur told the same story: 53.9 against 95.8.  Neither is wired
    into the score until Monday's labelled trial says where the cut is.
    """

    def setUp(self):
        self.settings = Settings()

    def evaluate(self, crop, quality=None, pose=None):
        face = {"landmarks_detected": 106, "total_landmarks": 106}
        if pose is not None:
            face["pose"] = pose
        return occlusion.evaluate_visual_occlusion(
            self.settings,
            detection(crop),
            quality if quality is not None else {"reason": None},
            {"status": "matched", "similarity": 0.9, "face": face},
        )

    def test_pose_is_reported_when_the_detector_supplies_it(self):
        pose = {"yaw": -1.42, "pitch": 0.51, "roll": 3.2, "eye_nose_asymmetry": 0.77}
        metrics = self.evaluate(face_like(128), pose=pose)["metrics"]
        self.assertEqual(metrics["yaw"], -1.42)
        self.assertEqual(metrics["pitch"], 0.51)
        self.assertEqual(metrics["roll"], 3.2)
        self.assertEqual(metrics["eye_nose_asymmetry"], 0.77)

    def test_pose_absent_degrades_to_none_not_to_an_error(self):
        metrics = self.evaluate(face_like(128))["metrics"]
        self.assertIsNone(metrics["yaw"])
        self.assertIsNone(metrics["eye_nose_asymmetry"])

    def test_blur_comes_from_the_quality_pass_already_computed(self):
        metrics = self.evaluate(face_like(128), quality={"reason": None, "blur": 70.53})["metrics"]
        self.assertEqual(metrics["blur"], 70.5)

    def test_a_profile_face_does_not_score_higher_for_being_a_profile(self):
        # The whole point of keeping pose out of the score: a turned head is a
        # pose problem, not a covered face, and the log must not conflate them.
        crop = face_like(128)
        frontal = self.evaluate(crop, pose={"yaw": 0.01, "pitch": 0.5, "roll": 0.0, "eye_nose_asymmetry": 0.02})
        profile = self.evaluate(crop, pose={"yaw": 1.46, "pitch": 0.5, "roll": 6.0, "eye_nose_asymmetry": 0.81})
        self.assertEqual(frontal["score"], profile["score"])
        self.assertEqual(frontal["suspected"], profile["suspected"])


class HeadPoseTests(unittest.TestCase):
    """_head_pose derives from the 5 keypoints the detector already returns."""

    @staticmethod
    def pose_module():
        spec = importlib.util.spec_from_file_location(
            "irisapp.recognizer", ROOT / "iris-app/app/recognizer.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def face_with(kps):
        return types.SimpleNamespace(kps=np.asarray(kps, dtype=np.float32))

    def setUp(self):
        self.module = self.pose_module()

    def test_a_frontal_face_has_yaw_near_zero(self):
        # eyes symmetric about the nose, mouth below it
        pose = self.module._head_pose(self.face_with(
            [[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]]
        ))["pose"]
        self.assertAlmostEqual(pose["yaw"], 0.0, delta=0.05)
        self.assertLess(pose["eye_nose_asymmetry"], 0.05)

    def test_a_turned_head_moves_the_nose_off_the_eye_midpoint(self):
        pose = self.module._head_pose(self.face_with(
            [[40, 50], [80, 50], [76, 70], [50, 90], [78, 90]]
        ))["pose"]
        self.assertGreater(abs(pose["yaw"]), 0.3)
        self.assertGreater(pose["eye_nose_asymmetry"], 0.5)

    def test_roll_follows_the_eye_line(self):
        pose = self.module._head_pose(self.face_with(
            [[40, 40], [80, 60], [60, 70], [45, 90], [75, 90]]
        ))["pose"]
        self.assertGreater(pose["roll"], 20)

    def test_missing_or_malformed_keypoints_return_none(self):
        self.assertIsNone(self.module._head_pose(types.SimpleNamespace(kps=None))["pose"])
        self.assertIsNone(self.module._head_pose(types.SimpleNamespace())["pose"])
        self.assertIsNone(
            self.module._head_pose(self.face_with([[1, 2], [3, 4]]))["pose"]
        )

    def test_degenerate_eyes_do_not_divide_by_zero(self):
        pose = self.module._head_pose(self.face_with(
            [[50, 50], [50, 50], [60, 70], [45, 90], [75, 90]]
        ))
        self.assertIsNone(pose["pose"])


if __name__ == "__main__":
    unittest.main()

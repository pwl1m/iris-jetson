"""The first real face must not pay for a cold TensorRT engine build."""
import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recognizer = load("iris_recognizer", ROOT / "iris-app/app/recognizer.py")


class WarmUpTests(unittest.TestCase):
    def build(self, det_size=(480, 480)):
        # No models are loaded: warm_up only has to drive every stage once.
        instance = object.__new__(recognizer.InsightFaceRecognizer)
        instance.det_size = det_size
        instance.detect_candidates = Mock(return_value=[])
        instance.inspect_candidate = Mock(return_value={})
        return instance

    def test_every_stage_runs_once_and_is_timed(self):
        instance = self.build()
        report = instance.warm_up()
        self.assertEqual(instance.detect_candidates.call_count, 1)
        self.assertEqual(instance.inspect_candidate.call_count, 1)
        self.assertEqual(sorted(report), ["detection_seconds", "landmark_and_recognition_seconds"])
        self.assertTrue(all(value >= 0 for value in report.values()))

    def test_synthetic_candidate_satisfies_inspect_candidate(self):
        # inspect_candidate rejects anything that is not (4,) bbox + (5, 2) kps,
        # so a malformed warm-up would silently skip the models it must build.
        instance = self.build()
        instance.warm_up()
        image, candidate = instance.inspect_candidate.call_args[0]
        self.assertEqual(np.asarray(candidate["bbox"], dtype=np.float32).shape, (4,))
        self.assertEqual(np.asarray(candidate["kps"], dtype=np.float32).shape, (5, 2))
        left, top, right, bottom = candidate["bbox"]
        self.assertLess(right, image.shape[1])
        self.assertLess(bottom, image.shape[0])
        self.assertGreater(recognizer.InsightFaceRecognizer._crop(
            instance, image, candidate["bbox"]).size, 0)

    def test_small_det_size_still_fits_the_alignment_crop(self):
        instance = self.build(det_size=(128, 128))
        instance.warm_up()
        image = instance.inspect_candidate.call_args[0][0]
        self.assertGreaterEqual(min(image.shape[:2]), 256)

    def test_runtime_warms_up_before_starting_the_stream(self):
        tree = ast.parse((ROOT / "iris-app/app/iris_runtime.py").read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IrisRuntime")
        start = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "start")
        calls = [
            node.func.attr for node in ast.walk(start)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("_warm_up_models", calls)
        self.assertLess(calls.index("_warm_up_models"), calls.index("start_stream"))


if __name__ == "__main__":
    unittest.main()

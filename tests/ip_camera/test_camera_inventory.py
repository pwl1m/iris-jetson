import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_cameras():
    package = types.ModuleType("app")
    package.__path__ = []
    sys.modules["app"] = package
    schemas = types.ModuleType("app.schemas")
    schemas.CameraConfig = object
    sys.modules["app.schemas"] = schemas
    settings = types.ModuleType("app.settings")
    settings.Settings = object
    sys.modules["app.settings"] = settings
    roi_spec = importlib.util.spec_from_file_location("app.roi", ROOT / "iris-app/app/roi.py")
    roi = importlib.util.module_from_spec(roi_spec)
    sys.modules["app.roi"] = roi
    roi_spec.loader.exec_module(roi)
    spec = importlib.util.spec_from_file_location("app.cameras", ROOT / "iris-app/app/cameras.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cameras = load_cameras()


class CameraInventoryTests(unittest.TestCase):
    def test_ip_engine_never_falls_back_to_private_source_url(self):
        self.assertEqual(
            cameras.inventory_stream_url("ip_engine", None, "http://iris-video-ip:8090/v1/frame"),
            "",
        )

    def test_ip_engine_uses_configured_public_preview_url(self):
        self.assertEqual(
            cameras.inventory_stream_url(
                "ip_engine", "https://jetson.tailnet.example/preview.mjpg", "http://iris-video-ip:8090/v1/frame"
            ),
            "https://jetson.tailnet.example/preview.mjpg",
        )

    def test_existing_non_ip_camera_keeps_legacy_fallback(self):
        self.assertEqual(
            cameras.inventory_stream_url("http_mjpeg", None, "http://iris-go2rtc:1984/api/stream.mjpeg"),
            "http://iris-go2rtc:1984/api/stream.mjpeg",
        )


if __name__ == "__main__":
    unittest.main()

import ast
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load("ip_engine", ROOT / "iris-video-ip/engine.py")
adapter = load("ip_adapter", ROOT / "iris-app/app/ip_capture.py")
retinaface = load("retinaface_decode", ROOT / "iris-video-ip/retinaface_decode.py")


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.store = engine.LatestFrame("entrada")
        self.payload = bytes(range(18))
        self.store.publish(self.payload, 3, 2)
        self.meta = self.store.snapshot()[0]

    def test_valid_frame(self):
        self.assertEqual(adapter.validate_frame(self.meta, self.payload, "entrada", None, 2),
                         (self.meta["stream_epoch"], 1))

    def test_rejects_bad_contract(self):
        for key, value in [("version", 2), ("camera_id", "other"), ("format", "RGB"),
                           ("width", -1), ("height", True), ("sequence", 0),
                           ("stream_epoch", ""), ("decoded_monotonic_ns", "now")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate_frame({**self.meta, key: value}, self.payload, "entrada", None, 2)

    def test_rejects_non_object_metadata(self):
        with self.assertRaises(ValueError):
            adapter.validate_frame([], self.payload, "entrada", None, 2)

    def test_rejects_truncated_payload(self):
        with self.assertRaises(ValueError):
            adapter.validate_frame(self.meta, self.payload[:-1], "entrada", None, 2)

    def test_rejects_stale_and_future(self):
        for delta in [-3_000_000_000, 3_000_000_000]:
            with self.assertRaises(ValueError):
                adapter.validate_frame({**self.meta, "decoded_monotonic_ns": time.monotonic_ns() + delta},
                                       self.payload, "entrada", None, 2)

    def test_rejects_duplicate_and_out_of_order(self):
        for seq in [1, 2]:
            with self.assertRaises(ValueError):
                adapter.validate_frame(self.meta, self.payload, "entrada", (self.meta["stream_epoch"], seq), 2)

    def test_reconnect_invalidates_frame_and_changes_epoch(self):
        previous = (self.meta["stream_epoch"], 1)
        self.store.reset()
        self.assertIsNone(self.store.snapshot())
        self.store.publish(self.payload, 3, 2)
        meta, payload = self.store.snapshot()
        self.assertNotEqual(meta["stream_epoch"], previous[0])
        adapter.validate_frame(meta, payload, "entrada", previous, 2)

    def test_only_latest_frame_is_retained(self):
        self.store.publish(bytes(18), 3, 2)
        self.assertEqual(self.store.snapshot()[0]["sequence"], 2)
        self.assertEqual(self.store.snapshot()[1], bytes(18))

    def test_store_expires_frame(self):
        with patch.object(engine.time, "monotonic_ns", return_value=self.meta["decoded_monotonic_ns"] + 3_000_000_000):
            self.assertIsNone(self.store.snapshot())

    def test_padded_rows_and_offset(self):
        self.assertEqual(engine.packed_bgr(b"XabcdefZZghijklZZ", 2, 2, 8, 1), b"abcdefghijkl")

    def test_padded_nv12_rows_and_decode(self):
        packed = engine.packed_nv12(b"X12aa34bbUVzz", 2, 2, 4, 4, 1, 9)
        self.assertEqual(packed, b"1234UV")
        store = engine.LatestFrame("entrada", frame_format="NV12")
        store.publish(packed, 2, 2)
        meta, payload = store.snapshot()
        self.assertEqual(meta["version"], 2)
        self.assertEqual(meta["format"], "NV12")
        self.assertEqual(adapter.validate_frame(meta, payload, "entrada", None, 2),
                         (meta["stream_epoch"], 1))
        frame = adapter.decode_frame(meta, payload)
        self.assertEqual(frame.shape, (2, 2, 3))

    def test_bad_layout(self):
        for args in [(b"", 2, 2, 6), (b"abcdef", 2, 1, 5), (b"", -1, 1, 6)]:
            with self.assertRaises(ValueError):
                engine.packed_bgr(*args)

        for args in [(b"", 2, 2, 2, 2), (b"", 3, 2, 3, 3), (b"", 2, 2, 1, 2)]:
            with self.assertRaises(ValueError):
                engine.packed_nv12(*args)

    def test_invalid_camera_id(self):
        with self.assertRaises(ValueError):
            engine.LatestFrame("camera\r\nheader")

    def test_output_fps_guard(self):
        self.assertEqual(engine.LatestFrame("entrada", output_fps=2).output_fps, 2)
        for value in [0, 31, "1"]:
            with self.assertRaises(ValueError):
                engine.LatestFrame("entrada", output_fps=value)

    def test_preview_store_rejects_invalid_jpeg_and_expires(self):
        preview = engine.LatestPreview(output_fps=5, quality=85)
        with self.assertRaises(ValueError):
            preview.publish(b"not-a-jpeg")
        preview.publish(b"\xff\xd8jpeg\xff\xd9")
        meta, payload = preview.snapshot()
        self.assertEqual(meta["sequence"], 1)
        self.assertEqual(payload, b"\xff\xd8jpeg\xff\xd9")
        with patch.object(engine.time, "monotonic_ns", return_value=meta["encoded_monotonic_ns"] + 3_000_000_000):
            self.assertIsNone(preview.snapshot())

    def test_retinaface_reference_decoder(self):
        outputs = [
            np.zeros((60 * 60 * 2, 1), dtype=np.float32),
            np.zeros((30 * 30 * 2, 1), dtype=np.float32),
            np.zeros((15 * 15 * 2, 1), dtype=np.float32),
            np.zeros((60 * 60 * 2, 4), dtype=np.float32),
            np.zeros((30 * 30 * 2, 4), dtype=np.float32),
            np.zeros((15 * 15 * 2, 4), dtype=np.float32),
            np.zeros((60 * 60 * 2, 10), dtype=np.float32),
            np.zeros((30 * 30 * 2, 10), dtype=np.float32),
            np.zeros((15 * 15 * 2, 10), dtype=np.float32),
        ]
        outputs[0][0] = 0.95
        outputs[0][1] = 0.90
        outputs[3][0, :] = [1, 1, 1, 1]
        outputs[3][1, :] = [1, 1, 1, 1]
        decoded = retinaface.decode_outputs(outputs)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["bbox"], [-8.0, -8.0, 8.0, 8.0])
        self.assertEqual(decoded[0]["landmarks"].shape, (5, 2))

    def test_retinaface_reference_decoder_rejects_wrong_heads(self):
        with self.assertRaises(ValueError):
            retinaface.decode_outputs([np.zeros((1,), dtype=np.float32)] * 8)


class FakeSocket:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = io.BytesIO()

    def makefile(self, *_args):
        return self.input

    def sendall(self, data):
        self.output.write(data)

    def settimeout(self, *_args):
        pass


class HttpTests(unittest.TestCase):
    def request(self, path, token=None, publish=False):
        store = engine.LatestFrame("entrada")
        if publish:
            store.publish(b"abc", 1, 1)
        auth = "" if token is None else "Authorization: Bearer " + token + "\r\n"
        sock = FakeSocket(("GET " + path + " HTTP/1.0\r\n" + auth + "\r\n").encode())
        engine.handler_for(store, "x" * 32)(sock, ("127.0.0.1", 1), object())
        return sock.output.getvalue()

    def test_authentication(self):
        for token in [None, "wrong", "é"]:
            self.assertIn(b"401", self.request("/v1/frame", token, True).split(b"\r\n")[0])

    def test_unavailable(self):
        self.assertIn(b"503", self.request("/v1/frame", "x" * 32))

    def test_frame_response(self):
        response = self.request("/v1/frame", "x" * 32, True)
        self.assertIn(b"200", response.split(b"\r\n")[0])
        self.assertIn(b"X-Iris-Frame:", response)
        self.assertTrue(response.endswith(b"abc"))

    def test_health_readiness(self):
        self.assertIn(b"503", self.request("/health"))
        self.assertIn(b"200", self.request("/health", publish=True))

    def test_preview_snapshot_response(self):
        store = engine.LatestFrame("entrada")
        preview = engine.LatestPreview()
        preview.publish(b"\xff\xd8jpeg\xff\xd9")
        sock = FakeSocket(b"GET /v1/preview.jpg HTTP/1.0\r\n\r\n")
        engine.handler_for(store, "x" * 32, preview)(sock, ("127.0.0.1", 1), object())
        response = sock.output.getvalue()
        self.assertIn(b"200", response.split(b"\r\n")[0])
        self.assertIn(b"Content-Type: image/jpeg", response)
        self.assertTrue(response.endswith(b"\xff\xd8jpeg\xff\xd9"))


class PreviewConcurrencyTests(unittest.TestCase):
    """The preview must never be able to starve the frame contract."""

    def serve(self, max_preview_clients=4):
        store = engine.LatestFrame("entrada", frame_format="NV12")
        store.publish(bytes(6), 2, 2)
        preview = engine.LatestPreview()
        preview.publish(b"\xff\xd8jpeg\xff\xd9")
        token = "x" * 32
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), engine.handler_for(store, token, preview, max_preview_clients)
        )
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return "http://127.0.0.1:%d" % server.server_address[1], token

    def open_preview(self, base):
        stream = urllib.request.urlopen(base + "/v1/preview.mjpeg", timeout=5)
        self.addCleanup(stream.close)
        self.assertIn(b"--iris-preview", stream.read(48))
        return stream

    def test_frame_contract_is_served_while_a_preview_is_open(self):
        base, token = self.serve()
        self.open_preview(base)
        request = urllib.request.Request(base + "/v1/frame", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=3) as response:
            self.assertEqual(response.read(), bytes(6))
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            self.assertTrue(json.loads(response.read())["ready"])

    def test_preview_clients_are_bounded(self):
        base, _token = self.serve(max_preview_clients=1)
        self.open_preview(base)
        with self.assertRaises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(base + "/v1/preview.mjpeg", timeout=3)
        self.assertEqual(refused.exception.code, 503)

    def test_engine_entrypoint_uses_a_threaded_server(self):
        # The e2e checks above build their own server; this pins main() itself,
        # which is what actually serves /v1/frame next to an open preview.
        tree = ast.parse((ROOT / "iris-video-ip/engine.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        servers = {
            node.func.id for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.endswith("HTTPServer")
        }
        self.assertEqual(servers, {"ThreadingHTTPServer"})

    def test_health_reports_the_served_contract_version(self):
        base, _token = self.serve()
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            self.assertEqual(json.loads(response.read())["contract_version"], 2)


class AdapterTests(unittest.TestCase):
    def test_fail_closed_without_secret(self):
        capture = adapter.IpEngineCapture("http://engine:8090/v1/frame", "entrada", "/missing/token")
        self.assertFalse(capture.isOpened())
        self.assertEqual(capture.read(), (False, None))

    def test_redirect_disabled(self):
        self.assertIsNone(adapter.NoRedirect().redirect_request(None))

    def test_read_and_duplicate(self):
        store = engine.LatestFrame("entrada")
        store.publish(b"abc", 1, 1)
        meta, payload = store.snapshot()
        headers = Message()
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = "3"
        headers["X-Iris-Frame"] = json.dumps(meta)
        with tempfile.NamedTemporaryFile() as token:
            token.write(b"x" * 32)
            token.flush()
            capture = adapter.IpEngineCapture("http://engine:8090/v1/frame", "entrada", token.name)
            def response(*_args, **_kwargs):
                body = io.BytesIO(payload)
                body.headers = headers
                return body
            with patch.object(capture.opener, "open", side_effect=response):
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (1, 1, 3))
                self.assertEqual(frame.tobytes(), payload)
                capture.next_read = 0
                self.assertEqual(capture.read(), (False, None))
                self.assertTrue(capture.transient_failure)
            capture.release()
            self.assertFalse(capture.isOpened())

    def test_runtime_dispatch_preserves_usb(self):
        # Extract only the dispatcher: no models, camera or global runtime imports.
        tree = ast.parse((ROOT / "iris-app/app/iris_runtime.py").read_text())
        runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IrisRuntime")
        method = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == "_open_capture")
        method.args.args[1].annotation = None
        namespace = {"IpEngineCapture": Mock(), "cv2": Mock(), "_GstUsbCapture": Mock()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "dispatcher", "exec"), namespace)
        settings = SimpleNamespace(ip_engine_token_file="/private/token", stream_capture_interval_seconds=1,
                                   ip_engine_max_frame_age_seconds=2, usb_camera_input_format="mjpeg",
                                   usb_camera_width=1920, usb_camera_height=1080, usb_camera_fps=15)
        owner = SimpleNamespace(settings=settings)
        camera = SimpleNamespace(source_kind="ip_engine", stream_url="http://engine/v1/frame",
                                 camera_id="entrada", device="/dev/video0")
        dispatch = namespace["_open_capture"]
        self.assertIs(dispatch(owner, camera), namespace["IpEngineCapture"].return_value)
        namespace["cv2"].VideoCapture.assert_not_called()
        camera.source_kind = "usb"
        dispatch(owner, camera)
        namespace["cv2"].VideoCapture.assert_called_with("/dev/video0", namespace["cv2"].CAP_V4L2)
        camera.source_kind = "gst_usb_sampled"
        dispatch(owner, camera)
        namespace["_GstUsbCapture"].assert_called_once_with(settings, camera, sampled=True)
        camera.source_kind = "rtsp"
        dispatch(owner, camera)
        namespace["cv2"].VideoCapture.assert_called_with(camera.stream_url, namespace["cv2"].CAP_FFMPEG)

    def test_ip_configuration_does_not_fallback_to_usb_device(self):
        source = (ROOT / "iris-app/app/cameras.py").read_text()
        self.assertIn("usb_source", source)
        self.assertIn("effective_device", source)


if __name__ == "__main__":
    unittest.main()

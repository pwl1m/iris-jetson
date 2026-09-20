"""Version-one local-host engine adapter; USB capture remains unchanged."""
import json
import http.client
import math
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

MAX_BYTES = 32 * 1024 * 1024
SUPPORTED_FORMATS = {"BGR", "NV12"}


class NoNewFrameError(ValueError):
    """The engine still exposes the already consumed latest frame."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def validate_frame(meta, payload, camera_id, previous, max_age):
    if not isinstance(meta, dict):
        raise ValueError("invalid frame metadata")
    frame_format = meta.get("format")
    version = meta.get("version")
    if (meta.get("camera_id") != camera_id or frame_format not in SUPPORTED_FORMATS or
            (frame_format == "BGR" and version != 1) or
            (frame_format == "NV12" and version != 2)):
        raise ValueError("incompatible frame contract")
    width, height = meta.get("width"), meta.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("invalid dimensions")
    if frame_format == "NV12" and (width % 2 or height % 2):
        raise ValueError("invalid NV12 dimensions")
    expected_size = width * height * 3 if frame_format == "BGR" else width * height * 3 // 2
    if not 0 < len(payload) == expected_size <= MAX_BYTES:
        raise ValueError("invalid payload size")
    epoch, sequence = meta.get("stream_epoch"), meta.get("sequence")
    if not isinstance(epoch, str) or not 1 <= len(epoch) <= 64 or type(sequence) is not int or sequence < 1:
        raise ValueError("invalid sequence")
    if previous and epoch == previous[0] and sequence <= previous[1]:
        raise NoNewFrameError("duplicate or out-of-order frame")
    decoded = meta.get("decoded_monotonic_ns")
    if type(decoded) is not int:
        raise ValueError("invalid timestamp")
    # Both containers must share this Jetson's monotonic clock (no time namespace).
    age = (time.monotonic_ns() - decoded) / 1e9
    if not math.isfinite(max_age) or max_age <= 0 or not 0 <= age <= max_age:
        raise ValueError("stale frame")
    return epoch, sequence


def decode_frame(meta, payload):
    """Decode the transport format into the BGR array expected by Iris."""
    width, height = meta["width"], meta["height"]
    if meta["format"] == "BGR":
        import numpy as np
        return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()

    import cv2
    import numpy as np
    nv12 = np.frombuffer(payload, dtype=np.uint8).reshape(height * 3 // 2, width)
    frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    if frame.shape != (height, width, 3):
        raise ValueError("invalid decoded frame shape")
    return frame


class IpEngineCapture:
    def __init__(self, url, camera_id, token_file, interval=1.0, max_age=2.0):
        self.url, self.camera_id = url, camera_id
        self.interval, self.max_age = max(0.05, interval), max_age
        self.previous, self.next_read = None, 0.0
        self.opened = False
        self.transient_failure = False
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            parsed = urlsplit(url)
            if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                return
            self.token = Path(token_file).read_text().strip()
            self.opened = len(self.token) >= 32 and self.token.isascii() and not any(c.isspace() for c in self.token)
        except (OSError, ValueError):
            pass

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    def read(self):
        if not self.opened:
            return False, None
        self.transient_failure = False
        time.sleep(max(0.0, self.next_read - time.monotonic()))
        self.next_read = time.monotonic() + self.interval
        request = urllib.request.Request(self.url, headers={"Authorization": "Bearer " + self.token})
        try:
            with self.opener.open(request, timeout=2) as response:
                if response.headers.get_content_type() != "application/octet-stream":
                    raise ValueError("invalid content type")
                meta = json.loads(response.headers["X-Iris-Frame"])
                size = int(response.headers["Content-Length"])
                if not 0 < size <= MAX_BYTES:
                    raise ValueError("invalid response size")
                payload = response.read(size + 1)
                if len(payload) != size:
                    raise ValueError("incomplete response")
            identity = validate_frame(meta, payload, self.camera_id, self.previous, self.max_age)
            frame = decode_frame(meta, payload)
            self.previous = identity
            return True, frame
        except NoNewFrameError:
            # At a cadence close to the engine output FPS, returning the
            # already consumed latest frame is normal scheduling jitter.
            self.transient_failure = True
            return False, None
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
            return False, None

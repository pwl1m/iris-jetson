from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    camera_id: str
    stream_url: str
    enabled: bool
    primary: bool = False
    source_kind: str | None = None
    device: str | None = None
    input_format: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    serial_number: str | None = None
    model_name: str | None = None
    stable_path: str | None = None


@dataclass
class DetectionResult:
    bbox: list[float]
    det_score: float
    embedding: np.ndarray
    crop: np.ndarray
    metadata: dict


@dataclass
class QualityResult:
    accepted: bool
    blur: float
    min_blur_score: float
    min_width: int
    min_height: int
    reason: str | None

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    camera_id: str
    stream_url: str
    enabled: bool
    primary: bool = False


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


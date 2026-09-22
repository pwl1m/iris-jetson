from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraConfig:
    camera_id: str
    stream_url: str
    enabled: bool
    public_stream_url: str | None = None
    public_stream_urls: dict[str, str] = field(default_factory=dict)
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
    # Fracoes (x1, y1, x2, y2), 0.0-1.0. None = sem filtro, censo conta o
    # frame inteiro. Ver app/roi.py.
    roi: tuple[float, float, float, float] | None = None


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

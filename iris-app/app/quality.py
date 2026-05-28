import cv2
import numpy as np

from .settings import Settings
from .schemas import DetectionResult, QualityResult


def _blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_face_quality(settings: Settings, detection: DetectionResult) -> QualityResult:
    bbox = detection.bbox
    width = int(max(0.0, bbox[2] - bbox[0]))
    height = int(max(0.0, bbox[3] - bbox[1]))
    blur = _blur_score(detection.crop)

    reason = None
    if detection.det_score < settings.face_min_det_score:
        reason = "det_score_baixo"
    elif width < settings.face_min_width:
        reason = "face_muito_pequena_largura"
    elif height < settings.face_min_height:
        reason = "face_muito_pequena_altura"
    elif blur < settings.face_min_blur_score:
        reason = "face_borrada"
    elif detection.metadata.get("occluded"):
        reason = "face_ocluida"

    return QualityResult(
        accepted=reason is None,
        blur=blur,
        min_blur_score=settings.face_min_blur_score,
        min_width=settings.face_min_width,
        min_height=settings.face_min_height,
        reason=reason,
    )

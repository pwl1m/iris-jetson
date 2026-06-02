import cv2
import numpy as np

from .schemas import DetectionResult
from .settings import Settings


def _ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def evaluate_visual_occlusion(
    settings: Settings,
    detection: DetectionResult,
    quality: dict,
    recognition: dict,
) -> dict:
    if not settings.visual_occlusion_enabled:
        return {"suspected": False, "score": 0.0, "signals": [], "method": "disabled"}

    crop = detection.crop
    if crop.size == 0:
        return {"suspected": False, "score": 0.0, "signals": ["empty_crop"], "method": "heuristic_v1"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    top = gray[: max(1, int(height * 0.35)), :]
    lower = gray[int(height * 0.45):, int(width * 0.2): int(width * 0.8)]
    center = crop[int(height * 0.2): int(height * 0.85), int(width * 0.18): int(width * 0.82)]

    dark_top_ratio = _ratio(top < settings.visual_occlusion_dark_pixel_threshold)
    dark_lower_ratio = _ratio(lower < settings.visual_occlusion_dark_pixel_threshold)

    ycrcb = cv2.cvtColor(center, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_mask = (
        (y > 40)
        & (cr >= settings.visual_occlusion_skin_cr_min)
        & (cr <= settings.visual_occlusion_skin_cr_max)
        & (cb >= settings.visual_occlusion_skin_cb_min)
        & (cb <= settings.visual_occlusion_skin_cb_max)
    )
    skin_ratio = _ratio(skin_mask)

    face = recognition.get("face", {}) or {}
    landmarks_detected = int(face.get("landmarks_detected") or 0)
    total_landmarks = int(face.get("total_landmarks") or 0)
    landmark_ratio = landmarks_detected / total_landmarks if total_landmarks else 0.0
    similarity = recognition.get("similarity")
    similarity_gap = None
    if similarity is not None:
        similarity_gap = max(0.0, float(settings.face_similarity_threshold) - float(similarity))

    signals = []
    score = 0.0

    if quality.get("reason") == "face_ocluida" or face.get("occluded"):
        signals.append("landmark_occlusion")
        score += 1.0
    if detection.det_score < settings.face_min_det_score + settings.visual_occlusion_det_score_margin:
        signals.append("det_score_near_min")
        score += 0.25
    if dark_lower_ratio >= settings.visual_occlusion_dark_lower_ratio:
        signals.append("dark_lower_face")
        score += 0.35
    if dark_top_ratio >= settings.visual_occlusion_dark_top_ratio:
        signals.append("dark_upper_face")
        score += 0.15
    if skin_ratio <= settings.visual_occlusion_min_skin_ratio:
        signals.append("low_skin_visibility")
        score += 0.35
    if (
        recognition.get("status") == "no_match"
        and landmark_ratio >= 0.95
        and similarity_gap is not None
        and similarity_gap <= settings.visual_occlusion_similarity_gap
    ):
        signals.append("full_landmarks_similarity_degraded")
        score += 0.25
    if recognition.get("status") == "no_match" and similarity is not None and float(similarity) >= settings.visual_occlusion_min_candidate_similarity:
        signals.append("known_candidate_below_threshold")
        score += 0.2

    suspected = score >= settings.visual_occlusion_score_threshold
    return {
        "suspected": suspected,
        "score": round(min(score, 1.0), 4),
        "signals": signals,
        "method": "heuristic_v1",
        "metrics": {
            "dark_top_ratio": round(dark_top_ratio, 4),
            "dark_lower_ratio": round(dark_lower_ratio, 4),
            "skin_ratio": round(skin_ratio, 4),
            "landmark_ratio": round(landmark_ratio, 4),
            "similarity_gap": round(similarity_gap, 4) if similarity_gap is not None else None,
        },
        "thresholds": {
            "score": settings.visual_occlusion_score_threshold,
            "dark_pixel": settings.visual_occlusion_dark_pixel_threshold,
            "dark_lower_ratio": settings.visual_occlusion_dark_lower_ratio,
            "dark_top_ratio": settings.visual_occlusion_dark_top_ratio,
            "min_skin_ratio": settings.visual_occlusion_min_skin_ratio,
            "similarity_gap": settings.visual_occlusion_similarity_gap,
        },
    }

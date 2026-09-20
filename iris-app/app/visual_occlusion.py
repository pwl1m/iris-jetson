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

    bbox_width = max(0.0, float(detection.bbox[2]) - float(detection.bbox[0]))
    bbox_height = max(0.0, float(detection.bbox[3]) - float(detection.bbox[1]))
    # Quantos rostos saem por aqui e a medida de cobertura da heuristica: com o
    # piso em 96 px eram 99,4% dos rostos reais medidos na linha USB. As metricas
    # sao invariantes de escala a partir de 48 px, entao o piso acompanha
    # face_min_width/height em vez de ser um valor proprio maior.
    if bbox_width < settings.visual_occlusion_min_width or bbox_height < settings.visual_occlusion_min_height:
        return {
            "suspected": False,
            "score": 0.0,
            "signals": ["face_too_small_for_visual_occlusion"],
            "method": "heuristic_v1",
            "metrics": {
                "bbox_width": round(bbox_width, 2),
                "bbox_height": round(bbox_height, 2),
            },
            "thresholds": {
                "min_width": settings.visual_occlusion_min_width,
                "min_height": settings.visual_occlusion_min_height,
            },
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    top = gray[: max(1, int(height * 0.35)), :]
    lower = gray[int(height * 0.45):, int(width * 0.2): int(width * 0.8)]
    center = crop[int(height * 0.2): int(height * 0.85), int(width * 0.18): int(width * 0.82)]

    dark_top_ratio = _ratio(top < settings.visual_occlusion_dark_pixel_threshold)
    dark_lower_ratio = _ratio(lower < settings.visual_occlusion_dark_pixel_threshold)
    # Observabilidade, sem peso no score. O limiar de escuro e absoluto, entao um
    # rosto em contraluz pontua como um rosto coberto. Estas duas metricas
    # separam os casos nos registros: luminancia baixa com assimetria ~0 e
    # subexposicao; assimetria positiva e o padrao de mascara ou mao.
    crop_luminance = float(gray.mean())
    dark_asymmetry = dark_lower_ratio - dark_top_ratio

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
    pose = face.get("pose") or {}
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

    # Metricas sao calculadas para todo rosto avaliado; o veredito so vale acima
    # do piso validado. Isso da cobertura de medicao de 100% sem mudar, hoje,
    # quantos eventos de oclusao sao escritos.
    verdict_floor = settings.visual_occlusion_verdict_min_size
    below_verdict_floor = bbox_width < verdict_floor or bbox_height < verdict_floor
    if below_verdict_floor:
        signals.append("below_verdict_floor")
    suspected = score >= settings.visual_occlusion_score_threshold and not below_verdict_floor
    return {
        "suspected": suspected,
        "score": round(min(score, 1.0), 4),
        "signals": signals,
        "method": "heuristic_v1",
        "metrics": {
            "dark_top_ratio": round(dark_top_ratio, 4),
            "dark_lower_ratio": round(dark_lower_ratio, 4),
            "dark_asymmetry": round(dark_asymmetry, 4),
            "crop_luminance": round(crop_luminance, 2),
            "skin_ratio": round(skin_ratio, 4),
            "landmark_ratio": round(landmark_ratio, 4),
            "similarity_gap": round(similarity_gap, 4) if similarity_gap is not None else None,
            # Observabilidade para o ensaio de calibracao; sem peso no score.
            # Perfil lateral e desfoque produzem hoje os mesmos sinais de pixel
            # que rosto coberto, e sem estas duas metricas o evento nao permite
            # distinguir as tres causas depois do fato.
            "blur": round(float(quality.get("blur")), 1) if quality.get("blur") is not None else None,
            "yaw": pose.get("yaw"),
            "pitch": pose.get("pitch"),
            "roll": pose.get("roll"),
            "eye_nose_asymmetry": pose.get("eye_nose_asymmetry"),
        },
        "thresholds": {
            "score": settings.visual_occlusion_score_threshold,
            "dark_pixel": settings.visual_occlusion_dark_pixel_threshold,
            "dark_lower_ratio": settings.visual_occlusion_dark_lower_ratio,
            "dark_top_ratio": settings.visual_occlusion_dark_top_ratio,
            "min_skin_ratio": settings.visual_occlusion_min_skin_ratio,
            "similarity_gap": settings.visual_occlusion_similarity_gap,
            "verdict_min_size": verdict_floor,
        },
    }

import cv2
import numpy as np

from .detector import InsightFaceDetector
from .quality import evaluate_face_quality
from .schemas import DetectionResult
from .settings import Settings
from .storage import FaceStore
from .visual_occlusion import evaluate_visual_occlusion


class IrisPipeline:
    def __init__(self, settings: Settings, detector: InsightFaceDetector, store: FaceStore):
        self.settings = settings
        self.detector = detector
        self.store = store

    def analyze_frame(self, image) -> tuple[DetectionResult, dict, dict]:
        candidates = self.detect_candidates(image, max_faces=1)
        if not candidates:
            from .recognizer import NoFaceDetectedError
            raise NoFaceDetectedError("nenhum rosto detectado")
        return self.analyze_candidate(image, candidates[0])

    def detect_candidates(self, image, max_faces: int) -> list[dict]:
        return self.detector.detect_all(image, max_faces=max_faces)

    def candidate_rank(self, candidate: dict) -> float:
        """Favor a face maior, nítida e com score alto em cada track."""
        crop = candidate["crop"]
        height, width = crop.shape[:2]
        area_quality = min(1.0, (width * height) / float(max(1, self.settings.face_min_width * self.settings.face_min_height * 4)))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_quality = min(1.0, blur / max(1.0, self.settings.face_min_blur_score * 2.0))
        return float(candidate["det_score"]) + area_quality * 0.35 + blur_quality * 0.15

    def analyze_candidate(self, image, candidate: dict) -> tuple[DetectionResult, dict, dict]:
        detection = self.detector.materialize(image, candidate)
        quality = evaluate_face_quality(self.settings, detection)
        recognition = self._compare_embedding(detection.embedding, detection.metadata)
        visual_occlusion = evaluate_visual_occlusion(self.settings, detection, quality.__dict__, recognition)
        recognition.setdefault("face", {})["visual_occlusion"] = visual_occlusion
        return detection, quality.__dict__, recognition

    def _compare_embedding(self, query_embedding, metadata: dict) -> dict:
        # A matriz vem do cache do FaceStore, reconstruido so quando a base muda.
        # Reconstruir aqui custava 133,7 ms por rosto com 1.000 embeddings nesta
        # Jetson, contra 1,3 ms do produto escalar, e escalava linearmente.
        subjects, stored_matrix, stored_norms, sources = self.store.embedding_matrix()

        if not subjects:
            return {
                "status": "no_match",
                "subject": None,
                "similarity": None,
                "face": metadata,
                "candidates": [],
                "model": self.settings.face_model_name,
            }

        query_norm = np.linalg.norm(query_embedding)

        norm_mask = (stored_norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(subjects), dtype=np.float64)
        if norm_mask.any():
            similarities[norm_mask] = np.dot(stored_matrix[norm_mask], query_embedding) / (stored_norms[norm_mask] * query_norm)

        top_k = min(self.settings.face_max_results, len(subjects))
        if top_k == len(subjects):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        matches = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim > 0 or len(matches) < top_k:
                matches.append({
                    "subject": subjects[idx],
                    "similarity": sim,
                    "source": sources[idx] if sources[idx] else None,
                })

        best = matches[0] if matches else None
        accepted = bool(best and best["similarity"] >= self.settings.face_similarity_threshold)
        return {
            "status": "matched" if accepted else "no_match",
            "subject": best["subject"] if accepted else None,
            "similarity": best["similarity"] if best else None,
            "face": metadata,
            "candidates": matches,
            "model": self.settings.face_model_name,
        }

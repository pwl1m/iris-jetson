import numpy as np

from .detector import InsightFaceDetector
from .quality import evaluate_face_quality
from .schemas import DetectionResult
from .settings import Settings
from .storage import FaceStore


class IrisPipeline:
    def __init__(self, settings: Settings, detector: InsightFaceDetector, store: FaceStore):
        self.settings = settings
        self.detector = detector
        self.store = store

    def analyze_frame(self, image) -> tuple[DetectionResult, dict, dict]:
        detection = self.detector.detect_best(image)
        quality = evaluate_face_quality(self.settings, detection)
        recognition = self._compare_embedding(detection.embedding, detection.metadata)
        return detection, quality.__dict__, recognition

    def _compare_embedding(self, query_embedding, metadata: dict) -> dict:
        subjects, embs, sources = [], [], []
        for subject, stored_embedding, source in self.store.embeddings():
            subjects.append(subject)
            embs.append(stored_embedding)
            sources.append(source)

        if not embs:
            return {
                "status": "no_match",
                "subject": None,
                "similarity": None,
                "face": metadata,
                "candidates": [],
                "model": self.settings.face_model_name,
            }

        stored_matrix = np.stack(embs, axis=0)
        query_norm = np.linalg.norm(query_embedding)
        stored_norms = np.linalg.norm(stored_matrix, axis=1)

        norm_mask = (stored_norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(embs), dtype=np.float64)
        if norm_mask.any():
            similarities[norm_mask] = np.dot(stored_matrix[norm_mask], query_embedding) / (stored_norms[norm_mask] * query_norm)

        top_k = min(self.settings.face_max_results, len(embs))
        if top_k == len(embs):
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

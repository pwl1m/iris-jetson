from .detector import InsightFaceDetector
from .quality import evaluate_face_quality
from .recognizer import cosine_similarity
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
        matches = []
        for subject, stored_embedding, source in self.store.embeddings():
            matches.append(
                {
                    "subject": subject,
                    "similarity": cosine_similarity(query_embedding, stored_embedding),
                    "source": source,
                }
            )

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        matches = matches[: self.settings.face_max_results]
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

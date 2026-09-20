from .recognizer import InsightFaceRecognizer
from .schemas import DetectionResult


class InsightFaceDetector:
    def __init__(self, recognizer: InsightFaceRecognizer, crop_padding: float):
        self.recognizer = recognizer
        self.crop_padding = crop_padding

    def detect_best(self, image) -> DetectionResult:
        candidates = self.detect_all(image, max_faces=1)
        if not candidates:
            from .recognizer import NoFaceDetectedError
            raise NoFaceDetectedError("nenhum rosto detectado")
        return self.materialize(image, candidates[0])

    def detect_all(self, image, max_faces: int) -> list[dict]:
        return self.recognizer.detect_candidates(image, max_faces=max_faces, padding=self.crop_padding)

    def materialize(self, image, candidate: dict) -> DetectionResult:
        candidate = self.recognizer.inspect_candidate(image, candidate, padding=self.crop_padding)
        return DetectionResult(
            bbox=candidate["metadata"]["bbox"],
            det_score=float(candidate["metadata"].get("det_score") or 0.0),
            embedding=candidate["embedding"],
            crop=candidate["crop"],
            metadata=candidate["metadata"],
        )

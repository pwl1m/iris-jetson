from .recognizer import InsightFaceRecognizer
from .schemas import DetectionResult


class InsightFaceDetector:
    def __init__(self, recognizer: InsightFaceRecognizer, crop_padding: float):
        self.recognizer = recognizer
        self.crop_padding = crop_padding

    def detect_best(self, image) -> DetectionResult:
        candidate = self.recognizer.inspect_best(image, padding=self.crop_padding)
        return DetectionResult(
            bbox=candidate["metadata"]["bbox"],
            det_score=float(candidate["metadata"].get("det_score") or 0.0),
            embedding=candidate["embedding"],
            crop=candidate["crop"],
            metadata=candidate["metadata"],
        )

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("imagem invalida")
    return image


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class InsightFaceRecognizer:
    def __init__(
        self,
        model_name: str,
        model_root: str,
        det_size: tuple[int, int],
        providers: list[str],
        ctx_id: int,
    ):
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        self.requested_providers = list(providers)
        self.available_providers = list(ort.get_available_providers())
        self.app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def _model_sessions_providers(self) -> dict[str, list[str]]:
        sessions: dict[str, list[str]] = {}
        models = getattr(self.app, "models", {}) or {}
        if isinstance(models, dict):
            for model_name, model in models.items():
                session = getattr(model, "session", None)
                if session is None:
                    continue
                try:
                    sessions[str(model_name)] = list(session.get_providers())
                except Exception:
                    sessions[str(model_name)] = []
        return sessions

    def provider_report(self) -> dict:
        model_sessions = self._model_sessions_providers()
        active = sorted({provider for values in model_sessions.values() for provider in values})
        missing_requested = [item for item in self.requested_providers if item not in self.available_providers]
        return {
            "requested": self.requested_providers,
            "available": self.available_providers,
            "active": active,
            "missing_requested": missing_requested,
            "models": model_sessions,
        }

    def extract_best(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        faces = self.app.get(image)
        if not faces:
            raise ValueError("nenhum rosto detectado")

        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        embedding = np.asarray(face.embedding, dtype=np.float32)
        metadata = {
            "bbox": [float(value) for value in face.bbox],
            "det_score": float(getattr(face, "det_score", 0.0)),
        }
        return embedding, metadata

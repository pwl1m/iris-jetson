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
    def __init__(self, model_name: str, model_root: str, det_size: tuple[int, int], providers: list[str]):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        self.app.prepare(ctx_id=-1, det_size=det_size)

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


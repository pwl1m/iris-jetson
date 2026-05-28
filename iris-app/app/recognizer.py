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


def _landmark_occlusion(face) -> dict:
    for attr in ("landmark_2d_106", "landmark_3d_68"):
        lm = getattr(face, attr, None)
        if lm is not None and isinstance(lm, np.ndarray) and lm.size > 0:
            total = int(lm.shape[0])
            if total == 0:
                continue
            valid = int(np.sum(np.any(lm != 0, axis=-1)))
            threshold = total // 2
            return {
                "landmarks_detected": valid,
                "total_landmarks": total,
                "landmark_model": attr,
                "occluded": valid < threshold,
                "occlusion_ratio": round(1.0 - valid / total, 4) if total > 0 else 1.0,
            }
    return {"landmarks_detected": 0, "total_landmarks": 0, "landmark_model": None, "occluded": False, "occlusion_ratio": 0.0}


class InsightFaceRecognizer:
    def __init__(
        self,
        model_name: str,
        model_root: str,
        det_size: tuple[int, int],
        providers: list[str],
        ctx_id: int,
        trt_fp16: bool = False,
        trt_engine_cache_path: str = "",
    ):
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        self.requested_providers = list(providers)
        self.available_providers = list(ort.get_available_providers())
        trt_opts = {}
        if trt_fp16:
            trt_opts["trt_fp16_enable"] = "True"
        if trt_engine_cache_path:
            trt_opts["trt_engine_cache_enable"] = "True"
            trt_opts["trt_engine_cache_path"] = trt_engine_cache_path
        provider_options = []
        for provider in providers:
            if "Tensorrt" in provider and trt_opts:
                provider_options.append(trt_opts)
            else:
                provider_options.append({})
        kwargs = {"providers": providers}
        if any(opts for opts in provider_options):
            kwargs["provider_options"] = provider_options
        self.app = FaceAnalysis(name=model_name, root=model_root, **kwargs)
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

    def _best_face(self, image: np.ndarray):
        faces = self.app.get(image)
        if not faces:
            raise ValueError("nenhum rosto detectado")
        return max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))

    def _metadata(self, face) -> dict:
        occlusion = _landmark_occlusion(face)
        return {
            "bbox": [float(value) for value in face.bbox],
            "det_score": float(getattr(face, "det_score", 0.0)),
            **occlusion,
        }

    def _crop(self, image: np.ndarray, bbox: list[float], padding: float = 0.0) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        x_pad = width * max(0.0, padding)
        y_pad = height * max(0.0, padding)
        left = max(0, int(x1 - x_pad))
        top = max(0, int(y1 - y_pad))
        right = min(image.shape[1], int(x2 + x_pad))
        bottom = min(image.shape[0], int(y2 + y_pad))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError("crop de rosto vazio")
        return crop

    def inspect_best(self, image: np.ndarray, padding: float = 0.0) -> dict:
        face = self._best_face(image)
        embedding = np.asarray(face.embedding, dtype=np.float32)
        metadata = self._metadata(face)
        crop = self._crop(image, metadata["bbox"], padding=padding)
        return {"embedding": embedding, "metadata": metadata, "crop": crop}

    def extract_best(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        candidate = self.inspect_best(image)
        embedding = candidate["embedding"]
        metadata = candidate["metadata"]
        return embedding, metadata

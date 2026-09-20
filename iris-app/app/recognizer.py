import time

import cv2
import numpy as np


class NoFaceDetectedError(ValueError):
    """A imagem foi processada corretamente, mas nao contem um rosto detectavel."""


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
    """Canal inerte, mantido pelo contrato do evento.

    Conta landmarks de valor exatamente (0,0). O `landmark_2d_106` e um regressor
    e sempre devolve 106 pontos finitos, entao `occluded` nunca fica True: medido
    em 1.997 eventos de oclusao da linha USB, `landmark_ratio` foi 1,000 em 100%
    deles. O sinal `landmark_occlusion` vale +1,0 e sozinho cruzaria o limiar,
    mas e inalcancavel. Trocar isso exige um modelo com visibilidade por ponto;
    ver docs/ip-camera/13_OCCLUSION_EVIDENCE_AND_RESOLUTION.md.
    """
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


def _head_pose(face) -> dict:
    """Pose de cabeca a partir dos 5 pontos que o detector ja devolve.

    Nao custa inferencia: os `kps` do SCRFD sao olho esquerdo, olho direito,
    nariz, canto esquerdo e canto direito da boca. Medido nos registros da linha
    USB, `|yaw|` mediano foi 1,457 nos eventos rotulados oclusao contra 0,512 nos
    normais da mesma faixa de tamanho: a maior parte daqueles eventos era perfil
    lateral, nao rosto coberto. As metricas entram como observabilidade para o
    ensaio de calibracao; nenhuma delas pontua no score hoje.
    """
    kps = getattr(face, "kps", None)
    if kps is None or not isinstance(kps, np.ndarray) or kps.shape != (5, 2):
        return {"pose": None}
    left_eye, right_eye, nose, left_mouth, right_mouth = [
        np.asarray(point, dtype=float) for point in kps
    ]
    eye_mid = (left_eye + right_eye) / 2.0
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance <= 0:
        return {"pose": None}
    mouth_mid = (left_mouth + right_mouth) / 2.0
    vertical = float(np.linalg.norm(mouth_mid - eye_mid)) or 1.0
    left_span = abs(float(nose[0] - left_eye[0]))
    right_span = abs(float(right_eye[0] - nose[0]))
    return {
        "pose": {
            "yaw": round(float((nose[0] - eye_mid[0]) / eye_distance), 4),
            "pitch": round(float((nose[1] - eye_mid[1]) / vertical), 4),
            "roll": round(float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))), 2),
            "eye_nose_asymmetry": round(float(abs(left_span - right_span) / max(left_span + right_span, 1e-6)), 4),
            "eye_distance": round(eye_distance, 2),
        }
    }


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
        kwargs = {
            "providers": providers,
            "allowed_modules": ["detection", "recognition", "landmark_2d_106"],
        }
        if any(opts for opts in provider_options):
            kwargs["provider_options"] = provider_options
        self.app = FaceAnalysis(name=model_name, root=model_root, **kwargs)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_size = det_size

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

    def _metadata(self, face) -> dict:
        occlusion = _landmark_occlusion(face)
        return {
            "bbox": [float(value) for value in face.bbox],
            "det_score": float(getattr(face, "det_score", 0.0)),
            **occlusion,
            **_head_pose(face),
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

    def detect_candidates(self, image: np.ndarray, max_faces: int, padding: float = 0.0) -> list[dict]:
        """Run only SCRFD detection; landmarks and embedding are deferred."""
        if type(max_faces) is not int or max_faces < 1:
            raise ValueError("invalid max faces")
        detector = (getattr(self.app, "models", {}) or {}).get("detection")
        if detector is None:
            raise RuntimeError("modelo de deteccao indisponivel")
        bboxes, keypoints = detector.detect(image, input_size=self.det_size, max_num=0)
        if bboxes is None or len(bboxes) == 0:
            return []
        results = []
        for index, raw_bbox in enumerate(bboxes[:max_faces]):
            bbox = [float(value) for value in raw_bbox[:4]]
            crop = self._crop(image, bbox, padding=padding)
            results.append({
                "bbox": bbox,
                "det_score": float(raw_bbox[4]),
                "kps": np.asarray(keypoints[index], dtype=np.float32),
                "crop": crop,
            })
        return results

    def inspect_candidate(self, image: np.ndarray, candidate: dict, padding: float = 0.0) -> dict:
        """Run landmarks and embedding for one detector candidate only."""
        from insightface.app.common import Face

        bbox = np.asarray(candidate["bbox"], dtype=np.float32)
        keypoints = np.asarray(candidate["kps"], dtype=np.float32)
        if bbox.shape != (4,) or keypoints.shape != (5, 2):
            raise ValueError("invalid face candidate")
        face = Face(bbox=bbox, kps=keypoints, det_score=float(candidate["det_score"]))
        models = getattr(self.app, "models", {}) or {}
        landmark_model = models.get("landmark_2d_106")
        if landmark_model is not None:
            landmark_model.get(image, face)
        recognition_model = models.get("recognition")
        if recognition_model is None:
            raise RuntimeError("modelo de reconhecimento indisponivel")
        recognition_model.get(image, face)
        embedding = np.asarray(face.embedding, dtype=np.float32)
        metadata = self._metadata(face)
        crop = self._crop(image, metadata["bbox"], padding=padding)
        return {"embedding": embedding, "metadata": metadata, "crop": crop}

    def inspect_best(self, image: np.ndarray, padding: float = 0.0) -> dict:
        candidates = self.detect_candidates(image, max_faces=1, padding=padding)
        if not candidates:
            raise NoFaceDetectedError("nenhum rosto detectado")
        return self.inspect_candidate(image, candidates[0], padding=padding)

    def warm_up(self) -> dict:
        """Run every model once so TensorRT builds its engines before real traffic.

        A cold w600k_r50/2d106det build costs minutes on the Jetson.  Detection
        runs on every frame and warms itself, but landmarks and recognition only
        execute once a face appears -- so without this the build lands on the
        first person to walk past the camera, stalling the worker while it holds
        the recognizer lock.
        """
        side = max(256, self.det_size[0], self.det_size[1])
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        # ArcFace's reference landmarks for a 112x112 crop, shifted into the
        # canvas: a non-degenerate alignment transform without a real face.
        origin = 64.0
        keypoints = np.asarray(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32,
        ) + origin
        candidate = {
            "bbox": [origin, origin, origin + 112.0, origin + 112.0],
            "det_score": 1.0,
            "kps": keypoints,
            "crop": canvas,
        }
        report = {}
        for stage, run in (
            ("detection", lambda: self.detect_candidates(canvas, max_faces=1)),
            ("landmark_and_recognition", lambda: self.inspect_candidate(canvas, candidate)),
        ):
            started = time.monotonic()
            run()
            report[stage + "_seconds"] = round(time.monotonic() - started, 3)
        return report

    def extract_best(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        candidate = self.inspect_best(image)
        embedding = candidate["embedding"]
        metadata = candidate["metadata"]
        return embedding, metadata

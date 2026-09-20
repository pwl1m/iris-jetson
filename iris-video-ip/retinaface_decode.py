"""Reference RetinaFace output decoder for the buffalo_m detector.

This module is deliberately independent from GStreamer and InsightFace. It is
the executable reference for the future DeepStream custom parser; it is not
part of the live acquisition path yet.
"""
from __future__ import annotations

import numpy as np


STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


def _centers(width: int, height: int, stride: int) -> np.ndarray:
    grid_width = width // stride
    grid_height = height // stride
    xv, yv = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    points = np.stack((xv, yv), axis=-1).astype(np.float32).reshape(-1, 2) * stride
    return np.repeat(points, NUM_ANCHORS, axis=0)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        left = np.maximum(boxes[current, 0], boxes[rest, 0])
        top = np.maximum(boxes[current, 1], boxes[rest, 1])
        right = np.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
        width = np.maximum(0.0, right - left + 1)
        height = np.maximum(0.0, bottom - top + 1)
        intersection = width * height
        overlap = intersection / (areas[current] + areas[rest] - intersection)
        order = rest[np.where(overlap <= threshold)[0]]
    return np.asarray(keep, dtype=np.int64)


def decode_outputs(
    outputs: list[np.ndarray] | tuple[np.ndarray, ...],
    input_size: tuple[int, int] = (480, 480),
    score_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> list[dict]:
    """Decode the nine buffalo_m RetinaFace output tensors.

    Output order follows InsightFace RetinaFace: three score tensors, three
    bbox-distance tensors, then three five-point landmark tensors.
    Coordinates are in the detector input space.
    """
    if len(outputs) != 9:
        raise ValueError("buffalo_m detector must expose nine outputs")
    width, height = input_size
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("input size must be positive and divisible by 32")
    if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= nms_threshold <= 1.0:
        raise ValueError("invalid detection thresholds")

    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    for index, stride in enumerate(STRIDES):
        score = np.asarray(outputs[index], dtype=np.float32).reshape(-1)
        bbox = np.asarray(outputs[index + 3], dtype=np.float32).reshape(-1, 4)
        kps = np.asarray(outputs[index + 6], dtype=np.float32).reshape(-1, 10)
        centers = _centers(width, height, stride)
        if score.size != centers.shape[0] or bbox.shape[0] != score.size or kps.shape[0] != score.size:
            raise ValueError("detector output shape does not match RetinaFace anchors")
        selected = np.flatnonzero(score >= score_threshold)
        if selected.size == 0:
            continue
        decoded = bbox[selected] * stride
        points = centers[selected]
        decoded = np.column_stack((
            points[:, 0] - decoded[:, 0],
            points[:, 1] - decoded[:, 1],
            points[:, 0] + decoded[:, 2],
            points[:, 1] + decoded[:, 3],
        ))
        kp = kps[selected] * stride
        kp = kp.reshape(-1, 5, 2) + np.repeat(points[:, None, :], 5, axis=1)
        boxes.append(decoded)
        scores.append(score[selected])
        landmarks.append(kp)

    if not boxes:
        return []
    all_boxes = np.concatenate(boxes, axis=0)
    all_scores = np.concatenate(scores, axis=0)
    all_landmarks = np.concatenate(landmarks, axis=0)
    keep = _nms(all_boxes, all_scores, nms_threshold)
    return [
        {
            "bbox": [float(value) for value in all_boxes[index]],
            "score": float(all_scores[index]),
            "landmarks": all_landmarks[index].astype(np.float32),
        }
        for index in keep
    ]

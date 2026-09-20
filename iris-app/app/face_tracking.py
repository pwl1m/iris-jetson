"""Bounded short-lived face tracks for the Iris stream worker.

This is intentionally an association layer, not biometric identity.  It joins
nearby detector boxes from adjacent frames so Iris can select one good frame for
recognition instead of running the full facial pipeline for every detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def iou(left: list[float], right: list[float]) -> float:
    """Return intersection-over-union for two [x1, y1, x2, y2] boxes."""
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class TrackCandidate:
    """A detector candidate plus its source frame, retained only while tracked.

    ``captured_at`` and ``capture_number`` travel with the frame so an event
    reports when its evidence was taken, not when the track was finally
    resolved -- those differ by up to ``ttl_seconds``.
    """

    bbox: list[float]
    rank: float
    payload: Any
    frame: Any
    captured_at: str
    capture_number: int


@dataclass
class FaceTrack:
    track_id: int
    bbox: list[float]
    first_seen: float
    last_seen: float
    frames_seen: int
    best: TrackCandidate
    last_attempt: float | None = None
    completed: bool = False
    last_landmarks: int = 0


def select_within_budget(ready_tracks: list["FaceTrack"], budget: int) -> tuple[list["FaceTrack"], int]:
    """Escolhe quais tracks materializar neste frame e quantos ficam para o proximo.

    Landmarks e embedding custam ~21 ms por rosto nesta Jetson e sao o unico
    item do pipeline que escala com a quantidade de gente: a deteccao custa
    ~43 ms por frame pedindo 3 ou 50 rostos.  Sem teto, um frame com muitos
    tracks prontos estoura o orcamento e a perda de frame vira nao
    deterministica, que e pior do que adiar de proposito.

    O excedente nao recebe `mark_attempt`, entao continua pronto no frame
    seguinte.  Como um track vive `ttl_seconds`, ele tem varias chances antes de
    expirar: a 5 FPS com TTL de 1,25 s sao ~6 frames, ou ~30 materializacoes por
    janela com orcamento 5.

    Ordem: quem nunca foi tentado primeiro, depois maior `rank`.  Um track novo
    nunca fica preso atras de retentativas, e entre iguais ganha o rosto maior,
    mais nitido e de melhor score.

    `budget` menor ou igual a zero desliga o teto.
    """
    if budget <= 0 or len(ready_tracks) <= budget:
        return list(ready_tracks), 0
    ordenados = sorted(
        ready_tracks,
        key=lambda track: (track.last_attempt is not None, -track.best.rank),
    )
    return ordenados[:budget], len(ordenados) - budget


class FaceTrackManager:
    """Greedy IoU tracker with bounded memory and recognition scheduling."""

    def __init__(
        self,
        max_tracks: int = 3,
        iou_threshold: float = 0.25,
        ttl_seconds: float = 1.25,
        min_frames: int = 2,
        retry_seconds: float = 0.75,
    ):
        if type(max_tracks) is not int or not 1 <= max_tracks <= 16:
            raise ValueError("invalid max tracks")
        if not 0.0 < iou_threshold <= 1.0 or ttl_seconds <= 0 or min_frames < 1 or retry_seconds <= 0:
            raise ValueError("invalid tracker settings")
        self.max_tracks = max_tracks
        self.iou_threshold = iou_threshold
        self.ttl_seconds = ttl_seconds
        self.min_frames = min_frames
        self.retry_seconds = retry_seconds
        self._next_id = 1
        self._tracks: dict[int, FaceTrack] = {}
        self.last_created_count = 0

    def update(self, candidates: list[TrackCandidate], now: float) -> list[FaceTrack]:
        """Associate candidates to active tracks and return tracks ready to analyze."""
        self.expire(now)
        self.last_created_count = 0
        unassigned = set(range(len(candidates)))
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for index, candidate in enumerate(candidates):
                overlap = iou(track.bbox, candidate.bbox)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_id, index))

        assigned_tracks: set[int] = set()
        for _overlap, track_id, index in sorted(pairs, reverse=True):
            if track_id in assigned_tracks or index not in unassigned:
                continue
            track = self._tracks[track_id]
            candidate = candidates[index]
            track.bbox = list(candidate.bbox)
            track.last_seen = now
            track.frames_seen += 1
            if candidate.rank > track.best.rank:
                track.best = candidate
            assigned_tracks.add(track_id)
            unassigned.remove(index)

        for index in sorted(unassigned, key=lambda item: candidates[item].rank, reverse=True):
            if len(self._tracks) >= self.max_tracks:
                break
            candidate = candidates[index]
            track = FaceTrack(
                track_id=self._next_id,
                bbox=list(candidate.bbox),
                first_seen=now,
                last_seen=now,
                frames_seen=1,
                best=candidate,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1
            self.last_created_count += 1

        return self.ready(now)

    def ready(self, now: float) -> list[FaceTrack]:
        return [
            track for track in self._tracks.values()
            if not track.completed
            and track.frames_seen >= self.min_frames
            and (track.last_attempt is None or now - track.last_attempt >= self.retry_seconds)
        ]

    def mark_attempt(self, track_id: int, now: float, completed: bool) -> None:
        track = self._tracks.get(track_id)
        if track is None:
            return
        track.last_attempt = now
        track.completed = bool(completed)

    def expire(self, now: float) -> list[int]:
        expired = [track_id for track_id, track in self._tracks.items() if now - track.last_seen > self.ttl_seconds]
        for track_id in expired:
            del self._tracks[track_id]
        return expired

    def clear(self) -> None:
        self._tracks.clear()

    @property
    def active_count(self) -> int:
        return len(self._tracks)

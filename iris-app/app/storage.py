import json
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def _iter_jsonl_reverse(path: Path, block_size: int = 1024 * 1024) -> Iterable[str]:
    with path.open("rb") as file:
        file.seek(0, 2)
        position = file.tell()
        pending = b""

        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            file.seek(position)
            chunk = file.read(read_size)
            lines = (chunk + pending).split(b"\n")
            pending = lines[0]

            for line in reversed(lines[1:]):
                if line:
                    yield line.decode("utf-8")

        if pending:
            yield pending.decode("utf-8")


class FaceStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_face_subject ON face_embeddings(subject)")

    def add_embedding(self, subject: str, embedding: np.ndarray, source: str | None = None) -> int:
        payload = json.dumps(embedding.astype(float).tolist(), separators=(",", ":"))
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO face_embeddings(subject, embedding, source) VALUES (?, ?, ?)",
                (subject, payload, source),
            )
            return int(cursor.lastrowid)

    def list_subjects(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT subject, COUNT(*) AS samples, MAX(created_at) AS last_sample_at
                FROM face_embeddings
                GROUP BY subject
                ORDER BY subject
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_subject(self, subject: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM face_embeddings WHERE subject = ?", (subject,))
            return int(cursor.rowcount)

    def samples(self, subject: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, subject, source, created_at
                FROM face_embeddings
                WHERE subject = ?
                ORDER BY created_at DESC, id DESC
                """,
                (subject,),
            ).fetchall()
        return [dict(row) for row in rows]

    def sample(self, sample_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, subject, source, created_at
                FROM face_embeddings
                WHERE id = ?
                LIMIT 1
                """,
                (sample_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_sample(self, sample_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM face_embeddings WHERE id = ?", (sample_id,))
            return int(cursor.rowcount)

    def embeddings(self) -> Iterable[tuple[str, np.ndarray, str | None]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT subject, embedding, source FROM face_embeddings").fetchall()
        for row in rows:
            yield row["subject"], np.asarray(json.loads(row["embedding"]), dtype=np.float32), row["source"]


def read_jsonl_events(
    path: Path,
    limit: int,
    *,
    since: str | None = None,
    predicate: Callable[[dict], bool] | None = None,
) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for row in _iter_jsonl_reverse(path):
        if not row.strip():
            continue
        event = json.loads(row)
        if predicate and not predicate(event):
            continue
        cursor = event.get("event_id") or event.get("capture_id") or event.get("captured_at")
        if since and cursor is not None and str(cursor) == since:
            break
        events.append(event)
        if not since and len(events) >= max(1, int(limit)):
            break
    if since:
        # A leitura fisica e reversa, mas a reconciliacao precisa receber o
        # lote mais antigo depois do cursor. Assim o proximo cursor avanca sem
        # saltar eventos quando o backlog e maior que o limite.
        events.reverse()
        return events[: max(1, int(limit))]
    return events


def read_jsonl_tail(path: Path, limit: int) -> list[dict]:
    return read_jsonl_events(path, limit)


def find_jsonl_event(path: Path, event_id: str) -> dict | None:
    if not path.exists():
        return None
    for row in _iter_jsonl_reverse(path):
        if not row.strip():
            continue
        event = json.loads(row)
        if event.get("event_id") == event_id:
            return event
        if event.get("capture_id") == event_id:
            return event
    return None

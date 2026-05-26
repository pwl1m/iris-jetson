import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


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

    def delete_sample(self, sample_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM face_embeddings WHERE id = ?", (sample_id,))
            return int(cursor.rowcount)

    def embeddings(self) -> Iterable[tuple[str, np.ndarray, str | None]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT subject, embedding, source FROM face_embeddings").fetchall()
        for row in rows:
            yield row["subject"], np.asarray(json.loads(row["embedding"]), dtype=np.float32), row["source"]

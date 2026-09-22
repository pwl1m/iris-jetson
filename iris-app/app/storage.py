import json
import sqlite3
import threading
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
        # A matriz de embeddings e reconstruida so quando a base muda.  Sem isso
        # cada rosto de cada frame fazia SELECT da tabela inteira e json.loads de
        # cada embedding: medido nesta Jetson, 133,7 ms para 1.000 embeddings e
        # 2.838 ms para 20.000, contra 1,3 e 16,5 ms do produto escalar em si.
        # Com 3 rostos a 10 FPS isso estourava o orcamento de frame ja em 1.000
        # rostos cadastrados, que e a ordem de grandeza da fase de recorrencia.
        self._cache_lock = threading.Lock()
        self._generation = 0
        self._cached_generation = -1
        self._cached_matrix: tuple[list[str], np.ndarray, np.ndarray, list[str | None]] | None = None
        self._init_db()

    def _invalidate(self) -> None:
        with self._cache_lock:
            self._generation += 1

    def embedding_matrix(self) -> tuple[list[str], np.ndarray, np.ndarray, list[str | None]]:
        """Subjects, matriz N x D, normas pre-calculadas e origens.

        Devolve sempre a mesma tupla enquanto a base nao muda.  Os arrays sao
        tratados como imutaveis pelos chamadores; nenhum deles escreve neles.
        """
        with self._cache_lock:
            generation = self._generation
            if self._cached_generation == generation and self._cached_matrix is not None:
                return self._cached_matrix

        subjects: list[str] = []
        vectors: list[np.ndarray] = []
        sources: list[str | None] = []
        for subject, embedding, source in self.embeddings():
            subjects.append(subject)
            vectors.append(embedding)
            sources.append(source)
        if vectors:
            matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
            norms = np.linalg.norm(matrix, axis=1)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
            norms = np.zeros((0,), dtype=np.float32)
        built = (subjects, matrix, norms, sources)

        with self._cache_lock:
            # Uma escrita concorrente durante a leitura invalida este resultado;
            # publica-se mesmo assim e a proxima chamada reconstroi.
            self._cached_matrix = built
            self._cached_generation = generation
        return built

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
            sample_id = int(cursor.lastrowid)
        self._invalidate()
        return sample_id

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
            removed = int(cursor.rowcount)
        self._invalidate()
        return removed

    def rename_subject(self, subject: str, new_subject: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE face_embeddings SET subject = ? WHERE subject = ?",
                (new_subject, subject),
            )
            renamed = int(cursor.rowcount)
        self._invalidate()
        return renamed

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
            removed = int(cursor.rowcount)
        self._invalidate()
        return removed

    def embeddings(self) -> Iterable[tuple[str, np.ndarray, str | None]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT subject, embedding, source FROM face_embeddings").fetchall()
        for row in rows:
            yield row["subject"], np.asarray(json.loads(row["embedding"]), dtype=np.float32), row["source"]


class FrameCensusStore:
    """Contagem de rostos por frame, persistida -- distinta de `/crowd`.

    `/crowd` le contadores em memoria que zeram a cada restart do worker.
    Esta tabela e o historico consultavel: uma linha por frame processado
    pelo worker (`source="worker"`) mais uma linha por chamada avulsa de
    `/people-count` (`source="upload"`). Nao guarda embedding nem identidade
    -- so "quantos rostos, em que camera, em que instante" -- por isso vive
    num arquivo proprio, fora de `faces.sqlite3`.
    """

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
                CREATE TABLE IF NOT EXISTS frame_census (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    capture_number INTEGER,
                    face_count INTEGER NOT NULL,
                    roi_applied INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'worker',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_census_camera_time ON frame_census(camera_id, id)")

    def record(
        self,
        camera_id: str,
        captured_at: str,
        face_count: int,
        *,
        capture_number: int | None = None,
        roi_applied: bool = False,
        source: str = "worker",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO frame_census(camera_id, captured_at, capture_number, face_count, roi_applied, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (camera_id, captured_at, capture_number, int(face_count), int(bool(roi_applied)), source),
            )
            return int(cursor.lastrowid)

    def list_recent(
        self,
        limit: int = 20,
        *,
        camera_id: str | None = None,
        since: int | None = None,
    ) -> list[dict]:
        """Historico do censo.

        Sem `since`: as `limit` linhas mais recentes, mais nova primeiro --
        mesma convencao de `read_jsonl_events` sem cursor. Com `since` (o
        `id` da ultima linha ja vista): so linhas mais novas que o cursor,
        em ordem cronologica, para paginacao incremental como `/events`.
        """
        limit = max(1, min(int(limit), 500))
        clauses = []
        params: list[object] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if since is not None:
            clauses.append("id > ?")
            params.append(int(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "ASC" if since is not None else "DESC"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM frame_census {where} ORDER BY id {order} LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]


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

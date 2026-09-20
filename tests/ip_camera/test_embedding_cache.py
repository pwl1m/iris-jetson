"""The embedding matrix is cached, and every write path invalidates it.

Measured on this Jetson before the cache: rebuilding the matrix cost 133.7 ms
with 1000 embeddings and 2838.6 ms with 20000, against 1.30 ms and 16.48 ms for
the dot product itself.  It ran once per face per frame, so with 3 faces the
comparison alone consumed 401 ms of a 200 ms frame budget at 1000 enrolled
faces -- the order of magnitude of the planned recurrence phase, where every
unidentified face is stored to be recognised on a later pass.
"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_storage():
    pkg = types.ModuleType("irisapp")
    pkg.__path__ = [str(ROOT / "iris-app/app")]
    sys.modules.setdefault("irisapp", pkg)
    spec = importlib.util.spec_from_file_location("irisapp.storage", ROOT / "iris-app/app/storage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = load_storage()


def vector(seed: int) -> np.ndarray:
    return np.full(8, float(seed), dtype=np.float32)


class EmbeddingMatrixCacheTests(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".sqlite3")
        self.store = storage.FaceStore(self.path)

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def test_empty_store_returns_empty_arrays_not_an_error(self):
        subjects, matrix, norms, sources = self.store.embedding_matrix()
        self.assertEqual(subjects, [])
        self.assertEqual(matrix.shape[0], 0)
        self.assertEqual(norms.shape[0], 0)
        self.assertEqual(sources, [])

    def test_matrix_norms_and_sources_line_up_with_subjects(self):
        self.store.add_embedding("ana", vector(3), source="captura")
        self.store.add_embedding("bruno", vector(4), None)
        subjects, matrix, norms, sources = self.store.embedding_matrix()
        self.assertEqual(subjects, ["ana", "bruno"])
        self.assertEqual(matrix.shape, (2, 8))
        self.assertEqual(sources, ["captura", None])
        np.testing.assert_allclose(norms, np.linalg.norm(matrix, axis=1))

    def test_repeated_reads_return_the_same_object_while_nothing_changes(self):
        self.store.add_embedding("ana", vector(3))
        first = self.store.embedding_matrix()
        self.assertIs(self.store.embedding_matrix(), first)

    def test_add_invalidates(self):
        self.store.add_embedding("ana", vector(3))
        first = self.store.embedding_matrix()
        self.store.add_embedding("bruno", vector(4))
        second = self.store.embedding_matrix()
        self.assertIsNot(second, first)
        self.assertEqual(second[0], ["ana", "bruno"])

    def test_delete_subject_invalidates(self):
        self.store.add_embedding("ana", vector(3))
        self.store.add_embedding("bruno", vector(4))
        self.store.embedding_matrix()
        self.store.delete_subject("ana")
        subjects, matrix, _, _ = self.store.embedding_matrix()
        self.assertEqual(subjects, ["bruno"])
        self.assertEqual(matrix.shape[0], 1)

    def test_rename_subject_invalidates(self):
        self.store.add_embedding("ana", vector(3))
        self.store.embedding_matrix()
        self.store.rename_subject("ana", "ana maria")
        self.assertEqual(self.store.embedding_matrix()[0], ["ana maria"])

    def test_delete_sample_invalidates(self):
        sample_id = self.store.add_embedding("ana", vector(3))
        self.store.add_embedding("ana", vector(5))
        self.store.embedding_matrix()
        self.store.delete_sample(sample_id)
        self.assertEqual(self.store.embedding_matrix()[1].shape[0], 1)

    def test_cache_agrees_with_the_uncached_iterator(self):
        # embeddings() stays the source of truth; the cache must not drift.
        for index, name in enumerate(("ana", "bruno", "ana")):
            self.store.add_embedding(name, vector(index + 1), source=f"s{index}")
        subjects, matrix, _, sources = self.store.embedding_matrix()
        direct = list(self.store.embeddings())
        self.assertEqual(subjects, [row[0] for row in direct])
        self.assertEqual(sources, [row[2] for row in direct])
        np.testing.assert_allclose(matrix, np.stack([row[1] for row in direct]))

    def test_rebuild_is_bounded_by_writes_not_by_reads(self):
        for index in range(20):
            self.store.add_embedding(f"s{index}", vector(index))
        built = []
        original = self.store.embeddings
        def counting():
            built.append(1)
            return original()
        self.store.embeddings = counting
        for _ in range(50):
            self.store.embedding_matrix()
        self.assertEqual(len(built), 1, "50 leituras deveriam reconstruir a matriz uma vez")
        self.store.add_embedding("novo", vector(99))
        self.store.embedding_matrix()
        self.assertEqual(len(built), 2)


if __name__ == "__main__":
    unittest.main()

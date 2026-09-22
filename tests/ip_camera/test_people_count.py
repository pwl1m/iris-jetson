"""Censo de pessoas persistido (`/people-count`) e ROI por câmera.

`/crowd` já existia, mas conta em memória e zera a cada restart do worker
(ver docstring de `crowd_status` em `iris_runtime.py`). Estes testes travam
o comportamento do que é novo: a tabela `frame_census`, o parsing/aplicação
da ROI (fração normalizada, não pixel — a resolução da câmera já mudou uma
vez nesta instalação, D-019 em docs/DECISIONS.md) e que uma ROI mal
configurada nunca derruba o censo, só desliga o filtro.
"""
import ast
import importlib.util
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "iris-app/app"


def settings_defaults():
    """Lê os defaults de `Settings` direto do source (mesma técnica de
    test_visual_occlusion.py) -- pydantic-settings não está instalado no
    host, e o ponto aqui é travar o que o deploy realmente publica."""
    tree = ast.parse((APP_DIR / "settings.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Settings")
    values = {}
    for node in klass.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    # `ast` só pega atribuições literais; `stream_source_kind_normalized` é
    # uma `@property` no `Settings` real. `configured_cameras()` a lê para
    # a câmera 1, então replicamos a mesma conta aqui em vez de travar um
    # atributo que o parser nunca vê.
    values["stream_source_kind_normalized"] = values.get("stream_source_kind", "").strip().lower()
    return types.SimpleNamespace(**values)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"irisapp_pc.{name}", APP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_modules():
    pkg = types.ModuleType("irisapp_pc")
    pkg.__path__ = [str(APP_DIR)]
    sys.modules["irisapp_pc"] = pkg
    roi = _load("roi")
    schemas = _load("schemas")
    sys.modules["irisapp_pc.settings"] = types.SimpleNamespace(Settings=settings_defaults)
    cameras = _load("cameras")
    storage = _load("storage")
    return roi, schemas, cameras, storage


roi, schemas, cameras, storage = load_modules()


class SettingsDefaultsTests(unittest.TestCase):
    def test_roi_defaults_to_disabled_for_every_camera_slot(self):
        defaults = settings_defaults()
        for index in range(1, 5):
            self.assertEqual(getattr(defaults, f"camera_{index}_roi"), "")

    def test_census_db_lives_next_to_events_not_inside_faces_db(self):
        defaults = settings_defaults()
        self.assertEqual(defaults.frame_census_db_path, "/data/events/census.sqlite3")


class ParseRoiTests(unittest.TestCase):
    def test_valid_fraction_string(self):
        self.assertEqual(roi.parse_roi("0.1,0.2,0.9,0.8"), (0.1, 0.2, 0.9, 0.8))

    def test_empty_or_missing_disables_filter(self):
        self.assertIsNone(roi.parse_roi(""))
        self.assertIsNone(roi.parse_roi(None))
        self.assertIsNone(roi.parse_roi("   "))

    def test_malformed_or_out_of_range_never_raises(self):
        for bad in ["not,a,roi,value", "0.1,0.2,0.9", "1.5,0,0.9,0.8", "0.5,0.5,0.4,0.9", "0.9,0.1,0.1,0.9"]:
            self.assertIsNone(roi.parse_roi(bad))


class BboxCenterInsideTests(unittest.TestCase):
    def test_center_inside_zone(self):
        zone = (0.0, 0.0, 0.5, 1.0)
        # Frame 1000x1000: bbox centrado em (200, 500) -- dentro da metade esquerda.
        self.assertTrue(roi.bbox_center_inside([100, 400, 300, 600], zone, 1000, 1000))

    def test_center_outside_zone(self):
        zone = (0.0, 0.0, 0.5, 1.0)
        # Centro em (800, 500) -- metade direita, fora da zona.
        self.assertFalse(roi.bbox_center_inside([700, 400, 900, 600], zone, 1000, 1000))

    def test_degenerate_frame_never_filters(self):
        zone = (0.0, 0.0, 0.5, 0.5)
        self.assertTrue(roi.bbox_center_inside([1, 1, 2, 2], zone, 0, 0))


class ConfiguredCamerasRoiTests(unittest.TestCase):
    def test_camera_1_roi_is_parsed_onto_camera_config(self):
        defaults = settings_defaults()
        defaults.camera_1_roi = "0.3,0.1,0.9,0.95"
        result = cameras.configured_cameras(defaults)
        entrada = next(c for c in result if c.camera_id == defaults.camera_1_id)
        self.assertEqual(entrada.roi, (0.3, 0.1, 0.9, 0.95))

    def test_invalid_roi_falls_back_to_unfiltered_not_a_crash(self):
        defaults = settings_defaults()
        defaults.camera_1_roi = "isso nao e uma roi"
        result = cameras.configured_cameras(defaults)
        entrada = next(c for c in result if c.camera_id == defaults.camera_1_id)
        self.assertIsNone(entrada.roi)

    def test_cameras_without_roi_configured_stay_unfiltered(self):
        defaults = settings_defaults()
        result = cameras.configured_cameras(defaults)
        for camera in result:
            if camera.camera_id != defaults.camera_1_id:
                self.assertIsNone(camera.roi)


class FrameCensusStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmpdir.name) / "census.sqlite3")
        self.store = storage.FrameCensusStore(self.db_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_creates_table_idempotently(self):
        # Reabrir o mesmo arquivo nao pode falhar nem duplicar a tabela.
        storage.FrameCensusStore(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='frame_census'"
            ).fetchall()
        self.assertEqual(len(tables), 1)

    def test_record_and_list_recent_default_order_is_newest_first(self):
        first = self.store.record("entrada", "2026-09-21T10:00:00+00:00", 1)
        second = self.store.record("entrada", "2026-09-21T10:00:01+00:00", 3)
        rows = self.store.list_recent(limit=10)
        self.assertEqual([row["id"] for row in rows], [second, first])
        self.assertEqual(rows[0]["face_count"], 3)

    def test_since_cursor_returns_only_newer_rows_in_chronological_order(self):
        first = self.store.record("entrada", "2026-09-21T10:00:00+00:00", 1)
        second = self.store.record("entrada", "2026-09-21T10:00:01+00:00", 2)
        third = self.store.record("entrada", "2026-09-21T10:00:02+00:00", 3)
        rows = self.store.list_recent(limit=10, since=first)
        self.assertEqual([row["id"] for row in rows], [second, third])

    def test_camera_id_filters_the_history(self):
        self.store.record("entrada", "2026-09-21T10:00:00+00:00", 1)
        self.store.record("cozinha", "2026-09-21T10:00:01+00:00", 5)
        rows = self.store.list_recent(limit=10, camera_id="cozinha")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["face_count"], 5)

    def test_roi_applied_flag_is_persisted(self):
        row_id = self.store.record("entrada", "2026-09-21T10:00:00+00:00", 2, roi_applied=True, source="upload")
        row = self.store.list_recent(limit=1)[0]
        self.assertEqual(row["id"], row_id)
        self.assertEqual(row["roi_applied"], 1)
        self.assertEqual(row["source"], "upload")


if __name__ == "__main__":
    unittest.main()

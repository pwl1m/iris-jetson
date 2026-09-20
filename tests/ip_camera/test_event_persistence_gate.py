"""Portões de persistência de evento: gravar só o que tem uso.

Cada evento persistido custa uma linha de JSONL e DOIS JPEGs, porque
`_save_capture` e `_save_face_crop` rodam antes da escrita. Medido na linha USB,
3,5 meses de uma câmera: 491.504 eventos, ~992 MB de JSONL e ~3,3 GB de imagem.

Dois desperdícios distintos nesses dados:

  1. Gente parada. Jose 101.489 eventos, Emanuel 52.579 — juntos 98% de tudo que
     foi identificado. Duas pessoas sentadas na mesa.
  2. Rosto pequeno demais. Dos 334.761 `no_match`, 84,8% têm 48-64 px e
     det_score medíocre; não sustentam veredito de oclusão nem embedding
     confiável, e alimentariam fusões erradas na fase de recorrência.

Os dois portões nascem DESLIGADOS de propósito: o ensaio de campo precisa gravar
tudo, porque é dele que sai o rótulo para calibrar. Estes testes travam tanto o
default desligado quanto o comportamento quando ligado.
"""
import ast
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "iris-app/app/iris_runtime.py"


def settings_defaults():
    tree = ast.parse((ROOT / "iris-app/app/settings.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Settings")
    values = {}
    for node in klass.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return types.SimpleNamespace(**values)


def gate_source() -> str:
    """Extrai o corpo de _event_persistence_gate sem importar o runtime.

    `iris_runtime` puxa fastapi, insightface e o mundo todo; o que interessa
    aqui é a regra, que é pura.
    """
    tree = ast.parse(RUNTIME.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_event_persistence_gate":
            return ast.unparse(node)
    raise AssertionError("_event_persistence_gate nao encontrado em iris_runtime.py")


class DefaultsTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings_defaults()

    def test_the_gates_ship_disabled(self):
        # O ensaio de campo grava tudo. Ligar so depois, com os cortes que o
        # scene_baseline.py indicar no cenario novo.
        self.assertEqual(self.settings.event_debounce_seconds, 0.0)
        self.assertEqual(self.settings.event_unmatched_min_width, 0)
        self.assertEqual(self.settings.event_unmatched_min_det_score, 0.0)

    def test_the_env_contract_documents_the_three_knobs(self):
        texto = (ROOT / ".env.example").read_text()
        for nome in ("EVENT_DEBOUNCE_SECONDS", "EVENT_UNMATCHED_MIN_WIDTH", "EVENT_UNMATCHED_MIN_DET_SCORE"):
            with self.subTest(variable=nome):
                self.assertIn(nome, texto)


class GateRuleTests(unittest.TestCase):
    """A regra em si, reconstruída a partir do fonte do runtime."""

    def setUp(self):
        self.fonte = gate_source()
        # As anotacoes da assinatura referenciam tipos do runtime; stubs bastam,
        # porque a regra e pura e nao toca em nenhum deles.
        escopo: dict = {"CameraConfig": object, "_CameraWorkerState": object}
        exec("import time\n" + self.fonte.replace("self._increment_stat", "self.increment"), escopo)
        self.gate = escopo["_event_persistence_gate"]

    def make_self(self, **settings):
        import threading
        base = dict(
            event_debounce_seconds=0.0,
            event_unmatched_min_width=0,
            event_unmatched_min_det_score=0.0,
        )
        base.update(settings)
        contador: dict[str, int] = {}
        fake = types.SimpleNamespace(
            settings=types.SimpleNamespace(**base),
            _event_debounce={},
            _event_debounce_lock=threading.Lock(),
            increment=lambda _state, nome: contador.__setitem__(nome, contador.get(nome, 0) + 1),
        )
        return fake, contador

    @staticmethod
    def detection(width=100.0, det_score=0.9):
        return types.SimpleNamespace(bbox=[0.0, 0.0, width, width], det_score=det_score)

    @staticmethod
    def camera(camera_id="entrada"):
        return types.SimpleNamespace(camera_id=camera_id)

    # ── desligado ────────────────────────────────────────────────────────────

    def test_disabled_never_suppresses_a_match(self):
        eu, _ = self.make_self()
        for _ in range(5):
            veredito = self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera(), None)
            self.assertIsNone(veredito)

    def test_disabled_never_suppresses_a_tiny_unmatched_face(self):
        eu, _ = self.make_self()
        veredito = self.gate(eu, {"status": "no_match"}, self.detection(width=20, det_score=0.1), self.camera(), None)
        self.assertIsNone(veredito)

    # ── debounce de identificado ─────────────────────────────────────────────

    def test_the_first_match_always_passes(self):
        eu, _ = self.make_self(event_debounce_seconds=30.0)
        self.assertIsNone(self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera(), None))

    def test_a_repeat_inside_the_window_is_suppressed(self):
        eu, contador = self.make_self(event_debounce_seconds=30.0)
        self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera(), None)
        veredito = self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera(), None)
        self.assertEqual(veredito, "debounce")
        self.assertEqual(contador.get("events_suppressed_debounce"), 1)

    def test_a_different_person_is_not_suppressed_by_someone_elses_event(self):
        eu, _ = self.make_self(event_debounce_seconds=30.0)
        self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera(), None)
        self.assertIsNone(self.gate(eu, {"status": "matched", "subject": "Paulo"}, self.detection(), self.camera(), None))

    def test_the_same_person_on_another_camera_is_not_suppressed(self):
        eu, _ = self.make_self(event_debounce_seconds=30.0)
        self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera("entrada"), None)
        self.assertIsNone(
            self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(), self.camera("saida"), None)
        )

    def test_a_match_without_a_subject_name_is_never_debounced(self):
        # Sem nome nao ha chave confiavel; gravar e mais seguro que suprimir.
        eu, _ = self.make_self(event_debounce_seconds=30.0)
        for _ in range(3):
            self.assertIsNone(self.gate(eu, {"status": "matched", "subject": None}, self.detection(), self.camera(), None))

    # ── qualidade de no_match ────────────────────────────────────────────────

    def test_unmatched_is_never_debounced_by_subject(self):
        # Todo no_match compartilha a ausencia de subject: debouncar por ele
        # faria uma pessoa suprimir outra.
        eu, contador = self.make_self(event_debounce_seconds=30.0)
        for _ in range(5):
            self.assertIsNone(self.gate(eu, {"status": "no_match"}, self.detection(), self.camera(), None))
        self.assertNotIn("events_suppressed_debounce", contador)

    def test_a_small_unmatched_face_is_suppressed(self):
        eu, contador = self.make_self(event_unmatched_min_width=64, event_unmatched_min_det_score=0.70)
        veredito = self.gate(eu, {"status": "no_match"}, self.detection(width=50, det_score=0.9), self.camera(), None)
        self.assertEqual(veredito, "quality")
        self.assertEqual(contador.get("events_suppressed_quality"), 1)

    def test_a_weak_detection_is_suppressed_even_when_large(self):
        eu, _ = self.make_self(event_unmatched_min_width=64, event_unmatched_min_det_score=0.70)
        self.assertEqual(
            self.gate(eu, {"status": "no_match"}, self.detection(width=200, det_score=0.5), self.camera(), None),
            "quality",
        )

    def test_a_good_unmatched_face_passes(self):
        eu, _ = self.make_self(event_unmatched_min_width=64, event_unmatched_min_det_score=0.70)
        self.assertIsNone(
            self.gate(eu, {"status": "no_match"}, self.detection(width=120, det_score=0.85), self.camera(), None)
        )

    def test_the_quality_gate_never_touches_a_match(self):
        # Pessoa identificada de longe continua valendo evento; o corte de
        # tamanho existe para desconhecido que nao servira para nada depois.
        eu, _ = self.make_self(event_unmatched_min_width=96, event_unmatched_min_det_score=0.80)
        self.assertIsNone(
            self.gate(eu, {"status": "matched", "subject": "Jose"}, self.detection(width=48, det_score=0.66), self.camera(), None)
        )


class CallerContractTests(unittest.TestCase):
    """O veredito escolhe se o track encerra ou continua elegível."""

    def test_debounce_closes_the_track_and_quality_keeps_it_open(self):
        fonte = RUNTIME.read_text()
        trecho = fonte.split("suppressed = self._event_persistence_gate")[1][:600]
        # Ja sabemos quem e: nao ha o que melhorar, encerra.
        self.assertIn('if suppressed == "debounce":', trecho)
        self.assertIn("return True", trecho.split('if suppressed == "quality":')[0])
        # Pode melhorar se a pessoa se aproximar: continua elegivel.
        self.assertIn("return False", trecho.split('if suppressed == "quality":')[1])

    def test_the_gate_runs_before_any_image_is_written(self):
        # O custo que se quer evitar sao os dois JPEGs, nao so a linha.
        fonte = RUNTIME.read_text()
        corpo = fonte.split("def _process_track")[1]
        pos_gate = corpo.index("self._event_persistence_gate")
        # O ramo de oclusao tambem chama _save_capture, e vem antes do portao de
        # proposito; a busca comeca depois do portao para achar o do evento.
        pos_save = corpo.index("image_path = self._save_capture", pos_gate)
        self.assertLess(pos_gate, pos_save, "o portao precisa vir antes de _save_capture")

    def test_occlusion_never_reaches_the_gate(self):
        # O ramo de oclusao retorna antes; sao 1.997 eventos no periodo todo.
        fonte = RUNTIME.read_text()
        corpo = fonte.split("def _process_track")[1]
        pos_occl = corpo.index("self._write_occlusion(occlusion_event)")
        pos_gate = corpo.index("self._event_persistence_gate")
        self.assertLess(pos_occl, pos_gate)


if __name__ == "__main__":
    unittest.main()

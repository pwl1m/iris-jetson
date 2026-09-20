"""Orçamento de materialização: detectar multidão é barato, reconhecer não é.

Medido nesta Jetson com `det_size` 960: a detecção custa ~43 ms por frame tanto
pedindo 3 rostos quanto pedindo 50, porque o SCRFD varre o frame inteiro e
`max_faces` só corta a lista depois.  Landmarks mais embedding custam ~21 ms por
rosto e são o único item que escala.  p95 por frame a 5 FPS, orçamento de 200 ms:
3 rostos 146,6 ms, 4 rostos 150,4 ms, 5 rostos 167,8 ms, 6 rostos 221,6 ms.

Daí a separação em três tetos: quantos rostos contar, quantos rastrear e quantos
materializar por frame.  Estes testes travam o terceiro.
"""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_tracking():
    spec = importlib.util.spec_from_file_location(
        "face_tracking_budget", ROOT / "iris-app/app/face_tracking.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tracking = load_tracking()


def settings_defaults():
    """Lê os defaults direto do fonte, sem depender de pydantic-settings."""
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


def track(track_id, rank, last_attempt=None):
    candidate = tracking.TrackCandidate(
        [0.0, 0.0, 10.0, 10.0], rank, {}, object(), "2026-09-19T00:00:00+00:00", 1
    )
    return tracking.FaceTrack(
        track_id=track_id, bbox=[0.0, 0.0, 10.0, 10.0],
        first_seen=0.0, last_seen=0.0, frames_seen=2,
        best=candidate, last_attempt=last_attempt,
    )


class BudgetTests(unittest.TestCase):
    def test_under_budget_passes_everything_through(self):
        pronto = [track(1, 1.0), track(2, 2.0)]
        escolhidos, adiados = tracking.select_within_budget(pronto, 5)
        self.assertEqual(len(escolhidos), 2)
        self.assertEqual(adiados, 0)

    def test_exactly_at_budget_defers_nothing(self):
        pronto = [track(i, float(i)) for i in range(5)]
        escolhidos, adiados = tracking.select_within_budget(pronto, 5)
        self.assertEqual(len(escolhidos), 5)
        self.assertEqual(adiados, 0)

    def test_over_budget_caps_and_reports_the_remainder(self):
        pronto = [track(i, float(i)) for i in range(9)]
        escolhidos, adiados = tracking.select_within_budget(pronto, 5)
        self.assertEqual(len(escolhidos), 5)
        self.assertEqual(adiados, 4)

    def test_higher_rank_wins_the_budget(self):
        # rank já pondera tamanho, nitidez e det_score: o melhor rosto do frame
        # é o que deve pagar o embedding primeiro.
        pronto = [track(1, 0.5), track(2, 9.0), track(3, 3.0)]
        escolhidos, _ = tracking.select_within_budget(pronto, 1)
        self.assertEqual(escolhidos[0].track_id, 2)

    def test_a_new_track_is_never_starved_behind_retries(self):
        # Sem isso, um track já tentado e de rank alto monopolizaria o orçamento
        # e uma pessoa nova nunca seria reconhecida num frame cheio.
        antigos = [track(i, 100.0, last_attempt=1.0) for i in range(5)]
        novo = track(99, 0.1)
        escolhidos, _ = tracking.select_within_budget(antigos + [novo], 1)
        self.assertEqual(escolhidos[0].track_id, 99)

    def test_zero_or_negative_budget_disables_the_cap(self):
        pronto = [track(i, float(i)) for i in range(9)]
        for budget in (0, -1):
            with self.subTest(budget=budget):
                escolhidos, adiados = tracking.select_within_budget(pronto, budget)
                self.assertEqual(len(escolhidos), 9)
                self.assertEqual(adiados, 0)

    def test_the_deferred_tracks_are_not_dropped_by_the_caller_contract(self):
        # O contrato é: quem não entra não recebe mark_attempt, logo continua
        # pronto no frame seguinte. A função não pode mutar nada.
        pronto = [track(i, float(i)) for i in range(9)]
        antes = [(t.track_id, t.last_attempt, t.completed) for t in pronto]
        tracking.select_within_budget(pronto, 5)
        depois = [(t.track_id, t.last_attempt, t.completed) for t in pronto]
        self.assertEqual(antes, depois)

    def test_empty_input(self):
        escolhidos, adiados = tracking.select_within_budget([], 5)
        self.assertEqual(escolhidos, [])
        self.assertEqual(adiados, 0)


class BudgetSettingsTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings_defaults()

    def test_the_census_reaches_at_least_the_tracker_capacity(self):
        self.assertLessEqual(
            self.settings.pipeline_max_faces, self.settings.pipeline_detect_max_faces,
            "o censo precisa alcancar pelo menos a capacidade do tracker",
        )

    def test_the_code_default_is_coherent_on_its_own(self):
        # Um clone sem configuracao nenhuma precisa rodar no perfil que foi
        # medido como cabivel: 5 tracks e 5 materializacoes por frame.
        self.assertEqual(self.settings.pipeline_max_faces, 5)
        self.assertEqual(self.settings.pipeline_materialize_budget, 5)

    def test_the_budget_is_clamped_to_tracker_capacity(self):
        # Baixar so a capacidade do tracker nao pode deixar um orcamento
        # pendurado maior que ela.
        fonte = (ROOT / "iris-app/app/settings.py").read_text()
        self.assertIn("min(budget, int(self.pipeline_max_faces))", fonte)

    def test_the_budget_matches_what_was_measured_to_fit(self):
        # 5 rostos deram p95 167,8 ms contra os 200 ms de orcamento a 5 FPS;
        # 6 deram 221,6 ms e estouraram.
        self.assertEqual(self.settings.pipeline_materialize_budget, 5)

    def test_tracker_capacity_stays_inside_what_FaceTrackManager_accepts(self):
        # FaceTrackManager rejeita max_tracks fora de 1..16.
        tracking.FaceTrackManager(max_tracks=self.settings.pipeline_max_faces)


class ComposeBudgetTests(unittest.TestCase):
    def test_the_ip_compose_declares_the_three_budgets(self):
        texto = (ROOT / "docker-compose.ip-camera.yml").read_text()
        for nome in ("PIPELINE_DETECT_MAX_FACES", "PIPELINE_MAX_FACES", "PIPELINE_MATERIALIZE_BUDGET"):
            with self.subTest(variable=nome):
                self.assertIn(nome, texto)


if __name__ == "__main__":
    unittest.main()

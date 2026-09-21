"""A dica de render mode é aditiva e nunca quebra o que o Onix já entrega.

O `GET /api/viewcare/iris/cameras/{id}/stream` do Onix hoje devolve
`render_mode => 'mjpeg'` e `content_type => 'multipart/x-mixed-replace'` como
literais fixas em `IrisController::getCameraStreamAction`. O `stream_url` em si
já é resolvido por rede a partir do que o device publica em `stream_urls`, e
esse caminho **não muda**.

O que o device passa a publicar é só uma dica: `stream_render_mode` e
`stream_content_type`. O Onix ignora campo desconhecido, então nada muda até que
o patch do lado dele seja aplicado. Estes testes travam exatamente isso: o
default continua `mjpeg`, o formato de `stream_urls` é preservado e um valor
inválido degrada para `mjpeg` em vez de derrubar o inventário.
"""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def load_cameras():
    pkg = types.ModuleType("irisapp_cam")
    pkg.__path__ = [str(ROOT / "iris-app/app")]
    sys.modules.setdefault("irisapp_cam", pkg)
    for nome in ("schemas", "settings"):
        spec = importlib.util.spec_from_file_location(f"irisapp_cam.{nome}", ROOT / f"iris-app/app/{nome}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"irisapp_cam.{nome}"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            # settings.py depende de pydantic, que nao esta no host; o modulo de
            # cameras so usa o tipo para anotacao.
            sys.modules[f"irisapp_cam.{nome}"] = types.SimpleNamespace(Settings=object)
    spec = importlib.util.spec_from_file_location("irisapp_cam.cameras", ROOT / "iris-app/app/cameras.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cameras = load_cameras()


class RenderModeTests(unittest.TestCase):
    def test_the_default_is_what_the_viewcare_already_plays(self):
        # O Onix devolve 'mjpeg' fixo hoje; o device nao pode discordar dele
        # antes do patch la.
        self.assertEqual(settings_defaults().camera_stream_render_mode, "mjpeg")

    def test_mjpeg_keeps_the_content_type_the_onix_hardcodes(self):
        modo, tipo = cameras.stream_render_mode("mjpeg")
        self.assertEqual(modo, "mjpeg")
        self.assertEqual(tipo, "multipart/x-mixed-replace")

    def test_hls_and_fmp4_are_offered(self):
        self.assertEqual(cameras.stream_render_mode("hls"), ("hls", "application/vnd.apple.mpegurl"))
        self.assertEqual(cameras.stream_render_mode("fmp4"), ("fmp4", "video/mp4"))

    def test_case_and_whitespace_do_not_matter(self):
        self.assertEqual(cameras.stream_render_mode("  HLS  ")[0], "hls")

    def test_an_unknown_mode_degrades_to_mjpeg_instead_of_breaking(self):
        # Uma configuracao errada nao pode derrubar o inventario inteiro, que e
        # de onde o Onix tira device, camera e saude.
        for ruim in ("webrtc", "", None, "rtsp", "   "):
            with self.subTest(modo=ruim):
                self.assertEqual(cameras.stream_render_mode(ruim), ("mjpeg", "multipart/x-mixed-replace"))


class ExistingContractTests(unittest.TestCase):
    """O que já foi entregue ao Onix continua idêntico."""

    def test_public_stream_urls_keeps_its_shape(self):
        # O resolver do Onix le exatamente estas duas chaves.
        urls = cameras.public_stream_urls("http://lan/x", "http://tail/x", "")
        self.assertEqual(urls, {"lan": "http://lan/x", "tailnet": "http://tail/x"})

    def test_the_legacy_url_is_still_the_lan_fallback(self):
        urls = cameras.public_stream_urls("", "", "http://legado/x")
        self.assertEqual(urls["lan"], "http://legado/x")

    def test_empty_transports_are_not_published(self):
        self.assertEqual(cameras.public_stream_urls("", "", ""), {})

    def test_the_private_frame_contract_is_never_published(self):
        # /v1/frame e autenticado e interno a rede Docker; entregar isso ao
        # ViewCare como se fosse preview de navegador seria um vazamento.
        self.assertEqual(
            cameras.inventory_stream_url("ip_engine", "", "http://iris-video-ip:8090/v1/frame"), ""
        )

    def test_a_configured_public_url_still_wins(self):
        self.assertEqual(
            cameras.inventory_stream_url("ip_engine", "http://lan/x", "http://interno/y"), "http://lan/x"
        )


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.texto = (ROOT / "docker-compose.ip-camera.yml").read_text()

    def test_the_render_mode_is_declared_and_defaults_to_mjpeg(self):
        self.assertIn('CAMERA_STREAM_RENDER_MODE: "${IRIS_IP_STREAM_RENDER_MODE:-mjpeg}"', self.texto)

    def test_go2rtc_stays_behind_a_profile(self):
        # Nao pode subir junto com `up -d`: enquanto nao houver decisao, o
        # transporte entregue continua sendo o MJPEG da engine.
        # "go2rtc" tambem aparece em comentarios e em URLs legadas; ancorar na
        # definicao do servico, com a indentacao de dois espacos.
        bloco = self.texto.split("\n  go2rtc:\n")[1][:800]
        self.assertIn("profiles: [preview]", bloco)
        self.assertIn("healthcheck", bloco)

    def test_go2rtc_binds_to_loopback_by_default(self):
        # GET /api/streams devolve a credencial RTSP da camera em texto puro.
        self.assertIn('"${IRIS_IP_PREVIEW_BIND:-127.0.0.1}:${IRIS_IP_PREVIEW_PORT:-1985}:1984"', self.texto)


if __name__ == "__main__":
    unittest.main()

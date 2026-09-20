"""The IP compose must declare every variable the Iris contract depends on.

The IP service carries no `env_file` on purpose, so anything absent from its
`environment:` block runs on a code default with no warning.  That failure was
real: MQTT publishing, capture retention and the in-app preview all ran on the
wrong value until measured (docs/ip-camera/12, section 2).  These tests pin the
variables whose silent default would change recognition or evidence.
"""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.ip-camera.yml"


def iris_app_environment() -> dict[str, str]:
    """Parse the `environment:` mapping of the iris-app service.

    Deliberately not PyYAML: it is not installed on the host and the point is to
    assert what the shipped file says, not what a parser can normalise it into.
    """
    lines = COMPOSE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "iris-app:")
    env_at = next(
        i for i, l in enumerate(lines[start:], start)
        if l.strip() == "environment:"
    )
    indent = len(lines[env_at]) - len(lines[env_at].lstrip())
    values = {}
    for line in lines[env_at + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        match = re.match(r'\s*([A-Z0-9_]+):\s*(.*)$', line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"')
    return values


class DeclaredEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = iris_app_environment()

    def test_detector_input_is_declared_not_left_to_the_code_default(self):
        # Without this line the container ran at 480x480, which downscales a
        # 1080p frame 4x and hands SCRFD a 12 px face.
        self.assertIn("FACE_DET_SIZE", self.env)

    def test_detector_input_is_the_measured_optimum(self):
        # 120 real frames: 960 gave det_score 0.780 and 2.3% discarded; 480 gave
        # 0.722 and 11.1%; 1280 regressed to 0.735 at 73.8 ms/frame.
        self.assertEqual(self.env["FACE_DET_SIZE"], "${IRIS_IP_FACE_DET_SIZE:-960,960}")

    def test_contract_critical_variables_stay_declared(self):
        for name in (
            "FACE_DET_SIZE",
            "MQTT_PUBLISH_ENABLED",
            "STREAM_MAX_CAPTURE_FILES",
            "STREAM_PREVIEW_ENABLED",
            "PIPELINE_MAX_FACES",
        ):
            with self.subTest(variable=name):
                self.assertIn(name, self.env)

    def test_every_override_has_a_default_so_a_bare_clone_still_boots(self):
        for name, value in self.env.items():
            if value.startswith("${") and "PUBLISH_HOST" not in name:
                with self.subTest(variable=name):
                    self.assertIn(":-", value, f"{name} has no default")


class EnvExampleTests(unittest.TestCase):
    """AGENTS.md defines .env.example as the contract; overrides must appear."""

    def test_detector_input_override_is_documented(self):
        text = (ROOT / ".env.example").read_text()
        self.assertIn("IRIS_IP_FACE_DET_SIZE=", text)


if __name__ == "__main__":
    unittest.main()

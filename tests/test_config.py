import tempfile
import unittest
from pathlib import Path

from vllm_evaluation.config import load_settings
from vllm_evaluation.errors import ConfigurationError


class LoadSettingsTests(unittest.TestCase):
    def test_defaults_are_available_without_a_config_file(self) -> None:
        settings = load_settings(environ={})

        self.assertEqual(settings.runner_type, "fake")
        self.assertEqual(settings.runner_concurrency, 1)
        self.assertEqual(settings.port, 8000)

    def test_toml_values_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                """
[server]
port = 8123

[runner]
timeout_seconds = 45.5
streaming = false
""".strip()
            )

            settings = load_settings(config_path, environ={})

        self.assertEqual(settings.port, 8123)
        self.assertEqual(settings.runner_timeout_seconds, 45.5)
        self.assertFalse(settings.runner_streaming)

    def test_environment_overrides_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text("[server]\nport = 8123\n")

            settings = load_settings(
                config_path,
                environ={"VLLM_EVAL_PORT": "9000"},
            )

        self.assertEqual(settings.port, 9000)

    def test_invalid_runner_concurrency_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_settings(environ={"VLLM_EVAL_RUNNER_CONCURRENCY": "0"})

import unittest

from vllm_evaluation.logging_config import configure_logging


class LoggingConfigTests(unittest.TestCase):
    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            configure_logging("not-a-level")


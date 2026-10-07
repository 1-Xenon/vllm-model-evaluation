"""Command-line bootstrap for the evaluation application."""

from __future__ import annotations

import argparse
import logging

from .config import load_settings, prepare_storage
from .logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the VLM evaluation application")
    parser.add_argument("--config", help="path to a TOML configuration file")
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_logging(settings.log_level)
    prepare_storage(settings)
    logging.getLogger(__name__).info(
        "configuration loaded for %s in %s environment", settings.app_name, settings.environment
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

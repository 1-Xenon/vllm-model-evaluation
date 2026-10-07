"""Logging setup for the application bootstrap and future services."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent human-readable application logging."""

    normalized_level = level.upper()
    numeric_level = getattr(logging, normalized_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unknown log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

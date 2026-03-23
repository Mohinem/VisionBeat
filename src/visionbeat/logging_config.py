"""Application logging helpers."""

from __future__ import annotations

import logging


def configure_logging(
    level: str = "INFO",
    *,
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> None:
    """Configure a process-wide logging format suitable for local development."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
    )

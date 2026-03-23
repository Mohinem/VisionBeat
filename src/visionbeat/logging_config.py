"""Application logging helpers."""

from __future__ import annotations

from visionbeat.observability import configure_root_logging


def configure_logging(
    level: str = "INFO",
    *,
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    structured: bool = True,
) -> None:
    """Configure process-wide logging for local development and observability."""
    configure_root_logging(level, log_format=log_format, structured=structured)

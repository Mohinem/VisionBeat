"""Command-line entry point for VisionBeat."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from visionbeat.app import VisionBeatApp
from visionbeat.config import AppConfig, load_config
from visionbeat.logging_config import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the VisionBeat gestural percussion instrument."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run",),
        default="run",
        help="Command to execute. Defaults to 'run'.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML or TOML configuration file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override the configured camera device index.",
    )
    return parser.parse_args(argv)


def build_config(config_path: str, *, camera_index: int | None = None) -> AppConfig:
    """Load application configuration and apply CLI overrides."""
    config = load_config(Path(config_path))
    if camera_index is None:
        return config
    return replace(config, camera=replace(config.camera, device_index=camera_index))


def main(argv: list[str] | None = None) -> None:
    """Initialize logging, load configuration, and start the application."""
    args = parse_args(argv)
    config = build_config(args.config, camera_index=args.camera_index)
    configure_logging(
        config.logging.level,
        log_format=config.logging.format,
        structured=config.logging.structured,
    )
    app = VisionBeatApp(config)
    app.run()


if __name__ == "__main__":
    main()

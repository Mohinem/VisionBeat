"""Command-line entry point for VisionBeat."""

from __future__ import annotations

import argparse

from visionbeat.app import VisionBeatApp
from visionbeat.config import AppConfig, load_config
from visionbeat.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the VisionBeat gestural percussion instrument."
    )
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to a TOML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize logging, load configuration, and start the application."""
    args = parse_args()
    config: AppConfig = load_config(args.config)
    configure_logging(config.log_level)
    app = VisionBeatApp(config)
    app.run()


if __name__ == "__main__":
    main()

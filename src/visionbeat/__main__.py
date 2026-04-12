"""Command-line entry point for VisionBeat."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from visionbeat.app import VisionBeatApp
from visionbeat.config import AppConfig, ConfigError, load_config
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import PoseBackendError, SUPPORTED_POSE_BACKENDS


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
    parser.add_argument(
        "--pose-backend",
        choices=SUPPORTED_POSE_BACKENDS,
        default=None,
        help="Select the pose tracking backend at runtime.",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug overlays and DEBUG log level.",
    )
    debug_group.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug overlays.",
    )
    parser.add_argument(
        "--sensitivity",
        choices=("conservative", "balanced", "expressive"),
        default="balanced",
        help="Gesture sensitivity preset for live demos.",
    )
    parser.add_argument(
        "--overlay-toggle-key",
        default="o",
        help="Single-character keyboard shortcut for toggling overlays (default: o).",
    )
    parser.add_argument(
        "--debug-toggle-key",
        default="d",
        help="Single-character keyboard shortcut for toggling debug panel (default: d).",
    )
    parser.add_argument(
        "--skeleton-only-hud",
        action="store_true",
        help=(
            "Show only skeleton landmarks in the preview HUD for dataset capture; "
            "hide text panels, landmark labels, and trigger flashes."
        ),
    )
    return parser.parse_args(argv)


def _apply_sensitivity_preset(config: AppConfig, preset: str) -> AppConfig:
    """Apply a named gesture sensitivity preset to a loaded config."""
    if preset == "balanced":
        return config

    thresholds = config.gestures.thresholds
    if preset == "conservative":
        updated_thresholds = replace(
            thresholds,
            punch_forward_delta_z=thresholds.punch_forward_delta_z * 1.15,
            strike_down_delta_y=thresholds.strike_down_delta_y * 1.15,
            snare_collision_distance=thresholds.snare_collision_distance * 0.9,
            min_velocity=thresholds.min_velocity * 1.1,
            axis_dominance_ratio=thresholds.axis_dominance_ratio * 1.1,
        )
    else:
        updated_thresholds = replace(
            thresholds,
            punch_forward_delta_z=thresholds.punch_forward_delta_z * 0.88,
            strike_down_delta_y=thresholds.strike_down_delta_y * 0.88,
            snare_collision_distance=thresholds.snare_collision_distance * 1.1,
            min_velocity=thresholds.min_velocity * 0.9,
            axis_dominance_ratio=max(1.0, thresholds.axis_dominance_ratio * 0.9),
        )
    return replace(config, gestures=replace(config.gestures, thresholds=updated_thresholds))


def build_config(
    config_path: str,
    *,
    camera_index: int | None = None,
    pose_backend: str | None = None,
    debug: bool = False,
    no_debug: bool = False,
    skeleton_only_hud: bool = False,
    sensitivity: str = "balanced",
) -> AppConfig:
    """Load application configuration and apply CLI overrides."""
    config = load_config(Path(config_path))
    if camera_index is not None:
        config = replace(config, camera=replace(config.camera, device_index=camera_index))
    if pose_backend is not None:
        config = replace(config, tracker=replace(config.tracker, backend=pose_backend.lower()))
    if debug:
        config = replace(
            config,
            logging=replace(config.logging, level="DEBUG"),
            debug=replace(
                config.debug,
                overlays=replace(config.debug.overlays, show_debug_panel=True),
            ),
        )
    if no_debug:
        config = replace(
            config,
            debug=replace(
                config.debug,
                overlays=replace(config.debug.overlays, show_debug_panel=False),
            ),
        )
    if skeleton_only_hud:
        config = replace(
            config,
            debug=replace(
                config.debug,
                overlays=replace(
                    config.debug.overlays,
                    draw_landmarks=True,
                    show_landmark_labels=False,
                    show_debug_panel=False,
                    show_trigger_flash=False,
                ),
            ),
        )
    return _apply_sensitivity_preset(config, sensitivity)


def main(argv: list[str] | None = None) -> None:
    """Initialize logging, load configuration, and start the application."""
    args = parse_args(argv)
    try:
        config = build_config(
            args.config,
            camera_index=args.camera_index,
            pose_backend=args.pose_backend,
            debug=args.debug,
            no_debug=args.no_debug,
            skeleton_only_hud=args.skeleton_only_hud,
            sensitivity=args.sensitivity,
        )
        configure_logging(
            config.logging.level,
            log_format=config.logging.format,
            structured=config.logging.structured,
        )
        app = VisionBeatApp(
            config,
            overlay_toggle_key=args.overlay_toggle_key,
            debug_toggle_key=args.debug_toggle_key,
        )
        app.run()
    except (ConfigError, PoseBackendError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

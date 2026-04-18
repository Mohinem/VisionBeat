"""Command-line entry point for VisionBeat."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from visionbeat.app import VisionBeatApp
from visionbeat.config import AppConfig, ConfigError, PredictiveConfig, load_config
from visionbeat.dataset_recording import record_dataset_video
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import SUPPORTED_POSE_BACKENDS, PoseBackendError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the VisionBeat gestural percussion instrument."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "record-dataset"),
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
        "--camera-backend",
        choices=("auto", "v4l2", "dshow", "msmf", "avfoundation", "gstreamer", "ffmpeg"),
        default=None,
        help="Override the configured OpenCV camera backend used for capture negotiation.",
    )
    parser.add_argument(
        "--camera-fourcc",
        default=None,
        help="Override the configured camera pixel format as a four-character code such as MJPG.",
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
            "Show only the skeleton overlay in the preview HUD for dataset capture; "
            "hide text panels and trigger flashes while keeping landmark labels."
        ),
    )
    parser.add_argument(
        "--predictive-mode",
        choices=("disabled", "shadow", "primary", "hybrid"),
        default=None,
        help=(
            "Predictive runtime mode: disabled=heuristics only, shadow=heuristics live plus "
            "predictive logging, primary=predictive CNN+decoder drives audio, "
            "hybrid=predictive arming plus completion-aligned firing."
        ),
    )
    parser.add_argument(
        "--timing-checkpoint",
        default=None,
        help="Override the predictive timing-model checkpoint path.",
    )
    parser.add_argument(
        "--gesture-checkpoint",
        default=None,
        help="Override the predictive gesture-classifier checkpoint path.",
    )
    parser.add_argument(
        "--predictive-threshold",
        type=float,
        default=None,
        help="Override the predictive timing threshold used by the streaming decoder.",
    )
    parser.add_argument(
        "--predictive-trigger-cooldown-frames",
        type=int,
        default=None,
        help="Override the predictive decoder cooldown in frames.",
    )
    parser.add_argument(
        "--predictive-trigger-max-gap-frames",
        type=int,
        default=None,
        help="Override the predictive decoder max-gap merge window in frames.",
    )
    parser.add_argument(
        "--predictive-device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override the predictive inference device.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Output raw-video path used by `record-dataset` mode.",
    )
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=0.0,
        help="Delay before `record-dataset` begins writing frames.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional fixed duration for `record-dataset` mode.",
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
    camera_backend: str | None = None,
    camera_fourcc: str | None = None,
    pose_backend: str | None = None,
    debug: bool = False,
    no_debug: bool = False,
    skeleton_only_hud: bool = False,
    sensitivity: str = "balanced",
    predictive_mode: str | None = None,
    timing_checkpoint: str | None = None,
    gesture_checkpoint: str | None = None,
    predictive_threshold: float | None = None,
    predictive_trigger_cooldown_frames: int | None = None,
    predictive_trigger_max_gap_frames: int | None = None,
    predictive_device: str | None = None,
) -> AppConfig:
    """Load application configuration and apply CLI overrides."""
    config = load_config(Path(config_path))
    if camera_index is not None or camera_backend is not None or camera_fourcc is not None:
        config = replace(
            config,
            camera=replace(
                config.camera,
                device_index=config.camera.device_index if camera_index is None else camera_index,
                backend=config.camera.backend if camera_backend is None else camera_backend.lower(),
                fourcc=(
                    config.camera.fourcc
                    if camera_fourcc is None
                    else camera_fourcc.strip().upper()
                ),
            ),
        )
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
                    show_landmark_labels=True,
                    show_debug_panel=False,
                    show_trigger_flash=False,
                ),
            ),
        )
    predictive_overrides: dict[str, object] = {}
    if predictive_mode is not None:
        predictive_overrides["mode"] = predictive_mode.lower()
    if timing_checkpoint is not None:
        predictive_overrides["timing_checkpoint_path"] = timing_checkpoint
    if gesture_checkpoint is not None:
        predictive_overrides["gesture_checkpoint_path"] = gesture_checkpoint
    if predictive_threshold is not None:
        predictive_overrides["threshold"] = predictive_threshold
    if predictive_trigger_cooldown_frames is not None:
        predictive_overrides["trigger_cooldown_frames"] = predictive_trigger_cooldown_frames
    if predictive_trigger_max_gap_frames is not None:
        predictive_overrides["trigger_max_gap_frames"] = predictive_trigger_max_gap_frames
    if predictive_device is not None:
        predictive_overrides["device"] = predictive_device.lower()
    if predictive_overrides:
        predictive_payload = config.predictive.to_dict()
        predictive_payload.update(predictive_overrides)
        if "mode" in predictive_overrides:
            predictive_payload.pop("enabled", None)
        config = replace(config, predictive=PredictiveConfig.from_mapping(predictive_payload))
    return _apply_sensitivity_preset(config, sensitivity)


def main(argv: list[str] | None = None) -> None:
    """Initialize logging, load configuration, and start the application."""
    args = parse_args(argv)
    try:
        config = build_config(
            args.config,
            camera_index=args.camera_index,
            camera_backend=args.camera_backend,
            camera_fourcc=args.camera_fourcc,
            pose_backend=args.pose_backend,
            debug=args.debug,
            no_debug=args.no_debug,
            skeleton_only_hud=args.skeleton_only_hud,
            sensitivity=args.sensitivity,
            predictive_mode=args.predictive_mode,
            timing_checkpoint=args.timing_checkpoint,
            gesture_checkpoint=args.gesture_checkpoint,
            predictive_threshold=args.predictive_threshold,
            predictive_trigger_cooldown_frames=args.predictive_trigger_cooldown_frames,
            predictive_trigger_max_gap_frames=args.predictive_trigger_max_gap_frames,
            predictive_device=args.predictive_device,
        )
        configure_logging(
            config.logging.level,
            log_format=config.logging.format,
            structured=config.logging.structured,
        )
        if args.command == "record-dataset":
            if args.output_video is None:
                raise SystemExit("--output-video is required for record-dataset mode.")
            result = record_dataset_video(
                config,
                output_path=args.output_video,
                start_delay_seconds=args.start_delay_seconds,
                duration_seconds=args.duration_seconds,
            )
            print(
                f"Recorded {result.frames_recorded} frames to {result.output_path} "
                f"({result.frame_width}x{result.frame_height} @ output {result.output_fps:.3f} fps, "
                f"target {result.target_fps} fps)"
            )
            print(f"Metadata: {result.metadata_path}")
            return
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

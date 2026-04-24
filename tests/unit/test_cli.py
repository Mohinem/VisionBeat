from __future__ import annotations

from pathlib import Path

from visionbeat.__main__ import build_config, parse_args
from visionbeat.config import GestureThresholdsConfig


def test_parse_args_supports_default_run_mode() -> None:
    args = parse_args(
        [
            "--config",
            "configs/default.yaml",
            "--camera-index",
            "2",
            "--camera-backend",
            "v4l2",
            "--camera-fourcc",
            "mjpg",
            "--pose-backend",
            "movenet",
            "--sensitivity",
            "expressive",
        ]
    )

    assert args.command == "run"
    assert args.config == "configs/default.yaml"
    assert args.camera_index == 2
    assert args.camera_backend == "v4l2"
    assert args.camera_fourcc == "mjpg"
    assert args.pose_backend == "movenet"
    assert args.sensitivity == "expressive"
    assert args.overlay_toggle_key == "o"
    assert args.debug_toggle_key == "d"
    assert args.skeleton_only_hud is False
    assert args.predictive_mode is None
    assert args.timing_checkpoint is None
    assert args.gesture_checkpoint is None
    assert args.predictive_threshold is None
    assert args.predictive_trigger_cooldown_frames is None
    assert args.predictive_trigger_max_gap_frames is None
    assert args.predictive_device is None
    assert args.output_video is None
    assert args.start_delay_seconds == 0.0
    assert args.duration_seconds is None


def test_parse_args_supports_explicit_run_command() -> None:
    args = parse_args(["run", "--camera-index", "3"])

    assert args.command == "run"
    assert args.camera_index == 3


def test_parse_args_supports_record_dataset_command() -> None:
    args = parse_args(
        [
            "record-dataset",
            "--output-video",
            "dataset/raw.mp4",
            "--start-delay-seconds",
            "3.5",
            "--duration-seconds",
            "12.0",
        ]
    )

    assert args.command == "record-dataset"
    assert args.output_video == "dataset/raw.mp4"
    assert args.start_delay_seconds == 3.5
    assert args.duration_seconds == 12.0


def test_parse_args_supports_skeleton_only_hud() -> None:
    args = parse_args(["--skeleton-only-hud"])

    assert args.skeleton_only_hud is True


def test_parse_args_supports_predictive_overrides() -> None:
    args = parse_args(
        [
            "--predictive-mode",
            "hybrid",
            "--timing-checkpoint",
            "models/timing.pt",
            "--gesture-checkpoint",
            "models/gesture.pt",
            "--predictive-threshold",
            "0.65",
            "--predictive-trigger-cooldown-frames",
            "8",
            "--predictive-trigger-max-gap-frames",
            "2",
            "--predictive-device",
            "cpu",
        ]
    )

    assert args.predictive_mode == "hybrid"
    assert args.timing_checkpoint == "models/timing.pt"
    assert args.gesture_checkpoint == "models/gesture.pt"
    assert args.predictive_threshold == 0.65
    assert args.predictive_trigger_cooldown_frames == 8
    assert args.predictive_trigger_max_gap_frames == 2
    assert args.predictive_device == "cpu"


def test_build_config_applies_camera_index_override(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text("camera:\n  device_index: 0\n", encoding="utf-8")

    config = build_config(config_path.as_posix(), camera_index=4)

    assert config.camera.device_index == 4


def test_build_config_applies_camera_backend_and_fourcc_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text("camera:\n  backend: auto\n  fourcc: null\n", encoding="utf-8")

    config = build_config(
        config_path.as_posix(),
        camera_backend="v4l2",
        camera_fourcc="mjpg",
    )

    assert config.camera.backend == "v4l2"
    assert config.camera.fourcc == "MJPG"


def test_build_config_applies_pose_backend_override(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text("tracker:\n  backend: mediapipe\n", encoding="utf-8")

    config = build_config(config_path.as_posix(), pose_backend="movenet")

    assert config.tracker.backend == "movenet"


def test_build_config_applies_debug_and_sensitivity_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text(
        "gestures:\n  thresholds:\n    punch_forward_delta_z: 0.2\n    strike_down_delta_y: 0.26\n",
        encoding="utf-8",
    )

    config = build_config(
        config_path.as_posix(),
        debug=True,
        sensitivity="conservative",
    )

    assert config.logging.level == "DEBUG"
    assert config.debug.overlays.show_debug_panel is True
    assert config.gestures.thresholds.punch_forward_delta_z > 0.2
    assert config.gestures.thresholds.strike_down_delta_y > 0.26
    assert (
        config.gestures.thresholds.snare_collision_distance
        < GestureThresholdsConfig().snare_collision_distance
    )


def test_build_config_applies_skeleton_only_hud_override(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text(
        "debug:\n"
        "  overlays:\n"
        "    draw_landmarks: false\n"
        "    show_landmark_labels: true\n"
        "    show_debug_panel: true\n"
        "    show_trigger_flash: true\n",
        encoding="utf-8",
    )

    config = build_config(config_path.as_posix(), debug=True, skeleton_only_hud=True)

    assert config.logging.level == "DEBUG"
    assert config.debug.overlays.draw_landmarks is True
    assert config.debug.overlays.show_landmark_labels is True
    assert config.debug.overlays.show_debug_panel is False
    assert config.debug.overlays.show_trigger_flash is False


def test_build_config_applies_predictive_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text(
        "predictive:\n"
        "  mode: disabled\n"
        "  timing_checkpoint_path: null\n"
        "  gesture_checkpoint_path: null\n",
        encoding="utf-8",
    )

    config = build_config(
        config_path.as_posix(),
        predictive_mode="hybrid",
        timing_checkpoint="models/timing.pt",
        gesture_checkpoint="models/gesture.pt",
        predictive_threshold=0.67,
        predictive_trigger_cooldown_frames=9,
        predictive_trigger_max_gap_frames=3,
        predictive_device="cpu",
    )

    assert config.predictive.enabled is True
    assert config.predictive.mode == "hybrid"
    assert config.predictive.timing_checkpoint_path == "models/timing.pt"
    assert config.predictive.gesture_checkpoint_path == "models/gesture.pt"
    assert config.predictive.threshold == 0.67
    assert config.predictive.trigger_cooldown_frames == 9
    assert config.predictive.trigger_max_gap_frames == 3
    assert config.predictive.device == "cpu"
    assert config.predictive.predictive_uses_completion_gate is True


def test_parse_args_rejects_conflicting_debug_flags() -> None:
    try:
        parse_args(["--debug", "--no-debug"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parse_args to reject conflicting debug flags.")

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
            "--pose-backend",
            "movenet",
            "--sensitivity",
            "expressive",
        ]
    )

    assert args.command == "run"
    assert args.config == "configs/default.yaml"
    assert args.camera_index == 2
    assert args.pose_backend == "movenet"
    assert args.sensitivity == "expressive"
    assert args.overlay_toggle_key == "o"
    assert args.debug_toggle_key == "d"
    assert args.skeleton_only_hud is False


def test_parse_args_supports_explicit_run_command() -> None:
    args = parse_args(["run", "--camera-index", "3"])

    assert args.command == "run"
    assert args.camera_index == 3


def test_parse_args_supports_skeleton_only_hud() -> None:
    args = parse_args(["--skeleton-only-hud"])

    assert args.skeleton_only_hud is True


def test_build_config_applies_camera_index_override(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text("camera:\n  device_index: 0\n", encoding="utf-8")

    config = build_config(config_path.as_posix(), camera_index=4)

    assert config.camera.device_index == 4


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
    assert config.debug.overlays.show_landmark_labels is False
    assert config.debug.overlays.show_debug_panel is False
    assert config.debug.overlays.show_trigger_flash is False


def test_parse_args_rejects_conflicting_debug_flags() -> None:
    try:
        parse_args(["--debug", "--no-debug"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parse_args to reject conflicting debug flags.")

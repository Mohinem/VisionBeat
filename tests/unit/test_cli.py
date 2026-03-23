from __future__ import annotations

from pathlib import Path

from visionbeat.__main__ import build_config, parse_args


def test_parse_args_supports_default_run_mode() -> None:
    args = parse_args(["--config", "configs/default.yaml", "--camera-index", "2"])

    assert args.command == "run"
    assert args.config == "configs/default.yaml"
    assert args.camera_index == 2


def test_parse_args_supports_explicit_run_command() -> None:
    args = parse_args(["run", "--camera-index", "3"])

    assert args.command == "run"
    assert args.camera_index == 3


def test_build_config_applies_camera_index_override(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.yaml"
    config_path.write_text("camera:\n  device_index: 0\n", encoding="utf-8")

    config = build_config(config_path.as_posix(), camera_index=4)

    assert config.camera.device_index == 4

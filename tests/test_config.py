from __future__ import annotations

import re
from pathlib import Path

import pytest

from visionbeat.config import AppConfig, ConfigError, load_config


def test_load_config_supports_nested_yaml_sections() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert isinstance(config, AppConfig)
    assert config.camera.window_name == "VisionBeat"
    assert config.gestures.thresholds.punch_forward_delta_z == pytest.approx(0.2)
    assert config.gestures.thresholds.strike_down_delta_y == pytest.approx(0.26)
    assert config.gestures.cooldowns.trigger_seconds == pytest.approx(0.2)
    assert config.audio.sample_mapping == {
        "kick": "assets/samples/kick.wav",
        "snare": "assets/samples/snare.wav",
    }
    assert config.debug.overlays.show_debug_panel is True
    assert config.logging.level == "INFO"


def test_load_config_supports_toml_with_nested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.toml"
    config_path.write_text(
        """
[camera]
window_name = "Studio View"

[gestures.thresholds]
punch_forward_delta_z = 0.24
strike_down_delta_y = 0.3

[gestures.cooldowns]
trigger_seconds = 0.28
analysis_window_seconds = 0.2
confirmation_window_seconds = 0.14

[audio.sample_mapping]
kick = "custom/kick.wav"
clap = "custom/clap.wav"

[debug.overlays]
draw_landmarks = false
show_debug_panel = true

[logging]
level = "debug"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.camera.window_name == "Studio View"
    assert config.gestures.thresholds.punch_forward_delta_z == pytest.approx(0.24)
    assert config.gestures.cooldowns.trigger_seconds == pytest.approx(0.28)
    assert config.audio.sample_mapping["clap"] == "custom/clap.wav"
    assert config.debug.overlays.draw_landmarks is False
    assert config.logging.level == "DEBUG"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "camera:\n  width: zero\n",
            "camera.width: expected an integer",
        ),
        (
            "gestures:\n  thresholds:\n    candidate_ratio: 1.5\n",
            "gestures.thresholds.candidate_ratio: must be less than or equal to 1.0",
        ),
        (
            "audio:\n  sample_mapping:\n    kick: ''\n",
            "audio.sample_mapping.kick: sample path must not be empty",
        ),
        (
            "debug:\n  overlays:\n    unknown: true\n",
            "debug.overlays: unknown field(s): unknown",
        ),
    ],
)
def test_load_config_reports_helpful_validation_errors(
    contents: str,
    message: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(config_path)


def test_load_config_rejects_unsupported_extensions(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported config file extension"):
        load_config(config_path)

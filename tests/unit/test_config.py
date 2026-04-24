from __future__ import annotations

import re
from pathlib import Path

import pytest

from visionbeat.config import (
    AppConfig,
    AudioConfig,
    CameraConfig,
    ConfigError,
    DebugConfig,
    GestureConfig,
    GestureCooldownsConfig,
    GestureThresholdsConfig,
    LoggingConfig,
    OverlayConfig,
    PredictiveConfig,
    RuntimeConfig,
    TrackerConfig,
    TransportConfig,
    _ConfigReader,
    _load_yaml_config,
    _parse_yaml_scalar,
    load_config,
)


def test_load_config_supports_nested_yaml_sections() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert isinstance(config, AppConfig)
    assert config.camera.window_name == "VisionBeat"
    assert config.camera.backend == "v4l2"
    assert config.camera.fourcc == "MJPG"
    assert config.runtime.async_pipeline is True
    assert config.runtime.target_render_fps == 30
    assert config.runtime.idle_sleep_seconds == pytest.approx(0.002)
    assert config.tracker.model_complexity == 0
    assert config.tracker.max_input_width == 640
    assert config.gestures.thresholds.punch_forward_delta_z == pytest.approx(0.006)
    assert config.gestures.thresholds.punch_max_vertical_drift == pytest.approx(0.75)
    assert config.gestures.thresholds.strike_down_delta_y == pytest.approx(0.06)
    assert config.gestures.thresholds.strike_confirmation_ratio == pytest.approx(0.4)
    assert config.gestures.thresholds.strike_max_depth_drift == pytest.approx(0.24)
    assert config.gestures.thresholds.snare_collision_distance == pytest.approx(0.26)
    assert config.gestures.thresholds.snare_confirmation_velocity_ratio == pytest.approx(0.8)
    assert config.gestures.thresholds.snare_collision_max_depth_gap == pytest.approx(0.24)
    assert config.gestures.thresholds.axis_dominance_ratio == pytest.approx(1.0)
    assert config.gestures.cooldowns.trigger_seconds == pytest.approx(0.2)
    assert config.gestures.cooldowns.analysis_window_seconds == pytest.approx(0.18)
    assert config.gestures.cooldowns.confirmation_window_seconds == pytest.approx(0.12)
    assert config.gestures.thresholds.min_velocity == pytest.approx(0.37)
    assert config.gestures.velocity_smoothing_alpha == pytest.approx(0.8)
    assert config.gestures.rearm_threshold_ratio == pytest.approx(0.45)
    assert config.audio.sample_mapping == {
        "kick": "assets/samples/kick.wav",
        "snare": "assets/samples/snare.wav",
    }
    assert config.debug.overlays.show_debug_panel is True
    assert config.logging.level == "INFO"
    assert config.debug.overlays.show_landmark_labels is True
    assert config.debug.overlays.show_trigger_flash is True
    assert config.logging.structured is True
    assert config.logging.event_log_path is None
    assert config.logging.event_log_format == "jsonl"
    assert config.logging.session_recording_path is None
    assert config.logging.session_recording_mode == "tracker_outputs"
    assert config.predictive.enabled is True
    assert config.predictive.mode == "hybrid"
    assert (
        config.predictive.timing_checkpoint_path
        == "outputs/future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001/checkpoints/best_model.pt"
    )
    assert (
        config.predictive.gesture_checkpoint_path
        == "outputs/gesture_classifier_future8_r1r2_train_r3_val/visionbeat_gesture_classifier_run_001/checkpoints/best_model.pt"
    )
    assert config.predictive.threshold == pytest.approx(0.6)
    assert config.predictive.trigger_cooldown_frames == 6
    assert config.predictive.trigger_max_gap_frames == 1
    assert config.predictive.device == "auto"


def test_load_config_supports_toml_with_nested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.toml"
    config_path.write_text(
        """
[camera]
window_name = "Studio View"

[runtime]
async_pipeline = true
target_render_fps = 45

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
show_landmark_labels = false
show_debug_panel = true
show_trigger_flash = false

[logging]
level = "debug"
structured = false
event_log_path = "logs/events.csv"
event_log_format = "csv"
session_recording_path = "logs/sessions"
session_recording_mode = "both"

[predictive]
mode = "primary"
timing_checkpoint_path = "models/timing.pt"
gesture_checkpoint_path = "models/gesture.pt"
threshold = 0.65
trigger_cooldown_frames = 8
trigger_max_gap_frames = 2
device = "cpu"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.camera.window_name == "Studio View"
    assert config.runtime.async_pipeline is True
    assert config.runtime.target_render_fps == 45
    assert config.gestures.thresholds.punch_forward_delta_z == pytest.approx(0.24)
    assert config.gestures.cooldowns.trigger_seconds == pytest.approx(0.28)
    assert config.audio.sample_mapping["clap"] == "custom/clap.wav"
    assert config.debug.overlays.draw_landmarks is False
    assert config.debug.overlays.show_landmark_labels is False
    assert config.logging.level == "DEBUG"
    assert config.debug.overlays.show_trigger_flash is False
    assert config.logging.structured is False
    assert config.logging.event_log_path == "logs/events.csv"
    assert config.logging.event_log_format == "csv"
    assert config.logging.session_recording_path == "logs/sessions"
    assert config.logging.session_recording_mode == "both"
    assert config.predictive.enabled is True
    assert config.predictive.mode == "primary"
    assert config.predictive.timing_checkpoint_path == "models/timing.pt"
    assert config.predictive.gesture_checkpoint_path == "models/gesture.pt"
    assert config.predictive.threshold == pytest.approx(0.65)
    assert config.predictive.trigger_cooldown_frames == 8
    assert config.predictive.trigger_max_gap_frames == 2
    assert config.predictive.device == "cpu"


@pytest.mark.parametrize(
    ("factory", "payload", "expected_type"),
    [
        (
            CameraConfig.from_dict,
            {"width": 1920, "height": 1080, "fps": 60, "backend": "v4l2", "fourcc": "mjpg"},
            CameraConfig,
        ),
        (
            RuntimeConfig.from_mapping,
            {"async_pipeline": True, "target_render_fps": 45, "idle_sleep_seconds": 0.001},
            RuntimeConfig,
        ),
        (
            TrackerConfig.from_dict,
            {
                "backend": "movenet",
                "model_complexity": 2,
                "max_input_width": 512,
                "min_tracking_confidence": 0.6,
            },
            TrackerConfig,
        ),
        (
            GestureThresholdsConfig.from_mapping,
            {"punch_forward_delta_z": 0.24, "candidate_ratio": 0.8},
            GestureThresholdsConfig,
        ),
        (
            GestureCooldownsConfig.from_mapping,
            {"trigger_seconds": 0.25, "analysis_window_seconds": 0.2},
            GestureCooldownsConfig,
        ),
        (
            GestureConfig.from_dict,
            {
                "history_size": 8,
                "active_hand": "left",
                "velocity_smoothing_alpha": 0.6,
                "rearm_threshold_ratio": 0.5,
            },
            GestureConfig,
        ),
        (
            AudioConfig.from_dict,
            {"sample_mapping": {"kick": "kick.wav", "snare": "snare.wav"}},
            AudioConfig,
        ),
        (TransportConfig.from_mapping, {"backend": "udp", "port": 9100}, TransportConfig),
        (
            OverlayConfig.from_dict,
            {"draw_landmarks": False, "show_landmark_labels": False, "show_trigger_flash": False},
            OverlayConfig,
        ),
        (DebugConfig.from_mapping, {"overlays": {"show_debug_panel": False}}, DebugConfig),
        (
            LoggingConfig.from_mapping,
            {"level": "debug", "session_recording_mode": "raw_frames"},
            LoggingConfig,
        ),
        (
            PredictiveConfig.from_mapping,
            {
                "mode": "primary",
                "timing_checkpoint_path": "models/timing.pt",
                "gesture_checkpoint_path": "models/gesture.pt",
                "device": "cpu",
            },
            PredictiveConfig,
        ),
    ],
)
def test_config_models_accept_valid_payloads(
    factory,
    payload: dict[str, object],
    expected_type: type,
) -> None:
    config = factory(payload)

    assert isinstance(config, expected_type)
    if isinstance(config, CameraConfig):
        assert config.backend == "v4l2"
        assert config.fourcc == "MJPG"
    if isinstance(config, TrackerConfig):
        assert config.backend == "movenet"
        assert config.max_input_width == 512
    if isinstance(config, RuntimeConfig):
        assert config.async_pipeline is True
        assert config.target_render_fps == 45
        assert config.idle_sleep_seconds == pytest.approx(0.001)
    if isinstance(config, PredictiveConfig):
        assert config.enabled is True
        assert config.mode == "primary"
        assert config.device == "cpu"


def test_predictive_config_keeps_enabled_backwards_compatible_as_shadow_mode() -> None:
    config = PredictiveConfig.from_mapping(
        {
            "enabled": True,
            "timing_checkpoint_path": "models/timing.pt",
            "gesture_checkpoint_path": "models/gesture.pt",
        }
    )

    assert config.enabled is True
    assert config.mode == "shadow"


def test_predictive_config_accepts_hybrid_mode() -> None:
    config = PredictiveConfig.from_mapping(
        {
            "mode": "hybrid",
            "timing_checkpoint_path": "models/timing.pt",
            "gesture_checkpoint_path": "models/gesture.pt",
        }
    )

    assert config.enabled is True
    assert config.mode == "hybrid"
    assert config.predictive_uses_completion_gate is True


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"device_index": -1}, "camera.device_index: must be greater than or equal to 0"),
        ({"width": True}, "camera.width: expected an integer"),
        ({"window_name": "   "}, "camera.window_name: must not be empty"),
        (
            {"backend": "wizard"},
            "camera.backend: must be one of 'auto', 'v4l2', 'dshow', 'msmf', 'avfoundation', 'gstreamer', or 'ffmpeg'.",
        ),
        ({"fourcc": "abc"}, "camera.fourcc: must be exactly four characters when set."),
    ],
)
def test_camera_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        CameraConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model_complexity": 3}, "tracker.model_complexity: must be one of 0, 1, 2"),
        ({"backend": "openpose"}, "tracker.backend: must be either 'mediapipe' or 'movenet'."),
        ({"max_input_width": -1}, "tracker.max_input_width: must be greater than or equal to 0"),
        (
            {"min_detection_confidence": 1.5},
            "tracker.min_detection_confidence: must be less than or equal to 1.0",
        ),
    ],
)
def test_tracker_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        TrackerConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"enabled": True},
            "predictive.timing_checkpoint_path: must be set when predictive is enabled.",
        ),
        (
            {"enabled": True, "timing_checkpoint_path": "timing.pt"},
            "predictive.gesture_checkpoint_path: must be set when predictive is enabled.",
        ),
        (
            {"mode": "wizard"},
            "predictive.mode: must be one of 'disabled', 'shadow', 'primary', or 'hybrid'.",
        ),
        (
            {
                "mode": "primary",
                "timing_checkpoint_path": "timing.pt",
                "gesture_checkpoint_path": "gesture.pt",
                "device": "metal",
            },
            "predictive.device: must be one of 'auto', 'cpu', or 'cuda'.",
        ),
        (
            {"enabled": False, "mode": "shadow"},
            "predictive.enabled: conflicts with predictive.mode.",
        ),
    ],
)
def test_predictive_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        PredictiveConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"candidate_ratio": 0.0}, "gestures.thresholds.candidate_ratio: must be greater than 0.0"),
        (
            {"strike_confirmation_ratio": 1.1},
            "gestures.thresholds.strike_confirmation_ratio: must be less than or equal to 1.0",
        ),
        (
            {"snare_confirmation_velocity_ratio": 1.1},
            "gestures.thresholds.snare_confirmation_velocity_ratio: "
            "must be less than or equal to 1.0",
        ),
        (
            {"axis_dominance_ratio": 0.9},
            "gestures.thresholds.axis_dominance_ratio: must be greater than or equal to 1.0",
        ),
        (
            {"unknown": 1},
            "gestures.thresholds: unknown field(s): unknown. Allowed fields:",
        ),
    ],
)
def test_gesture_threshold_config_validation_errors(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        GestureThresholdsConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"active_hand": "center"},
            "gestures.active_hand: must be either 'left' or 'right'.",
        ),
        ({"history_size": 0}, "gestures.history_size: must be greater than 0"),
        (
            {"velocity_smoothing_alpha": 0.0},
            "gestures.velocity_smoothing_alpha: must be greater than 0.0",
        ),
        (
            {"rearm_threshold_ratio": 1.5},
            "gestures.rearm_threshold_ratio: must be less than or equal to 1.0",
        ),
    ],
)
def test_gesture_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        GestureConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"backend": "alsa"}, "audio.backend: must be 'pygame'."),
        (
            {"sample_mapping": {}},
            "audio.sample_mapping: must define at least one gesture-to-sample mapping.",
        ),
        (
            {"sample_mapping": {"kick": "   "}},
            "audio.sample_mapping.kick: sample path must not be empty.",
        ),
    ],
)
def test_audio_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        AudioConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"backend": "osc"}, "transport.backend: must be either 'none' or 'udp'."),
        ({"port": 0}, "transport.port: must be greater than 0"),
    ],
)
def test_transport_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        TransportConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"level": "info", "event_log_format": "parquet"}, "logging.event_log_format"),
        ({"session_recording_mode": "frames"}, "logging.session_recording_mode"),
        ({"structured": "yes"}, "logging.structured: expected a boolean"),
    ],
)
def test_logging_config_validation_errors(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        LoggingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("camera:\n  width: zero\n", "camera.width: expected an integer"),
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
        (
            "logging:\n  event_log_format: parquet\n",
            "logging.event_log_format: must be either 'jsonl' or 'csv'",
        ),
        (
            "logging:\n  session_recording_mode: frames\n",
            "logging.session_recording_mode: must be one of "
            "'tracker_outputs', 'raw_frames', or 'both'",
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


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("42", 42),
        ("4.2", 4.2),
        ("true", True),
        ("null", None),
        ('"kick.wav"', "kick.wav"),
        ("bare-string", "bare-string"),
    ],
)
def test_parse_yaml_scalar_supports_supported_scalar_types(
    raw_value: str,
    expected: object,
) -> None:
    assert _parse_yaml_scalar(raw_value, path=Path("config.yaml"), lineno=1) == expected


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("camera:\n\twidth: 1280\n", "contains a tab indentation"),
        ("camera:\n width: 1280\n", "must use multiples of two spaces"),
        ("camera\n", "must contain a ':' separator"),
        (": value\n", "has an empty key"),
    ],
)
def test_yaml_loader_reports_parse_failures(contents: str, message: str, tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=re.escape(message)):
        _load_yaml_config(config_path)


@pytest.mark.parametrize(
    ("payload", "method_name", "message"),
    [
        ({"value": "text"}, "integer", "root.value: expected an integer"),
        ({"value": 2}, "number", "root.value: must be less than 1.0"),
        ({"value": 1}, "string", "root.value: expected a string"),
        ({"value": 1}, "boolean", "root.value: expected a boolean"),
    ],
)
def test_config_reader_errors_include_fully_qualified_paths(
    payload: dict[str, object],
    method_name: str,
    message: str,
) -> None:
    reader = _ConfigReader(payload, path="root")

    with pytest.raises(ConfigError, match=re.escape(message)):
        if method_name == "integer":
            reader.integer("value", default=0)
        elif method_name == "number":
            reader.number("value", default=0.0, maximum=1.0, inclusive_max=False)
        elif method_name == "string":
            reader.string("value", default=None)
        else:
            reader.boolean("value", default=False)


def test_load_config_rejects_unsupported_extensions(tmp_path: Path) -> None:
    config_path = tmp_path / "visionbeat.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported config file extension"):
        load_config(config_path)

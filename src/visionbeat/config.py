"""Configuration loading and validation for VisionBeat."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when VisionBeat configuration loading or validation fails."""


_MISSING = object()


@dataclass(frozen=True, slots=True)
class ConfigFieldError:
    """Structured validation error with a human-readable config path."""

    path: str
    message: str

    def render(self) -> str:
        """Return a formatted validation error string."""
        return f"{self.path}: {self.message}"


class _ConfigReader:
    """Utility that validates mapping access and type conversion with path-aware errors."""

    def __init__(self, payload: Mapping[str, Any], *, path: str) -> None:
        self.payload = dict(payload)
        self.path = path

    def reject_unknown_keys(self, allowed_keys: set[str]) -> None:
        """Raise a helpful error when unexpected keys are present."""
        unknown_keys = sorted(set(self.payload) - allowed_keys)
        if not unknown_keys:
            return
        allowed = ", ".join(sorted(allowed_keys))
        unknown = ", ".join(unknown_keys)
        raise ConfigError(
            f"{self.path}: unknown field(s): {unknown}. Allowed fields: {allowed}."
        )

    def child_mapping(self, key: str, *, default: Mapping[str, Any] | None = None) -> _ConfigReader:
        """Return a nested mapping reader for the requested key."""
        value = self.payload.get(key, _MISSING)
        child_path = f"{self.path}.{key}" if self.path else key
        if value is _MISSING:
            if default is None:
                raise ConfigError(f"{child_path}: missing required section.")
            value = default
        if not isinstance(value, Mapping):
            raise ConfigError(f"{child_path}: expected a mapping/table/object.")
        return _ConfigReader(value, path=child_path)

    def _get_raw(self, key: str, *, required: bool, default: Any) -> Any:
        value = self.payload.get(key, _MISSING)
        field_path = f"{self.path}.{key}" if self.path else key
        if value is _MISSING:
            if required:
                raise ConfigError(f"{field_path}: missing required field.")
            return default
        return value

    def string(
        self,
        key: str,
        *,
        default: str | None = None,
        required: bool = False,
        non_empty: bool = False,
    ) -> str | None:
        """Read and validate a string field."""
        value = self._get_raw(key, required=required, default=default)
        field_path = f"{self.path}.{key}" if self.path else key
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"{field_path}: expected a string.")
        normalized = value.strip()
        if non_empty and not normalized:
            raise ConfigError(f"{field_path}: must not be empty.")
        return normalized

    def boolean(self, key: str, *, default: bool) -> bool:
        """Read and validate a boolean field."""
        value = self._get_raw(key, required=False, default=default)
        field_path = f"{self.path}.{key}" if self.path else key
        if not isinstance(value, bool):
            raise ConfigError(f"{field_path}: expected a boolean.")
        return value

    def integer(
        self,
        key: str,
        *,
        default: int,
        minimum: int | None = None,
        allowed: set[int] | None = None,
    ) -> int:
        """Read and validate an integer field."""
        value = self._get_raw(key, required=False, default=default)
        field_path = f"{self.path}.{key}" if self.path else key
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{field_path}: expected an integer.")
        if minimum is not None and value < minimum:
            comparator = "greater than" if minimum == 1 else "greater than or equal to"
            target = 0 if minimum == 1 else minimum
            raise ConfigError(f"{field_path}: must be {comparator} {target}.")
        if allowed is not None and value not in allowed:
            allowed_text = ", ".join(str(item) for item in sorted(allowed))
            raise ConfigError(f"{field_path}: must be one of {allowed_text}.")
        return int(value)

    def number(
        self,
        key: str,
        *,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
        inclusive_min: bool = True,
        inclusive_max: bool = True,
    ) -> float:
        """Read and validate a numeric field."""
        value = self._get_raw(key, required=False, default=default)
        field_path = f"{self.path}.{key}" if self.path else key
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{field_path}: expected a number.")
        numeric = float(value)
        if minimum is not None:
            if inclusive_min and numeric < minimum:
                raise ConfigError(f"{field_path}: must be greater than or equal to {minimum}.")
            if not inclusive_min and numeric <= minimum:
                raise ConfigError(f"{field_path}: must be greater than {minimum}.")
        if maximum is not None:
            if inclusive_max and numeric > maximum:
                raise ConfigError(f"{field_path}: must be less than or equal to {maximum}.")
            if not inclusive_max and numeric >= maximum:
                raise ConfigError(f"{field_path}: must be less than {maximum}.")
        return numeric


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera device and preview-window configuration."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    window_name: str = "VisionBeat"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CameraConfig:
        """Build camera configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="camera")
        reader.reject_unknown_keys(
            {"device_index", "width", "height", "fps", "mirror", "window_name"}
        )
        return cls(
            device_index=reader.integer("device_index", default=0, minimum=0),
            width=reader.integer("width", default=1280, minimum=1),
            height=reader.integer("height", default=720, minimum=1),
            fps=reader.integer("fps", default=30, minimum=1),
            mirror=reader.boolean("mirror", default=True),
            window_name=reader.string("window_name", default="VisionBeat", non_empty=True)
            or "VisionBeat",
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraConfig:
        """Compatibility constructor from a plain dictionary."""
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "mirror": self.mirror,
            "window_name": self.window_name,
        }


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """MediaPipe tracking model configuration."""

    model_complexity: int = 1
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    enable_segmentation: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TrackerConfig:
        """Build tracker configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="tracker")
        reader.reject_unknown_keys(
            {
                "model_complexity",
                "min_detection_confidence",
                "min_tracking_confidence",
                "enable_segmentation",
            }
        )
        return cls(
            model_complexity=reader.integer(
                "model_complexity", default=1, allowed={0, 1, 2}
            ),
            min_detection_confidence=reader.number(
                "min_detection_confidence", default=0.55, minimum=0.0, maximum=1.0
            ),
            min_tracking_confidence=reader.number(
                "min_tracking_confidence", default=0.55, minimum=0.0, maximum=1.0
            ),
            enable_segmentation=reader.boolean("enable_segmentation", default=False),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrackerConfig:
        """Compatibility constructor from a plain dictionary."""
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "model_complexity": self.model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "enable_segmentation": self.enable_segmentation,
        }


@dataclass(frozen=True, slots=True)
class GestureThresholdsConfig:
    """Motion thresholds for gesture detection and confirmation."""

    punch_forward_delta_z: float = 0.2
    punch_max_vertical_drift: float = 0.1
    strike_down_delta_y: float = 0.26
    strike_max_depth_drift: float = 0.1
    min_velocity: float = 0.75
    candidate_ratio: float = 0.7
    axis_dominance_ratio: float = 1.7

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GestureThresholdsConfig:
        """Build gesture threshold configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="gestures.thresholds")
        reader.reject_unknown_keys(
            {
                "punch_forward_delta_z",
                "punch_max_vertical_drift",
                "strike_down_delta_y",
                "strike_max_depth_drift",
                "min_velocity",
                "candidate_ratio",
                "axis_dominance_ratio",
            }
        )
        return cls(
            punch_forward_delta_z=reader.number(
                "punch_forward_delta_z", default=0.2, minimum=0.0, inclusive_min=False
            ),
            punch_max_vertical_drift=reader.number(
                "punch_max_vertical_drift", default=0.1, minimum=0.0, inclusive_min=False
            ),
            strike_down_delta_y=reader.number(
                "strike_down_delta_y", default=0.26, minimum=0.0, inclusive_min=False
            ),
            strike_max_depth_drift=reader.number(
                "strike_max_depth_drift", default=0.1, minimum=0.0, inclusive_min=False
            ),
            min_velocity=reader.number(
                "min_velocity", default=0.75, minimum=0.0, inclusive_min=False
            ),
            candidate_ratio=reader.number(
                "candidate_ratio", default=0.7, minimum=0.0, maximum=1.0, inclusive_min=False
            ),
            axis_dominance_ratio=reader.number(
                "axis_dominance_ratio", default=1.7, minimum=1.0
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "punch_forward_delta_z": self.punch_forward_delta_z,
            "punch_max_vertical_drift": self.punch_max_vertical_drift,
            "strike_down_delta_y": self.strike_down_delta_y,
            "strike_max_depth_drift": self.strike_max_depth_drift,
            "min_velocity": self.min_velocity,
            "candidate_ratio": self.candidate_ratio,
            "axis_dominance_ratio": self.axis_dominance_ratio,
        }


@dataclass(frozen=True, slots=True)
class GestureCooldownsConfig:
    """Timing windows that control gesture buffering and cooldown behavior."""

    trigger_seconds: float = 0.2
    analysis_window_seconds: float = 0.18
    confirmation_window_seconds: float = 0.12

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GestureCooldownsConfig:
        """Build gesture timing configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="gestures.cooldowns")
        reader.reject_unknown_keys(
            {"trigger_seconds", "analysis_window_seconds", "confirmation_window_seconds"}
        )
        return cls(
            trigger_seconds=reader.number(
                "trigger_seconds", default=0.2, minimum=0.0, inclusive_min=False
            ),
            analysis_window_seconds=reader.number(
                "analysis_window_seconds", default=0.18, minimum=0.0, inclusive_min=False
            ),
            confirmation_window_seconds=reader.number(
                "confirmation_window_seconds", default=0.12, minimum=0.0, inclusive_min=False
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "trigger_seconds": self.trigger_seconds,
            "analysis_window_seconds": self.analysis_window_seconds,
            "confirmation_window_seconds": self.confirmation_window_seconds,
        }


@dataclass(frozen=True, slots=True)
class GestureConfig:
    """Thresholds and timing options for gesture recognition."""

    thresholds: GestureThresholdsConfig = field(default_factory=GestureThresholdsConfig)
    cooldowns: GestureCooldownsConfig = field(default_factory=GestureCooldownsConfig)
    history_size: int = 6
    active_hand: str = "right"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GestureConfig:
        """Build gesture configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="gestures")
        reader.reject_unknown_keys({"thresholds", "cooldowns", "history_size", "active_hand"})
        active_hand = reader.string("active_hand", default="right", non_empty=True) or "right"
        active_hand = active_hand.lower()
        if active_hand not in {"left", "right"}:
            raise ConfigError("gestures.active_hand: must be either 'left' or 'right'.")
        return cls(
            thresholds=GestureThresholdsConfig.from_mapping(
                reader.child_mapping("thresholds", default={}).payload
            ),
            cooldowns=GestureCooldownsConfig.from_mapping(
                reader.child_mapping("cooldowns", default={}).payload
            ),
            history_size=reader.integer("history_size", default=6, minimum=1),
            active_hand=active_hand,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GestureConfig:
        """Compatibility constructor from a plain dictionary."""
        return cls.from_mapping(payload)

    @property
    def punch_forward_delta_z(self) -> float:
        """Return the punch forward-distance threshold."""
        return self.thresholds.punch_forward_delta_z

    @property
    def punch_max_vertical_drift(self) -> float:
        """Return the maximum vertical punch drift."""
        return self.thresholds.punch_max_vertical_drift

    @property
    def strike_down_delta_y(self) -> float:
        """Return the downward-strike travel threshold."""
        return self.thresholds.strike_down_delta_y

    @property
    def strike_max_depth_drift(self) -> float:
        """Return the maximum depth drift for downward strikes."""
        return self.thresholds.strike_max_depth_drift

    @property
    def min_velocity(self) -> float:
        """Return the shared minimum gesture velocity threshold."""
        return self.thresholds.min_velocity

    @property
    def candidate_ratio(self) -> float:
        """Return the ratio used while a gesture is only a candidate."""
        return self.thresholds.candidate_ratio

    @property
    def axis_dominance_ratio(self) -> float:
        """Return the minimum dominance required on the primary motion axis."""
        return self.thresholds.axis_dominance_ratio

    @property
    def cooldown_seconds(self) -> float:
        """Return the post-trigger cooldown duration in seconds."""
        return self.cooldowns.trigger_seconds

    @property
    def analysis_window_seconds(self) -> float:
        """Return the rolling gesture analysis window in seconds."""
        return self.cooldowns.analysis_window_seconds

    @property
    def confirmation_window_seconds(self) -> float:
        """Return the maximum confirmation window in seconds."""
        return self.cooldowns.confirmation_window_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "thresholds": self.thresholds.to_dict(),
            "cooldowns": self.cooldowns.to_dict(),
            "history_size": self.history_size,
            "active_hand": self.active_hand,
        }


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio subsystem configuration for sample playback."""

    backend: str = "pygame"
    sample_rate: int = 44_100
    buffer_size: int = 256
    output_channels: int = 2
    simultaneous_voices: int = 16
    output_device_name: str | None = None
    sample_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "kick": "assets/samples/kick.wav",
            "snare": "assets/samples/snare.wav",
        }
    )
    volume: float = 0.9

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AudioConfig:
        """Build audio configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="audio")
        reader.reject_unknown_keys(
            {
                "backend",
                "sample_rate",
                "buffer_size",
                "output_channels",
                "simultaneous_voices",
                "output_device_name",
                "sample_mapping",
                "volume",
            }
        )
        backend = reader.string("backend", default="pygame", non_empty=True) or "pygame"
        backend = backend.lower()
        if backend != "pygame":
            raise ConfigError("audio.backend: must be 'pygame'.")
        sample_mapping_reader = reader.child_mapping(
            "sample_mapping",
            default={"kick": "assets/samples/kick.wav", "snare": "assets/samples/snare.wav"},
        )
        sample_mapping = _validate_sample_mapping(sample_mapping_reader)
        output_device_name = reader.string("output_device_name", default=None)
        return cls(
            backend=backend,
            sample_rate=reader.integer("sample_rate", default=44_100, minimum=1),
            buffer_size=reader.integer("buffer_size", default=256, minimum=1),
            output_channels=reader.integer("output_channels", default=2, minimum=1),
            simultaneous_voices=reader.integer("simultaneous_voices", default=16, minimum=1),
            output_device_name=output_device_name or None,
            sample_mapping=sample_mapping,
            volume=reader.number("volume", default=0.9, minimum=0.0, maximum=1.0),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AudioConfig:
        """Compatibility constructor from a plain dictionary."""
        return cls.from_mapping(payload)

    @property
    def sample_paths(self) -> dict[str, str]:
        """Return the configured sample-path mapping keyed by trigger name."""
        return dict(self.sample_mapping)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "backend": self.backend,
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size,
            "output_channels": self.output_channels,
            "simultaneous_voices": self.simultaneous_voices,
            "output_device_name": self.output_device_name,
            "sample_mapping": dict(self.sample_mapping),
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """External gesture transport configuration for downstream audio engines."""

    backend: str = "none"
    host: str = "127.0.0.1"
    port: int = 9000
    source: str = "visionbeat"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TransportConfig:
        """Build transport configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="transport")
        reader.reject_unknown_keys({"backend", "host", "port", "source"})
        backend = (reader.string("backend", default="none", non_empty=True) or "none").lower()
        if backend not in {"none", "udp"}:
            raise ConfigError("transport.backend: must be either 'none' or 'udp'.")
        host = reader.string("host", default="127.0.0.1", non_empty=True) or "127.0.0.1"
        source = reader.string("source", default="visionbeat", non_empty=True) or "visionbeat"
        return cls(
            backend=backend,
            host=host,
            port=reader.integer("port", default=9000, minimum=1),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transport configuration into a dictionary."""
        return {
            "backend": self.backend,
            "host": self.host,
            "port": self.port,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Options controlling the real-time debug overlay."""

    draw_landmarks: bool = True
    draw_velocity_vectors: bool = True
    show_debug_panel: bool = True

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        path: str = "debug.overlays",
    ) -> OverlayConfig:
        """Build overlay configuration from a validated mapping."""
        reader = _ConfigReader(payload, path=path)
        reader.reject_unknown_keys({"draw_landmarks", "draw_velocity_vectors", "show_debug_panel"})
        return cls(
            draw_landmarks=reader.boolean("draw_landmarks", default=True),
            draw_velocity_vectors=reader.boolean("draw_velocity_vectors", default=True),
            show_debug_panel=reader.boolean("show_debug_panel", default=True),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OverlayConfig:
        """Compatibility constructor from a plain dictionary."""
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "draw_landmarks": self.draw_landmarks,
            "draw_velocity_vectors": self.draw_velocity_vectors,
            "show_debug_panel": self.show_debug_panel,
        }


@dataclass(frozen=True, slots=True)
class DebugConfig:
    """Debug and diagnostics configuration."""

    overlays: OverlayConfig = field(default_factory=OverlayConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DebugConfig:
        """Build debug configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="debug")
        reader.reject_unknown_keys({"overlays"})
        return cls(
            overlays=OverlayConfig.from_mapping(
                reader.child_mapping("overlays", default={}).payload
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {"overlays": self.overlays.to_dict()}


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Process logging and observability configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    structured: bool = True
    event_log_path: str | None = None
    event_log_format: str = "jsonl"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> LoggingConfig:
        """Build logging configuration from a validated mapping."""
        reader = _ConfigReader(payload, path="logging")
        reader.reject_unknown_keys(
            {"level", "format", "structured", "event_log_path", "event_log_format"}
        )
        level = reader.string("level", default="INFO", non_empty=True) or "INFO"
        event_log_format = (
            reader.string("event_log_format", default="jsonl", non_empty=True) or "jsonl"
        ).lower()
        if event_log_format not in {"jsonl", "csv"}:
            raise ConfigError("logging.event_log_format: must be either 'jsonl' or 'csv'.")
        event_log_path = reader.string("event_log_path", default=None)
        return cls(
            level=level.upper(),
            format=reader.string(
                "format",
                default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                non_empty=True,
            )
            or "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            structured=reader.boolean("structured", default=True),
            event_log_path=event_log_path or None,
            event_log_format=event_log_format,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a dictionary."""
        return {
            "level": self.level,
            "format": self.format,
            "structured": self.structured,
            "event_log_path": self.event_log_path,
            "event_log_format": self.event_log_format,
        }


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration aggregating all subsystem settings."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def overlay(self) -> OverlayConfig:
        """Compatibility accessor for the overlay renderer."""
        return self.debug.overlays

    @property
    def log_level(self) -> str:
        """Compatibility accessor for code using the old top-level log level."""
        return self.logging.level

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AppConfig:
        """Create an application configuration from a nested mapping."""
        reader = _ConfigReader(payload, path="")
        reader.reject_unknown_keys(
            {"camera", "tracker", "gestures", "audio", "transport", "debug", "logging"}
        )
        return cls(
            camera=CameraConfig.from_mapping(reader.child_mapping("camera", default={}).payload),
            tracker=TrackerConfig.from_mapping(reader.child_mapping("tracker", default={}).payload),
            gestures=GestureConfig.from_mapping(
                reader.child_mapping("gestures", default={}).payload
            ),
            audio=AudioConfig.from_mapping(reader.child_mapping("audio", default={}).payload),
            transport=TransportConfig.from_mapping(
                reader.child_mapping("transport", default={}).payload
            ),
            debug=DebugConfig.from_mapping(reader.child_mapping("debug", default={}).payload),
            logging=LoggingConfig.from_mapping(reader.child_mapping("logging", default={}).payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the application configuration into a dictionary."""
        return {
            "camera": self.camera.to_dict(),
            "tracker": self.tracker.to_dict(),
            "gestures": self.gestures.to_dict(),
            "audio": self.audio.to_dict(),
            "transport": self.transport.to_dict(),
            "debug": self.debug.to_dict(),
            "logging": self.logging.to_dict(),
        }


def _validate_sample_mapping(reader: _ConfigReader) -> dict[str, str]:
    """Validate the audio sample mapping section."""
    if not reader.payload:
        raise ConfigError(
            "audio.sample_mapping: must define at least one gesture-to-sample mapping."
        )
    normalized: dict[str, str] = {}
    for key, value in reader.payload.items():
        field_path = f"{reader.path}.{key}" if reader.path else key
        if not isinstance(key, str):
            raise ConfigError(f"{reader.path}: all keys must be strings.")
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise ConfigError(f"{field_path}: mapping keys must not be empty.")
        if not isinstance(value, str):
            raise ConfigError(f"{field_path}: expected a string file path.")
        normalized_value = value.strip()
        if not normalized_value:
            raise ConfigError(f"{field_path}: sample path must not be empty.")
        normalized[normalized_key] = normalized_value
    return normalized


def _strip_yaml_comment(line: str) -> str:
    """Remove YAML comments while respecting quoted strings."""
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index].rstrip()
    return line.rstrip()


def _parse_yaml_scalar(raw_value: str, *, path: Path, lineno: int) -> Any:
    """Parse a minimal YAML scalar used by VisionBeat config files."""
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError as err:
        if ":" in value:
            raise ConfigError(
                f"Failed to parse config file {path}: line {lineno} uses unsupported YAML syntax."
            ) from err
        return value


def _load_yaml_config(path: Path) -> Mapping[str, Any]:
    """Load a restricted YAML mapping without external dependencies."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "	" in raw_line:
            raise ConfigError(
                f"Failed to parse config file {path}: line {lineno} contains a tab indentation."
            )
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ConfigError(
                f"Failed to parse config file {path}: line {lineno} "
                "must use multiples of two spaces."
            )
        while indent < stack[-1][0]:
            stack.pop()
        if indent > stack[-1][0]:
            raise ConfigError(
                f"Failed to parse config file {path}: line {lineno} has unexpected indentation."
            )
        content = line.strip()
        if ":" not in content:
            raise ConfigError(
                f"Failed to parse config file {path}: line {lineno} must contain a ':' separator."
            )
        key_text, value_text = content.split(":", 1)
        key = key_text.strip()
        if not key:
            raise ConfigError(
                f"Failed to parse config file {path}: line {lineno} has an empty key."
            )
        container = stack[-1][1]
        if not value_text.strip():
            child: dict[str, Any] = {}
            container[key] = child
            stack.append((indent + 2, child))
            continue
        container[key] = _parse_yaml_scalar(value_text, path=path, lineno=lineno)
    return root


def _load_raw_config(path: Path) -> Mapping[str, Any]:
    """Read a YAML or TOML configuration file into a mapping."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            data = _load_yaml_config(path)
        elif suffix == ".toml":
            with path.open("rb") as config_file:
                data = tomllib.load(config_file)
        else:
            raise ConfigError(
                f"Unsupported config file extension '{suffix or '<none>'}'. "
                "Use .yaml, .yml, or .toml."
            )
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse config file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Config file {path} must contain a top-level mapping/object, "
            f"not {type(data).__name__}."
        )
    return data


def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from a YAML or TOML file on disk."""
    config_path = Path(path)
    raw_config = _load_raw_config(config_path)
    try:
        return AppConfig.from_dict(raw_config)
    except ConfigError as exc:
        raise ConfigError(f"Invalid config file {config_path}: {exc}") from exc

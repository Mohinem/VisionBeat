"""Configuration loading and validation for VisionBeat."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require_positive_int(value: int, *, field_name: str) -> int:
    """Return a validated positive integer."""
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value

def _require_non_negative_int(value: int, *, field_name: str) -> int:
    """Return a validated non-negative integer."""
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")
    return value

def _require_probability(value: float, *, field_name: str) -> float:
    """Return a validated probability-like value in the inclusive range [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return value


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera device and preview-window configuration."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    window_name: str = "VisionBeat"

    def __post_init__(self) -> None:
        """Validate camera dimensions, frame rate, and window metadata."""
        object.__setattr__(
            self,
            "device_index",
            _require_non_negative_int(self.device_index, field_name="device_index"),
        )
        object.__setattr__(self, "width", _require_positive_int(self.width, field_name="width"))
        object.__setattr__(self, "height", _require_positive_int(self.height, field_name="height"))
        object.__setattr__(self, "fps", _require_positive_int(self.fps, field_name="fps"))
        if not self.window_name.strip():
            raise ValueError("window_name must not be empty.")
        object.__setattr__(self, "window_name", self.window_name.strip())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a JSON-friendly dictionary."""
        return {
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "mirror": self.mirror,
            "window_name": self.window_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraConfig:
        """Create a camera configuration from a mapping."""
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """MediaPipe tracking model configuration."""

    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False

    def __post_init__(self) -> None:
        """Validate tracker model complexity and confidence thresholds."""
        if self.model_complexity not in {0, 1, 2}:
            raise ValueError("model_complexity must be one of 0, 1, or 2.")
        object.__setattr__(
            self,
            "min_detection_confidence",
            _require_probability(
                self.min_detection_confidence,
                field_name="min_detection_confidence",
            ),
        )
        object.__setattr__(
            self,
            "min_tracking_confidence",
            _require_probability(
                self.min_tracking_confidence,
                field_name="min_tracking_confidence",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a JSON-friendly dictionary."""
        return {
            "model_complexity": self.model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "enable_segmentation": self.enable_segmentation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrackerConfig:
        """Create a tracker configuration from a mapping."""
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class GestureConfig:
    """Thresholds and behavioral options for gesture recognition."""

    punch_forward_delta_z: float = 0.18
    punch_max_vertical_drift: float = 0.12
    strike_down_delta_y: float = 0.22
    strike_max_depth_drift: float = 0.14
    min_velocity: float = 0.5
    cooldown_seconds: float = 0.18
    analysis_window_seconds: float = 0.18
    confirmation_window_seconds: float = 0.12
    candidate_ratio: float = 0.7
    axis_dominance_ratio: float = 1.5
    history_size: int = 6
    active_hand: str = "right"

    def __post_init__(self) -> None:
        """Validate gesture thresholds and active hand selection."""
        for field_name in (
            "punch_forward_delta_z",
            "punch_max_vertical_drift",
            "strike_down_delta_y",
            "strike_max_depth_drift",
            "min_velocity",
            "cooldown_seconds",
            "analysis_window_seconds",
            "confirmation_window_seconds",
            "candidate_ratio",
            "axis_dominance_ratio",
        ):
            value = float(getattr(self, field_name))
            if value <= 0.0:
                raise ValueError(f"{field_name} must be greater than zero.")
            object.__setattr__(self, field_name, value)
        if self.candidate_ratio > 1.0:
            raise ValueError("candidate_ratio must be less than or equal to 1.0.")
        if self.axis_dominance_ratio < 1.0:
            raise ValueError("axis_dominance_ratio must be greater than or equal to 1.0.")
        object.__setattr__(
            self,
            "history_size",
            _require_positive_int(self.history_size, field_name="history_size"),
        )
        active_hand = self.active_hand.strip().lower()
        if active_hand not in {"left", "right"}:
            raise ValueError("active_hand must be either 'left' or 'right'.")
        object.__setattr__(self, "active_hand", active_hand)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a JSON-friendly dictionary."""
        return {
            "punch_forward_delta_z": self.punch_forward_delta_z,
            "punch_max_vertical_drift": self.punch_max_vertical_drift,
            "strike_down_delta_y": self.strike_down_delta_y,
            "strike_max_depth_drift": self.strike_max_depth_drift,
            "min_velocity": self.min_velocity,
            "cooldown_seconds": self.cooldown_seconds,
            "analysis_window_seconds": self.analysis_window_seconds,
            "confirmation_window_seconds": self.confirmation_window_seconds,
            "candidate_ratio": self.candidate_ratio,
            "axis_dominance_ratio": self.axis_dominance_ratio,
            "history_size": self.history_size,
            "active_hand": self.active_hand,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GestureConfig:
        """Create a gesture configuration from a mapping."""
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio subsystem configuration for sample playback."""

    sample_rate: int = 44_100
    buffer_size: int = 256
    kick_sample: str = "assets/samples/kick.wav"
    snare_sample: str = "assets/samples/snare.wav"
    volume: float = 0.9

    def __post_init__(self) -> None:
        """Validate playback settings and sample path strings."""
        object.__setattr__(
            self,
            "sample_rate",
            _require_positive_int(self.sample_rate, field_name="sample_rate"),
        )
        object.__setattr__(
            self,
            "buffer_size",
            _require_positive_int(self.buffer_size, field_name="buffer_size"),
        )
        if not self.kick_sample.strip():
            raise ValueError("kick_sample must not be empty.")
        if not self.snare_sample.strip():
            raise ValueError("snare_sample must not be empty.")
        object.__setattr__(self, "kick_sample", self.kick_sample.strip())
        object.__setattr__(self, "snare_sample", self.snare_sample.strip())
        object.__setattr__(self, "volume", _require_probability(self.volume, field_name="volume"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a JSON-friendly dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size,
            "kick_sample": self.kick_sample,
            "snare_sample": self.snare_sample,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AudioConfig:
        """Create an audio configuration from a mapping."""
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Options controlling the real-time debug overlay."""

    draw_landmarks: bool = True
    draw_velocity_vectors: bool = True
    show_debug_panel: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into a JSON-friendly dictionary."""
        return {
            "draw_landmarks": self.draw_landmarks,
            "draw_velocity_vectors": self.draw_velocity_vectors,
            "show_debug_panel": self.show_debug_panel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OverlayConfig:
        """Create an overlay configuration from a mapping."""
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration aggregating all subsystem settings."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate nested configuration objects and normalize log level casing."""
        if not self.log_level.strip():
            raise ValueError("log_level must not be empty.")
        object.__setattr__(self, "log_level", self.log_level.strip().upper())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the application configuration into a dictionary."""
        return {
            "camera": self.camera.to_dict(),
            "tracker": self.tracker.to_dict(),
            "gestures": self.gestures.to_dict(),
            "audio": self.audio.to_dict(),
            "overlay": self.overlay.to_dict(),
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AppConfig:
        """Create an application configuration from a nested mapping."""
        return cls(
            camera=CameraConfig.from_dict(payload.get("camera", {})),
            tracker=TrackerConfig.from_dict(payload.get("tracker", {})),
            gestures=GestureConfig.from_dict(payload.get("gestures", {})),
            audio=AudioConfig.from_dict(payload.get("audio", {})),
            overlay=OverlayConfig.from_dict(payload.get("overlay", {})),
            log_level=payload.get("log_level", "INFO"),
        )

def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from a TOML file on disk."""
    with Path(path).open("rb") as config_file:
        data = tomllib.load(config_file)

    return AppConfig.from_dict(data)

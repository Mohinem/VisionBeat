"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CameraConfig:
    """Camera and display configuration."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    window_name: str = "VisionBeat"


@dataclass(slots=True)
class TrackerConfig:
    """MediaPipe tracker configuration."""

    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False


@dataclass(slots=True)
class GestureConfig:
    """Thresholds for gesture recognition."""

    punch_forward_delta_z: float = 0.18
    punch_max_vertical_drift: float = 0.12
    strike_down_delta_y: float = 0.22
    strike_max_depth_drift: float = 0.14
    min_velocity: float = 0.5
    cooldown_seconds: float = 0.18
    history_size: int = 6
    active_hand: str = "right"


@dataclass(slots=True)
class AudioConfig:
    """Audio playback configuration."""

    sample_rate: int = 44_100
    buffer_size: int = 256
    kick_sample: str = "assets/samples/kick.wav"
    snare_sample: str = "assets/samples/snare.wav"
    volume: float = 0.9


@dataclass(slots=True)
class OverlayConfig:
    """UI overlay configuration."""

    draw_landmarks: bool = True
    draw_velocity_vectors: bool = True
    show_debug_panel: bool = True


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    log_level: str = "INFO"


def _construct(dataclass_type: type[Any], values: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a dictionary of keyword values."""
    return dataclass_type(**(values or {}))


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a TOML file."""
    with Path(path).open("rb") as config_file:
        data = tomllib.load(config_file)

    return AppConfig(
        camera=_construct(CameraConfig, data.get("camera")),
        tracker=_construct(TrackerConfig, data.get("tracker")),
        gestures=_construct(GestureConfig, data.get("gestures")),
        audio=_construct(AudioConfig, data.get("audio")),
        overlay=_construct(OverlayConfig, data.get("overlay")),
        log_level=data.get("log_level", "INFO"),
    )

"""Audio engine abstractions and pygame-backed sample playback for VisionBeat."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionbeat.config import AudioConfig
from visionbeat.models import AudioTrigger, GestureType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SampleAsset:
    """Resolved metadata for a named audio sample asset."""

    name: str
    path: Path
    exists: bool


class AudioEngine(ABC):
    """Abstract interface for VisionBeat audio playback backends."""

    @abstractmethod
    def trigger_sound(self, name: str, intensity: float = 1.0) -> bool:
        """Trigger a named sound and return whether playback was started."""

    def trigger(
        self,
        trigger: AudioTrigger | GestureType | str,
        intensity: float | None = None,
    ) -> bool:
        """Trigger playback from a structured trigger, gesture enum, or raw sample name."""
        if isinstance(trigger, AudioTrigger):
            return self.trigger_sound(trigger.gesture.value, intensity=trigger.intensity)
        if isinstance(trigger, GestureType):
            return self.trigger_sound(
                trigger.value,
                intensity=1.0 if intensity is None else intensity,
            )
        return self.trigger_sound(
            trigger,
            intensity=1.0 if intensity is None else intensity,
        )

    @abstractmethod
    def close(self) -> None:
        """Release any backend-specific audio resources."""


@dataclass(slots=True)
class PygameAudioEngine(AudioEngine):
    """Concrete audio engine that loads and plays samples through ``pygame.mixer``."""

    config: AudioConfig
    pygame_module: Any | None = None
    _pygame: Any = field(init=False)
    _sounds: dict[str, Any] = field(init=False, default_factory=dict)
    _assets: dict[str, SampleAsset] = field(init=False, default_factory=dict)
    _mixer_ready: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initialize the pygame mixer and attempt to load configured sample assets."""
        self._pygame = self.pygame_module or self._import_pygame()
        self._assets = self._resolve_assets()
        self._initialize_mixer()
        self._sounds = self._load_available_sounds()

    def _import_pygame(self) -> Any:
        """Import pygame lazily so unit tests can inject a mock module."""
        import pygame

        return pygame

    def _resolve_assets(self) -> dict[str, SampleAsset]:
        """Resolve sample paths into named asset descriptors."""
        return {
            name: SampleAsset(name=name, path=Path(path), exists=Path(path).exists())
            for name, path in self.config.sample_paths.items()
        }

    def _initialize_mixer(self) -> None:
        """Initialize pygame's mixer subsystem if possible."""
        try:
            self._pygame.mixer.init(
                frequency=self.config.sample_rate,
                buffer=self.config.buffer_size,
                channels=self.config.output_channels,
                devicename=self.config.output_device_name,
            )
            self._pygame.mixer.set_num_channels(self.config.simultaneous_voices)
            self._mixer_ready = True
        except Exception as exc:  # pragma: no cover - defensive for environment variance
            logger.warning("Audio mixer initialization failed: %s", exc)
            self._mixer_ready = False

    def _load_available_sounds(self) -> dict[str, Any]:
        """Load any configured samples that are present on disk."""
        sounds: dict[str, Any] = {}
        for name, asset in self._assets.items():
            if not asset.exists:
                logger.warning(
                    "Audio sample '%s' is missing at %s. Playback for this sound is disabled.",
                    name,
                    asset.path,
                )
                continue
            if not self._mixer_ready:
                logger.warning(
                    "Audio sample '%s' is available, but the mixer is not ready. "
                    "Playback is disabled.",
                    name,
                )
                continue
            try:
                sound = self._pygame.mixer.Sound(asset.path.as_posix())
            except Exception as exc:  # pragma: no cover - depends on backend decoder behavior
                logger.warning(
                    "Failed to load audio sample '%s' from %s: %s",
                    name,
                    asset.path,
                    exc,
                )
                continue
            sound.set_volume(self.config.volume)
            sounds[name] = sound
        return sounds

    def available_sounds(self) -> tuple[str, ...]:
        """Return the names of samples that were loaded successfully."""
        return tuple(sorted(self._sounds))

    def missing_sounds(self) -> tuple[str, ...]:
        """Return the names of samples that could not be loaded."""
        return tuple(sorted(name for name in self._assets if name not in self._sounds))

    def trigger_sound(self, name: str, intensity: float = 1.0) -> bool:
        """Trigger a named sample with optional intensity scaling."""
        sound = self._sounds.get(name.strip().lower())
        if sound is None:
            logger.warning("Requested audio sample '%s' is unavailable.", name)
            return False
        normalized_intensity = max(0.0, min(float(intensity), 1.0))
        sound.set_volume(self.config.volume * normalized_intensity)
        sound.play()
        return True

    def close(self) -> None:
        """Shutdown the mixer subsystem if it was initialized."""
        if self._mixer_ready:
            self._pygame.mixer.quit()
            self._mixer_ready = False


def create_audio_engine(config: AudioConfig) -> AudioEngine:
    """Instantiate the configured audio engine backend."""
    if config.backend == "pygame":
        return PygameAudioEngine(config)
    raise ValueError(f"Unsupported audio backend: {config.backend}.")

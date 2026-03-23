"""Low-latency drum sample playback using pygame.mixer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionbeat.config import AudioConfig
from visionbeat.models import GestureType


@dataclass(slots=True)
class AudioEngine:
    """Load and trigger drum samples in response to gesture events."""

    config: AudioConfig
    _pygame: Any = field(init=False)
    _sounds: dict[GestureType, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize pygame's mixer subsystem and load sound assets."""
        import pygame

        self._validate_sample_paths()
        self._pygame = pygame
        self._pygame.mixer.init(
            frequency=self.config.sample_rate,
            buffer=self.config.buffer_size,
        )
        self._sounds = {
            GestureType.KICK: self._pygame.mixer.Sound(Path(self.config.kick_sample).as_posix()),
            GestureType.SNARE: self._pygame.mixer.Sound(Path(self.config.snare_sample).as_posix()),
        }
        for sound in self._sounds.values():
            sound.set_volume(self.config.volume)

    def _validate_sample_paths(self) -> None:
        """Ensure configured sample files exist before initializing audio."""
        missing = [
            path
            for path in (self.config.kick_sample, self.config.snare_sample)
            if not Path(path).exists()
        ]
        if missing:
            missing_display = ", ".join(missing)
            raise FileNotFoundError(
                "Missing sample files: "
                f"{missing_display}. Run `python scripts/generate_demo_samples.py` first."
            )

    def trigger(self, gesture: GestureType) -> None:
        """Play the sample assigned to a gesture."""
        self._sounds[gesture].play()

    def close(self) -> None:
        """Shutdown the mixer subsystem."""
        self._pygame.mixer.quit()

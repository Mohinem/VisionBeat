"""Core typed models shared across the VisionBeat runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class GestureType(StrEnum):
    """Supported gestural drum events."""

    KICK = "kick"
    SNARE = "snare"


@dataclass(slots=True)
class LandmarkPoint:
    """Single normalized landmark point from a tracker."""

    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass(slots=True)
class PoseFrame:
    """Tracker output normalized into a unit-testable structure."""

    timestamp: float
    landmarks: dict[str, LandmarkPoint] = field(default_factory=dict)

    def get(self, name: str) -> LandmarkPoint | None:
        """Return a landmark if present."""
        return self.landmarks.get(name)


@dataclass(slots=True)
class GestureEvent:
    """Detected gesture with an associated drum target."""

    gesture: GestureType
    confidence: float
    hand: str
    timestamp: float
    label: str

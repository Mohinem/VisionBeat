"""Serializable domain models used across VisionBeat subsystems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any


class GestureType(StrEnum):
    """Enumeration of supported gestures that can trigger audio events."""

    KICK = "kick"
    SNARE = "snare"


def _coerce_float(value: float | int, *, field_name: str) -> float:
    """Return a finite floating-point value for a model field."""
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class FrameTimestamp:
    """Monotonic timestamp attached to camera frames and derived events."""

    seconds: float

    def __post_init__(self) -> None:
        """Validate that the timestamp is finite and non-negative."""
        seconds = _coerce_float(self.seconds, field_name="seconds")
        if seconds < 0:
            raise ValueError("seconds must be greater than or equal to zero.")
        object.__setattr__(self, "seconds", seconds)

    def to_dict(self) -> dict[str, float]:
        """Serialize the timestamp into a JSON-friendly dictionary."""
        return {"seconds": self.seconds}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrameTimestamp:
        """Create a timestamp instance from a serialized mapping."""
        return cls(seconds=payload["seconds"])


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """Single normalized landmark point produced by a body or hand tracker."""

    x: float
    y: float
    z: float
    visibility: float = 1.0

    def __post_init__(self) -> None:
        """Validate numeric coordinates and normalized visibility."""
        object.__setattr__(self, "x", _coerce_float(self.x, field_name="x"))
        object.__setattr__(self, "y", _coerce_float(self.y, field_name="y"))
        object.__setattr__(self, "z", _coerce_float(self.z, field_name="z"))
        visibility = _coerce_float(self.visibility, field_name="visibility")
        if not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must be between 0.0 and 1.0.")
        object.__setattr__(self, "visibility", visibility)

    def to_dict(self) -> dict[str, float]:
        """Serialize the landmark into a JSON-friendly dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LandmarkPoint:
        """Create a landmark instance from a serialized mapping."""
        return cls(
            x=payload["x"],
            y=payload["y"],
            z=payload["z"],
            visibility=payload.get("visibility", 1.0),
        )

    def mirrored_horizontally(self) -> LandmarkPoint:
        """Return a copy mirrored across the vertical image centerline."""
        return LandmarkPoint(
            x=1.0 - self.x,
            y=self.y,
            z=self.z,
            visibility=self.visibility,
        )


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    """A scored candidate generated before a gesture becomes a final event."""

    gesture: GestureType
    confidence: float
    hand: str
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the confidence score and tracked hand label."""
        confidence = _coerce_float(self.confidence, field_name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        hand = self.hand.strip().lower()
        if hand not in {"left", "right"}:
            raise ValueError("hand must be either 'left' or 'right'.")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "hand", hand)
        object.__setattr__(self, "label", self.label.strip())

    def to_dict(self) -> dict[str, str | float]:
        """Serialize the candidate into a JSON-friendly dictionary."""
        return {
            "gesture": self.gesture.value,
            "confidence": self.confidence,
            "hand": self.hand,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DetectionCandidate:
        """Create a detection candidate from a serialized mapping."""
        return cls(
            gesture=GestureType(payload["gesture"]),
            confidence=payload["confidence"],
            hand=payload["hand"],
            label=payload.get("label", ""),
        )


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """A confirmed gesture emitted by the gesture detector."""

    gesture: GestureType
    confidence: float
    hand: str
    timestamp: FrameTimestamp
    label: str

    def __post_init__(self) -> None:
        """Validate confidence, hand selection, and timestamp payloads."""
        confidence = _coerce_float(self.confidence, field_name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        hand = self.hand.strip().lower()
        if hand not in {"left", "right"}:
            raise ValueError("hand must be either 'left' or 'right'.")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "hand", hand)
        object.__setattr__(self, "label", self.label.strip())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the gesture event into a JSON-friendly dictionary."""
        return {
            "gesture": self.gesture.value,
            "confidence": self.confidence,
            "hand": self.hand,
            "timestamp": self.timestamp.to_dict(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GestureEvent:
        """Create a gesture event from a serialized mapping."""
        return cls(
            gesture=GestureType(payload["gesture"]),
            confidence=payload["confidence"],
            hand=payload["hand"],
            timestamp=FrameTimestamp.from_dict(payload["timestamp"]),
            label=payload["label"],
        )


@dataclass(frozen=True, slots=True)
class AudioTrigger:
    """An audio playback request derived from a gesture or sequenced event."""

    gesture: GestureType
    timestamp: FrameTimestamp
    intensity: float = 1.0

    def __post_init__(self) -> None:
        """Validate trigger intensity and normalize the timestamp payload."""
        intensity = _coerce_float(self.intensity, field_name="intensity")
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("intensity must be between 0.0 and 1.0.")
        object.__setattr__(self, "intensity", intensity)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trigger into a JSON-friendly dictionary."""
        return {
            "gesture": self.gesture.value,
            "timestamp": self.timestamp.to_dict(),
            "intensity": self.intensity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AudioTrigger:
        """Create an audio trigger from a serialized mapping."""
        return cls(
            gesture=GestureType(payload["gesture"]),
            timestamp=FrameTimestamp.from_dict(payload["timestamp"]),
            intensity=payload.get("intensity", 1.0),
        )


@dataclass(frozen=True, slots=True)
class TrackerOutput:
    """Serializable output from a landmark tracker for a single processed frame."""

    timestamp: FrameTimestamp
    landmarks: dict[str, LandmarkPoint] = field(default_factory=dict)
    candidates: tuple[DetectionCandidate, ...] = ()
    person_detected: bool = False
    status: str = "no_person_detected"

    def __post_init__(self) -> None:
        """Normalize payload values and tracking-status metadata."""
        normalized_landmarks = {
            name: value if isinstance(value, LandmarkPoint) else LandmarkPoint.from_dict(value)
            for name, value in self.landmarks.items()
        }
        normalized_candidates = tuple(
            candidate
            if isinstance(candidate, DetectionCandidate)
            else DetectionCandidate.from_dict(candidate)
            for candidate in self.candidates
        )
        status = self.status.strip() or "unknown"
        person_detected = bool(self.person_detected or normalized_landmarks)
        object.__setattr__(self, "landmarks", normalized_landmarks)
        object.__setattr__(self, "candidates", normalized_candidates)
        object.__setattr__(self, "person_detected", person_detected)
        object.__setattr__(self, "status", status)

    def get(self, name: str) -> LandmarkPoint | None:
        """Return a landmark by name if it is present in the frame."""
        return self.landmarks.get(name)

    def mirrored_horizontally(self) -> TrackerOutput:
        """Return tracker output mirrored across the vertical image centerline."""
        return TrackerOutput(
            timestamp=self.timestamp,
            landmarks={
                name: point.mirrored_horizontally()
                for name, point in self.landmarks.items()
            },
            candidates=self.candidates,
            person_detected=self.person_detected,
            status=self.status,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker output into a JSON-friendly dictionary."""
        return {
            "timestamp": self.timestamp.to_dict(),
            "landmarks": {name: point.to_dict() for name, point in self.landmarks.items()},
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "person_detected": self.person_detected,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrackerOutput:
        """Create tracker output from a serialized mapping."""
        return cls(
            timestamp=FrameTimestamp.from_dict(payload["timestamp"]),
            landmarks={
                name: LandmarkPoint.from_dict(point)
                for name, point in payload.get("landmarks", {}).items()
            },
            candidates=tuple(
                DetectionCandidate.from_dict(candidate)
                for candidate in payload.get("candidates", [])
            ),
            person_detected=payload.get("person_detected", False),
            status=payload.get("status", "no_person_detected"),
        )


@dataclass(frozen=True, slots=True)
class RenderState:
    """Business-state payload consumed by the overlay renderer."""

    pose: TrackerOutput
    frame_index: int
    fps: float | None = None
    current_candidate: DetectionCandidate | None = None
    confirmed_gesture: GestureEvent | None = None
    cooldown_remaining_seconds: float = 0.0
    detector_status: str | None = None
    audio_status: str | None = None

    def __post_init__(self) -> None:
        """Validate loop-derived overlay state values."""
        frame_index = int(self.frame_index)
        if frame_index < 0:
            raise ValueError("frame_index must be greater than or equal to zero.")
        object.__setattr__(self, "frame_index", frame_index)
        if self.fps is not None:
            fps = _coerce_float(self.fps, field_name="fps")
            if fps <= 0.0:
                raise ValueError("fps must be greater than zero when provided.")
            object.__setattr__(self, "fps", fps)
        cooldown = _coerce_float(
            self.cooldown_remaining_seconds,
            field_name="cooldown_remaining_seconds",
        )
        if cooldown < 0.0:
            raise ValueError("cooldown_remaining_seconds must be greater than or equal to zero.")
        object.__setattr__(self, "cooldown_remaining_seconds", cooldown)
        if self.detector_status is not None:
            object.__setattr__(self, "detector_status", self.detector_status.strip() or None)
        if self.audio_status is not None:
            object.__setattr__(self, "audio_status", self.audio_status.strip() or None)


PoseFrame = TrackerOutput

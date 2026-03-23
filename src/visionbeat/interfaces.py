"""Abstract protocol definitions for VisionBeat subsystems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from visionbeat.models import AudioTrigger, FrameTimestamp, GestureEvent, TrackerOutput

Frame = Any


@runtime_checkable
class CameraSource(Protocol):
    """Interface for components that provide sequential camera frames to the app."""

    def open(self) -> None:
        """Acquire underlying camera resources and prepare frame capture."""

    def read(self) -> Frame:
        """Return the next available frame from the configured capture source."""

    def close(self) -> None:
        """Release any camera resources held by the implementation."""


@runtime_checkable
class LandmarkTracker(Protocol):
    """Interface for trackers that convert raw frames into normalized landmarks."""

    def process(self, frame: Frame, timestamp: FrameTimestamp) -> TrackerOutput:
        """Produce tracker output for a captured frame and its timestamp."""

    def close(self) -> None:
        """Release model or native resources held by the tracker."""


@runtime_checkable
class GestureDetector(Protocol):
    """Interface for detectors that transform tracker output into gesture events."""

    def update(self, frame: TrackerOutput) -> Sequence[GestureEvent]:
        """Consume tracker output and emit zero or more recognized gestures."""


@runtime_checkable
class AudioEngine(Protocol):
    """Interface for audio backends that respond to gesture-driven trigger events."""

    def trigger(self, trigger: AudioTrigger) -> None:
        """Play or schedule audio for the provided trigger payload."""

    def close(self) -> None:
        """Release audio resources held by the playback backend."""


@runtime_checkable
class OverlayRenderer(Protocol):
    """Interface for renderers that draw tracker and gesture state onto frames."""

    def render(
        self,
        frame: Frame,
        pose: TrackerOutput,
        events: Sequence[GestureEvent],
    ) -> Frame:
        """Return a rendered frame containing visual debugging or performance overlays."""

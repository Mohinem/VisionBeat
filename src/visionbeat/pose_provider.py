"""Pose-tracking backend abstractions and provider selection helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

from visionbeat.config import TrackerConfig
from visionbeat.models import FrameTimestamp, TrackerOutput

Frame = Any
SUPPORTED_POSE_BACKENDS: Final[tuple[str, ...]] = ("mediapipe", "movenet")


class PoseBackendError(RuntimeError):
    """Base error raised for unsupported or unavailable pose backends."""


class PoseBackendUnavailableError(PoseBackendError):
    """Raised when a selected pose backend exists but cannot be used."""


class PoseProvider(ABC):
    """Backend interface that emits project-native normalized landmarks.

    The normalized contract is `TrackerOutput`, which keeps gesture detection
    independent from backend-specific SDK objects and landmark enums.
    """

    @abstractmethod
    def process(self, frame: Frame, timestamp: FrameTimestamp | float) -> TrackerOutput:
        """Extract normalized landmarks from a camera frame."""

    @abstractmethod
    def close(self) -> None:
        """Release native resources held by the backend."""


def create_pose_provider(config: TrackerConfig) -> PoseProvider:
    """Create the configured pose backend implementation."""
    backend = config.backend.lower()
    if backend == "mediapipe":
        from visionbeat.mediapipe_provider import MediaPipePoseProvider

        return MediaPipePoseProvider(config)
    if backend == "movenet":
        from visionbeat.movenet_provider import MoveNetPoseProvider

        return MoveNetPoseProvider(config)
    raise PoseBackendError(
        f"Unsupported pose backend '{config.backend}'. "
        f"Expected one of: {', '.join(SUPPORTED_POSE_BACKENDS)}."
    )

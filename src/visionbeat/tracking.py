"""MediaPipe-based landmark tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from visionbeat.config import TrackerConfig
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput

Frame = Any
_POSE_LANDMARKS: Final[dict[str, int]] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


@dataclass(slots=True)
class PoseTracker:
    """Wrapper around MediaPipe Pose that emits normalized landmarks."""

    config: TrackerConfig
    _cv2: Any | None = field(init=False, default=None)
    _pose: Any = field(init=False)

    def __post_init__(self) -> None:
        """Construct the underlying MediaPipe pose tracker."""
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_segmentation=self.config.enable_segmentation,
        )

    def process(self, frame: Frame, timestamp: FrameTimestamp | float) -> TrackerOutput:
        """Extract tracker output from a BGR webcam frame."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2

        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb_frame)
        landmarks: dict[str, LandmarkPoint] = {}
        if result.pose_landmarks is not None:
            for name, index in _POSE_LANDMARKS.items():
                landmark = result.pose_landmarks.landmark[index]
                landmarks[name] = LandmarkPoint(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility,
                )
        frame_timestamp = (
            timestamp
            if isinstance(timestamp, FrameTimestamp)
            else FrameTimestamp(seconds=timestamp)
        )
        return TrackerOutput(timestamp=frame_timestamp, landmarks=landmarks)

    def close(self) -> None:
        """Close MediaPipe resources."""
        self._pose.close()

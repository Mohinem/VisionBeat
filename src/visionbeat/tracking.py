"""MediaPipe-based landmark tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final

from visionbeat.config import TrackerConfig
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput

logger = logging.getLogger(__name__)
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
    """Wrapper around MediaPipe Pose that emits the minimum upper-body landmarks."""

    config: TrackerConfig
    _cv2: Any | None = field(init=False, default=None)
    _pose: Any = field(init=False)

    @staticmethod
    def _load_pose_factory() -> Any:
        """Load the MediaPipe pose factory across supported package layouts."""
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "pose"):
            return solutions.pose.Pose

        # Some wheels expose solutions only via `mediapipe.solutions`.
        try:
            from mediapipe import solutions as top_level_solutions
        except ImportError:
            top_level_solutions = None
        if top_level_solutions is not None and hasattr(top_level_solutions, "pose"):
            return top_level_solutions.pose.Pose

        # MediaPipe wheels may expose solutions via `mediapipe.python.solutions`.
        try:
            from mediapipe.python import solutions as python_solutions
        except ImportError as exc:
            msg = (
                "Unable to locate MediaPipe Pose API. Expected one of "
                "`mediapipe.solutions.pose` or `mediapipe.python.solutions.pose`."
            )
            raise RuntimeError(msg) from exc

        return python_solutions.pose.Pose

    def __post_init__(self) -> None:
        """Construct the underlying MediaPipe pose tracker."""
        logger.info(
            "Initializing pose tracker complexity=%s detection_threshold=%.2f "
            "tracking_threshold=%.2f",
            self.config.model_complexity,
            self.config.min_detection_confidence,
            self.config.min_tracking_confidence,
        )
        pose_factory = self._load_pose_factory()
        self._pose = pose_factory(
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
        frame_timestamp = (
            timestamp
            if isinstance(timestamp, FrameTimestamp)
            else FrameTimestamp(seconds=timestamp)
        )

        if result.pose_landmarks is None:
            logger.debug(
                "Tracking status=no_person_detected timestamp=%.3f",
                frame_timestamp.seconds,
            )
            return TrackerOutput(
                timestamp=frame_timestamp,
                landmarks={},
                person_detected=False,
                status="no_person_detected",
            )

        landmarks: dict[str, LandmarkPoint] = {}
        for name, index in _POSE_LANDMARKS.items():
            landmark = result.pose_landmarks.landmark[index]
            if landmark.visibility < self.config.min_tracking_confidence:
                continue
            landmarks[name] = LandmarkPoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
            )

        status = "tracking" if landmarks else "landmarks_below_confidence_threshold"
        logger.debug(
            "Tracking status=%s timestamp=%.3f landmarks=%s",
            status,
            frame_timestamp.seconds,
            sorted(landmarks),
        )
        return TrackerOutput(
            timestamp=frame_timestamp,
            landmarks=landmarks,
            person_detected=True,
            status=status,
        )

    def close(self) -> None:
        """Close MediaPipe resources."""
        logger.info("Closing pose tracker")
        self._pose.close()

"""Frame overlay rendering utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from visionbeat.config import OverlayConfig
from visionbeat.models import GestureEvent, PoseFrame

Frame = Any

_LANDMARK_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


class OverlayRenderer:
    """Draw tracking and gesture feedback onto webcam frames."""

    def __init__(self, config: OverlayConfig, *, cv2_module: Any | None = None) -> None:
        """Store overlay configuration and an optional OpenCV-compatible module."""
        self.config = config
        self._cv2 = cv2_module

    def render(self, frame: Frame, pose: PoseFrame, events: Sequence[GestureEvent]) -> Frame:
        """Render landmarks, status, and the latest gesture labels on a frame copy."""
        output = frame.copy()

        if self.config.draw_landmarks:
            draw_pose_landmarks(output, pose, cv2_module=self._cv2)

        if self.config.show_debug_panel:
            labels = [event.label for event in events[-2:]]
            if not labels:
                labels = [
                    "Tracking lost"
                    if not pose.person_detected
                    else pose.status.replace("_", " ")
                ]
            draw_labels(output, [f"VisionBeat ({pose.status})", *labels], cv2_module=self._cv2)

        return output


def draw_pose_landmarks(frame: Frame, pose: PoseFrame, *, cv2_module: Any | None = None) -> Frame:
    """Draw tracked upper-body landmarks and simple bone connections."""
    cv2 = cv2_module
    if cv2 is None:
        import cv2 as cv2  # type: ignore[no-redef]

    height, width = frame.shape[:2]
    for start_name, end_name in _LANDMARK_CONNECTIONS:
        start = pose.get(start_name)
        end = pose.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(
            frame,
            (int(start.x * width), int(start.y * height)),
            (int(end.x * width), int(end.y * height)),
            (80, 180, 255),
            2,
        )

    for name, landmark in pose.landmarks.items():
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 6, (50, 220, 120), -1)
        cv2.putText(
            frame,
            name.replace("_", " "),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    return frame


def draw_labels(
    frame: Frame,
    labels: Iterable[str],
    *,
    cv2_module: Any | None = None,
    origin: tuple[int, int] = (12, 12),
) -> Frame:
    """Draw a compact label panel for tracker status and recent events."""
    cv2 = cv2_module
    if cv2 is None:
        import cv2 as cv2  # type: ignore[no-redef]

    label_list = [label for label in labels if label.strip()]
    if not label_list:
        return frame

    x0, y0 = origin
    line_height = 28
    panel_height = 20 + line_height * len(label_list)
    panel_width = 440
    cv2.rectangle(frame, (x0, y0), (x0 + panel_width, y0 + panel_height), (20, 20, 20), -1)
    for index, label in enumerate(label_list, start=1):
        cv2.putText(
            frame,
            label,
            (x0 + 12, y0 + 12 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (100, 220, 255) if index > 1 else (255, 255, 255),
            2,
        )
    return frame

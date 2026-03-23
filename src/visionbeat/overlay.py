"""Frame overlay rendering utilities."""

from __future__ import annotations

from typing import Any

from visionbeat.config import OverlayConfig
from visionbeat.models import GestureEvent, PoseFrame

Frame = Any


class OverlayRenderer:
    """Draw tracking and gesture feedback onto webcam frames."""

    def __init__(self, config: OverlayConfig) -> None:
        """Store overlay configuration."""
        self.config = config

    def render(self, frame: Frame, pose: PoseFrame, events: list[GestureEvent]) -> Frame:
        """Render landmarks and the latest gesture labels on a frame copy."""
        import cv2

        output = frame.copy()
        height, width = output.shape[:2]

        if self.config.draw_landmarks:
            for landmark in pose.landmarks.values():
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                cv2.circle(output, (cx, cy), 8, (50, 220, 120), -1)

        if self.config.show_debug_panel:
            cv2.rectangle(output, (12, 12), (420, 120), (20, 20, 20), -1)
            cv2.putText(
                output,
                "VisionBeat",
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            labels = [event.label for event in events[-2:]] or ["Listening for gestures..."]
            for index, label in enumerate(labels, start=1):
                cv2.putText(
                    output,
                    label,
                    (24, 44 + 28 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 220, 255),
                    2,
                )

        return output

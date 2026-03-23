"""Webcam capture abstraction for VisionBeat."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from visionbeat.config import CameraConfig

logger = logging.getLogger(__name__)
Frame = Any


@dataclass(slots=True)
class CameraStream:
    """Thin wrapper around ``cv2.VideoCapture`` for easier testing and teardown."""

    config: CameraConfig
    _capture: Any = None

    def open(self) -> None:
        """Open the configured capture device."""
        import cv2

        self._capture = cv2.VideoCapture(self.config.device_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open webcam device {self.config.device_index}.")
        logger.info("Opened webcam device %s", self.config.device_index)

    def read(self) -> Frame:
        """Read the next frame from the webcam."""
        if self._capture is None:
            raise RuntimeError("Camera stream has not been opened.")
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError("Failed to read frame from webcam.")
        if self.config.mirror:
            import cv2

            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        """Release the capture device if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Closed webcam device")

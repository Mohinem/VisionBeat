"""Webcam capture abstractions for VisionBeat."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from visionbeat.config import CameraConfig
from visionbeat.interfaces import CameraSource as CameraSourceProtocol
from visionbeat.observability import ObservabilityRecorder

logger = logging.getLogger(__name__)
Frame = Any


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """Single frame captured from the configured camera source."""

    image: Frame
    captured_at: float
    frame_index: int


@dataclass(slots=True)
class CameraSource(CameraSourceProtocol):
    """OpenCV-backed camera source with configurable frame size and FPS targets."""

    config: CameraConfig
    recorder: ObservabilityRecorder | None = None
    _cv2: Any | None = field(default=None)
    _capture: Any = field(default=None)
    _frame_index: int = field(default=0, init=False)

    def open(self) -> None:
        """Open the configured capture device and apply capture settings."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2

        logger.info(
            "Starting camera device=%s width=%s height=%s target_fps=%s",
            self.config.device_index,
            self.config.width,
            self.config.height,
            self.config.fps,
        )
        self._capture = self._cv2.VideoCapture(self.config.device_index)
        self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(self._cv2.CAP_PROP_FPS, self.config.fps)
        if not self._capture.isOpened():
            if self.recorder is not None:
                self.recorder.log_camera_initialization(
                    device_index=self.config.device_index,
                    width=self.config.width,
                    height=self.config.height,
                    target_fps=self.config.fps,
                    mirror=self.config.mirror,
                    opened=False,
                )
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Unable to open webcam device {self.config.device_index}.")
        self._frame_index = 0
        if self.recorder is not None:
            self.recorder.log_camera_initialization(
                device_index=self.config.device_index,
                width=self.config.width,
                height=self.config.height,
                target_fps=self.config.fps,
                mirror=self.config.mirror,
                opened=True,
            )
        logger.info("Camera device %s opened successfully", self.config.device_index)

    def read(self) -> Frame:
        """Read and return the next image frame from the camera source."""
        return self.read_frame().image

    def read_frame(self) -> CameraFrame:
        """Read the next frame together with capture metadata for logging and tests."""
        if self._capture is None:
            raise RuntimeError("Camera source has not been opened.")

        success, frame = self._capture.read()
        if not success:
            logger.warning("Camera read failed for device %s", self.config.device_index)
            raise RuntimeError("Failed to read frame from webcam.")

        if self.config.mirror:
            assert self._cv2 is not None
            frame = self._cv2.flip(frame, 1)

        metadata = CameraFrame(
            image=frame,
            captured_at=time.monotonic(),
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        logger.debug("Captured camera frame index=%s", metadata.frame_index)
        return metadata

    def close(self) -> None:
        """Release the capture device if it is currently open."""
        if self._capture is not None:
            logger.info("Closing camera device %s", self.config.device_index)
            self._capture.release()
            self._capture = None


CameraStream = CameraSource

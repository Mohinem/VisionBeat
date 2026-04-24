"""Webcam capture abstractions for VisionBeat."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionbeat.config import CameraConfig
from visionbeat.interfaces import CameraSource as CameraSourceProtocol
from visionbeat.observability import ObservabilityRecorder

logger = logging.getLogger(__name__)
Frame = Any
_LATEST_FRAME_BUFFER_SIZE = 1
_BACKEND_CONSTANTS = {
    "v4l2": "CAP_V4L2",
    "dshow": "CAP_DSHOW",
    "msmf": "CAP_MSMF",
    "avfoundation": "CAP_AVFOUNDATION",
    "gstreamer": "CAP_GSTREAMER",
    "ffmpeg": "CAP_FFMPEG",
}


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """Single frame captured from the configured camera source."""

    image: Frame
    captured_at: float
    frame_index: int
    display_image: Frame | None = None
    mirrored_for_display: bool = False


@dataclass(frozen=True, slots=True)
class CameraCaptureMode:
    """Resolved camera mode accepted by OpenCV after negotiation."""

    backend: str | None
    width: int | None
    height: int | None
    fps: float | None
    fourcc: str | None


@dataclass(slots=True)
class CameraSource(CameraSourceProtocol):
    """OpenCV-backed camera source with configurable frame size and FPS targets."""

    config: CameraConfig
    recorder: ObservabilityRecorder | None = None
    _cv2: Any | None = field(default=None)
    _capture: Any = field(default=None)
    _frame_index: int = field(default=0, init=False)
    _capture_mode: CameraCaptureMode | None = field(default=None, init=False)

    def open(self) -> None:
        """Open the configured capture device and apply capture settings."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2

        logger.info(
            "Starting camera device=%s width=%s height=%s target_fps=%s backend=%s fourcc=%s",
            self.config.device_index,
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.backend,
            self.config.fourcc or "auto",
        )
        self._capture = self._open_capture()
        self._apply_capture_settings()
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
            raise RuntimeError(
                "Unable to open webcam device "
                f"{self.config.device_index}. Check camera permissions, whether another app "
                "is using the webcam, or try --camera-index with a different device."
            )
        self._capture_mode = self._read_capture_mode()
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
        if self._capture_mode is not None:
            logger.info(
                "Camera device %s opened successfully negotiated_backend=%s "
                "resolution=%sx%s fps=%.3f fourcc=%s",
                self.config.device_index,
                self._capture_mode.backend or "unknown",
                self._capture_mode.width or self.config.width,
                self._capture_mode.height or self.config.height,
                self._capture_mode.fps or 0.0,
                self._capture_mode.fourcc or "unknown",
            )
            self._warn_if_mode_mismatch(self._capture_mode)
        else:
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
            raise RuntimeError(
                "Failed to read frame from webcam. Ensure the webcam remains connected and not "
                "locked by another application."
            )

        display_frame = frame
        if self.config.mirror:
            assert self._cv2 is not None
            display_frame = self._cv2.flip(frame, 1)

        metadata = CameraFrame(
            image=frame,
            display_image=display_frame,
            captured_at=time.monotonic(),
            frame_index=self._frame_index,
            mirrored_for_display=self.config.mirror,
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
            self._capture_mode = None

    def capture_mode(self) -> CameraCaptureMode | None:
        """Return the last negotiated camera mode when available."""
        return self._capture_mode

    def _open_capture(self) -> Any:
        assert self._cv2 is not None
        if self.config.backend == "auto":
            return self._cv2.VideoCapture(self.config.device_index)
        constant_name = _BACKEND_CONSTANTS[self.config.backend]
        api_preference = getattr(self._cv2, constant_name, None)
        if api_preference is None:
            raise RuntimeError(
                f"Configured camera backend '{self.config.backend}' is not available in this OpenCV build."
            )
        capture = self._cv2.VideoCapture(self.config.device_index, int(api_preference))
        if bool(getattr(capture, "isOpened", lambda: False)()):
            return capture
        fallback_source = self._device_path_fallback_source()
        if fallback_source is None:
            return capture
        logger.warning(
            "Camera backend %s could not open device index %s directly; retrying with device path %s",
            self.config.backend,
            self.config.device_index,
            fallback_source,
        )
        try:
            capture.release()
        except Exception:
            pass
        return self._cv2.VideoCapture(fallback_source, int(api_preference))

    def _apply_capture_settings(self) -> None:
        assert self._capture is not None
        assert self._cv2 is not None
        fourcc_property = getattr(self._cv2, "CAP_PROP_FOURCC", None)
        if self.config.fourcc is not None and fourcc_property is not None:
            encoded_fourcc = self._cv2.VideoWriter_fourcc(*self.config.fourcc)
            self._capture.set(fourcc_property, encoded_fourcc)
        self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(self._cv2.CAP_PROP_FPS, self.config.fps)
        buffer_size_property = getattr(self._cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_size_property is not None:
            # Ask OpenCV to keep the capture queue shallow so the runtime processes
            # the newest frame instead of building perceptible lag behind the camera.
            self._capture.set(buffer_size_property, _LATEST_FRAME_BUFFER_SIZE)

    def _read_capture_mode(self) -> CameraCaptureMode | None:
        assert self._capture is not None
        assert self._cv2 is not None
        width = self._read_property(getattr(self._cv2, "CAP_PROP_FRAME_WIDTH", None))
        height = self._read_property(getattr(self._cv2, "CAP_PROP_FRAME_HEIGHT", None))
        fps = self._read_property(getattr(self._cv2, "CAP_PROP_FPS", None))
        backend_code = self._read_property(getattr(self._cv2, "CAP_PROP_BACKEND", None))
        fourcc_value = self._read_property(getattr(self._cv2, "CAP_PROP_FOURCC", None))
        if width is None and height is None and fps is None and backend_code is None and fourcc_value is None:
            return None
        return CameraCaptureMode(
            backend=_decode_backend_name(self._cv2, backend_code),
            width=int(round(width)) if width not in {None, 0.0} else None,
            height=int(round(height)) if height not in {None, 0.0} else None,
            fps=fps if fps not in {None, 0.0} else None,
            fourcc=_decode_fourcc(fourcc_value),
        )

    def _read_property(self, property_id: int | None) -> float | None:
        if property_id is None or self._capture is None:
            return None
        getter = getattr(self._capture, "get", None)
        if not callable(getter):
            return None
        try:
            value = getter(property_id)
        except Exception:
            return None
        if value is None:
            return None
        return float(value)

    def _warn_if_mode_mismatch(self, capture_mode: CameraCaptureMode) -> None:
        if capture_mode.width not in {None, self.config.width} or capture_mode.height not in {
            None,
            self.config.height,
        }:
            logger.warning(
                "Camera resolution request was not honored: requested=%sx%s negotiated=%sx%s",
                self.config.width,
                self.config.height,
                capture_mode.width or "?",
                capture_mode.height or "?",
            )
        if capture_mode.fps is not None and capture_mode.fps + 1.0 < float(self.config.fps):
            logger.warning(
                "Camera FPS request was not honored: requested=%s negotiated=%.3f. "
                "This usually means the current webcam mode cannot sustain the requested rate.",
                self.config.fps,
                capture_mode.fps,
            )
        if self.config.fourcc is not None and capture_mode.fourcc not in {None, self.config.fourcc}:
            logger.warning(
                "Camera FOURCC request was not honored: requested=%s negotiated=%s",
                self.config.fourcc,
                capture_mode.fourcc,
            )

    def _device_path_fallback_source(self) -> str | None:
        if self.config.backend != "v4l2":
            return None
        if self.config.device_index < 0:
            return None
        device_path = Path(f"/dev/video{self.config.device_index}")
        if not device_path.exists():
            return None
        return device_path.as_posix()


def _decode_fourcc(raw_value: float | None) -> str | None:
    if raw_value in {None, 0.0}:
        return None
    integer = int(round(raw_value))
    decoded = "".join(chr((integer >> (8 * offset)) & 0xFF) for offset in range(4))
    cleaned = decoded.replace("\x00", "").strip()
    return cleaned or None


def _decode_backend_name(cv2_module: Any, raw_value: float | None) -> str | None:
    if raw_value is None:
        return None
    backend_code = int(round(raw_value))
    for backend_name, constant_name in _BACKEND_CONSTANTS.items():
        constant_value = getattr(cv2_module, constant_name, None)
        if constant_value is not None and int(constant_value) == backend_code:
            return backend_name
    cap_any = getattr(cv2_module, "CAP_ANY", None)
    if cap_any is not None and int(cap_any) == backend_code:
        return "auto"
    return str(backend_code)


CameraStream = CameraSource

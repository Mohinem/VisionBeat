"""MoveNet pose backend that normalizes landmarks into TrackerOutput."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Final
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

from visionbeat.config import TrackerConfig
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput
from visionbeat.pose_provider import PoseBackendUnavailableError, PoseProvider

logger = logging.getLogger(__name__)
Frame = Any
_MOVENET_MODEL_URL: Final[str] = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/"
    "tflite/float16/4?lite-format=tflite"
)
_MOVENET_MODEL_FILENAME: Final[str] = "movenet_singlepose_lightning_float16.tflite"
_MOVENET_KEYPOINTS: Final[dict[str, int]] = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
}


@dataclass(slots=True)
class MoveNetPoseProvider(PoseProvider):
    """TensorFlow Lite-backed MoveNet single-pose provider."""

    config: TrackerConfig
    _cv2: Any | None = field(init=False, default=None)
    _interpreter: Any = field(init=False)
    _input_details: dict[str, Any] = field(init=False)
    _output_details: dict[str, Any] = field(init=False)
    _input_height: int = field(init=False)
    _input_width: int = field(init=False)

    @staticmethod
    def _load_interpreter_class(import_failures: list[str]) -> Any:
        """Load a TensorFlow Lite interpreter from the available runtime."""
        try:
            module = import_module("tflite_runtime.interpreter")
        except ImportError as exc:
            import_failures.append(f"tflite_runtime.interpreter: {exc}")
        else:
            interpreter_cls = getattr(module, "Interpreter", None)
            if interpreter_cls is not None:
                return interpreter_cls
            import_failures.append("tflite_runtime.interpreter: missing Interpreter")

        try:
            tensorflow = import_module("tensorflow")
        except ImportError as exc:
            import_failures.append(f"tensorflow: {exc}")
        else:
            lite = getattr(tensorflow, "lite", None)
            interpreter_cls = getattr(lite, "Interpreter", None) if lite is not None else None
            if interpreter_cls is not None:
                return interpreter_cls
            import_failures.append("tensorflow.lite: missing Interpreter")

        failure_lines = (
            "\n".join(f"- {failure}" for failure in import_failures)
            or "- no import errors captured"
        )
        msg = (
            "Pose backend 'movenet' requires a TensorFlow Lite runtime.\n"
            "Install it with `python -m pip install -e .[movenet]` or install TensorFlow.\n"
            f"Import attempts:\n{failure_lines}"
        )
        raise PoseBackendUnavailableError(msg)

    @staticmethod
    def _ensure_model_asset() -> Path:
        """Ensure the MoveNet model asset is present locally."""
        model_dir = Path.home() / ".cache" / "visionbeat" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / _MOVENET_MODEL_FILENAME
        if model_path.exists():
            return model_path

        try:
            urlretrieve(_MOVENET_MODEL_URL, model_path)
        except (URLError, OSError) as exc:
            msg = f"Failed to download MoveNet model from {_MOVENET_MODEL_URL}: {exc}"
            raise PoseBackendUnavailableError(msg) from exc
        return model_path

    def __post_init__(self) -> None:
        """Initialize the MoveNet interpreter and inspect tensor shapes."""
        import_failures: list[str] = []
        interpreter_cls = self._load_interpreter_class(import_failures)
        model_path = self._ensure_model_asset()
        try:
            self._interpreter = interpreter_cls(model_path=str(model_path))
        except Exception as exc:
            msg = (
                "Pose backend 'movenet' could not initialize its TensorFlow Lite "
                "interpreter. If you installed `tflite-runtime`, make sure the "
                "environment uses `numpy<2`.\n"
                f"Original error: {exc}"
            )
            raise PoseBackendUnavailableError(msg) from exc
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]
        input_shape = self._input_details["shape"]
        self._input_height = int(input_shape[1])
        self._input_width = int(input_shape[2])
        logger.info(
            "Initializing pose backend=%s model=%s input=%sx%s",
            self.config.backend,
            model_path.name,
            self._input_width,
            self._input_height,
        )

    def process(self, frame: Frame, timestamp: FrameTimestamp | float) -> TrackerOutput:
        """Run MoveNet on a BGR frame and emit normalized project landmarks."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2

        frame_timestamp = (
            timestamp
            if isinstance(timestamp, FrameTimestamp)
            else FrameTimestamp(seconds=timestamp)
        )
        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        input_tensor, scale, pad_top, pad_left = self._prepare_input(rgb_frame)

        self._interpreter.set_tensor(self._input_details["index"], input_tensor)
        self._interpreter.invoke()
        keypoints = np.asarray(
            self._interpreter.get_tensor(self._output_details["index"]),
            dtype=np.float32,
        )[0, 0]
        person_detected = float(np.max(keypoints[:, 2])) >= self.config.min_detection_confidence
        if not person_detected:
            return TrackerOutput(
                timestamp=frame_timestamp,
                landmarks={},
                person_detected=False,
                status="no_person_detected",
            )

        frame_height, frame_width = rgb_frame.shape[:2]
        landmarks: dict[str, LandmarkPoint] = {}
        for name, index in _MOVENET_KEYPOINTS.items():
            score = float(keypoints[index, 2])
            if score < self.config.min_tracking_confidence:
                continue
            raw_y = float(keypoints[index, 0]) * self._input_height
            raw_x = float(keypoints[index, 1]) * self._input_width
            y = ((raw_y - pad_top) / scale) / frame_height
            x = ((raw_x - pad_left) / scale) / frame_width
            landmarks[name] = LandmarkPoint(
                x=self._clamp_unit_interval(x),
                y=self._clamp_unit_interval(y),
                # MoveNet is 2D-only, so the shared z axis is left neutral.
                z=0.0,
                visibility=self._clamp_unit_interval(score),
            )

        status = "tracking" if landmarks else "landmarks_below_confidence_threshold"
        return TrackerOutput(
            timestamp=frame_timestamp,
            landmarks=landmarks,
            person_detected=True,
            status=status,
        )

    def close(self) -> None:
        """Release interpreter resources when supported by the runtime."""
        closer = getattr(self._interpreter, "close", None)
        if callable(closer):
            closer()

    def _prepare_input(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize with padding so MoveNet keeps the original aspect ratio."""
        frame_height, frame_width = rgb_frame.shape[:2]
        scale = min(self._input_width / frame_width, self._input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = self._cv2.resize(rgb_frame, (resized_width, resized_height))

        padded = np.zeros((self._input_height, self._input_width, 3), dtype=np.uint8)
        pad_top = (self._input_height - resized_height) // 2
        pad_left = (self._input_width - resized_width) // 2
        padded[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized

        input_tensor = np.expand_dims(padded, axis=0)
        input_dtype = self._input_details["dtype"]
        quantization = self._input_details.get("quantization", (0.0, 0))
        scale_factor, zero_point = float(quantization[0]), int(quantization[1])
        if np.issubdtype(input_dtype, np.integer) and scale_factor > 0.0:
            quantized = np.round(input_tensor.astype(np.float32) / scale_factor) + zero_point
            input_tensor = quantized.astype(input_dtype)
        else:
            input_tensor = input_tensor.astype(input_dtype)
        return input_tensor, scale, pad_top, pad_left

    @staticmethod
    def _clamp_unit_interval(value: float) -> float:
        """Clamp a floating-point value into [0.0, 1.0]."""
        return min(1.0, max(0.0, value))

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visionbeat.config import TrackerConfig
from visionbeat.models import FrameTimestamp
from visionbeat.movenet_provider import MoveNetPoseProvider
from visionbeat.pose_provider import PoseBackendUnavailableError


class FakeCV2:
    COLOR_BGR2RGB = 99

    def __init__(self) -> None:
        self.cvt_calls: list[tuple[object, int]] = []
        self.resize_calls: list[tuple[tuple[int, ...], tuple[int, int]]] = []

    def cvtColor(self, frame: object, code: int) -> object:  # noqa: N802
        self.cvt_calls.append((frame, code))
        return frame

    def resize(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        self.resize_calls.append((frame.shape, size))
        width, height = size
        return np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)


class FakeInterpreter:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.closed = False
        self.tensors: list[tuple[int, np.ndarray]] = []
        self.invocations = 0

    def set_tensor(self, index: int, tensor: np.ndarray) -> None:
        self.tensors.append((index, tensor))

    def invoke(self) -> None:
        self.invocations += 1

    def get_tensor(self, index: int) -> np.ndarray:
        assert index == 1
        return self.output

    def close(self) -> None:
        self.closed = True


def make_provider(output: np.ndarray) -> MoveNetPoseProvider:
    provider = object.__new__(MoveNetPoseProvider)
    provider.config = TrackerConfig(
        backend="movenet",
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    provider._cv2 = FakeCV2()
    provider._interpreter = FakeInterpreter(output)
    provider._input_details = {
        "index": 0,
        "shape": np.array([1, 4, 4, 3]),
        "dtype": np.uint8,
        "quantization": (0.0, 0),
    }
    provider._output_details = {"index": 1}
    provider._input_height = 4
    provider._input_width = 4
    return provider


def test_movenet_provider_returns_structured_landmarks() -> None:
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    keypoints[0, 0, 5] = [0.25, 0.5, 0.90]
    keypoints[0, 0, 9] = [0.5, 0.25, 0.95]
    provider = make_provider(keypoints)
    frame = np.zeros((2, 4, 3), dtype=np.uint8)

    output = provider.process(frame, FrameTimestamp(seconds=1.0))

    assert output.person_detected is True
    assert output.status == "tracking"
    assert set(output.landmarks) == {"left_shoulder", "left_wrist"}
    assert output.get("left_shoulder").x == pytest.approx(0.5)
    assert output.get("left_shoulder").y == pytest.approx(0.0)
    assert output.get("left_wrist").x == pytest.approx(0.25)
    assert output.get("left_wrist").y == pytest.approx(0.5)
    assert output.get("left_wrist").z == pytest.approx(0.0)


def test_movenet_provider_handles_missing_person_gracefully() -> None:
    keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
    keypoints[0, 0, 5] = [0.25, 0.5, 0.20]
    provider = make_provider(keypoints)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    output = provider.process(frame, 2.0)

    assert output.person_detected is False
    assert output.status == "no_person_detected"
    assert output.landmarks == {}


def test_movenet_provider_close_releases_interpreter_resources() -> None:
    provider = make_provider(np.zeros((1, 1, 17, 3), dtype=np.float32))

    provider.close()

    assert provider._interpreter.closed is True


def test_movenet_provider_raises_clear_error_when_runtime_missing(monkeypatch) -> None:
    def fake_import_module(name: str) -> SimpleNamespace:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr("visionbeat.movenet_provider.import_module", fake_import_module)

    with pytest.raises(PoseBackendUnavailableError, match="\\[movenet\\]"):
        MoveNetPoseProvider._load_interpreter_class([])

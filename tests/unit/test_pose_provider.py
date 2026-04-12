from __future__ import annotations

import numpy as np
import pytest

from visionbeat.config import TrackerConfig
from visionbeat.pose_provider import (
    PoseBackendUnavailableError,
    create_pose_provider,
    resize_frame_for_tracking,
)


def test_create_pose_provider_builds_mediapipe_backend(monkeypatch) -> None:
    class StubProvider:
        def __init__(self, config: TrackerConfig) -> None:
            self.config = config

    monkeypatch.setattr(
        "visionbeat.mediapipe_provider.MediaPipePoseProvider",
        StubProvider,
    )

    provider = create_pose_provider(TrackerConfig(backend="mediapipe"))

    assert isinstance(provider, StubProvider)
    assert provider.config.backend == "mediapipe"


def test_create_pose_provider_builds_movenet_backend(monkeypatch) -> None:
    class StubProvider:
        def __init__(self, config: TrackerConfig) -> None:
            self.config = config

    monkeypatch.setattr(
        "visionbeat.movenet_provider.MoveNetPoseProvider",
        StubProvider,
    )

    provider = create_pose_provider(TrackerConfig(backend="movenet"))

    assert isinstance(provider, StubProvider)
    assert provider.config.backend == "movenet"


def test_create_pose_provider_propagates_backend_unavailable_errors(monkeypatch) -> None:
    def _raise(_: TrackerConfig) -> None:
        raise PoseBackendUnavailableError("runtime missing")

    monkeypatch.setattr(
        "visionbeat.movenet_provider.MoveNetPoseProvider",
        _raise,
    )

    with pytest.raises(PoseBackendUnavailableError, match="runtime missing"):
        create_pose_provider(TrackerConfig(backend="movenet"))


def test_resize_frame_for_tracking_downscales_wide_frames() -> None:
    class FakeCV2:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], tuple[int, int]]] = []

        def resize(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
            self.calls.append((frame.shape, size))
            width, height = size
            return np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake_cv2 = FakeCV2()

    resized = resize_frame_for_tracking(frame, cv2_module=fake_cv2, max_input_width=640)

    assert resized.shape == (360, 640, 3)
    assert fake_cv2.calls == [((720, 1280, 3), (640, 360))]

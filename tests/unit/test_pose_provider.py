from __future__ import annotations

import pytest

from visionbeat.config import TrackerConfig
from visionbeat.pose_provider import (
    PoseBackendUnavailableError,
    create_pose_provider,
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

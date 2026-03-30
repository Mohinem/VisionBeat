from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import pytest

from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput

MotionPoint: TypeAlias = tuple[float, tuple[float, float, float]]
MotionSequence: TypeAlias = tuple[MotionPoint, ...]


@pytest.fixture
def motion_sequences() -> dict[str, MotionSequence]:
    return {
        "stationary": (
            (0.00, (0.50, 0.40, -0.10)),
            (0.05, (0.50, 0.40, -0.10)),
            (0.10, (0.50, 0.40, -0.10)),
            (0.15, (0.50, 0.40, -0.10)),
        ),
        "outward_jab": (
            (0.00, (0.50, 0.40, -0.08)),
            (0.05, (0.60, 0.41, -0.09)),
            (0.10, (0.66, 0.41, -0.10)),
        ),
        "downward_strike": (
            (0.00, (0.55, 0.20, -0.05)),
            (0.05, (0.56, 0.36, -0.06)),
            (0.10, (0.57, 0.45, -0.07)),
        ),
        "noisy_movement": (
            (0.00, (0.50, 0.30, -0.10)),
            (0.03, (0.58, 0.36, -0.18)),
            (0.06, (0.46, 0.25, -0.07)),
            (0.09, (0.59, 0.40, -0.21)),
            (0.12, (0.48, 0.31, -0.12)),
            (0.15, (0.60, 0.43, -0.19)),
        ),
        "borderline_threshold_movement": (
            (0.00, (0.50, 0.40, -0.10)),
            (0.05, (0.50, 0.40, -0.226)),
            (0.10, (0.50, 0.40, -0.280)),
        ),
    }


@pytest.fixture
def tracker_output_factory() -> Callable[..., TrackerOutput]:
    def build(
        timestamp: float,
        *,
        right_wrist: tuple[float, float, float] | None = None,
        left_wrist: tuple[float, float, float] | None = None,
        right_shoulder: tuple[float, float, float] | None = None,
        left_shoulder: tuple[float, float, float] | None = None,
        right_visibility: float = 1.0,
        left_visibility: float = 1.0,
        right_shoulder_visibility: float = 1.0,
        left_shoulder_visibility: float = 1.0,
        status: str = "tracking",
        person_detected: bool | None = None,
    ) -> TrackerOutput:
        landmarks: dict[str, LandmarkPoint] = {}
        if right_wrist is not None and right_shoulder is None:
            right_shoulder = (0.40, 0.20, -0.02)
        if left_wrist is not None and left_shoulder is None:
            left_shoulder = (0.60, 0.20, -0.02)
        if right_wrist is not None:
            x, y, z = right_wrist
            landmarks["right_wrist"] = LandmarkPoint(
                x=x,
                y=y,
                z=z,
                visibility=right_visibility,
            )
        if left_wrist is not None:
            x, y, z = left_wrist
            landmarks["left_wrist"] = LandmarkPoint(
                x=x,
                y=y,
                z=z,
                visibility=left_visibility,
            )
        if right_shoulder is not None:
            x, y, z = right_shoulder
            landmarks["right_shoulder"] = LandmarkPoint(
                x=x,
                y=y,
                z=z,
                visibility=right_shoulder_visibility,
            )
        if left_shoulder is not None:
            x, y, z = left_shoulder
            landmarks["left_shoulder"] = LandmarkPoint(
                x=x,
                y=y,
                z=z,
                visibility=left_shoulder_visibility,
            )
        return TrackerOutput(
            timestamp=FrameTimestamp(seconds=timestamp),
            landmarks=landmarks,
            person_detected=bool(landmarks) if person_detected is None else person_detected,
            status=status,
        )

    return build


@pytest.fixture
def sequence_to_frames(
    tracker_output_factory: Callable[..., TrackerOutput],
) -> Callable[..., list[TrackerOutput]]:
    def build(
        sequence: Sequence[MotionPoint],
        *,
        hand: str = "right",
        visibility: float = 1.0,
    ) -> list[TrackerOutput]:
        frames: list[TrackerOutput] = []
        for timestamp, wrist in sequence:
            kwargs = {
                f"{hand}_wrist": wrist,
                f"{hand}_visibility": visibility,
            }
            frames.append(tracker_output_factory(timestamp, **kwargs))
        return frames

    return build

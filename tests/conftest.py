from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from tests.synthetic_motion import MotionPoint, MotionSequence, SyntheticMotionGenerator
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput


@pytest.fixture
def synthetic_motion_generator() -> SyntheticMotionGenerator:
    return SyntheticMotionGenerator(frame_interval=0.05, seed=11)


@pytest.fixture
def motion_sequences(
    synthetic_motion_generator: SyntheticMotionGenerator,
) -> dict[str, MotionSequence]:
    return {
        "stationary": synthetic_motion_generator.stationary_hand(duration=0.15),
        "forward_punch": synthetic_motion_generator.forward_punch(duration=0.10, velocity=2.5),
        "downward_strike": synthetic_motion_generator.downward_strike(duration=0.10, velocity=2.5),
        "noisy_movement": synthetic_motion_generator.jitter_noise(duration=0.15, noise=0.06),
        "borderline_threshold_movement": synthetic_motion_generator.forward_punch(
            duration=0.10,
            velocity=1.8,
            noise=0.0,
        ),
    }


@pytest.fixture
def tracker_output_factory() -> Callable[..., TrackerOutput]:
    def build(
        timestamp: float,
        *,
        right_wrist: tuple[float, float, float] | None = None,
        left_wrist: tuple[float, float, float] | None = None,
        right_visibility: float = 1.0,
        left_visibility: float = 1.0,
        status: str = "tracking",
        person_detected: bool | None = None,
    ) -> TrackerOutput:
        landmarks: dict[str, LandmarkPoint] = {}
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

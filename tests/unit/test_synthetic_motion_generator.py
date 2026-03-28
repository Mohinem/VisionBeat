from __future__ import annotations

import pytest

from tests.synthetic_motion import SyntheticMotionGenerator


def test_generator_is_deterministic_for_same_seed() -> None:
    generator_a = SyntheticMotionGenerator(seed=99)
    generator_b = SyntheticMotionGenerator(seed=99)

    assert generator_a.jitter_noise(duration=0.2, noise=0.02) == generator_b.jitter_noise(
        duration=0.2,
        noise=0.02,
    )


def test_generator_velocity_and_duration_control_displacement() -> None:
    generator = SyntheticMotionGenerator(frame_interval=0.05, seed=1)

    sequence = generator.forward_punch(velocity=2.0, duration=0.15, noise=0.0)

    assert len(sequence) == 4
    start_z = sequence[0][1][2]
    end_z = sequence[-1][1][2]
    assert (end_z - start_z) == pytest.approx(-0.30)


def test_generator_noise_changes_samples() -> None:
    generator = SyntheticMotionGenerator(seed=10)

    clean = generator.stationary_hand(duration=0.1, noise=0.0)
    noisy = generator.stationary_hand(duration=0.1, noise=0.02)

    assert clean != noisy


@pytest.mark.parametrize(
    ("builder", "expected_delta_sign"),
    [
        ("forward_punch", "negative_z"),
        ("downward_strike", "positive_y"),
        ("stationary_hand", "none"),
    ],
)
def test_generator_gesture_shapes_match_expected_axes(
    builder: str,
    expected_delta_sign: str,
) -> None:
    generator = SyntheticMotionGenerator(seed=3)
    build = getattr(generator, builder)
    sequence = build(duration=0.1, noise=0.0)
    start = sequence[0][1]
    end = sequence[-1][1]

    if expected_delta_sign == "negative_z":
        assert end[2] < start[2]
    elif expected_delta_sign == "positive_y":
        assert end[1] > start[1]
    else:
        assert end == start


def test_to_tracker_outputs_matches_gesture_detector_structure() -> None:
    generator = SyntheticMotionGenerator(seed=5)
    sequence = generator.forward_punch(duration=0.1, velocity=2.5)

    frames = generator.to_tracker_outputs(sequence, hand="right", visibility=0.9)

    assert len(frames) == len(sequence)
    assert frames[0].status == "tracking"
    assert frames[0].person_detected is True
    wrist = frames[0].landmarks["right_wrist"]
    assert wrist.visibility == pytest.approx(0.9)


def test_invalid_duration_raises_value_error() -> None:
    generator = SyntheticMotionGenerator()

    with pytest.raises(ValueError, match="duration"):
        generator.forward_punch(duration=0.0)

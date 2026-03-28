from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest

from visionbeat.config import GestureConfig, GestureCooldownsConfig, GestureThresholdsConfig
from visionbeat.gestures import COMPARISON_EPSILON, GestureDetector, MotionSample
from visionbeat.models import FrameTimestamp, GestureEvent, GestureType, TrackerOutput
from visionbeat.observability import GestureObservationEvent


@dataclass
class FakeGestureObserver:
    candidates: list[GestureObservationEvent] = field(default_factory=list)
    triggers: list[GestureObservationEvent] = field(default_factory=list)
    cooldown_suppressions: list[GestureObservationEvent] = field(default_factory=list)

    def log_gesture_candidate(self, event: GestureObservationEvent) -> None:
        self.candidates.append(event)

    def log_confirmed_trigger(self, event: GestureObservationEvent) -> None:
        self.triggers.append(event)

    def log_cooldown_suppression(self, event: GestureObservationEvent) -> None:
        self.cooldown_suppressions.append(event)


def feed_frames(detector: GestureDetector, frames: list[TrackerOutput]) -> list[GestureEvent]:
    events: list[GestureEvent] = []
    for frame in frames:
        events.extend(detector.update(frame))
    return events


def test_no_trigger_when_stationary(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldowns=GestureCooldownsConfig(analysis_window_seconds=0.2),
        )
    )

    events = feed_frames(detector, sequence_to_frames(motion_sequences["stationary"]))

    assert events == []
    assert detector.candidates == ()


@pytest.mark.parametrize(
    ("sequence_name", "expected_gesture", "expected_label"),
    [
        ("forward_punch", GestureType.KICK, "Forward punch → kick"),
        ("downward_strike", GestureType.SNARE, "Downward strike → snare"),
    ],
)
def test_detector_confirms_expected_gestures_for_synthetic_sequences(
    sequence_name: str,
    expected_gesture: GestureType,
    expected_label: str,
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.15,
                trigger_seconds=0.05,
            ),
            thresholds=GestureThresholdsConfig(
                strike_down_delta_y=0.15,
                min_velocity=0.5,
            ),
        )
    )

    events = feed_frames(detector, sequence_to_frames(motion_sequences[sequence_name]))

    assert len(events) == 1
    assert events[0].gesture is expected_gesture
    assert events[0].label == expected_label


@pytest.mark.parametrize(
    ("travel_scale", "expected_candidate", "expected_confirmed"),
    [
        (0.999, True, False),
        (1.0, True, True),
        (1.001, True, True),
    ],
)
def test_punch_threshold_boundaries_are_stable(
    travel_scale: float,
    expected_candidate: bool,
    expected_confirmed: bool,
    tracker_output_factory,
) -> None:
    config = GestureConfig(
        history_size=6,
        cooldowns=GestureCooldownsConfig(
            analysis_window_seconds=0.2,
            confirmation_window_seconds=0.15,
            trigger_seconds=0.05,
        ),
        thresholds=GestureThresholdsConfig(
            punch_forward_delta_z=0.20,
            punch_max_vertical_drift=0.10,
            min_velocity=0.75,
            candidate_ratio=0.7,
            axis_dominance_ratio=1.7,
        ),
    )
    detector = GestureDetector(config)
    target_delta_z = config.punch_forward_delta_z * travel_scale
    frames = [
        tracker_output_factory(0.00, right_wrist=(0.50, 0.40, -0.05)),
        tracker_output_factory(0.10, right_wrist=(0.505, 0.405, -0.05 - target_delta_z)),
        tracker_output_factory(0.20, right_wrist=(0.505, 0.405, -0.05 - target_delta_z)),
    ]

    first = detector.update(frames[0])
    second = detector.update(frames[1])
    candidate_count_after_second = len(detector.candidates)
    third = detector.update(frames[2])

    assert first == []
    assert second == []
    assert (candidate_count_after_second == 1) is expected_candidate
    assert (len(third) == 1) is expected_confirmed


@pytest.mark.parametrize(
    ("travel_scale", "expected_candidate", "expected_confirmed"),
    [
        (0.999, True, False),
        (1.0, True, True),
        (1.001, True, True),
    ],
)
def test_strike_threshold_boundaries_are_stable(
    travel_scale: float,
    expected_candidate: bool,
    expected_confirmed: bool,
    tracker_output_factory,
) -> None:
    config = GestureConfig(
        history_size=6,
        cooldowns=GestureCooldownsConfig(
            analysis_window_seconds=0.2,
            confirmation_window_seconds=0.15,
            trigger_seconds=0.05,
        ),
        thresholds=GestureThresholdsConfig(
            strike_down_delta_y=0.22,
            strike_max_depth_drift=0.10,
            min_velocity=0.5,
            candidate_ratio=0.7,
            axis_dominance_ratio=1.7,
        ),
    )
    detector = GestureDetector(config)
    target_delta_y = config.strike_down_delta_y * travel_scale
    frames = [
        tracker_output_factory(0.00, right_wrist=(0.50, 0.20, -0.05)),
        tracker_output_factory(0.10, right_wrist=(0.505, 0.20 + target_delta_y, -0.055)),
        tracker_output_factory(0.20, right_wrist=(0.505, 0.20 + target_delta_y, -0.055)),
    ]

    first = detector.update(frames[0])
    second = detector.update(frames[1])
    candidate_count_after_second = len(detector.candidates)
    third = detector.update(frames[2])

    assert first == []
    assert second == []
    assert (candidate_count_after_second == 1) is expected_candidate
    assert (len(third) == 1) is expected_confirmed


def test_duplicate_trigger_regression_respects_cooldown_and_logs_suppression(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    observer = FakeGestureObserver()
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.30,
                analysis_window_seconds=0.25,
                confirmation_window_seconds=0.15,
            ),
        ),
        observer=observer,
    )
    frames = sequence_to_frames(motion_sequences["forward_punch"])
    repeated_frames = frames + sequence_to_frames(
        (
            (0.20, (0.50, 0.40, -0.10)),
            (0.25, (0.50, 0.41, -0.22)),
            (0.28, (0.50, 0.42, -0.34)),
        )
    )

    events = feed_frames(detector, repeated_frames)

    assert [event.gesture for event in events] == [GestureType.KICK]
    assert len(observer.cooldown_suppressions) >= 1
    assert observer.cooldown_suppressions[-1].reason == "cooldown_active"


@pytest.mark.parametrize(
    "sequence_name",
    ["noisy_movement", "borderline_threshold_movement"],
)
def test_non_trigger_sequences_do_not_emit_false_positives(
    sequence_name: str,
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=8,
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.12,
            ),
            thresholds=GestureThresholdsConfig(
                punch_forward_delta_z=0.2,
                strike_down_delta_y=0.22,
                min_velocity=0.75,
                axis_dominance_ratio=1.7,
            ),
        )
    )

    events = feed_frames(detector, sequence_to_frames(motion_sequences[sequence_name]))

    assert events == []


def test_inactive_hand_is_ignored(tracker_output_factory) -> None:
    detector = GestureDetector(GestureConfig(active_hand="left"))

    events = detector.update(tracker_output_factory(0.1, right_wrist=(0.5, 0.5, -0.5)))

    assert events == []
    assert detector.candidates == ()


def test_low_visibility_clears_pending_candidate(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    tracker_output_factory,
) -> None:
    detector = GestureDetector(GestureConfig())
    candidate_frames = motion_sequences["forward_punch"][:2]

    for timestamp, wrist in candidate_frames:
        detector.update(tracker_output_factory(timestamp, right_wrist=wrist))

    assert len(detector.candidates) == 1
    detector.update(
        tracker_output_factory(
            0.10,
            right_wrist=cast(tuple[float, float, float], candidate_frames[-1][1]),
            right_visibility=0.1,
        )
    )

    assert detector.candidates == ()


def test_candidate_expires_when_confirmation_window_is_exceeded(
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.05,
                trigger_seconds=0.2,
            )
        )
    )
    detector.update(tracker_output_factory(0.00, right_wrist=(0.50, 0.40, -0.08)))
    detector.update(tracker_output_factory(0.05, right_wrist=(0.51, 0.42, -0.22)))

    candidate = detector.candidates[0]
    events = detector.update(tracker_output_factory(0.20, right_wrist=(0.65, 0.42, -0.10)))

    assert candidate.gesture is GestureType.KICK
    assert events == []
    assert detector.candidates == ()


def test_cooldown_remaining_tracks_last_trigger(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.25,
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.15,
            )
        )
    )
    events = feed_frames(detector, sequence_to_frames(motion_sequences["forward_punch"]))

    assert len(events) == 1
    assert detector.cooldown_remaining(FrameTimestamp(seconds=0.10)) == pytest.approx(0.25)
    assert detector.cooldown_remaining(0.20) == pytest.approx(0.15)


def test_compute_metrics_returns_expected_velocity_profile() -> None:
    detector = GestureDetector(GestureConfig())
    samples = [
        MotionSample(timestamp=0.0, x=0.50, y=0.40, z=-0.10),
        MotionSample(timestamp=0.1, x=0.52, y=0.41, z=-0.25),
        MotionSample(timestamp=0.2, x=0.53, y=0.42, z=-0.34),
    ]

    metrics = detector._compute_metrics(samples)

    assert metrics is not None
    assert metrics.delta_x == pytest.approx(0.03)
    assert metrics.delta_y == pytest.approx(0.02)
    assert metrics.delta_z == pytest.approx(-0.24)
    assert metrics.forward_velocity > 0.0
    assert metrics.punch_axis_ratio > 1.0


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([MotionSample(timestamp=0.0, x=0.0, y=0.0, z=0.0)], None),
        (
            [
                MotionSample(timestamp=0.1, x=0.0, y=0.0, z=0.0),
                MotionSample(timestamp=0.1, x=0.2, y=0.2, z=0.2),
            ],
            None,
        ),
    ],
)
def test_compute_metrics_handles_short_or_non_monotonic_histories(
    samples: list[MotionSample],
    expected: object,
) -> None:
    detector = GestureDetector(GestureConfig())

    assert detector._compute_metrics(samples) is expected


def test_epsilon_inclusive_threshold_logic_at_exact_boundary() -> None:
    detector = GestureDetector(GestureConfig())
    metrics = detector._compute_metrics(
        [
            MotionSample(timestamp=0.0, x=0.5, y=0.4, z=-0.1),
            MotionSample(
                timestamp=0.1,
                x=0.5,
                y=0.4 + COMPARISON_EPSILON / 2,
                z=-0.1 - detector.config.punch_forward_delta_z,
            ),
        ]
    )

    assert metrics is not None
    assert detector._is_confirmed(GestureType.KICK, metrics) is True
    assert detector._meets_punch_candidate(metrics) is True



@pytest.mark.parametrize(
    ("builder", "velocity", "duration", "noise", "expected"),
    [
        ("forward_punch", 2.5, 0.10, 0.0, GestureType.KICK),
        ("downward_strike", 2.6, 0.10, 0.0, GestureType.SNARE),
        ("non_trigger_movement", 1.3, 0.10, 0.0, None),
        ("jitter_noise", 0.0, 0.15, 0.05, None),
    ],
)
def test_generator_sequences_drive_expected_detector_outcomes(
    builder: str,
    velocity: float,
    duration: float,
    noise: float,
    expected: GestureType | None,
    synthetic_motion_generator,
) -> None:
    detector = GestureDetector(GestureConfig(history_size=8))
    sequence_builder = getattr(synthetic_motion_generator, builder)
    kwargs = {"duration": duration, "noise": noise}
    if builder != "jitter_noise":
        kwargs["velocity"] = velocity
    sequence = sequence_builder(**kwargs)

    events = feed_frames(detector, synthetic_motion_generator.to_tracker_outputs(sequence))

    if expected is None:
        assert events == []
    else:
        assert len(events) == 1
        assert events[0].gesture is expected

def test_observer_receives_candidate_and_trigger_events(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    observer = FakeGestureObserver()
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.15,
                trigger_seconds=0.05,
            ),
            thresholds=GestureThresholdsConfig(
                strike_down_delta_y=0.15,
                min_velocity=0.5,
            ),
        ),
        observer=observer,
    )

    events = feed_frames(detector, sequence_to_frames(motion_sequences["forward_punch"]))

    assert len(events) == 1
    assert len(observer.candidates) == 1
    assert len(observer.triggers) == 1
    candidate_event = observer.candidates[0]
    trigger_event = observer.triggers[0]
    assert candidate_event.event_kind == "candidate"
    assert trigger_event.event_kind == "trigger"
    assert candidate_event.gesture_type is GestureType.KICK
    assert trigger_event.accepted is True
    assert trigger_event.confidence == pytest.approx(events[0].confidence)

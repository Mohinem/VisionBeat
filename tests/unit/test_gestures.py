from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest

from visionbeat.config import GestureConfig, GestureCooldownsConfig, GestureThresholdsConfig
from visionbeat.gestures import COMPARISON_EPSILON, GestureDetector, MotionMetrics, MotionSample
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
        ("outward_jab", GestureType.KICK, "Outward jab → kick"),
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
        (0.699, False, False),
        (0.7, True, True),
        (0.701, True, True),
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
    target_delta_x = config.punch_forward_delta_z * travel_scale
    frames = [
        tracker_output_factory(0.00, right_wrist=(0.50, 0.40, -0.05)),
        tracker_output_factory(0.10, right_wrist=(0.50 + target_delta_x, 0.405, -0.055)),
        tracker_output_factory(0.20, right_wrist=(0.50 + target_delta_x, 0.405, -0.055)),
    ]

    first = detector.update(frames[0])
    second = detector.update(frames[1])
    candidate_count_after_second = len(detector.candidates)
    third = detector.update(frames[2])
    triggered_on_second = len(second) == 1
    confirmed_count = len(second) + len(third)
    candidate_observed = candidate_count_after_second == 1 or triggered_on_second

    assert first == []
    if not triggered_on_second:
        assert second == []
    assert candidate_observed is expected_candidate
    assert (confirmed_count == 1) is expected_confirmed


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
    triggered_on_second = len(second) == 1
    confirmed_count = len(second) + len(third)
    candidate_observed = candidate_count_after_second == 1 or triggered_on_second

    assert first == []
    if not triggered_on_second:
        assert second == []
    assert candidate_observed is expected_candidate
    assert (confirmed_count == 1) is expected_confirmed


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
    frames = sequence_to_frames(motion_sequences["outward_jab"])
    repeated_frames = frames + sequence_to_frames(
        (
            (0.18, (0.53, 0.40, -0.08)),
            (0.24, (0.62, 0.41, -0.09)),
            (0.28, (0.71, 0.42, -0.10)),
        )
    )

    events = feed_frames(detector, repeated_frames)

    assert [event.gesture for event in events] == [GestureType.KICK]
    assert len(observer.cooldown_suppressions) >= 1
    assert observer.cooldown_suppressions[-1].reason == "cooldown_active"


def test_recovery_gate_blocks_retrigger_until_hand_resets(tracker_output_factory) -> None:
    detector = GestureDetector(
        GestureConfig(
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.05,
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.15,
            )
        )
    )
    frames = [
        tracker_output_factory(0.00, right_wrist=(0.50, 0.40, -0.08)),
        tracker_output_factory(0.05, right_wrist=(0.63, 0.41, -0.09)),
        tracker_output_factory(0.10, right_wrist=(0.76, 0.41, -0.10)),
        tracker_output_factory(0.18, right_wrist=(0.77, 0.41, -0.10)),
        tracker_output_factory(0.26, right_wrist=(0.79, 0.42, -0.11)),
        tracker_output_factory(0.34, right_wrist=(0.81, 0.42, -0.11)),
    ]

    events = feed_frames(detector, frames)

    assert [event.gesture for event in events] == [GestureType.KICK]
    assert detector.status_summary(FrameTimestamp(seconds=0.34)) == "recovering kick"


def test_shoulder_relative_motion_reduces_body_sway_false_positives(
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.05,
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.15,
            )
        )
    )
    frames = [
        tracker_output_factory(
            0.00,
            right_wrist=(0.60, 0.40, -0.08),
            right_shoulder=(0.50, 0.20, -0.02),
        ),
        tracker_output_factory(
            0.05,
            right_wrist=(0.69, 0.42, -0.10),
            right_shoulder=(0.59, 0.22, -0.04),
        ),
        tracker_output_factory(
            0.10,
            right_wrist=(0.78, 0.43, -0.11),
            right_shoulder=(0.68, 0.23, -0.05),
        ),
    ]

    events = feed_frames(detector, frames)

    assert events == []
    assert detector.candidates == ()


def test_outward_jab_still_confirms_with_moderate_shoulder_followthrough(
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.05,
                analysis_window_seconds=0.24,
                confirmation_window_seconds=0.18,
            )
        )
    )
    frames = [
        tracker_output_factory(
            0.00,
            right_wrist=(0.60, 0.40, -0.08),
            right_shoulder=(0.50, 0.20, -0.02),
        ),
        tracker_output_factory(
            0.06,
            right_wrist=(0.70, 0.41, -0.09),
            right_shoulder=(0.55, 0.21, -0.03),
        ),
        tracker_output_factory(
            0.12,
            right_wrist=(0.80, 0.41, -0.10),
            right_shoulder=(0.56, 0.22, -0.04),
        ),
    ]

    events = feed_frames(detector, frames)

    assert [event.gesture for event in events] == [GestureType.KICK]


def test_shoulder_visibility_mode_change_clears_pending_kick_history(
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.05,
                analysis_window_seconds=0.24,
                confirmation_window_seconds=0.18,
            ),
            thresholds=GestureThresholdsConfig(
                punch_forward_delta_z=0.12,
                punch_max_vertical_drift=0.12,
                min_velocity=0.4,
                candidate_ratio=0.6,
                axis_dominance_ratio=1.1,
            ),
        )
    )
    frames = [
        tracker_output_factory(
            0.00,
            right_wrist=(0.60, 0.40, -0.08),
            right_shoulder=(0.50, 0.20, -0.02),
        ),
        tracker_output_factory(
            0.05,
            right_wrist=(0.68, 0.41, -0.09),
            right_shoulder=(0.505, 0.205, -0.06),
        ),
        tracker_output_factory(
            0.10,
            right_wrist=(0.70, 0.41, -0.10),
            right_shoulder=(0.505, 0.205, -0.06),
            right_shoulder_visibility=0.1,
        ),
    ]

    assert detector.update(frames[0]) == []
    events = detector.update(frames[1])
    assert [event.gesture for event in events] == [GestureType.KICK]

    events = detector.update(frames[2])

    assert events == []
    assert detector.candidates == ()


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
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                punch_forward_delta_z=0.20,
                punch_max_vertical_drift=0.18,
                min_velocity=0.45,
                candidate_ratio=0.6,
                axis_dominance_ratio=1.2,
            ),
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.4,
                confirmation_window_seconds=0.18,
                trigger_seconds=0.2,
            ),
        )
    )
    candidate_frames = (
        (0.00, (0.55, 0.20, -0.05)),
        (0.20, (0.56, 0.35, -0.055)),
    )

    for timestamp, wrist in candidate_frames:
        detector.update(tracker_output_factory(timestamp, right_wrist=wrist))

    assert len(detector.candidates) == 1
    detector.update(
        tracker_output_factory(
            0.21,
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
            thresholds=GestureThresholdsConfig(
                punch_forward_delta_z=0.20,
                punch_max_vertical_drift=0.18,
                min_velocity=0.45,
                candidate_ratio=0.6,
                axis_dominance_ratio=1.2,
            ),
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.4,
                confirmation_window_seconds=0.05,
                trigger_seconds=0.2,
            )
        )
    )
    detector.update(tracker_output_factory(0.00, right_wrist=(0.55, 0.20, -0.05)))
    detector.update(tracker_output_factory(0.20, right_wrist=(0.56, 0.35, -0.055)))

    candidate = detector.candidates[0]
    events = detector.update(tracker_output_factory(0.40, right_wrist=(0.56, 0.23, -0.055)))

    assert candidate.gesture is GestureType.SNARE
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
    events = feed_frames(detector, sequence_to_frames(motion_sequences["outward_jab"]))

    assert len(events) == 1
    trigger_time = events[0].timestamp.seconds
    assert detector.cooldown_remaining(FrameTimestamp(seconds=trigger_time)) == pytest.approx(0.25)
    assert detector.cooldown_remaining(0.20) == pytest.approx(0.25 - (0.20 - trigger_time))


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
    assert metrics.delta_abs_x == pytest.approx(0.03)
    assert metrics.delta_y == pytest.approx(0.02)
    assert metrics.delta_z == pytest.approx(-0.24)
    assert metrics.outward_velocity > 0.0


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
                x=0.5 + detector.config.kick_outward_delta_x,
                y=0.4 + COMPARISON_EPSILON / 2,
                z=-0.1 + COMPARISON_EPSILON / 2,
            ),
        ]
    )

    assert metrics is not None
    assert detector._is_confirmed(GestureType.KICK, metrics, shoulder_relative=True) is True
    assert detector._meets_kick_candidate(metrics, shoulder_relative=True) is True


def test_logged_outward_jab_regression_now_confirms_as_kick() -> None:
    detector = GestureDetector(GestureConfig())
    metrics = MotionMetrics(
        elapsed=0.12501151600008598,
        delta_x=-0.053355634212493896,
        delta_abs_x=0.053355634212493896,
        delta_y=0.04259753227233887,
        delta_z=-0.0003672957420348677,
        net_velocity=0.7704927138616684,
        peak_x_velocity=-0.23474316411658805,
        peak_abs_x_velocity=0.23474316411658805,
        peak_y_velocity=0.18741187611683918,
        peak_z_velocity=-0.0016159523904902089,
    )

    assert detector._meets_kick_candidate(metrics, shoulder_relative=True) is True
    assert detector._is_confirmed(GestureType.KICK, metrics, shoulder_relative=True) is True


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

    events = feed_frames(detector, sequence_to_frames(motion_sequences["outward_jab"]))

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

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from visionbeat.config import GestureConfig, GestureCooldownsConfig, GestureThresholdsConfig
from visionbeat.gestures import (
    COMPARISON_EPSILON,
    CollisionRecoveryState,
    GestureDetector,
    MotionMetrics,
    MotionSample,
    WristCollisionMetrics,
    WristCollisionSample,
)
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


def build_collision_frames(
    tracker_output_factory,
    right_sequence: tuple[tuple[float, tuple[float, float, float]], ...],
    left_sequence: tuple[tuple[float, tuple[float, float, float]], ...],
) -> list[TrackerOutput]:
    frames: list[TrackerOutput] = []
    for (right_timestamp, right_wrist), (left_timestamp, left_wrist) in zip(
        right_sequence,
        left_sequence,
        strict=True,
    ):
        assert right_timestamp == left_timestamp
        frames.append(
            tracker_output_factory(
                right_timestamp,
                right_wrist=right_wrist,
                left_wrist=left_wrist,
            )
        )
    return frames


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


def test_downward_strike_confirms_as_kick(
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

    events = feed_frames(detector, sequence_to_frames(motion_sequences["downward_strike"]))

    assert len(events) == 1
    assert events[0].gesture is GestureType.KICK
    assert events[0].label == "Downward strike → kick"


def test_wrist_collision_confirms_as_snare(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.2,
                confirmation_window_seconds=0.15,
                trigger_seconds=0.05,
            ),
        )
    )
    frames = build_collision_frames(
        tracker_output_factory,
        motion_sequences["wrist_collision_right"],
        motion_sequences["wrist_collision_left"],
    )

    events = feed_frames(detector, frames)

    assert len(events) == 1
    assert events[0].gesture is GestureType.SNARE
    assert events[0].label == "Wrist collision → snare"

@pytest.mark.parametrize(
    ("travel_scale", "expected_candidate", "expected_confirmed"),
    [
        (0.699, False, False),
        (0.7, True, False),
        (1.0, True, True),
    ],
)
def test_kick_threshold_boundaries_are_stable(
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
            strike_confirmation_ratio=0.8,
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


@pytest.mark.parametrize(
    ("distance_scale", "expected_confirmed"),
    [
        (1.001, False),
        (1.0, True),
        (0.999, True),
    ],
)
def test_snare_collision_confirmation_threshold_is_stable(
    distance_scale: float,
    expected_confirmed: bool,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                snare_collision_distance=0.12,
                snare_collision_max_depth_gap=0.10,
                min_velocity=0.5,
                candidate_ratio=0.7,
            )
        )
    )
    metrics = WristCollisionMetrics(
        elapsed=0.10,
        delta_x_gap=-0.25,
        delta_y_gap=0.0,
        delta_z_gap=0.0,
        delta_distance_xy=-0.25,
        current_distance_xy=detector.config.snare_collision_distance * distance_scale,
        current_depth_gap=0.02,
        net_velocity=2.5,
        peak_x_velocity=-2.5,
        peak_y_velocity=0.0,
        peak_z_velocity=0.0,
        peak_distance_velocity=-2.5,
    )

    assert detector._meets_snare_candidate(metrics) is True
    assert detector._is_snare_confirmed(metrics) is expected_confirmed


@pytest.mark.parametrize(
    ("velocity_scale", "expected_confirmed"),
    [
        (0.999, False),
        (1.0, True),
        (1.001, True),
    ],
)
def test_snare_collision_velocity_confirmation_threshold_is_stable(
    velocity_scale: float,
    expected_confirmed: bool,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                snare_collision_distance=0.12,
                snare_confirmation_velocity_ratio=0.9,
                snare_collision_max_depth_gap=0.10,
                min_velocity=0.5,
                candidate_ratio=0.7,
            )
        )
    )
    threshold_velocity = detector.config.min_velocity * detector.config.snare_confirmation_velocity_ratio
    metrics = WristCollisionMetrics(
        elapsed=0.10,
        delta_x_gap=-0.12,
        delta_y_gap=0.0,
        delta_z_gap=0.0,
        delta_distance_xy=-0.12,
        current_distance_xy=0.10,
        current_depth_gap=0.02,
        net_velocity=1.2,
        peak_x_velocity=-(threshold_velocity * velocity_scale),
        peak_y_velocity=0.0,
        peak_z_velocity=0.0,
        peak_distance_velocity=-(threshold_velocity * velocity_scale),
    )

    assert detector._meets_snare_candidate(metrics) is True
    assert detector._is_snare_confirmed(metrics) is expected_confirmed


def test_inward_jab_no_longer_triggers(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    sequence_to_frames,
) -> None:
    detector = GestureDetector(GestureConfig())

    events = feed_frames(detector, sequence_to_frames(motion_sequences["inward_jab"]))

    assert events == []
    assert detector.candidates == ()


def test_collision_candidate_blocks_simultaneous_kick(
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                strike_down_delta_y=0.15,
                snare_collision_distance=0.12,
                min_velocity=0.39,
                candidate_ratio=0.6,
            )
        )
    )
    frames = [
        tracker_output_factory(0.00, right_wrist=(0.75, 0.20, -0.05), left_wrist=(0.25, 0.20, -0.05)),
        tracker_output_factory(0.05, right_wrist=(0.59, 0.33, -0.05), left_wrist=(0.41, 0.31, -0.05)),
        tracker_output_factory(0.10, right_wrist=(0.54, 0.41, -0.05), left_wrist=(0.46, 0.39, -0.05)),
    ]

    events = feed_frames(detector, frames)

    assert [event.gesture for event in events] == [GestureType.SNARE]


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
    frames = sequence_to_frames(motion_sequences["downward_strike"])
    repeated_frames = frames + [
        sequence_to_frames(
            (
                (0.18, (0.56, 0.24, -0.05)),
                (0.24, (0.57, 0.39, -0.06)),
                (0.28, (0.58, 0.48, -0.07)),
            )
        )[index]
        for index in range(3)
    ]

    events = feed_frames(detector, repeated_frames)

    assert [event.gesture for event in events] == [GestureType.KICK]
    assert len(observer.cooldown_suppressions) >= 1
    assert observer.cooldown_suppressions[-1].reason == "cooldown_active"


def test_snare_recovery_gate_blocks_retrigger_until_wrists_separate(tracker_output_factory) -> None:
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
        tracker_output_factory(0.00, right_wrist=(0.75, 0.40, -0.05), left_wrist=(0.25, 0.40, -0.05)),
        tracker_output_factory(0.05, right_wrist=(0.60, 0.40, -0.05), left_wrist=(0.40, 0.40, -0.05)),
        tracker_output_factory(0.10, right_wrist=(0.54, 0.40, -0.05), left_wrist=(0.46, 0.40, -0.05)),
        tracker_output_factory(0.18, right_wrist=(0.53, 0.40, -0.05), left_wrist=(0.47, 0.40, -0.05)),
        tracker_output_factory(0.26, right_wrist=(0.52, 0.40, -0.05), left_wrist=(0.48, 0.40, -0.05)),
    ]

    events = feed_frames(detector, frames)

    assert [event.gesture for event in events] == [GestureType.SNARE]
    assert detector.status_summary(FrameTimestamp(seconds=0.26)) == "recovering snare"


@pytest.mark.parametrize(
    "sequence_name",
    ["noisy_movement", "borderline_threshold_movement", "inward_jab"],
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
                strike_down_delta_y=0.22,
                snare_collision_distance=0.10,
                min_velocity=0.75,
                axis_dominance_ratio=1.7,
            ),
        )
    )

    events = feed_frames(detector, sequence_to_frames(motion_sequences[sequence_name]))

    assert events == []


def test_inactive_hand_is_ignored_for_kick(tracker_output_factory) -> None:
    detector = GestureDetector(GestureConfig(active_hand="left"))

    events = detector.update(tracker_output_factory(0.1, right_wrist=(0.5, 0.5, -0.05)))

    assert events == []
    assert detector.candidates == ()


def test_bilateral_snare_still_works_when_active_hand_is_left(tracker_output_factory) -> None:
    detector = GestureDetector(GestureConfig(active_hand="left"))
    frames = [
        tracker_output_factory(0.00, right_wrist=(0.75, 0.40, -0.05), left_wrist=(0.25, 0.40, -0.05)),
        tracker_output_factory(0.05, right_wrist=(0.60, 0.40, -0.05), left_wrist=(0.40, 0.40, -0.05)),
        tracker_output_factory(0.10, right_wrist=(0.54, 0.40, -0.05), left_wrist=(0.46, 0.40, -0.05)),
    ]

    events = feed_frames(detector, frames)

    assert [event.gesture for event in events] == [GestureType.SNARE]
    assert events[0].hand == "left"


def test_low_visibility_clears_pending_candidate(
    tracker_output_factory,
) -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                strike_down_delta_y=0.20,
                strike_max_depth_drift=0.18,
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
        (0.20, (0.56, 0.33, -0.055)),
    )

    for timestamp, wrist in candidate_frames:
        detector.update(tracker_output_factory(timestamp, right_wrist=wrist))

    assert len(detector.candidates) == 1
    detector.update(
        tracker_output_factory(
            0.21,
            right_wrist=candidate_frames[-1][1],
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
                strike_down_delta_y=0.20,
                strike_max_depth_drift=0.18,
                min_velocity=0.45,
                candidate_ratio=0.6,
                axis_dominance_ratio=1.2,
            ),
            cooldowns=GestureCooldownsConfig(
                analysis_window_seconds=0.4,
                confirmation_window_seconds=0.05,
                trigger_seconds=0.2,
            ),
        )
    )
    detector.update(tracker_output_factory(0.00, right_wrist=(0.55, 0.20, -0.05)))
    detector.update(tracker_output_factory(0.20, right_wrist=(0.56, 0.33, -0.055)))

    candidate = detector.candidates[0]
    events = detector.update(tracker_output_factory(0.40, right_wrist=(0.56, 0.23, -0.055)))

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
    events = feed_frames(detector, sequence_to_frames(motion_sequences["downward_strike"]))

    assert len(events) == 1
    trigger_time = events[0].timestamp.seconds
    assert detector.cooldown_remaining(FrameTimestamp(seconds=trigger_time)) == pytest.approx(0.25)
    assert detector.cooldown_remaining(0.20) == pytest.approx(0.25 - (0.20 - trigger_time))


def test_compute_metrics_returns_expected_velocity_profile() -> None:
    detector = GestureDetector(GestureConfig())
    samples = [
        MotionSample(timestamp=0.0, x=0.50, y=0.20, z=-0.10),
        MotionSample(timestamp=0.1, x=0.51, y=0.31, z=-0.12),
        MotionSample(timestamp=0.2, x=0.52, y=0.42, z=-0.14),
    ]

    metrics = detector._compute_metrics(samples)

    assert metrics is not None
    assert metrics.delta_x == pytest.approx(0.02)
    assert metrics.delta_abs_x == pytest.approx(0.02)
    assert metrics.delta_y == pytest.approx(0.22)
    assert metrics.delta_z == pytest.approx(-0.04)
    assert metrics.downward_velocity > 0.0


def test_compute_collision_metrics_returns_expected_profile() -> None:
    detector = GestureDetector(GestureConfig())
    samples = [
        WristCollisionSample(timestamp=0.0, x_gap=0.50, y_gap=0.00, z_gap=0.00, distance_xy=0.50),
        WristCollisionSample(timestamp=0.1, x_gap=0.28, y_gap=0.02, z_gap=0.01, distance_xy=0.2807),
        WristCollisionSample(timestamp=0.2, x_gap=0.08, y_gap=0.02, z_gap=0.01, distance_xy=0.0825),
    ]

    metrics = detector._compute_collision_metrics(samples)

    assert metrics is not None
    assert metrics.delta_x_gap == pytest.approx(-0.42)
    assert metrics.delta_y_gap == pytest.approx(0.02)
    assert metrics.current_distance_xy == pytest.approx(0.0825)
    assert metrics.closing_velocity > 0.0


def test_compute_collision_metrics_keeps_fast_closing_peak_after_rebound() -> None:
    detector = GestureDetector(GestureConfig(velocity_smoothing_alpha=1.0))
    samples = [
        WristCollisionSample(timestamp=0.0, x_gap=0.40, y_gap=0.00, z_gap=0.00, distance_xy=0.40),
        WristCollisionSample(timestamp=0.1, x_gap=0.10, y_gap=0.00, z_gap=0.00, distance_xy=0.10),
        WristCollisionSample(timestamp=0.15, x_gap=0.28, y_gap=0.00, z_gap=0.00, distance_xy=0.28),
    ]

    metrics = detector._compute_collision_metrics(samples)

    assert metrics is not None
    assert metrics.peak_distance_velocity == pytest.approx(3.6)
    assert metrics.closing_velocity == pytest.approx(3.0)
    assert metrics.opening_velocity == pytest.approx(3.6)


def test_logged_wrist_collision_regression_now_confirms_as_snare() -> None:
    detector = GestureDetector(GestureConfig())
    metrics = WristCollisionMetrics(
        elapsed=0.19496659700007513,
        delta_x_gap=-0.09316384792327881,
        delta_y_gap=-0.01547229290008545,
        delta_z_gap=-0.0952039361000061,
        delta_distance_xy=-0.09444197793685669,
        current_distance_xy=detector.config.snare_collision_distance * 0.92,
        current_depth_gap=0.0952039361000061,
        net_velocity=1.0455128214772698,
        peak_x_velocity=-0.3576111020025244,
        peak_y_velocity=-0.15706172698065266,
        peak_z_velocity=-0.5778048294438297,
        peak_distance_velocity=-0.3576111020025244,
    )

    assert detector._meets_snare_candidate(metrics) is True
    assert detector._is_snare_confirmed(metrics) is True


def test_fast_snare_rebound_still_confirms_when_recent_closing_peak_is_strong() -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                snare_collision_distance=0.12,
                snare_confirmation_velocity_ratio=0.9,
                snare_collision_max_depth_gap=0.10,
                min_velocity=0.5,
                candidate_ratio=0.7,
            )
        )
    )
    threshold_velocity = detector.config.min_velocity * detector.config.snare_confirmation_velocity_ratio
    metrics = WristCollisionMetrics(
        elapsed=0.10,
        delta_x_gap=-0.08,
        delta_y_gap=0.0,
        delta_z_gap=0.0,
        delta_distance_xy=-0.08,
        current_distance_xy=0.10,
        current_depth_gap=0.02,
        net_velocity=0.8,
        peak_x_velocity=-(threshold_velocity * 1.05),
        peak_y_velocity=0.0,
        peak_z_velocity=0.0,
        peak_distance_velocity=threshold_velocity * 1.4,
        peak_closing_velocity=-(threshold_velocity * 1.05),
        peak_opening_velocity=threshold_velocity * 1.4,
    )

    assert detector._meets_snare_candidate(metrics) is True
    assert detector._is_snare_confirmed(metrics) is True


def test_snare_recovery_uses_opening_peak_even_when_closing_was_faster() -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                snare_collision_distance=0.12,
                min_velocity=0.5,
            )
        )
    )
    recovery = CollisionRecoveryState(anchor_distance_xy=0.10)
    metrics = WristCollisionMetrics(
        elapsed=0.10,
        delta_x_gap=0.04,
        delta_y_gap=0.0,
        delta_z_gap=0.0,
        delta_distance_xy=0.04,
        current_distance_xy=0.14,
        current_depth_gap=0.02,
        net_velocity=0.6,
        peak_x_velocity=0.0,
        peak_y_velocity=0.0,
        peak_z_velocity=0.0,
        peak_distance_velocity=-1.2,
        peak_closing_velocity=-1.2,
        peak_opening_velocity=0.3,
    )

    assert detector._collision_recovery_complete(recovery, metrics) is True


def test_logged_downward_strike_regression_now_confirms_as_kick() -> None:
    detector = GestureDetector(GestureConfig())
    metrics = MotionMetrics(
        elapsed=0.19580230000008214,
        delta_x=-0.01634138822555542,
        delta_abs_x=0.0,
        delta_y=0.09949994087219238,
        delta_z=0.007269076630473048,
        net_velocity=0.6287485168875402,
        peak_x_velocity=-0.07526097155467111,
        peak_abs_x_velocity=0.0,
        peak_y_velocity=0.3942727619442893,
        peak_z_velocity=0.3812577362128058,
    )

    assert detector._meets_kick_candidate(metrics) is True
    assert detector._is_kick_confirmed(metrics) is True


def test_epsilon_inclusive_threshold_logic_at_exact_boundaries() -> None:
    detector = GestureDetector(
        GestureConfig(
            thresholds=GestureThresholdsConfig(
                strike_down_delta_y=0.20,
                strike_confirmation_ratio=1.0,
                strike_max_depth_drift=0.10,
                snare_collision_distance=0.12,
                snare_confirmation_velocity_ratio=1.0,
                snare_collision_max_depth_gap=0.10,
                min_velocity=0.5,
                candidate_ratio=0.7,
                axis_dominance_ratio=1.7,
            )
        )
    )
    kick_metrics = MotionMetrics(
        elapsed=0.10,
        delta_x=0.01,
        delta_abs_x=0.01,
        delta_y=0.20,
        delta_z=COMPARISON_EPSILON / 2,
        net_velocity=0.5,
        peak_x_velocity=0.1,
        peak_abs_x_velocity=0.1,
        peak_y_velocity=0.5,
        peak_z_velocity=COMPARISON_EPSILON / 2,
    )
    snare_metrics = WristCollisionMetrics(
        elapsed=0.10,
        delta_x_gap=-0.20,
        delta_y_gap=0.0,
        delta_z_gap=COMPARISON_EPSILON / 2,
        delta_distance_xy=-0.20,
        current_distance_xy=0.12,
        current_depth_gap=COMPARISON_EPSILON / 2,
        net_velocity=0.5,
        peak_x_velocity=-2.0,
        peak_y_velocity=0.0,
        peak_z_velocity=COMPARISON_EPSILON / 2,
        peak_distance_velocity=-0.5,
    )

    assert detector._is_kick_confirmed(kick_metrics) is True
    assert detector._meets_snare_candidate(snare_metrics) is True
    assert detector._is_snare_confirmed(snare_metrics) is True


def test_observer_receives_candidate_and_trigger_events_for_snare(
    motion_sequences: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    tracker_output_factory,
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
        ),
        observer=observer,
    )
    frames = build_collision_frames(
        tracker_output_factory,
        motion_sequences["wrist_collision_right"],
        motion_sequences["wrist_collision_left"],
    )

    events = feed_frames(detector, frames)

    assert len(events) == 1
    assert len(observer.candidates) == 1
    assert len(observer.triggers) == 1
    candidate_event = observer.candidates[0]
    trigger_event = observer.triggers[0]
    assert candidate_event.event_kind == "candidate"
    assert trigger_event.event_kind == "trigger"
    assert candidate_event.gesture_type is GestureType.SNARE
    assert trigger_event.accepted is True
    assert trigger_event.confidence == pytest.approx(events[0].confidence)

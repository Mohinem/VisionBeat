from __future__ import annotations

import pytest

from visionbeat.models import FrameTimestamp, GestureEvent, GestureType
from visionbeat.rhythm_prediction import (
    RhythmObservation,
    RhythmPredictionConfig,
    RhythmPredictionTracker,
)


def observe(
    tracker: RhythmPredictionTracker,
    gesture: GestureType,
    timestamp_seconds: float,
    *,
    confidence: float = 1.0,
) -> object:
    return tracker.observe(
        RhythmObservation(
            gesture=gesture,
            timestamp_seconds=timestamp_seconds,
            confidence=confidence,
        )
    )


def test_tracker_does_not_predict_before_enough_repetitions() -> None:
    tracker = RhythmPredictionTracker()

    assert observe(tracker, GestureType.KICK, 0.0) is None
    assert observe(tracker, GestureType.KICK, 0.5) is None
    assert tracker.active_prediction(GestureType.KICK, timestamp_seconds=0.5) is None


def test_stable_repeated_pulse_predicts_next_expected_beat() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    prediction = observe(tracker, GestureType.KICK, 1.0)

    assert prediction is not None
    assert prediction.gesture is GestureType.KICK
    assert prediction.interval_seconds == pytest.approx(0.5)
    assert prediction.expected_timestamp_seconds == pytest.approx(1.5)
    assert prediction.last_observed_timestamp_seconds == pytest.approx(1.0)
    assert prediction.expires_after_seconds == pytest.approx(1.875)
    assert prediction.jitter_ratio == pytest.approx(0.0)
    assert prediction.confidence == pytest.approx(1.0)
    assert prediction.observation_count == 3
    assert prediction.repetition_count == 2


def test_tracker_rejects_jittery_timing() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.4)
    prediction = observe(tracker, GestureType.KICK, 1.2)

    assert prediction is None
    assert tracker.active_prediction(GestureType.KICK, timestamp_seconds=1.2) is None


def test_skipped_expected_beat_weakens_then_expires_prediction() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    prediction = observe(tracker, GestureType.KICK, 1.0)

    assert prediction is not None
    late_prediction = tracker.active_prediction(GestureType.KICK, timestamp_seconds=1.6)
    assert late_prediction is not None
    assert late_prediction.expected_timestamp_seconds == pytest.approx(1.5)
    assert late_prediction.confidence < prediction.confidence

    expired = tracker.advance(timestamp_seconds=1.9)

    assert len(expired) == 1
    assert expired[0].gesture is GestureType.KICK
    assert expired[0].expected_timestamp_seconds == pytest.approx(1.5)
    assert tracker.active_prediction(GestureType.KICK, timestamp_seconds=1.9) is None
    assert tracker.history_for(GestureType.KICK) == ()


def test_late_observation_after_expiry_starts_new_history_without_old_pulse() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    observe(tracker, GestureType.KICK, 1.0)

    prediction = observe(tracker, GestureType.KICK, 2.0)

    assert prediction is None
    assert [item.timestamp_seconds for item in tracker.history_for(GestureType.KICK)] == [2.0]


def test_tempo_change_adapts_after_recent_stable_intervals() -> None:
    tracker = RhythmPredictionTracker(RhythmPredictionConfig(history_size=4))

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    observe(tracker, GestureType.KICK, 1.0)
    observe(tracker, GestureType.KICK, 1.4)
    prediction = observe(tracker, GestureType.KICK, 1.8)

    assert prediction is not None
    assert prediction.interval_seconds == pytest.approx(0.4)
    assert prediction.expected_timestamp_seconds == pytest.approx(2.2)
    assert [item.timestamp_seconds for item in tracker.history_for(GestureType.KICK)] == [
        0.5,
        1.0,
        1.4,
        1.8,
    ]


def test_separate_gesture_histories_do_not_corrupt_each_other() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    kick_prediction = observe(tracker, GestureType.KICK, 1.0)

    observe(tracker, GestureType.SNARE, 0.1)
    observe(tracker, GestureType.SNARE, 0.7)
    snare_prediction = observe(tracker, GestureType.SNARE, 1.3)

    assert kick_prediction is not None
    assert snare_prediction is not None
    active_predictions = tracker.active_predictions(timestamp_seconds=1.3)

    assert {prediction.gesture for prediction in active_predictions} == {
        GestureType.KICK,
        GestureType.SNARE,
    }
    assert tracker.active_prediction(
        GestureType.KICK,
        timestamp_seconds=1.3,
    ).expected_timestamp_seconds == pytest.approx(1.5)
    assert tracker.active_prediction(
        GestureType.SNARE,
        timestamp_seconds=1.3,
    ).expected_timestamp_seconds == pytest.approx(1.9)


def test_tracker_accepts_confirmed_gesture_events() -> None:
    tracker = RhythmPredictionTracker()

    for timestamp_seconds in (0.0, 0.5):
        tracker.observe_event(
            GestureEvent(
                gesture=GestureType.SNARE,
                confidence=0.8,
                hand="right",
                timestamp=FrameTimestamp(seconds=timestamp_seconds),
                label="confirmed",
            )
        )
    prediction = tracker.observe_event(
        GestureEvent(
            gesture=GestureType.SNARE,
            confidence=0.8,
            hand="right",
            timestamp=FrameTimestamp(seconds=1.0),
            label="confirmed",
        ),
        source="predictive_completion",
        frame_index=12,
    )

    assert prediction is not None
    assert prediction.gesture is GestureType.SNARE
    assert prediction.expected_timestamp_seconds == pytest.approx(1.5)
    assert prediction.confidence == pytest.approx(0.8)


def test_non_increasing_timestamps_for_same_gesture_are_rejected() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        observe(tracker, GestureType.KICK, 1.0)


def test_prediction_is_marked_matched_when_hit_arrives_near_expected_time() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    pending_update = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.KICK, timestamp_seconds=1.0)
    )
    matched_update = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.KICK, timestamp_seconds=1.55)
    )

    assert [outcome.outcome for outcome in pending_update.outcomes] == ["pending"]
    matched = matched_update.outcomes[0]
    assert matched.outcome == "matched"
    assert matched.predicted_time_seconds == pytest.approx(1.5)
    assert matched.actual_time_seconds == pytest.approx(1.55)
    assert matched.actual_gesture is GestureType.KICK
    assert matched.error_ms == pytest.approx(50.0)
    assert matched_update.outcomes[1].outcome == "pending"


def test_prediction_is_marked_missed_after_skipped_beat() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    observe(tracker, GestureType.KICK, 1.0)

    update = tracker.advance_with_outcomes(timestamp_seconds=1.63)

    assert len(update.outcomes) == 1
    missed = update.outcomes[0]
    assert missed.outcome == "missed"
    assert missed.predicted_time_seconds == pytest.approx(1.5)
    assert missed.actual_time_seconds is None
    assert missed.actual_gesture is None
    assert missed.error_ms is None


def test_late_hit_outside_tolerance_marks_prediction_missed_with_actual_time() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    observe(tracker, GestureType.KICK, 1.0)
    update = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.KICK, timestamp_seconds=1.7)
    )

    missed = update.outcomes[0]
    assert missed.outcome == "missed"
    assert missed.predicted_time_seconds == pytest.approx(1.5)
    assert missed.actual_time_seconds == pytest.approx(1.7)
    assert missed.actual_gesture is GestureType.KICK
    assert missed.error_ms == pytest.approx(200.0)


def test_wrong_gesture_inside_tolerance_marks_prediction_missed_with_actual_gesture() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    observe(tracker, GestureType.KICK, 1.0)
    update = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.SNARE, timestamp_seconds=1.51)
    )

    assert len(update.outcomes) == 1
    missed = update.outcomes[0]
    assert missed.outcome == "missed"
    assert missed.gesture is GestureType.KICK
    assert missed.actual_gesture is GestureType.SNARE
    assert missed.actual_time_seconds == pytest.approx(1.51)
    assert missed.error_ms == pytest.approx(10.0)


def test_multiple_predictions_do_not_duplicate_matches() -> None:
    tracker = RhythmPredictionTracker()

    observe(tracker, GestureType.KICK, 0.0)
    observe(tracker, GestureType.KICK, 0.5)
    first_pending = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.KICK, timestamp_seconds=1.0)
    ).outcomes[0]
    first_match_update = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.KICK, timestamp_seconds=1.5)
    )
    second_match_update = tracker.observe_with_outcomes(
        RhythmObservation(gesture=GestureType.KICK, timestamp_seconds=2.0)
    )

    matched_ids = [
        outcome.prediction_id
        for update in (first_match_update, second_match_update)
        for outcome in update.outcomes
        if outcome.outcome == "matched"
    ]
    assert matched_ids == [
        first_pending.prediction_id,
        first_match_update.outcomes[1].prediction_id,
    ]
    assert len(set(matched_ids)) == len(matched_ids)

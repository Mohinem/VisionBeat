from __future__ import annotations

import numpy as np

from visionbeat.models import GestureType
from visionbeat.predictive_shadow import (
    CompletionTriggerDecoder,
    PredictiveStatus,
    PrimaryTriggerDecoder,
    StreamingTriggerDecoder,
)


def test_predictive_status_summary_reports_warmup_before_window_is_ready() -> None:
    status = PredictiveStatus(
        available_window_frames=12,
        required_window_size=24,
        threshold=0.6,
    )

    assert status.summary() == "warmup 12/24"


def test_predictive_status_summary_reports_probability_and_top_label() -> None:
    status = PredictiveStatus(
        available_window_frames=24,
        required_window_size=24,
        threshold=0.3,
        timing_probability=0.23,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.71,
    )

    assert status.summary() == "p=0.23/0.30 top=kick 0.71"


def test_streaming_trigger_decoder_replaces_weaker_candidate_inside_cooldown_window() -> None:
    decoder = StreamingTriggerDecoder(
        threshold=0.6,
        cooldown_frames=6,
        max_gap_frames=1,
    )
    window = np.zeros((24, 40), dtype=np.float32)

    assert decoder.update(
        frame_index=10,
        timestamp_seconds=10 / 30.0,
        timing_probability=0.70,
        window_matrix=window,
        heuristic_gesture_types=("kick",),
    ) == ()
    assert decoder.update(
        frame_index=11,
        timestamp_seconds=11 / 30.0,
        timing_probability=0.40,
        window_matrix=window,
        heuristic_gesture_types=(),
    ) == ()
    assert decoder.update(
        frame_index=14,
        timestamp_seconds=14 / 30.0,
        timing_probability=0.85,
        window_matrix=window,
        heuristic_gesture_types=("snare",),
    ) == ()
    assert decoder.update(
        frame_index=15,
        timestamp_seconds=15 / 30.0,
        timing_probability=0.20,
        window_matrix=window,
        heuristic_gesture_types=(),
    ) == ()

    emitted = decoder.flush()

    assert len(emitted) == 1
    assert emitted[0].frame_index == 14
    assert emitted[0].timing_probability == 0.85
    assert emitted[0].heuristic_gesture_types_on_peak_frame == ("snare",)


def test_streaming_trigger_decoder_emits_when_cooldown_window_is_clear() -> None:
    decoder = StreamingTriggerDecoder(
        threshold=0.6,
        cooldown_frames=2,
        max_gap_frames=1,
    )
    window = np.zeros((24, 40), dtype=np.float32)

    decoder.update(
        frame_index=10,
        timestamp_seconds=10 / 30.0,
        timing_probability=0.72,
        window_matrix=window,
        heuristic_gesture_types=("kick",),
    )
    decoder.update(
        frame_index=11,
        timestamp_seconds=11 / 30.0,
        timing_probability=0.10,
        window_matrix=window,
        heuristic_gesture_types=(),
    )
    emitted = decoder.update(
        frame_index=13,
        timestamp_seconds=13 / 30.0,
        timing_probability=0.10,
        window_matrix=window,
        heuristic_gesture_types=(),
    )

    assert len(emitted) == 1
    assert emitted[0].frame_index == 10
    assert emitted[0].run_length == 1


def test_primary_trigger_decoder_fires_on_causal_peak_drop() -> None:
    decoder = PrimaryTriggerDecoder(
        threshold=0.6,
        cooldown_frames=6,
        max_gap_frames=1,
        horizon_frames=8,
    )

    assert decoder.update(
        frame_index=10,
        timestamp_seconds=10 / 30.0,
        timing_probability=0.20,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    ) == ()
    assert decoder.update(
        frame_index=11,
        timestamp_seconds=11 / 30.0,
        timing_probability=0.65,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    ) == ()
    assert decoder.update(
        frame_index=12,
        timestamp_seconds=12 / 30.0,
        timing_probability=0.76,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.93,
        class_probabilities={"kick": 0.93, "snare": 0.07},
    ) == ()

    emitted = decoder.update(
        frame_index=13,
        timestamp_seconds=13 / 30.0,
        timing_probability=0.67,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.92,
        class_probabilities={"kick": 0.92, "snare": 0.08},
    )

    assert len(emitted) == 1
    assert emitted[0].frame_index == 12
    assert emitted[0].timing_probability == 0.76
    assert emitted[0].gesture is GestureType.KICK
    assert emitted[0].gesture_confidence == 0.93


def test_primary_trigger_decoder_cancels_held_plateau_without_emitting() -> None:
    decoder = PrimaryTriggerDecoder(
        threshold=0.6,
        cooldown_frames=6,
        max_gap_frames=1,
        horizon_frames=4,
    )
    values = (0.65, 0.72, 0.71, 0.70, 0.69, 0.68, 0.10)
    emitted = ()
    for index, value in enumerate(values, start=10):
        emitted = decoder.update(
            frame_index=index,
            timestamp_seconds=index / 30.0,
            timing_probability=value,
            predicted_gesture=GestureType.SNARE,
            predicted_gesture_confidence=0.88,
            class_probabilities={"kick": 0.12, "snare": 0.88},
        )
        if index < 16:
            assert emitted == ()

    assert emitted == ()
    assert decoder.flush() == ()


def test_primary_trigger_decoder_respects_refractory_cooldown() -> None:
    decoder = PrimaryTriggerDecoder(
        threshold=0.6,
        cooldown_frames=2,
        max_gap_frames=1,
        horizon_frames=8,
    )
    first = decoder.update(
        frame_index=10,
        timestamp_seconds=10 / 30.0,
        timing_probability=0.64,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    )
    assert first == ()
    decoder.update(
        frame_index=11,
        timestamp_seconds=11 / 30.0,
        timing_probability=0.75,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.92,
        class_probabilities={"kick": 0.92, "snare": 0.08},
    )
    emitted = decoder.update(
        frame_index=12,
        timestamp_seconds=12 / 30.0,
        timing_probability=0.66,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.90,
        class_probabilities={"kick": 0.90, "snare": 0.10},
    )
    assert len(emitted) == 1

    assert decoder.update(
        frame_index=13,
        timestamp_seconds=13 / 30.0,
        timing_probability=0.72,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.92,
        class_probabilities={"kick": 0.92, "snare": 0.08},
    ) == ()
    assert decoder.update(
        frame_index=14,
        timestamp_seconds=14 / 30.0,
        timing_probability=0.78,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.93,
        class_probabilities={"kick": 0.93, "snare": 0.07},
    ) == ()


def test_primary_trigger_decoder_fires_on_live_scale_peak_resolution() -> None:
    decoder = PrimaryTriggerDecoder(
        threshold=0.6,
        cooldown_frames=6,
        max_gap_frames=1,
        horizon_frames=8,
    )

    assert decoder.update(
        frame_index=20,
        timestamp_seconds=20 / 30.0,
        timing_probability=0.54,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.89,
        class_probabilities={"kick": 0.11, "snare": 0.89},
    ) == ()
    assert decoder.update(
        frame_index=21,
        timestamp_seconds=21 / 30.0,
        timing_probability=0.58,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.90,
        class_probabilities={"kick": 0.10, "snare": 0.90},
    ) == ()
    assert decoder.update(
        frame_index=22,
        timestamp_seconds=22 / 30.0,
        timing_probability=0.56,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.88,
        class_probabilities={"kick": 0.12, "snare": 0.88},
    ) == ()

    emitted = decoder.update(
        frame_index=23,
        timestamp_seconds=23 / 30.0,
        timing_probability=0.42,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.87,
        class_probabilities={"kick": 0.13, "snare": 0.87},
    )

    assert len(emitted) == 1
    assert emitted[0].frame_index == 21
    assert emitted[0].timing_probability == 0.58
    assert emitted[0].gesture is GestureType.SNARE


def test_completion_trigger_decoder_emits_on_threshold_crossing() -> None:
    decoder = CompletionTriggerDecoder(
        threshold=0.6,
        cooldown_frames=6,
    )

    assert decoder.update(
        frame_index=30,
        timestamp_seconds=30 / 30.0,
        timing_probability=0.55,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.90,
        class_probabilities={"kick": 0.90, "snare": 0.10},
    ) == ()

    emitted = decoder.update(
        frame_index=31,
        timestamp_seconds=31 / 30.0,
        timing_probability=0.64,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    )

    assert len(emitted) == 1
    assert emitted[0].frame_index == 31
    assert emitted[0].timing_probability == 0.64
    assert emitted[0].gesture is GestureType.KICK


def test_completion_trigger_decoder_requires_reset_before_rearming() -> None:
    decoder = CompletionTriggerDecoder(
        threshold=0.6,
        cooldown_frames=2,
    )

    emitted = decoder.update(
        frame_index=40,
        timestamp_seconds=40 / 30.0,
        timing_probability=0.62,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.90,
        class_probabilities={"kick": 0.10, "snare": 0.90},
    )
    assert len(emitted) == 1
    assert decoder.update(
        frame_index=41,
        timestamp_seconds=41 / 30.0,
        timing_probability=0.63,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.09, "snare": 0.91},
    ) == ()
    assert decoder.update(
        frame_index=42,
        timestamp_seconds=42 / 30.0,
        timing_probability=0.61,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.92,
        class_probabilities={"kick": 0.08, "snare": 0.92},
    ) == ()
    assert decoder.update(
        frame_index=43,
        timestamp_seconds=43 / 30.0,
        timing_probability=0.45,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.90,
        class_probabilities={"kick": 0.10, "snare": 0.90},
    ) == ()

    emitted = decoder.update(
        frame_index=44,
        timestamp_seconds=44 / 30.0,
        timing_probability=0.66,
        predicted_gesture=GestureType.SNARE,
        predicted_gesture_confidence=0.93,
        class_probabilities={"kick": 0.07, "snare": 0.93},
    )

    assert len(emitted) == 1
    assert emitted[0].frame_index == 44
    assert emitted[0].timing_probability == 0.66

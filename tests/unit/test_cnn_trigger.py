from __future__ import annotations

import numpy as np

from visionbeat.cnn_trigger import decode_trigger_events, evaluate_decoded_triggers


def test_decode_trigger_events_collapses_local_peaks_and_applies_cooldown() -> None:
    recording_ids = np.asarray(["rec"] * 8, dtype="<U16")
    frame_indices = np.arange(31, 39, dtype=np.int64)
    timestamps = frame_indices.astype(np.float32) / 30.0
    probabilities = np.asarray([0.1, 0.8, 0.75, 0.2, 0.7, 0.1, 0.85, 0.2], dtype=np.float32)

    decoded = decode_trigger_events(
        recording_ids=recording_ids,
        window_end_frame_indices=frame_indices,
        window_end_timestamps_seconds=timestamps,
        probabilities=probabilities,
        threshold=0.6,
        cooldown_frames=2,
        max_gap_frames=1,
    )

    assert [event.window_end_frame_index for event in decoded] == [32, 37]
    assert [round(event.probability, 2) for event in decoded] == [0.8, 0.85]


def test_evaluate_decoded_triggers_uses_event_neighborhoods() -> None:
    recording_ids = np.asarray(["rec"] * 8, dtype="<U16")
    frame_indices = np.arange(31, 39, dtype=np.int64)
    labels = np.asarray([0, 1, 1, 0, 0, 0, 1, 0], dtype=np.int64)
    timestamps = frame_indices.astype(np.float32) / 30.0
    probabilities = np.asarray([0.1, 0.8, 0.75, 0.2, 0.7, 0.1, 0.85, 0.2], dtype=np.float32)

    decoded = decode_trigger_events(
        recording_ids=recording_ids,
        window_end_frame_indices=frame_indices,
        window_end_timestamps_seconds=timestamps,
        probabilities=probabilities,
        threshold=0.6,
        cooldown_frames=2,
        max_gap_frames=1,
    )
    metrics = evaluate_decoded_triggers(
        decoded_triggers=decoded,
        recording_ids=recording_ids,
        window_end_frame_indices=frame_indices,
        labels=labels,
        match_tolerance_frames=0,
        max_gap_frames=1,
    )

    assert metrics["decoded_trigger_count"] == 2
    assert metrics["positive_event_count"] == 2
    assert metrics["detected_positive_event_count"] == 2
    assert metrics["false_positive_trigger_count"] == 0
    assert metrics["missed_positive_event_count"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

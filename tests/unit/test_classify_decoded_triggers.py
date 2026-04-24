from __future__ import annotations

import numpy as np

from visionbeat.classify_decoded_triggers import (
    PositiveGestureSpan,
    build_positive_gesture_spans,
    match_classified_triggers_to_positive_spans,
    summarize_matched_gesture_predictions,
)


def test_build_positive_gesture_spans_assigns_one_label_per_positive_run() -> None:
    spans = build_positive_gesture_spans(
        recording_ids=np.asarray(["rec-1"] * 6, dtype="<U16"),
        window_end_frame_indices=np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
        labels=np.asarray([1, 1, 0, 1, 1, 0], dtype=np.int64),
        target_gesture_labels=np.asarray(
            ["kick", "kick", "", "snare", "snare", ""],
            dtype="<U16",
        ),
        max_gap_frames=1,
    )

    assert spans == (
        PositiveGestureSpan(
            recording_id="rec-1",
            start_frame_index=10,
            end_frame_index=11,
            gesture_label="kick",
        ),
        PositiveGestureSpan(
            recording_id="rec-1",
            start_frame_index=13,
            end_frame_index=14,
            gesture_label="snare",
        ),
    )


def test_match_classified_triggers_and_summarize_type_metrics() -> None:
    positive_spans = (
        PositiveGestureSpan(
            recording_id="rec-1",
            start_frame_index=10,
            end_frame_index=11,
            gesture_label="kick",
        ),
        PositiveGestureSpan(
            recording_id="rec-1",
            start_frame_index=13,
            end_frame_index=14,
            gesture_label="snare",
        ),
    )
    rows = (
        {
            "trigger_index": 0,
            "recording_id": "rec-1",
            "window_index": 100,
            "window_end_frame_index": 10,
            "window_end_timestamp_seconds": 10 / 30.0,
            "timing_probability": 0.9,
            "predicted_gesture_label": "kick",
            "predicted_gesture_confidence": 0.8,
        },
        {
            "trigger_index": 1,
            "recording_id": "rec-1",
            "window_index": 101,
            "window_end_frame_index": 13,
            "window_end_timestamp_seconds": 13 / 30.0,
            "timing_probability": 0.85,
            "predicted_gesture_label": "kick",
            "predicted_gesture_confidence": 0.7,
        },
        {
            "trigger_index": 2,
            "recording_id": "rec-1",
            "window_index": 102,
            "window_end_frame_index": 18,
            "window_end_timestamp_seconds": 18 / 30.0,
            "timing_probability": 0.8,
            "predicted_gesture_label": "snare",
            "predicted_gesture_confidence": 0.9,
        },
    )

    matched = match_classified_triggers_to_positive_spans(
        rows=rows,
        positive_spans=positive_spans,
        match_tolerance_frames=0,
    )
    summary = summarize_matched_gesture_predictions(
        matched_rows=matched,
        false_trigger_count=1,
        positive_event_count=2,
        class_labels=("kick", "snare"),
    )

    assert len(matched) == 2
    assert matched[0].correct is True
    assert matched[1].correct is False
    assert summary["matched_trigger_count"] == 2
    assert summary["matched_accuracy"] == 0.5
    assert summary["correctly_typed_event_count"] == 1
    assert summary["correctly_typed_event_recall"] == 0.5
    assert summary["false_trigger_count"] == 1

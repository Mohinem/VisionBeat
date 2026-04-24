from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionbeat.analyze_decoder_timing import (
    CompletionEvent,
    SpanCompletionMapping,
    _classify_timing,
    load_completion_events,
    map_positive_spans_to_completion_events,
    match_triggers_to_completion_timings,
)
from visionbeat.predict_cnn import InferenceDataset


def test_map_positive_spans_to_completion_events_uses_future_horizon_lookup() -> None:
    dataset = InferenceDataset(
        path=None,  # type: ignore[arg-type]
        X=np.zeros((9, 4, 2), dtype=np.float32),
        recording_ids=np.asarray(["rec-1"] * 9, dtype="<U16"),
        window_end_frame_indices=np.asarray([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int64),
        window_end_timestamps_seconds=np.asarray(
            [frame / 30.0 for frame in range(10, 19)],
            dtype=np.float32,
        ),
        y=np.asarray([1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64),
        target_gesture_labels=np.asarray([""] * 9, dtype="<U16"),
        feature_names=None,
        schema_version=None,
        target_name="completion_within_next_k_frames",
        horizon_frames=3,
        stride=1,
    )
    completion_events = (
        CompletionEvent(recording_id="rec-1", completion_frame_index=13, gesture_label="kick"),
        CompletionEvent(recording_id="rec-1", completion_frame_index=18, gesture_label="snare"),
    )

    mappings = map_positive_spans_to_completion_events(
        dataset=dataset,
        completion_events=completion_events,
        max_gap_frames=1,
    )

    assert len(mappings) == 2
    assert mappings[0].recording_id == "rec-1"
    assert mappings[0].start_frame_index == 10
    assert mappings[0].end_frame_index == 12
    assert mappings[0].completion_frame_index == 13
    assert mappings[0].completion_timestamp_seconds == pytest.approx(13 / 30.0)
    assert mappings[0].gesture_label == "kick"
    assert mappings[1].recording_id == "rec-1"
    assert mappings[1].start_frame_index == 15
    assert mappings[1].end_frame_index == 16
    assert mappings[1].completion_frame_index == 18
    assert mappings[1].completion_timestamp_seconds == pytest.approx(18 / 30.0)
    assert mappings[1].gesture_label == "snare"


def test_match_triggers_to_completion_timings_marks_early_and_late_cases() -> None:
    matched = match_triggers_to_completion_timings(
        config_slug="th_0p60_cd_6_gap_1",
        decoded_triggers=(
            {
                "trigger_index": 0,
                "recording_id": "rec-1",
                "window_end_frame_index": 10,
                "window_end_timestamp_seconds": 10 / 30.0,
                "probability": 0.9,
            },
            {
                "trigger_index": 1,
                "recording_id": "rec-1",
                "window_end_frame_index": 16,
                "window_end_timestamp_seconds": 16 / 30.0,
                "probability": 0.8,
            },
        ),
        span_mappings=(
            SpanCompletionMapping(
                recording_id="rec-1",
                start_frame_index=9,
                end_frame_index=12,
                completion_frame_index=14,
                completion_timestamp_seconds=14 / 30.0,
                gesture_label="kick",
            ),
            SpanCompletionMapping(
                recording_id="rec-1",
                start_frame_index=15,
                end_frame_index=18,
                completion_frame_index=15,
                completion_timestamp_seconds=15 / 30.0,
                gesture_label="snare",
            ),
        ),
        match_tolerance_frames=0,
        too_early_frame_threshold=3,
        too_late_frame_threshold=0,
    )

    assert [row.delta_frames for row in matched] == [-4, 1]
    assert matched[0].too_early is True
    assert matched[0].too_late is False
    assert matched[0].timing_class == "before_completion"
    assert matched[1].too_early is False
    assert matched[1].too_late is True
    assert matched[1].timing_class == "after_completion"


def test_classify_timing_covers_all_relative_positions() -> None:
    assert _classify_timing(-1) == "before_completion"
    assert _classify_timing(0) == "at_completion"
    assert _classify_timing(1) == "after_completion"


def test_load_completion_events_accepts_v2_completion_frame_column(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels_v2.csv"
    labels_path.write_text(
        "recording_id,event_id,gesture_label,arm_start_frame,completion_frame\n"
        "session_01,evt-001,kick,10,12\n",
        encoding="utf-8",
    )

    events = load_completion_events(
        Path(labels_path),
        default_recording_id="session_01",
    )

    assert events == (
        CompletionEvent(
            recording_id="session_01",
            completion_frame_index=12,
            gesture_label="kick",
        ),
    )

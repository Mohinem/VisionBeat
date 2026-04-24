from __future__ import annotations

import numpy as np
import pytest

from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from visionbeat.train_cnn import CombinedDataset
from visionbeat.train_gesture_classifier import (
    prepare_gesture_classification_dataset,
    summarize_gesture_classes,
)


def test_prepare_gesture_classification_dataset_filters_positive_windows() -> None:
    dataset = _build_combined_dataset(
        y=np.asarray([0, 1, 1, 0, 1], dtype=np.int64),
        target_gesture_labels=np.asarray(["", "kick", "snare", "", "kick"], dtype="<U16"),
    )

    filtered = prepare_gesture_classification_dataset(
        dataset,
        indices=np.arange(5, dtype=np.int64),
    )

    assert filtered.sample_count == 3
    assert filtered.window_end_frame_indices.tolist() == [11, 12, 14]
    assert filtered.gesture_labels.tolist() == ["kick", "snare", "kick"]
    assert filtered.y.tolist() == [0, 1, 0]


def test_summarize_gesture_classes_rejects_missing_required_class() -> None:
    with pytest.raises(ValueError, match="missing one or more required classes"):
        summarize_gesture_classes(
            np.asarray([0, 0, 0], dtype=np.int64),
            class_labels=("kick", "snare"),
        )


def _build_combined_dataset(
    *,
    y: np.ndarray,
    target_gesture_labels: np.ndarray,
) -> CombinedDataset:
    sample_count = int(y.shape[0])
    X = np.zeros((sample_count, 24, len(FEATURE_NAMES)), dtype=np.float32)
    recording_ids = np.asarray(["rec-1"] * sample_count, dtype="<U16")
    window_end_frame_indices = np.arange(10, 10 + sample_count, dtype=np.int64)
    window_end_timestamps_seconds = window_end_frame_indices.astype(np.float32) / 30.0
    return CombinedDataset(
        X=X,
        y=y,
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        window_end_timestamps_seconds=window_end_timestamps_seconds,
        target_gesture_labels=target_gesture_labels,
        feature_names=FEATURE_NAMES,
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_count=len(FEATURE_NAMES),
        target_name="completion_within_next_k_frames",
        window_size=24,
        stride=1,
        horizon_frames=8,
    )

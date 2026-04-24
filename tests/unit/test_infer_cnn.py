from __future__ import annotations

from pathlib import Path

import numpy as np

from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from visionbeat.infer_cnn import (
    analyze_thresholds,
    build_threshold_grid,
    _infer_hidden_channels_from_state_dict,
    load_inference_dataset,
    summarize_threshold_analysis,
)


def test_load_inference_dataset_supports_optional_ground_truth(tmp_path: Path) -> None:
    dataset_path = tmp_path / "windows.npz"
    sample_count = 6
    X = np.arange(sample_count * 32 * len(FEATURE_NAMES), dtype=np.float32).reshape(
        sample_count,
        32,
        len(FEATURE_NAMES),
    )
    np.savez_compressed(
        dataset_path,
        X=X,
        recording_ids=np.asarray(["recording-a"] * sample_count, dtype="<U128"),
        window_end_frame_indices=np.arange(31, 31 + sample_count, dtype=np.int64),
        window_end_timestamps_seconds=np.arange(sample_count, dtype=np.float32) / 30.0,
        feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype="<U64"),
        target_name=np.asarray("completion_frame_binary", dtype="<U64"),
    )

    dataset = load_inference_dataset(dataset_path)

    assert dataset.sample_count == sample_count
    assert dataset.input_shape == (32, len(FEATURE_NAMES))
    assert dataset.y is None
    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.schema_version == FEATURE_SCHEMA_VERSION
    assert dataset.recording_ids[0] == "recording-a"
    assert dataset.horizon_frames == 0
    assert dataset.stride == 1


def test_load_inference_dataset_accepts_future_completion_target(tmp_path: Path) -> None:
    dataset_path = tmp_path / "future_windows.npz"
    sample_count = 4
    X = np.arange(sample_count * 32 * len(FEATURE_NAMES), dtype=np.float32).reshape(
        sample_count,
        32,
        len(FEATURE_NAMES),
    )
    np.savez_compressed(
        dataset_path,
        X=X,
        recording_ids=np.asarray(["recording-a"] * sample_count, dtype="<U128"),
        window_end_frame_indices=np.arange(31, 31 + sample_count, dtype=np.int64),
        window_end_timestamps_seconds=np.arange(sample_count, dtype=np.float32) / 30.0,
        feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype="<U64"),
        target_name=np.asarray("completion_within_next_k_frames", dtype="<U64"),
        horizon_frames=np.asarray(4, dtype=np.int64),
        stride=np.asarray(1, dtype=np.int64),
    )

    dataset = load_inference_dataset(dataset_path)

    assert dataset.target_name == "completion_within_next_k_frames"
    assert dataset.horizon_frames == 4


def test_load_inference_dataset_accepts_recent_completion_target(tmp_path: Path) -> None:
    dataset_path = tmp_path / "recent_windows.npz"
    sample_count = 4
    X = np.arange(sample_count * 32 * len(FEATURE_NAMES), dtype=np.float32).reshape(
        sample_count,
        32,
        len(FEATURE_NAMES),
    )
    np.savez_compressed(
        dataset_path,
        X=X,
        recording_ids=np.asarray(["recording-a"] * sample_count, dtype="<U128"),
        window_end_frame_indices=np.arange(31, 31 + sample_count, dtype=np.int64),
        window_end_timestamps_seconds=np.arange(sample_count, dtype=np.float32) / 30.0,
        feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype="<U64"),
        target_name=np.asarray("completion_within_last_k_frames", dtype="<U64"),
        horizon_frames=np.asarray(3, dtype=np.int64),
        stride=np.asarray(1, dtype=np.int64),
    )

    dataset = load_inference_dataset(dataset_path)

    assert dataset.target_name == "completion_within_last_k_frames"
    assert dataset.horizon_frames == 3


def test_load_inference_dataset_accepts_arm_target(tmp_path: Path) -> None:
    dataset_path = tmp_path / "arm_windows.npz"
    sample_count = 4
    X = np.arange(sample_count * 32 * len(FEATURE_NAMES), dtype=np.float32).reshape(
        sample_count,
        32,
        len(FEATURE_NAMES),
    )
    np.savez_compressed(
        dataset_path,
        X=X,
        recording_ids=np.asarray(["recording-a"] * sample_count, dtype="<U128"),
        window_end_frame_indices=np.arange(31, 31 + sample_count, dtype=np.int64),
        window_end_timestamps_seconds=np.arange(sample_count, dtype=np.float32) / 30.0,
        feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype="<U64"),
        target_name=np.asarray("arm_within_next_k_frames", dtype="<U64"),
        horizon_frames=np.asarray(3, dtype=np.int64),
        stride=np.asarray(1, dtype=np.int64),
    )

    dataset = load_inference_dataset(dataset_path)

    assert dataset.target_name == "arm_within_next_k_frames"
    assert dataset.horizon_frames == 3


def test_infer_hidden_channels_from_state_dict_uses_saved_shapes() -> None:
    state_dict = {
        "features.0.weight": np.zeros((64, len(FEATURE_NAMES), 3), dtype=np.float32),
        "classifier.2.weight": np.zeros((1, 128), dtype=np.float32),
    }

    hidden_channels = _infer_hidden_channels_from_state_dict(
        state_dict=state_dict,
        feature_count=len(FEATURE_NAMES),
    )

    assert hidden_channels == 64


def test_threshold_analysis_finds_best_f1_and_low_fp_thresholds() -> None:
    thresholds = build_threshold_grid(start=0.1, stop=0.3, step=0.1)
    assert thresholds == [0.1, 0.2, 0.3]

    analysis_rows = analyze_thresholds(
        y_true=np.asarray([0, 0, 1, 1], dtype=np.int64),
        probabilities=np.asarray([0.15, 0.25, 0.35, 0.85], dtype=np.float32),
        thresholds=thresholds,
    )
    summary = summarize_threshold_analysis(
        analysis_rows=analysis_rows,
        selected_threshold=0.2,
    )

    assert summary["best_f1_threshold"]["threshold"] == 0.3
    assert summary["best_f1_threshold"]["f1"] == 1.0
    assert summary["lowest_false_positive_threshold"]["threshold"] == 0.3
    assert summary["lowest_false_positive_threshold"]["false_positive"] == 0
    assert summary["selected_threshold"]["threshold"] == 0.2

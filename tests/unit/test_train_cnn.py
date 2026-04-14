from __future__ import annotations

from pathlib import Path

import numpy as np

from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from visionbeat.train_cnn import (
    _binary_classification_metrics,
    _choose_binary_loss_mitigation,
    _curate_training_negatives,
    _prepare_run_directory,
    _summarize_binary_targets,
    combine_archives,
    load_archive,
    split_dataset,
)


def test_default_split_uses_last_recording_for_validation_and_purges_overlap(
    tmp_path: Path,
) -> None:
    archive_one = tmp_path / "recording_1_actual.npz"
    archive_two = tmp_path / "recording_2_actual.npz"
    _write_archive(
        archive_one,
        recording_id="recording-1",
        frame_start=31,
        sample_count=80,
        positive_positions=(20, 60),
    )
    _write_archive(
        archive_two,
        recording_id="recording-2",
        frame_start=31,
        sample_count=100,
        positive_positions=(25, 26, 27, 55, 56, 57, 85),
    )

    combined = combine_archives((load_archive(archive_one), load_archive(archive_two)))
    split = split_dataset(combined, validation_fraction=0.25)

    assert combined.X.shape == (180, 32, len(FEATURE_NAMES))
    assert combined.feature_names == FEATURE_NAMES
    assert combined.schema_version == FEATURE_SCHEMA_VERSION
    assert split.train_indices.size > 0
    assert split.validation_indices.size > 0
    assert split.purge_gap_frames == 31
    assert split.policy == "grouped_tail_with_validation_recording"
    assert split.validation_recording_id == "recording-2"

    train_recording_1 = sorted(
        int(combined.window_end_frame_indices[index])
        for index in split.train_indices
        if combined.recording_ids[index] == "recording-1"
    )
    validation_recording_1 = sorted(
        int(combined.window_end_frame_indices[index])
        for index in split.validation_indices
        if combined.recording_ids[index] == "recording-1"
    )
    assert train_recording_1
    assert not validation_recording_1
    assert train_recording_1 == list(range(31, 111))

    train_recording_2 = sorted(
        int(combined.window_end_frame_indices[index])
        for index in split.train_indices
        if combined.recording_ids[index] == "recording-2"
    )
    validation_recording_2 = sorted(
        int(combined.window_end_frame_indices[index])
        for index in split.validation_indices
        if combined.recording_ids[index] == "recording-2"
    )
    assert train_recording_2
    assert validation_recording_2
    assert max(train_recording_2) <= min(validation_recording_2) - combined.window_size
    validation_positive_frames = sorted(
        int(combined.window_end_frame_indices[index])
        for index in split.validation_indices
        if combined.recording_ids[index] == "recording-2" and combined.y[index] == 1
    )
    assert validation_positive_frames == [116]


def test_validation_recording_id_override_is_respected(tmp_path: Path) -> None:
    archive_one = tmp_path / "recording_1_actual.npz"
    archive_two = tmp_path / "recording_2_actual.npz"
    _write_archive(
        archive_one,
        recording_id="recording-1",
        frame_start=31,
        sample_count=80,
        positive_positions=(20, 50, 70),
    )
    _write_archive(
        archive_two,
        recording_id="recording-2",
        frame_start=31,
        sample_count=100,
        positive_positions=(30, 60, 90),
    )

    combined = combine_archives((load_archive(archive_one), load_archive(archive_two)))
    split = split_dataset(
        combined,
        validation_fraction=0.34,
        validation_recording_id="recording-1",
    )

    assert split.validation_recording_id == "recording-1"
    assert np.all(combined.recording_ids[split.validation_indices] == "recording-1")
    assert np.any(combined.recording_ids[split.train_indices] == "recording-2")


def test_binary_label_summary_and_loss_mitigation() -> None:
    imbalanced_stats = _summarize_binary_targets(np.asarray([0, 0, 0, 0, 1], dtype=np.int64))
    assert imbalanced_stats.total_count == 5
    assert imbalanced_stats.negative_count == 4
    assert imbalanced_stats.positive_count == 1
    assert imbalanced_stats.positive_rate == 0.2
    assert imbalanced_stats.negative_to_positive_ratio == 4.0
    strategy, pos_weight = _choose_binary_loss_mitigation(imbalanced_stats)
    assert strategy == "bce_with_logits_pos_weight"
    assert pos_weight == 4.0

    balanced_stats = _summarize_binary_targets(np.asarray([0, 1, 0, 1], dtype=np.int64))
    strategy, pos_weight = _choose_binary_loss_mitigation(balanced_stats)
    assert strategy == "standard_bce_with_logits"
    assert pos_weight == 1.0


def test_binary_classification_metrics_include_confusion_and_roc_auc() -> None:
    metrics = _binary_classification_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float32),
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(float(metrics["f1"]), 6) == round(2.0 / 3.0, 6)
    assert metrics["roc_auc"] == 0.75
    assert metrics["true_negative"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["detected_positive_count"] == 1
    assert metrics["missed_positive_count"] == 1
    assert metrics["predicted_positive_count"] == 1
    assert metrics["predicted_positive_rate"] == 0.25


def test_load_archive_accepts_future_completion_targets(tmp_path: Path) -> None:
    archive_path = tmp_path / "future_target.npz"
    _write_archive(
        archive_path,
        recording_id="recording-1",
        frame_start=31,
        sample_count=12,
        positive_positions=(4, 5),
        target_name="completion_within_next_k_frames",
        horizon_frames=4,
    )

    archive = load_archive(archive_path)

    assert archive.target_name == "completion_within_next_k_frames"
    assert archive.horizon_frames == 4


def test_curate_training_negatives_keeps_hard_negatives_and_downsamples_easy_ones(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "curation.npz"
    _write_archive(
        archive_path,
        recording_id="recording-1",
        frame_start=31,
        sample_count=20,
        positive_positions=(9, 10),
    )
    dataset = combine_archives((load_archive(archive_path),))
    train_indices = np.arange(20, dtype=np.int64)

    curation = _curate_training_negatives(
        dataset,
        train_indices=train_indices,
        seed=7,
        max_negative_positive_ratio=2.0,
        hard_negative_margin_frames=2,
    )

    kept_frames = dataset.window_end_frame_indices[curation.kept_indices].tolist()
    assert curation.kept_positive_count == 2
    assert curation.kept_hard_negative_count == 4
    assert curation.kept_easy_negative_count == 0
    assert curation.dropped_easy_negative_count == 14
    assert kept_frames == [38, 39, 40, 41, 42, 43]


def test_prepare_run_directory_auto_increments(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_one = _prepare_run_directory(outputs_dir)
    assert run_one.name == "visionbeat_cnn_run_001"
    assert run_one.is_dir()

    run_two = _prepare_run_directory(outputs_dir)
    assert run_two.name == "visionbeat_cnn_run_002"
    assert run_two.is_dir()


def _write_archive(
    path: Path,
    *,
    recording_id: str,
    frame_start: int,
    sample_count: int,
    positive_positions: tuple[int, ...],
    target_name: str = "completion_frame_binary",
    horizon_frames: int = 4,
) -> None:
    X = np.arange(sample_count * 32 * len(FEATURE_NAMES), dtype=np.float32).reshape(
        sample_count,
        32,
        len(FEATURE_NAMES),
    )
    y = np.zeros(sample_count, dtype=np.int64)
    y[list(positive_positions)] = 1
    target_gesture_labels = np.asarray(
        ["kick" if label == 1 else "" for label in y],
        dtype="<U128",
    )
    recording_ids = np.asarray([recording_id] * sample_count, dtype="<U128")
    window_end_frame_indices = np.arange(frame_start, frame_start + sample_count, dtype=np.int64)
    window_end_timestamps_seconds = window_end_frame_indices.astype(np.float32) / 30.0
    np.savez_compressed(
        path,
        X=X,
        y=y,
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        window_end_timestamps_seconds=window_end_timestamps_seconds,
        target_gesture_labels=target_gesture_labels,
        feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype="<U64"),
        feature_count=np.asarray(len(FEATURE_NAMES), dtype=np.int64),
        target_name=np.asarray(target_name, dtype="<U64"),
        window_size=np.asarray(32, dtype=np.int64),
        stride=np.asarray(1, dtype=np.int64),
        horizon_frames=np.asarray(horizon_frames, dtype=np.int64),
    )

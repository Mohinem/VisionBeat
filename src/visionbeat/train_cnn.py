"""Train a small 1D CNN baseline on trusted VisionBeat NPZ archives."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Final

import numpy as np

from visionbeat.cnn_trigger import decode_trigger_events, evaluate_decoded_triggers
from visionbeat.cnn_model import (
    VisionBeatCnnSpec,
    binary_classification_metrics as _binary_classification_metrics,
    build_checkpoint_payload,
    build_completion_cnn,
    format_optional_metric as _format_optional_metric,
    require_torch as _require_torch,
    resolve_device as _resolve_device,
)
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

_SUPPORTED_TARGET_NAMES: Final[tuple[str, ...]] = (
    "completion_frame_binary",
    "completion_within_next_k_frames",
    "completion_within_last_k_frames",
    "arm_frame_binary",
    "arm_within_next_k_frames",
    "arm_within_last_k_frames",
)
_REQUIRED_ARCHIVE_KEYS: Final[tuple[str, ...]] = (
    "X",
    "y",
    "recording_ids",
    "window_end_frame_indices",
    "window_end_timestamps_seconds",
    "target_gesture_labels",
    "feature_names",
    "schema_version",
    "feature_count",
    "target_name",
    "window_size",
    "stride",
    "horizon_frames",
)
_DEFAULT_ARCHIVES: Final[tuple[str, str]] = (
    "/tmp/visionbeat-diff-check/recording_1_actual.npz",
    "/tmp/visionbeat-diff-check/recording_2_actual.npz",
)
_EPSILON: Final[float] = 1e-8
_RUN_DIR_PREFIX: Final[str] = "visionbeat_cnn_run_"


@dataclass(frozen=True, slots=True)
class LoadedArchive:
    """One validated CNN-ready archive."""

    path: Path
    X: np.ndarray
    y: np.ndarray
    recording_ids: np.ndarray
    window_end_frame_indices: np.ndarray
    window_end_timestamps_seconds: np.ndarray
    target_gesture_labels: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str
    feature_count: int
    target_name: str
    window_size: int
    stride: int
    horizon_frames: int


@dataclass(frozen=True, slots=True)
class CombinedDataset:
    """Combined archives with shared metadata."""

    X: np.ndarray
    y: np.ndarray
    recording_ids: np.ndarray
    window_end_frame_indices: np.ndarray
    window_end_timestamps_seconds: np.ndarray
    target_gesture_labels: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str
    feature_count: int
    target_name: str
    window_size: int
    stride: int
    horizon_frames: int

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the per-sample input shape stored in the archives."""
        return (self.window_size, self.feature_count)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Leak-aware train/validation split metadata."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    purge_gap_frames: int
    policy: str
    validation_recording_id: str


@dataclass(frozen=True, slots=True)
class RecordingGroup:
    """One contiguous local gesture neighborhood inside a recording."""

    indices: tuple[int, ...]
    first_end_frame: int
    last_end_frame: int
    positive_count: int


@dataclass(frozen=True, slots=True)
class BinaryLabelStats:
    """Binary class counts and rates for logging and loss configuration."""

    total_count: int
    negative_count: int
    positive_count: int
    positive_rate: float
    negative_to_positive_ratio: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "total_count": self.total_count,
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
            "positive_rate": self.positive_rate,
            "negative_to_positive_ratio": self.negative_to_positive_ratio,
        }


@dataclass(frozen=True, slots=True)
class NegativeCurationResult:
    """Train-only negative curation summary."""

    kept_indices: np.ndarray
    kept_positive_count: int
    kept_hard_negative_count: int
    kept_easy_negative_count: int
    dropped_easy_negative_count: int
    max_negative_positive_ratio: float | None
    hard_negative_margin_frames: int

    @property
    def kept_negative_count(self) -> int:
        return self.kept_hard_negative_count + self.kept_easy_negative_count

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "kept_sample_count": int(self.kept_indices.size),
            "kept_positive_count": self.kept_positive_count,
            "kept_negative_count": self.kept_negative_count,
            "kept_hard_negative_count": self.kept_hard_negative_count,
            "kept_easy_negative_count": self.kept_easy_negative_count,
            "dropped_easy_negative_count": self.dropped_easy_negative_count,
            "max_negative_positive_ratio": self.max_negative_positive_ratio,
            "hard_negative_margin_frames": self.hard_negative_margin_frames,
        }


@dataclass(frozen=True, slots=True)
class BinaryLossMitigationPlan:
    """Resolved per-class loss weights for one training run."""

    strategy: str
    positive_weight: float
    negative_weight: float
    majority_downsample_factor: float | None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "strategy": self.strategy,
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "majority_downsample_factor": self.majority_downsample_factor,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for CNN training."""

    parser = argparse.ArgumentParser(
        description="Train a baseline 1D CNN on trusted VisionBeat NPZ archives."
    )
    parser.add_argument(
        "archives",
        nargs="*",
        default=list(_DEFAULT_ARCHIVES),
        help=(
            "Trusted NPZ archives to load. Defaults to the two canonical archives in "
            "/tmp/visionbeat-diff-check."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Base output directory. If the final path component does not already look "
            "like a run directory, the script creates the next numbered "
            "visionbeat_cnn_run_### folder under it."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. Default: 20.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size. Default: 256.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Default: 1e-3.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default: 1e-4.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate in the classifier head. Default: 0.2.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Base channel width for the CNN. Default: 64.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of grouped local gesture neighborhoods held out for validation "
            "inside the validation recording. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--validation-recording-id",
        default="",
        help=(
            "Recording id that should supply the grouped validation holdout. By "
            "default, the script uses the last recording encountered in the archives, "
            "which matches the canonical recording_1 train / recording_2 validation "
            "setup."
        ),
    )
    parser.add_argument(
        "--holdout-recording-id",
        action="append",
        default=[],
        help=(
            "Optional recording id to place entirely in validation. Repeatable. "
            "This is the safest split if you want zero cross-recording leakage."
        ),
    )
    parser.add_argument(
        "--max-train-negative-positive-ratio",
        type=float,
        default=6.0,
        help=(
            "Train-only negative curation ratio. All positives and all hard negatives are kept, "
            "then easy negatives are downsampled so negatives are at most this multiple of "
            "positives. Use 0 or a negative value to disable curation. Default: 6.0."
        ),
    )
    parser.add_argument(
        "--hard-negative-margin-frames",
        type=int,
        default=-1,
        help=(
            "Frame distance around each positive window end used to keep hard negatives. "
            "Defaults to the dataset window_size when omitted."
        ),
    )
    parser.add_argument(
        "--imbalance-strategy",
        default="positive_pos_weight",
        choices=("positive_pos_weight", "majority_downsample_upweight"),
        help=(
            "How to compensate for class imbalance during training. "
            "`positive_pos_weight` keeps the existing positive-class BCE weighting. "
            "`majority_downsample_upweight` follows the downsample-majority and "
            "upweight-kept-majority recipe."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        default="f1",
        choices=(
            "loss",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "decoded_trigger_f1",
            "decoded_trigger_precision",
            "decoded_trigger_recall",
        ),
        help=(
            "Metric used to choose the saved best checkpoint. "
            "`loss` minimizes validation loss; all other options maximize the named "
            "validation metric. Decoded-trigger metrics evaluate event quality on the "
            "validation split after local-peak decoding. Default: f1."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-trigger-cooldown-frames",
        type=int,
        default=-1,
        help=(
            "Cooldown in frames for validation decoded-trigger checkpoint selection. "
            "Defaults to half the dataset window size."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-trigger-max-gap-frames",
        type=int,
        default=-1,
        help=(
            "Maximum gap in frames used to merge nearby positive windows when "
            "computing validation decoded-trigger metrics. Defaults to dataset stride."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-trigger-match-tolerance-frames",
        type=int,
        default=0,
        help=(
            "Extra frame tolerance when matching decoded validation triggers to labeled "
            "positive neighborhoods. Default: 0."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic training seed. Default: 7.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. Default: auto.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Default: 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for model training."""

    args = parse_args(argv)
    try:
        run_dir = _prepare_run_directory(Path(args.output_dir))
        archives = tuple(load_archive(Path(path)) for path in args.archives)
        dataset = combine_archives(archives)
        split = split_dataset(
            dataset,
            validation_fraction=args.validation_fraction,
            validation_recording_id=args.validation_recording_id,
            holdout_recording_ids=tuple(args.holdout_recording_id),
        )
        result = train_model(
            dataset,
            split=split,
            output_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            max_negative_positive_ratio=args.max_train_negative_positive_ratio,
            hard_negative_margin_frames=args.hard_negative_margin_frames,
            imbalance_strategy=args.imbalance_strategy,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
            checkpoint_selection_trigger_cooldown_frames=(
                args.checkpoint_selection_trigger_cooldown_frames
            ),
            checkpoint_selection_trigger_max_gap_frames=(
                args.checkpoint_selection_trigger_max_gap_frames
            ),
            checkpoint_selection_trigger_match_tolerance_frames=(
                args.checkpoint_selection_trigger_match_tolerance_frames
            ),
            config=_build_run_config(args=args, dataset=dataset, split=split, output_dir=run_dir),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print("Training complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Best checkpoint: {result['best_checkpoint_path']}")
    print(f"Config: {result['config_path']}")
    print(f"Epoch metrics: {result['epoch_metrics_path']}")
    print(f"Evaluation report: {result['evaluation_report_path']}")
    print(
        "Checkpoint selection: "
        f"{result['checkpoint_selection_metric']}={result['checkpoint_selection_score']:.6f}"
    )
    print(f"Selected checkpoint validation loss: {result['best_val_loss']:.6f}")


def _prepare_run_directory(base_path: Path) -> Path:
    """Create a clear run directory without overwriting existing artifacts."""

    if base_path.name.startswith(_RUN_DIR_PREFIX):
        if base_path.exists() and any(base_path.iterdir()):
            raise ValueError(
                f"Run directory already exists and is not empty: {base_path}"
            )
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    base_path.mkdir(parents=True, exist_ok=True)
    next_index = 1
    for child in base_path.iterdir():
        if not child.is_dir() or not child.name.startswith(_RUN_DIR_PREFIX):
            continue
        suffix = child.name[len(_RUN_DIR_PREFIX) :]
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    run_dir = base_path / f"{_RUN_DIR_PREFIX}{next_index:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _build_run_config(
    *,
    args: argparse.Namespace,
    dataset: CombinedDataset,
    split: DatasetSplit,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a serializable record of the configuration used for one run."""

    return {
        "archives": [str(path) for path in args.archives],
        "output_dir": str(output_dir),
        "schema_version": dataset.schema_version,
        "feature_count": dataset.feature_count,
        "input_shape": list(dataset.input_shape),
        "target_name": dataset.target_name,
        "horizon_frames": dataset.horizon_frames,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_channels": args.hidden_channels,
            "max_train_negative_positive_ratio": args.max_train_negative_positive_ratio,
            "hard_negative_margin_frames": args.hard_negative_margin_frames,
            "imbalance_strategy": args.imbalance_strategy,
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "checkpoint_selection_trigger_cooldown_frames": (
                args.checkpoint_selection_trigger_cooldown_frames
            ),
            "checkpoint_selection_trigger_max_gap_frames": (
                args.checkpoint_selection_trigger_max_gap_frames
            ),
            "checkpoint_selection_trigger_match_tolerance_frames": (
                args.checkpoint_selection_trigger_match_tolerance_frames
            ),
            "seed": args.seed,
            "device": args.device,
            "num_workers": args.num_workers,
        },
        "split": {
            "validation_fraction": args.validation_fraction,
            "validation_recording_id": split.validation_recording_id,
            "holdout_recording_ids": list(args.holdout_recording_id),
            "policy": split.policy,
            "purge_gap_frames": split.purge_gap_frames,
        },
    }


def load_archive(path: Path) -> LoadedArchive:
    """Load and validate one trusted NPZ archive."""

    if not path.exists():
        raise FileNotFoundError(f"Archive does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing_keys = [key for key in _REQUIRED_ARCHIVE_KEYS if key not in archive.files]
        if missing_keys:
            raise ValueError(
                f"Archive {path} is missing required keys: {', '.join(missing_keys)}."
            )

        X = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.int64)
        recording_ids = np.asarray(archive["recording_ids"])
        window_end_frame_indices = np.asarray(
            archive["window_end_frame_indices"],
            dtype=np.int64,
        )
        window_end_timestamps_seconds = np.asarray(
            archive["window_end_timestamps_seconds"],
            dtype=np.float32,
        )
        target_gesture_labels = np.asarray(archive["target_gesture_labels"])
        feature_names = tuple(str(name) for name in archive["feature_names"].tolist())
        schema_version = str(archive["schema_version"].item())
        feature_count = int(archive["feature_count"].item())
        target_name = str(archive["target_name"].item())
        window_size = int(archive["window_size"].item())
        stride = int(archive["stride"].item())
        horizon_frames = int(archive["horizon_frames"].item())

    _validate_archive_shapes(
        path=path,
        X=X,
        y=y,
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        window_end_timestamps_seconds=window_end_timestamps_seconds,
        target_gesture_labels=target_gesture_labels,
        feature_names=feature_names,
        schema_version=schema_version,
        feature_count=feature_count,
        target_name=target_name,
        window_size=window_size,
        stride=stride,
        horizon_frames=horizon_frames,
    )
    return LoadedArchive(
        path=path,
        X=X,
        y=y,
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        window_end_timestamps_seconds=window_end_timestamps_seconds,
        target_gesture_labels=target_gesture_labels,
        feature_names=feature_names,
        schema_version=schema_version,
        feature_count=feature_count,
        target_name=target_name,
        window_size=window_size,
        stride=stride,
        horizon_frames=horizon_frames,
    )


def combine_archives(archives: tuple[LoadedArchive, ...]) -> CombinedDataset:
    """Combine multiple compatible archives into one training dataset."""

    if not archives:
        raise ValueError("At least one archive is required for training.")

    reference = archives[0]
    for archive in archives[1:]:
        if archive.feature_names != reference.feature_names:
            raise ValueError(
                f"Feature name mismatch between {reference.path} and {archive.path}."
            )
        if archive.schema_version != reference.schema_version:
            raise ValueError(
                f"Schema version mismatch between {reference.path} and {archive.path}."
            )
        if archive.feature_count != reference.feature_count:
            raise ValueError(
                f"Feature count mismatch between {reference.path} and {archive.path}."
            )
        if archive.window_size != reference.window_size:
            raise ValueError(
                f"Window size mismatch between {reference.path} and {archive.path}."
            )
        if archive.stride != reference.stride:
            raise ValueError(
                f"Stride mismatch between {reference.path} and {archive.path}."
            )
        if archive.target_name != reference.target_name:
            raise ValueError(
                f"Target mismatch between {reference.path} and {archive.path}."
            )
        if archive.horizon_frames != reference.horizon_frames:
            raise ValueError(
                f"Horizon mismatch between {reference.path} and {archive.path}."
            )

    return CombinedDataset(
        X=np.concatenate([archive.X for archive in archives], axis=0),
        y=np.concatenate([archive.y for archive in archives], axis=0),
        recording_ids=np.concatenate([archive.recording_ids for archive in archives], axis=0),
        window_end_frame_indices=np.concatenate(
            [archive.window_end_frame_indices for archive in archives],
            axis=0,
        ),
        window_end_timestamps_seconds=np.concatenate(
            [archive.window_end_timestamps_seconds for archive in archives],
            axis=0,
        ),
        target_gesture_labels=np.concatenate(
            [archive.target_gesture_labels for archive in archives],
            axis=0,
        ),
        feature_names=reference.feature_names,
        schema_version=reference.schema_version,
        feature_count=reference.feature_count,
        target_name=reference.target_name,
        window_size=reference.window_size,
        stride=reference.stride,
        horizon_frames=reference.horizon_frames,
    )


def split_dataset(
    dataset: CombinedDataset,
    *,
    validation_fraction: float,
    validation_recording_id: str = "",
    holdout_recording_ids: tuple[str, ...] = (),
) -> DatasetSplit:
    """Split by recording and grouped completion neighborhoods to reduce leakage."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than zero and less than one.")

    purge_gap_frames = dataset.window_size - 1
    holdout_set = {recording_id for recording_id in holdout_recording_ids if recording_id != ""}
    if holdout_set:
        validation_mask = np.isin(dataset.recording_ids, list(holdout_set))
        train_mask = ~validation_mask
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if train_indices.size == 0:
            raise ValueError("Holdout split leaves zero training samples.")
        if validation_indices.size == 0:
            raise ValueError("Holdout split leaves zero validation samples.")
        return DatasetSplit(
            train_indices=train_indices.astype(np.int64, copy=False),
            validation_indices=validation_indices.astype(np.int64, copy=False),
            purge_gap_frames=purge_gap_frames,
            policy="full_recording_holdout",
            validation_recording_id=",".join(sorted(holdout_set)),
        )

    by_recording: dict[str, list[int]] = defaultdict(list)
    for index, recording_id in enumerate(dataset.recording_ids.tolist()):
        by_recording[str(recording_id)].append(index)

    resolved_validation_recording_id = _resolve_validation_recording_id(
        by_recording=by_recording,
        preferred_recording_id=validation_recording_id,
        recording_ids=dataset.recording_ids,
    )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for recording_id, indices in by_recording.items():
        ordered = sorted(indices, key=lambda idx: int(dataset.window_end_frame_indices[idx]))
        if len(ordered) < 2:
            raise ValueError(
                f"Recording {recording_id!r} does not have enough windows for a split."
            )
        if recording_id != resolved_validation_recording_id:
            train_indices.extend(ordered)
            continue

        safe_training_chunk, validation_chunk = _split_validation_recording(
            ordered_indices=ordered,
            y=dataset.y,
            window_end_frame_indices=dataset.window_end_frame_indices,
            validation_fraction=validation_fraction,
            purge_gap_frames=purge_gap_frames,
            group_gap_frames=dataset.window_size,
            recording_id=recording_id,
        )
        train_indices.extend(safe_training_chunk)
        validation_indices.extend(validation_chunk)

    if not train_indices:
        raise ValueError("Split leaves zero training samples.")
    if not validation_indices:
        raise ValueError("Split leaves zero validation samples.")

    return DatasetSplit(
        train_indices=np.asarray(sorted(train_indices), dtype=np.int64),
        validation_indices=np.asarray(sorted(validation_indices), dtype=np.int64),
        purge_gap_frames=purge_gap_frames,
        policy="grouped_tail_with_validation_recording",
        validation_recording_id=resolved_validation_recording_id,
    )


def _resolve_validation_recording_id(
    *,
    by_recording: dict[str, list[int]],
    preferred_recording_id: str,
    recording_ids: np.ndarray,
) -> str:
    """Pick the recording that will contribute the validation holdout."""

    if preferred_recording_id:
        if preferred_recording_id not in by_recording:
            raise ValueError(
                f"validation_recording_id {preferred_recording_id!r} is not present in the archives."
            )
        return preferred_recording_id

    ordered_recordings: list[str] = []
    seen: set[str] = set()
    for recording_id in recording_ids.tolist():
        recording_id = str(recording_id)
        if recording_id not in seen:
            seen.add(recording_id)
            ordered_recordings.append(recording_id)
    if not ordered_recordings:
        raise ValueError("No recording ids were found in the combined dataset.")
    return ordered_recordings[-1]


def _split_validation_recording(
    ordered_indices: list[int],
    *,
    y: np.ndarray,
    window_end_frame_indices: np.ndarray,
    validation_fraction: float,
    purge_gap_frames: int,
    group_gap_frames: int,
    recording_id: str,
) -> tuple[list[int], list[int]]:
    """Split one recording by grouped completion neighborhoods plus a purge boundary."""

    groups = _build_recording_groups(
        ordered_indices=ordered_indices,
        y=y,
        window_end_frame_indices=window_end_frame_indices,
        group_gap_frames=group_gap_frames,
    )
    if len(groups) < 2:
        return _split_validation_recording_by_tail(
            ordered_indices=ordered_indices,
            y=y,
            window_end_frame_indices=window_end_frame_indices,
            validation_fraction=validation_fraction,
            purge_gap_frames=purge_gap_frames,
            recording_id=recording_id,
        )

    validation_group_count = max(1, int(ceil(len(groups) * validation_fraction)))
    validation_group_count = min(validation_group_count, len(groups) - 1)
    validation_groups = groups[-validation_group_count:]
    validation_chunk = [index for group in validation_groups for index in group.indices]
    first_validation_end_frame = validation_groups[0].first_end_frame
    safe_training_chunk = [
        index
        for group in groups[:-validation_group_count]
        for index in group.indices
        if int(window_end_frame_indices[index]) <= first_validation_end_frame - purge_gap_frames - 1
    ]
    if not validation_chunk:
        raise ValueError(
            f"Grouped split produced zero validation windows for recording {recording_id!r}."
        )
    if not safe_training_chunk:
        raise ValueError(
            "Grouped split produced zero safe training windows for recording "
            f"{recording_id!r}. Try a smaller validation_fraction or use "
            "--holdout-recording-id."
        )
    return safe_training_chunk, validation_chunk


def _build_recording_groups(
    *,
    ordered_indices: list[int],
    y: np.ndarray,
    window_end_frame_indices: np.ndarray,
    group_gap_frames: int,
) -> list[RecordingGroup]:
    """Partition a recording into contiguous local gesture neighborhoods."""

    ordered_array = np.asarray(ordered_indices, dtype=np.int64)
    frames = window_end_frame_indices[ordered_array].astype(np.int64, copy=False)
    labels = y[ordered_array].astype(np.int64, copy=False)
    positive_positions = np.flatnonzero(labels == 1)
    if positive_positions.size < 2:
        return []

    positive_clusters: list[tuple[int, int]] = []
    cluster_start = int(positive_positions[0])
    cluster_end = int(positive_positions[0])
    for position in positive_positions[1:]:
        position = int(position)
        current_frame = int(frames[position])
        previous_frame = int(frames[cluster_end])
        if current_frame - previous_frame <= group_gap_frames:
            cluster_end = position
            continue
        positive_clusters.append((cluster_start, cluster_end))
        cluster_start = position
        cluster_end = position
    positive_clusters.append((cluster_start, cluster_end))

    if len(positive_clusters) < 2:
        return []

    groups: list[RecordingGroup] = []
    start_position = 0
    for cluster_index, (cluster_start_pos, cluster_end_pos) in enumerate(positive_clusters):
        if cluster_index == len(positive_clusters) - 1:
            end_position = len(ordered_array) - 1
        else:
            next_cluster_start_pos = positive_clusters[cluster_index + 1][0]
            boundary_frame = (
                int(frames[cluster_end_pos]) + int(frames[next_cluster_start_pos])
            ) // 2
            end_position = int(np.searchsorted(frames, boundary_frame, side="right") - 1)
        group_positions = ordered_array[start_position : end_position + 1]
        group_labels = labels[start_position : end_position + 1]
        groups.append(
            RecordingGroup(
                indices=tuple(int(index) for index in group_positions.tolist()),
                first_end_frame=int(frames[start_position]),
                last_end_frame=int(frames[end_position]),
                positive_count=int(group_labels.sum()),
            )
        )
        start_position = end_position + 1
    return groups


def _split_validation_recording_by_tail(
    *,
    ordered_indices: list[int],
    y: np.ndarray,
    window_end_frame_indices: np.ndarray,
    validation_fraction: float,
    purge_gap_frames: int,
    recording_id: str,
) -> tuple[list[int], list[int]]:
    """Fallback for sparse recordings without enough positive groups."""

    validation_count = max(1, int(ceil(len(ordered_indices) * validation_fraction)))
    start = min(len(ordered_indices) - 1, max(1, len(ordered_indices) - validation_count))
    labels = y[np.asarray(ordered_indices, dtype=np.int64)]
    if labels.sum() > 0 and labels[start:].sum() == 0:
        positive_positions = np.flatnonzero(labels == 1)
        if positive_positions.size > 0:
            start = max(1, int(positive_positions[-1]))
    validation_chunk = ordered_indices[start:]
    first_validation_end_frame = int(window_end_frame_indices[validation_chunk[0]])
    safe_training_chunk = [
        index
        for index in ordered_indices[:start]
        if int(window_end_frame_indices[index]) <= first_validation_end_frame - purge_gap_frames - 1
    ]
    if not safe_training_chunk:
        raise ValueError(
            "Chronology-aware split produced zero safe training windows for "
            f"recording {recording_id!r}. Try a smaller validation_fraction or "
            "use --holdout-recording-id."
        )
    return safe_training_chunk, validation_chunk


def _summarize_binary_targets(y: np.ndarray) -> BinaryLabelStats:
    """Summarize binary labels for logging and imbalance handling."""

    values = np.asarray(y, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"Expected rank-1 binary targets, got shape {values.shape}.")
    unique_targets = set(np.unique(values).tolist())
    if not unique_targets <= {0, 1}:
        raise ValueError(
            f"Expected binary targets containing only 0/1, got {sorted(unique_targets)}."
        )

    total_count = int(values.size)
    positive_count = int(np.sum(values == 1))
    negative_count = int(np.sum(values == 0))
    positive_rate = positive_count / max(total_count, 1)
    negative_to_positive_ratio = (
        negative_count / positive_count if positive_count > 0 else None
    )
    return BinaryLabelStats(
        total_count=total_count,
        negative_count=negative_count,
        positive_count=positive_count,
        positive_rate=positive_rate,
        negative_to_positive_ratio=negative_to_positive_ratio,
    )


def _choose_binary_loss_mitigation(stats: BinaryLabelStats) -> tuple[str, float]:
    """Choose a simple loss strategy based on training-set imbalance."""

    if stats.positive_count == 0:
        raise ValueError("Training split contains no positive samples.")
    if stats.negative_count == 0:
        raise ValueError("Training split contains no negative samples.")
    ratio = stats.negative_to_positive_ratio
    if ratio is None:
        raise ValueError("Training split ratio is undefined because positives are missing.")
    if ratio > 1.5:
        return "bce_with_logits_pos_weight", float(ratio)
    return "standard_bce_with_logits", 1.0


def _plan_binary_loss_mitigation(
    *,
    strategy: str,
    raw_train_stats: BinaryLabelStats,
    curated_train_stats: BinaryLabelStats,
) -> BinaryLossMitigationPlan:
    """Resolve the train-time weighting plan after negative curation."""

    if curated_train_stats.positive_count == 0:
        raise ValueError("Training split contains no positive samples.")
    if curated_train_stats.negative_count == 0:
        raise ValueError("Training split contains no negative samples.")

    majority_downsample_factor = None
    if raw_train_stats.negative_count > 0:
        majority_downsample_factor = (
            raw_train_stats.negative_count / curated_train_stats.negative_count
        )

    if strategy == "positive_pos_weight":
        resolved_strategy, positive_weight = _choose_binary_loss_mitigation(curated_train_stats)
        return BinaryLossMitigationPlan(
            strategy=resolved_strategy,
            positive_weight=float(positive_weight),
            negative_weight=1.0,
            majority_downsample_factor=majority_downsample_factor,
        )
    if strategy == "majority_downsample_upweight":
        return BinaryLossMitigationPlan(
            strategy="majority_downsample_upweight",
            positive_weight=1.0,
            negative_weight=(
                1.0 if majority_downsample_factor is None else float(majority_downsample_factor)
            ),
            majority_downsample_factor=majority_downsample_factor,
        )
    raise ValueError(f"Unsupported imbalance strategy {strategy!r}.")


def _resolve_checkpoint_selection_score(
    *,
    selection_metric: str,
    val_loss: float,
    metrics: dict[str, int | float | None],
    decoded_trigger_metrics: dict[str, int | float] | None = None,
) -> float:
    """Return the scalar score used to rank checkpoints for persistence."""

    if selection_metric == "loss":
        return -float(val_loss)
    if selection_metric == "decoded_trigger_f1":
        return float(decoded_trigger_metrics["f1"]) if decoded_trigger_metrics is not None else float(
            "-inf"
        )
    if selection_metric == "decoded_trigger_precision":
        return (
            float(decoded_trigger_metrics["precision"])
            if decoded_trigger_metrics is not None
            else float("-inf")
        )
    if selection_metric == "decoded_trigger_recall":
        return (
            float(decoded_trigger_metrics["recall"])
            if decoded_trigger_metrics is not None
            else float("-inf")
        )
    if selection_metric not in {"f1", "precision", "recall", "roc_auc"}:
        raise ValueError(f"Unsupported checkpoint_selection_metric {selection_metric!r}.")
    value = metrics.get(selection_metric)
    if value is None:
        return float("-inf")
    return float(value)


def _format_checkpoint_selection_value(*, selection_metric: str, value: float) -> str:
    """Render the active checkpoint-selection score for logging."""

    if selection_metric == "loss":
        return f"val_loss={-value:.6f}"
    if selection_metric.startswith("decoded_trigger_"):
        return f"{selection_metric}={value:.4f}"
    return f"val_{selection_metric}={value:.4f}"


def _resolve_decoded_trigger_config(
    *,
    window_size: int,
    stride: int,
    cooldown_frames: int,
    max_gap_frames: int,
    match_tolerance_frames: int,
) -> dict[str, int]:
    """Resolve validation decoded-trigger settings using predict-time defaults."""

    if cooldown_frames < -1:
        raise ValueError("checkpoint_selection_trigger_cooldown_frames must be -1 or greater.")
    if max_gap_frames < -1:
        raise ValueError("checkpoint_selection_trigger_max_gap_frames must be -1 or greater.")
    if match_tolerance_frames < 0:
        raise ValueError(
            "checkpoint_selection_trigger_match_tolerance_frames must be greater than or equal to zero."
        )
    return {
        "cooldown_frames": max(window_size // 2, 0) if cooldown_frames < 0 else cooldown_frames,
        "max_gap_frames": max(stride, 1) if max_gap_frames < 0 else max_gap_frames,
        "match_tolerance_frames": match_tolerance_frames,
    }


def _prefixed_decoded_trigger_metrics(
    decoded_trigger_metrics: dict[str, int | float],
) -> dict[str, int | float]:
    """Namespace decoded-trigger metrics for training history logs."""

    return {
        f"decoded_trigger_{key}": value for key, value in decoded_trigger_metrics.items()
    }


def _format_binary_label_stats(name: str, stats: BinaryLabelStats) -> str:
    """Format one class-distribution line for stdout logging."""

    ratio = (
        f"{stats.negative_to_positive_ratio:.2f}"
        if stats.negative_to_positive_ratio is not None
        else "undefined"
    )
    return (
        f"{name}: samples={stats.total_count} "
        f"neg={stats.negative_count} "
        f"pos={stats.positive_count} "
        f"pos_rate={stats.positive_rate:.2%} "
        f"neg_to_pos={ratio}"
    )


def _describe_positive_class_meaning(*, target_name: str, horizon_frames: int) -> str:
    """Describe what label 1 means for the active training target."""

    if target_name == "completion_frame_binary":
        return "gesture completion occurs at the last frame of the window"
    if target_name == "completion_within_next_k_frames":
        return f"gesture completion occurs within the next {horizon_frames} frame(s) after the window"
    if target_name == "completion_within_last_k_frames":
        return (
            "gesture completion occurred within the last "
            f"{horizon_frames} frame(s) ending at the window"
        )
    if target_name == "arm_frame_binary":
        return "early-arm timing is active at the last frame of the window"
    if target_name == "arm_within_next_k_frames":
        return f"early-arm timing becomes active within the next {horizon_frames} frame(s) after the window"
    if target_name == "arm_within_last_k_frames":
        return (
            "early-arm timing became active within the last "
            f"{horizon_frames} frame(s) ending at the window"
        )
    raise ValueError(f"Unsupported target_name {target_name!r}.")


def _describe_negative_class_meaning(*, target_name: str, horizon_frames: int) -> str:
    """Describe what label 0 means for the active training target."""

    if target_name == "completion_frame_binary":
        return "no gesture completion occurs at the last frame of the window"
    if target_name == "completion_within_next_k_frames":
        return (
            "no gesture completion occurs within the next "
            f"{horizon_frames} frame(s) after the window"
        )
    if target_name == "completion_within_last_k_frames":
        return (
            "no gesture completion occurred within the last "
            f"{horizon_frames} frame(s) ending at the window"
        )
    if target_name == "arm_frame_binary":
        return "early-arm timing is not active at the last frame of the window"
    if target_name == "arm_within_next_k_frames":
        return (
            "early-arm timing does not become active within the next "
            f"{horizon_frames} frame(s) after the window"
        )
    if target_name == "arm_within_last_k_frames":
        return (
            "early-arm timing did not become active within the last "
            f"{horizon_frames} frame(s) ending at the window"
        )
    raise ValueError(f"Unsupported target_name {target_name!r}.")


def _curate_training_negatives(
    dataset: CombinedDataset,
    *,
    train_indices: np.ndarray,
    seed: int,
    max_negative_positive_ratio: float,
    hard_negative_margin_frames: int,
) -> NegativeCurationResult:
    """Keep all positives, preserve hard negatives, and downsample easy negatives."""

    ordered_indices = np.asarray(train_indices, dtype=np.int64)
    labels = dataset.y[ordered_indices].astype(np.int64, copy=False)
    positive_indices = ordered_indices[labels == 1]
    negative_indices = ordered_indices[labels == 0]
    if positive_indices.size == 0 or negative_indices.size == 0:
        return NegativeCurationResult(
            kept_indices=np.sort(ordered_indices),
            kept_positive_count=int(positive_indices.size),
            kept_hard_negative_count=0,
            kept_easy_negative_count=int(negative_indices.size),
            dropped_easy_negative_count=0,
            max_negative_positive_ratio=None if max_negative_positive_ratio <= 0.0 else max_negative_positive_ratio,
            hard_negative_margin_frames=hard_negative_margin_frames,
        )

    hard_negative_mask = np.zeros(negative_indices.shape[0], dtype=bool)
    for recording_id in np.unique(dataset.recording_ids[ordered_indices]).tolist():
        recording_mask = dataset.recording_ids[ordered_indices] == recording_id
        recording_train_indices = ordered_indices[recording_mask]
        recording_labels = dataset.y[recording_train_indices].astype(np.int64, copy=False)
        recording_positive_frames = np.sort(
            dataset.window_end_frame_indices[recording_train_indices[recording_labels == 1]].astype(
                np.int64,
                copy=False,
            )
        )
        if recording_positive_frames.size == 0:
            continue
        negative_positions = np.flatnonzero(
            (dataset.recording_ids[negative_indices] == recording_id)
        )
        if negative_positions.size == 0:
            continue
        negative_frames = dataset.window_end_frame_indices[
            negative_indices[negative_positions]
        ].astype(np.int64, copy=False)
        insertion_positions = np.searchsorted(recording_positive_frames, negative_frames)
        nearest_distances = np.full(negative_frames.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        has_left = insertion_positions > 0
        if np.any(has_left):
            left_distances = negative_frames[has_left] - recording_positive_frames[insertion_positions[has_left] - 1]
            nearest_distances[has_left] = np.minimum(nearest_distances[has_left], left_distances)
        has_right = insertion_positions < recording_positive_frames.shape[0]
        if np.any(has_right):
            right_distances = recording_positive_frames[insertion_positions[has_right]] - negative_frames[has_right]
            nearest_distances[has_right] = np.minimum(nearest_distances[has_right], right_distances)
        hard_negative_mask[negative_positions] = nearest_distances <= hard_negative_margin_frames

    hard_negative_indices = negative_indices[hard_negative_mask]
    easy_negative_indices = negative_indices[~hard_negative_mask]
    if max_negative_positive_ratio <= 0.0:
        kept_easy_negative_indices = easy_negative_indices
    else:
        max_negative_count = int(ceil(positive_indices.size * max_negative_positive_ratio))
        easy_negative_budget = max(0, max_negative_count - hard_negative_indices.size)
        if easy_negative_indices.size <= easy_negative_budget:
            kept_easy_negative_indices = easy_negative_indices
        else:
            rng = np.random.default_rng(seed)
            kept_easy_negative_indices = np.sort(
                rng.choice(
                    easy_negative_indices,
                    size=easy_negative_budget,
                    replace=False,
                ).astype(np.int64, copy=False)
            )

    kept_indices = np.sort(
        np.concatenate(
            [
                positive_indices,
                hard_negative_indices,
                kept_easy_negative_indices,
            ],
            axis=0,
        )
    ).astype(np.int64, copy=False)
    return NegativeCurationResult(
        kept_indices=kept_indices,
        kept_positive_count=int(positive_indices.size),
        kept_hard_negative_count=int(hard_negative_indices.size),
        kept_easy_negative_count=int(kept_easy_negative_indices.size),
        dropped_easy_negative_count=int(easy_negative_indices.size - kept_easy_negative_indices.size),
        max_negative_positive_ratio=(
            None if max_negative_positive_ratio <= 0.0 else float(max_negative_positive_ratio)
        ),
        hard_negative_margin_frames=hard_negative_margin_frames,
    )


def train_model(
    dataset: CombinedDataset,
    *,
    split: DatasetSplit,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_channels: int,
    seed: int,
    device: str,
    num_workers: int,
    max_negative_positive_ratio: float,
    hard_negative_margin_frames: int,
    imbalance_strategy: str,
    checkpoint_selection_metric: str,
    checkpoint_selection_trigger_cooldown_frames: int,
    checkpoint_selection_trigger_max_gap_frames: int,
    checkpoint_selection_trigger_match_tolerance_frames: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train the baseline CNN and save the best checkpoint."""

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be greater than or equal to zero.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0).")
    if hidden_channels <= 0:
        raise ValueError("hidden_channels must be greater than zero.")
    if hard_negative_margin_frames < -1:
        raise ValueError("hard_negative_margin_frames must be -1 or greater.")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    _set_deterministic_seed(seed, torch)

    resolved_device = _resolve_device(device, torch)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    config_dir = output_dir / "config"
    for directory in (checkpoints_dir, reports_dir, plots_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "best_model.pt"
    config_path = config_dir / "config.json"
    metadata_path = reports_dir / "training_metadata.json"
    epoch_metrics_path = reports_dir / "epoch_metrics.json"
    evaluation_report_path = reports_dir / "evaluation_report.json"
    best_confusion_matrix_path = reports_dir / "best_confusion_matrix.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    combined_stats = _summarize_binary_targets(dataset.y)
    raw_train_stats = _summarize_binary_targets(dataset.y[split.train_indices])
    validation_stats = _summarize_binary_targets(dataset.y[split.validation_indices])
    resolved_hard_negative_margin_frames = (
        dataset.window_size if hard_negative_margin_frames < 0 else hard_negative_margin_frames
    )
    curation = _curate_training_negatives(
        dataset,
        train_indices=split.train_indices,
        seed=seed,
        max_negative_positive_ratio=max_negative_positive_ratio,
        hard_negative_margin_frames=resolved_hard_negative_margin_frames,
    )
    curated_train_indices = curation.kept_indices
    train_stats = _summarize_binary_targets(dataset.y[curated_train_indices])

    positive_class_meaning = _describe_positive_class_meaning(
        target_name=dataset.target_name,
        horizon_frames=dataset.horizon_frames,
    )
    negative_class_meaning = _describe_negative_class_meaning(
        target_name=dataset.target_name,
        horizon_frames=dataset.horizon_frames,
    )

    train_X = torch.from_numpy(dataset.X[curated_train_indices]).float()
    train_y = torch.from_numpy(dataset.y[curated_train_indices]).float()
    val_X = torch.from_numpy(dataset.X[split.validation_indices]).float()
    val_y = torch.from_numpy(dataset.y[split.validation_indices]).float()
    validation_recording_ids = dataset.recording_ids[split.validation_indices]
    validation_window_end_frame_indices = dataset.window_end_frame_indices[
        split.validation_indices
    ]
    validation_window_end_timestamps_seconds = dataset.window_end_timestamps_seconds[
        split.validation_indices
    ]

    if train_stats.positive_count == 0:
        raise ValueError("Training split contains no positive samples.")
    if validation_stats.positive_count == 0:
        raise ValueError(
            "Validation split contains no positive samples. Adjust the split settings."
        )

    print("Class distribution:")
    print(f"  {_format_binary_label_stats('combined', combined_stats)}")
    print(f"  {_format_binary_label_stats('train_raw', raw_train_stats)}")
    print(f"  {_format_binary_label_stats('train_curated', train_stats)}")
    print(f"  {_format_binary_label_stats('validation', validation_stats)}")
    if curation.max_negative_positive_ratio is None:
        print(
            "Training sample curation: disabled; keeping all train windows "
            f"(hard_negative_margin_frames={curation.hard_negative_margin_frames})."
        )
    else:
        print(
            "Training sample curation: keep all positives and hard negatives, "
            f"downsample easy negatives to max neg:pos={curation.max_negative_positive_ratio:.2f} "
            f"within hard_negative_margin_frames={curation.hard_negative_margin_frames}. "
            f"Kept hard_neg={curation.kept_hard_negative_count}, "
            f"kept easy_neg={curation.kept_easy_negative_count}, "
            f"dropped easy_neg={curation.dropped_easy_negative_count}."
        )
    print(f"Target meaning: positive={positive_class_meaning}")

    decoded_trigger_config = _resolve_decoded_trigger_config(
        window_size=dataset.window_size,
        stride=dataset.stride,
        cooldown_frames=checkpoint_selection_trigger_cooldown_frames,
        max_gap_frames=checkpoint_selection_trigger_max_gap_frames,
        match_tolerance_frames=checkpoint_selection_trigger_match_tolerance_frames,
    )
    print(
        "Validation decoded-trigger config: "
        f"cooldown_frames={decoded_trigger_config['cooldown_frames']} "
        f"max_gap_frames={decoded_trigger_config['max_gap_frames']} "
        f"match_tolerance_frames={decoded_trigger_config['match_tolerance_frames']}"
    )

    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    pin_memory = resolved_device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model_spec = VisionBeatCnnSpec(
        feature_count=dataset.feature_count,
        window_size=dataset.window_size,
        hidden_channels=hidden_channels,
        dropout=dropout,
        schema_version=dataset.schema_version,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        horizon_frames=dataset.horizon_frames,
    )
    model = build_completion_cnn(nn, model_spec)
    model.to(resolved_device)

    loss_plan = _plan_binary_loss_mitigation(
        strategy=imbalance_strategy,
        raw_train_stats=raw_train_stats,
        curated_train_stats=train_stats,
    )
    train_criterion = nn.BCEWithLogitsLoss(reduction="none")
    val_criterion = nn.BCEWithLogitsLoss(reduction="none")
    if loss_plan.strategy == "bce_with_logits_pos_weight":
        print(
            "Imbalance mitigation: BCEWithLogitsLoss with "
            f"positive_weight={loss_plan.positive_weight:.4f} "
            "(training negative/positive ratio)."
        )
    elif loss_plan.strategy == "majority_downsample_upweight":
        print(
            "Imbalance mitigation: majority downsample plus majority upweight "
            f"(negative_weight={loss_plan.negative_weight:.4f}, "
            f"effective_downsample_factor={loss_plan.negative_weight:.4f})."
        )
    else:
        print(
            "Imbalance mitigation: standard BCEWithLogitsLoss "
            "(training split is close to balanced)."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: list[dict[str, int | float | None]] = []
    best_val_loss = float("inf")
    best_selection_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, int | float | None] | None = None
    best_decoded_trigger_metrics: dict[str, int | float] | None = None
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=resolved_device,
            torch=torch,
            train=True,
            positive_weight=loss_plan.positive_weight,
            negative_weight=loss_plan.negative_weight,
        )
        val_loss, metrics, y_score, y_true = _evaluate(
            model=model,
            loader=val_loader,
            criterion=val_criterion,
            device=resolved_device,
            torch=torch,
            positive_weight=1.0,
            negative_weight=1.0,
        )
        decoded_triggers = decode_trigger_events(
            recording_ids=validation_recording_ids,
            window_end_frame_indices=validation_window_end_frame_indices,
            window_end_timestamps_seconds=validation_window_end_timestamps_seconds,
            probabilities=y_score,
            threshold=0.5,
            cooldown_frames=decoded_trigger_config["cooldown_frames"],
            max_gap_frames=decoded_trigger_config["max_gap_frames"],
        )
        decoded_trigger_metrics = evaluate_decoded_triggers(
            decoded_triggers=decoded_triggers,
            recording_ids=validation_recording_ids,
            window_end_frame_indices=validation_window_end_frame_indices,
            labels=y_true,
            match_tolerance_frames=decoded_trigger_config["match_tolerance_frames"],
            max_gap_frames=decoded_trigger_config["max_gap_frames"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                **metrics,
                **_prefixed_decoded_trigger_metrics(decoded_trigger_metrics),
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_accuracy={metrics['accuracy']:.4f} "
            f"val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} "
            f"val_f1={metrics['f1']:.4f} "
            f"val_roc_auc={_format_optional_metric(metrics['roc_auc'])}"
        )
        print(
            "  "
            f"val_confusion=TN:{metrics['true_negative']} "
            f"FP:{metrics['false_positive']} "
            f"FN:{metrics['false_negative']} "
            f"TP:{metrics['true_positive']} "
            f"detected_positives={metrics['detected_positive_count']} "
            f"false_positives={metrics['false_positive']} "
            f"false_negatives={metrics['false_negative']}"
        )
        print(
            "  "
            f"decoded_trigger_precision={decoded_trigger_metrics['precision']:.4f} "
            f"decoded_trigger_recall={decoded_trigger_metrics['recall']:.4f} "
            f"decoded_trigger_f1={decoded_trigger_metrics['f1']:.4f} "
            f"decoded_trigger_count={decoded_trigger_metrics['decoded_trigger_count']} "
            f"decoded_false_triggers={decoded_trigger_metrics['false_positive_trigger_count']}"
        )
        selection_score = _resolve_checkpoint_selection_score(
            selection_metric=checkpoint_selection_metric,
            val_loss=val_loss,
            metrics=metrics,
            decoded_trigger_metrics=decoded_trigger_metrics,
        )

        if selection_score > best_selection_score or (
            selection_score == best_selection_score and val_loss < best_val_loss
        ):
            best_selection_score = selection_score
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_decoded_trigger_metrics = dict(decoded_trigger_metrics)
            checkpoint_payload = build_checkpoint_payload(
                spec=model_spec,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                extra={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": metrics,
                    "val_decoded_trigger_metrics": decoded_trigger_metrics,
                    "checkpoint_selection_metric": checkpoint_selection_metric,
                    "checkpoint_selection_score": selection_score,
                    "decoded_trigger_config": decoded_trigger_config,
                    "class_distribution": {
                        "combined": combined_stats.as_dict(),
                        "train_raw": raw_train_stats.as_dict(),
                        "train_curated": train_stats.as_dict(),
                        "validation": validation_stats.as_dict(),
                    },
                    "training_sample_curation": curation.as_dict(),
                    "imbalance_strategy": loss_plan.strategy,
                    "loss_weighting": loss_plan.as_dict(),
                    "positive_class_meaning": positive_class_meaning,
                    "negative_class_meaning": negative_class_meaning,
                },
            )
            torch.save(checkpoint_payload, checkpoint_path)

    if best_metrics is None or best_decoded_trigger_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics.")

    print(
        "Best checkpoint metrics: "
        f"epoch={best_epoch:02d} "
        f"{_format_checkpoint_selection_value(selection_metric=checkpoint_selection_metric, value=best_selection_score)} "
        f"selected_val_loss={best_val_loss:.6f} "
        f"val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_precision={best_metrics['precision']:.4f} "
        f"val_recall={best_metrics['recall']:.4f} "
        f"val_f1={best_metrics['f1']:.4f} "
        f"val_roc_auc={_format_optional_metric(best_metrics['roc_auc'])}"
    )
    print(
        "  "
        f"best_val_confusion=TN:{best_metrics['true_negative']} "
        f"FP:{best_metrics['false_positive']} "
        f"FN:{best_metrics['false_negative']} "
        f"TP:{best_metrics['true_positive']} "
        f"detected_positives={best_metrics['detected_positive_count']} "
        f"false_positives={best_metrics['false_positive']} "
        f"false_negatives={best_metrics['false_negative']}"
    )
    print(
        "  "
        f"best_decoded_trigger_precision={best_decoded_trigger_metrics['precision']:.4f} "
        f"best_decoded_trigger_recall={best_decoded_trigger_metrics['recall']:.4f} "
        f"best_decoded_trigger_f1={best_decoded_trigger_metrics['f1']:.4f} "
        f"best_decoded_trigger_count={best_decoded_trigger_metrics['decoded_trigger_count']} "
        f"best_decoded_false_triggers={best_decoded_trigger_metrics['false_positive_trigger_count']}"
    )

    evaluation_report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "checkpoint_selection_score": (
            -best_selection_score if checkpoint_selection_metric == "loss" else best_selection_score
        ),
        "decoded_trigger_config": decoded_trigger_config,
        "best_val_metrics": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
            "roc_auc": best_metrics["roc_auc"],
            "confusion_matrix": {
                "true_negative": best_metrics["true_negative"],
                "false_positive": best_metrics["false_positive"],
                "false_negative": best_metrics["false_negative"],
                "true_positive": best_metrics["true_positive"],
            },
            "positive_event_behavior": {
                "detected_positive_count": best_metrics["detected_positive_count"],
                "missed_positive_count": best_metrics["missed_positive_count"],
                "false_positive_count": best_metrics["false_positive"],
                "positive_detection_rate": best_metrics["positive_detection_rate"],
                "predicted_positive_count": best_metrics["predicted_positive_count"],
                "predicted_positive_rate": best_metrics["predicted_positive_rate"],
            },
        },
        "best_decoded_trigger_metrics": dict(best_decoded_trigger_metrics),
        "validation_recording_id": split.validation_recording_id,
        "split_policy": split.policy,
        "target_name": dataset.target_name,
        "horizon_frames": dataset.horizon_frames,
        "threshold": 0.5,
        "report_meaning": (
            "All metrics are window-level. "
            f"A positive window means {positive_class_meaning}."
        ),
    }
    evaluation_report_path.write_text(
        json.dumps(evaluation_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best_confusion_matrix = {
        "true_negative": best_metrics["true_negative"],
        "false_positive": best_metrics["false_positive"],
        "false_negative": best_metrics["false_negative"],
        "true_positive": best_metrics["true_positive"],
        "threshold": 0.5,
        "meaning": f"Window-level confusion matrix. Positive means {positive_class_meaning}.",
    }
    best_confusion_matrix_path.write_text(
        json.dumps(best_confusion_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    epoch_metrics_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_outputs = _maybe_save_training_plots(history=history, plots_dir=plots_dir)
    if plot_outputs["matplotlib_available"]:
        print(f"Saved plots: {', '.join(plot_outputs['paths'])}")
    else:
        print("Plots skipped: matplotlib is not available.")

    metadata = {
        "recordings": [str(index) for index in np.unique(dataset.recording_ids).tolist()],
        "run_dir": str(output_dir),
        "input_shape": list(dataset.input_shape),
        "num_features": dataset.feature_count,
        "feature_names": list(dataset.feature_names),
        "schema_version": dataset.schema_version,
        "target_name": dataset.target_name,
        "horizon_frames": dataset.horizon_frames,
        "train_samples": int(curated_train_indices.size),
        "train_samples_raw": int(split.train_indices.size),
        "train_samples_curated": int(curated_train_indices.size),
        "validation_samples": int(split.validation_indices.size),
        "purge_gap_frames": split.purge_gap_frames,
        "split_policy": split.policy,
        "validation_recording_id": split.validation_recording_id,
        "class_distribution": {
            "combined": combined_stats.as_dict(),
            "train_raw": raw_train_stats.as_dict(),
            "train_curated": train_stats.as_dict(),
            "validation": validation_stats.as_dict(),
        },
        "training_sample_curation": curation.as_dict(),
        "checkpoint_model_metadata": model_spec.to_checkpoint_metadata(),
        "imbalance_strategy": loss_plan.strategy,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "checkpoint_selection_score": (
            -best_selection_score if checkpoint_selection_metric == "loss" else best_selection_score
        ),
        "decoded_trigger_config": decoded_trigger_config,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "hidden_channels": hidden_channels,
        "loss_weighting": loss_plan.as_dict(),
        "pos_weight": (
            loss_plan.positive_weight if loss_plan.strategy == "bce_with_logits_pos_weight" else None
        ),
        "history": history,
        "config_path": str(config_path),
        "best_checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "best_decoded_trigger_metrics": best_decoded_trigger_metrics,
        "selected_val_loss": best_val_loss,
        "epoch_metrics_path": str(epoch_metrics_path),
        "evaluation_report_path": str(evaluation_report_path),
        "best_confusion_matrix_path": str(best_confusion_matrix_path),
        "plot_paths": plot_outputs["paths"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_dir": str(output_dir),
        "config_path": str(config_path),
        "best_checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "checkpoint_selection_score": (
            -best_selection_score if checkpoint_selection_metric == "loss" else best_selection_score
        ),
        "metadata_path": str(metadata_path),
        "epoch_metrics_path": str(epoch_metrics_path),
        "evaluation_report_path": str(evaluation_report_path),
        "best_confusion_matrix_path": str(best_confusion_matrix_path),
        "plot_paths": plot_outputs["paths"],
    }


def _validate_archive_shapes(
    *,
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
    window_end_timestamps_seconds: np.ndarray,
    target_gesture_labels: np.ndarray,
    feature_names: tuple[str, ...],
    schema_version: str,
    feature_count: int,
    target_name: str,
    window_size: int,
    stride: int,
    horizon_frames: int,
) -> None:
    if X.ndim != 3:
        raise ValueError(f"Archive {path} has invalid X shape {X.shape}; expected rank 3.")
    if y.ndim != 1:
        raise ValueError(f"Archive {path} has invalid y shape {y.shape}; expected rank 1.")
    sample_count = X.shape[0]
    for name, values in (
        ("y", y),
        ("recording_ids", recording_ids),
        ("window_end_frame_indices", window_end_frame_indices),
        ("window_end_timestamps_seconds", window_end_timestamps_seconds),
        ("target_gesture_labels", target_gesture_labels),
    ):
        if values.shape[0] != sample_count:
            raise ValueError(
                f"Archive {path} has inconsistent sample count in {name}: "
                f"expected {sample_count}, got {values.shape[0]}."
            )
    if X.shape[1] != window_size:
        raise ValueError(
            f"Archive {path} window dimension mismatch. Expected {window_size}, got {X.shape[1]}."
        )
    if X.shape[2] != feature_count:
        raise ValueError(
            "Archive "
            f"{path} feature dimension mismatch. Expected {feature_count}, got {X.shape[2]}."
        )
    if feature_count != len(feature_names):
        raise ValueError(
            f"Archive {path} feature_count does not match feature_names length."
        )
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            f"Archive {path} feature_names do not match the canonical schema."
        )
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Archive {path} schema_version mismatch. Expected {FEATURE_SCHEMA_VERSION}, "
            f"got {schema_version}."
        )
    if target_name not in _SUPPORTED_TARGET_NAMES:
        raise ValueError(
            f"Archive {path} has unsupported target_name {target_name!r}. "
            f"Expected one of {_SUPPORTED_TARGET_NAMES}."
        )
    if target_name in {
        "completion_within_next_k_frames",
        "completion_within_last_k_frames",
    } and horizon_frames <= 0:
        raise ValueError(
            f"Archive {path} has invalid horizon_frames {horizon_frames} for tolerant completion targets."
        )
    if stride <= 0:
        raise ValueError(f"Archive {path} has invalid stride {stride}.")
    if not np.isfinite(X).all():
        raise ValueError(f"Archive {path} contains non-finite feature values.")
    if not np.isfinite(y).all():
        raise ValueError(f"Archive {path} contains non-finite targets.")
    unique_targets = set(np.unique(y).tolist())
    if not unique_targets <= {0, 1}:
        raise ValueError(
            f"Archive {path} has unsupported targets {sorted(unique_targets)}; expected binary 0/1."
        )

def _set_deterministic_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    torch: Any,
    train: bool,
    positive_weight: float,
    negative_weight: float,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = _compute_weighted_binary_loss(
                logits=logits,
                targets=targets,
                criterion=criterion,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
                torch=torch,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
    return total_loss / max(total_examples, 1)


def _evaluate(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    positive_weight: float,
    negative_weight: float,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = _compute_weighted_binary_loss(
                logits=logits,
                targets=targets,
                criterion=criterion,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
                torch=torch,
            )
            probabilities = torch.sigmoid(logits)
            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0).astype(np.int64, copy=False)
    y_score = np.concatenate(all_probabilities, axis=0)
    metrics = _binary_classification_metrics(y_true, y_score)
    return total_loss / max(total_examples, 1), metrics, y_score, y_true


def _compute_weighted_binary_loss(
    *,
    logits: Any,
    targets: Any,
    criterion: Any,
    positive_weight: float,
    negative_weight: float,
    torch: Any,
):
    """Apply optional per-class weights on top of unreduced BCE loss."""

    losses = criterion(logits, targets)
    if positive_weight == 1.0 and negative_weight == 1.0:
        return losses.mean()
    positive_weights = torch.full_like(targets, float(positive_weight))
    negative_weights = torch.full_like(targets, float(negative_weight))
    class_weights = torch.where(targets > 0.5, positive_weights, negative_weights)
    return (losses * class_weights).mean()


def _maybe_save_training_plots(
    *,
    history: list[dict[str, int | float | None]],
    plots_dir: Path,
) -> dict[str, Any]:
    """Save simple loss and metric plots when matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"matplotlib_available": False, "paths": []}

    epochs = [int(entry["epoch"]) for entry in history]
    train_loss = [float(entry["train_loss"]) for entry in history]
    val_loss = [float(entry["val_loss"]) for entry in history]
    loss_curve_path = plots_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("VisionBeat CNN Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_curve_path, dpi=150)
    plt.close(fig)

    metric_curve_path = plots_dir / "validation_metrics_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = [
            float(value) if value is not None else float("nan")
            for value in (entry.get(metric_name) for entry in history)
        ]
        ax.plot(epochs, values, label=metric_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("VisionBeat CNN Validation Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(metric_curve_path, dpi=150)
    plt.close(fig)

    return {
        "matplotlib_available": True,
        "paths": [str(loss_curve_path), str(metric_curve_path)],
    }


if __name__ == "__main__":
    main()

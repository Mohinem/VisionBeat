"""Build sliding-window 1D-CNN training samples from canonical feature tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import numpy as np

from visionbeat.config import ConfigError, load_config
from visionbeat.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    CanonicalFeatureSchema,
    FeatureSchemaError,
    assert_feature_schemas_match,
    get_canonical_feature_schema,
)
from visionbeat.logging_config import configure_logging

TrainingTarget = Literal[
    "completion_frame_binary",
    "completion_within_next_k_frames",
    "completion_within_last_k_frames",
    "arm_frame_binary",
    "arm_within_next_k_frames",
    "arm_within_last_k_frames",
]

DEFAULT_WINDOW_SIZE: Final[int] = 32
DEFAULT_STRIDE: Final[int] = 1
DEFAULT_TARGET: Final[TrainingTarget] = "completion_frame_binary"
DEFAULT_HORIZON_FRAMES: Final[int] = 4
SUPPORTED_TRAINING_TARGETS: Final[tuple[TrainingTarget, ...]] = (
    "completion_frame_binary",
    "completion_within_next_k_frames",
    "completion_within_last_k_frames",
    "arm_frame_binary",
    "arm_within_next_k_frames",
    "arm_within_last_k_frames",
)
_REQUIRED_FRAME_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "recording_id",
    "frame_index",
    "timestamp_seconds",
)
_GESTURE_LABEL_COLUMNS: Final[tuple[str, ...]] = ("gesture_label", "gesture")
_IS_COMPLETION_COLUMNS: Final[tuple[str, ...]] = ("is_completion_frame", "is_completion")
_IS_ARM_COLUMNS: Final[tuple[str, ...]] = ("is_arm_frame",)


@dataclass(frozen=True, slots=True)
class FrameFeatureRow:
    """One aligned per-frame feature row loaded from the offline CSV."""

    recording_id: str
    frame_index: int
    timestamp_seconds: float
    gesture_label: str
    is_completion_frame: bool
    is_arm_frame: bool
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainingSampleDataset:
    """Training-ready sliding-window tensors and associated metadata."""

    X: np.ndarray
    y: np.ndarray
    recording_ids: np.ndarray
    window_end_frame_indices: np.ndarray
    window_end_timestamps_seconds: np.ndarray
    target_gesture_labels: np.ndarray
    feature_schema: CanonicalFeatureSchema
    target_name: str
    window_size: int
    stride: int
    horizon_frames: int

    def save(self, output_path: str | Path) -> Path:
        """Persist the dataset as a compressed NumPy archive."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            X=self.X,
            y=self.y,
            recording_ids=self.recording_ids,
            window_end_frame_indices=self.window_end_frame_indices,
            window_end_timestamps_seconds=self.window_end_timestamps_seconds,
            target_gesture_labels=self.target_gesture_labels,
            feature_names=np.asarray(self.feature_schema.feature_names, dtype="<U64"),
            schema_version=np.asarray(self.feature_schema.version, dtype="<U64"),
            feature_count=np.asarray(self.feature_schema.feature_count, dtype=np.int64),
            target_name=np.asarray(self.target_name, dtype="<U64"),
            window_size=np.asarray(self.window_size, dtype=np.int64),
            stride=np.asarray(self.stride, dtype=np.int64),
            horizon_frames=np.asarray(self.horizon_frames, dtype=np.int64),
        )
        return destination


@dataclass(frozen=True, slots=True)
class TrainingSampleGenerationResult:
    """Summary of one completed training-sample generation job."""

    frame_table_path: Path
    output_path: Path
    target_name: str
    window_size: int
    stride: int
    horizon_frames: int
    sample_count: int
    feature_schema: CanonicalFeatureSchema
    X_shape: tuple[int, int, int]
    y_shape: tuple[int, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training-sample generation."""
    parser = argparse.ArgumentParser(
        description="Build sliding-window 1D-CNN samples from VisionBeat feature CSV files."
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Path to a per-frame canonical feature CSV generated offline.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output `.npz` path. Defaults to '<frames-stem>.samples.npz'.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Number of frames per sample window. Default: {DEFAULT_WINDOW_SIZE}.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Sliding-window stride in frames. Default: {DEFAULT_STRIDE}.",
    )
    parser.add_argument(
        "--target",
        choices=SUPPORTED_TRAINING_TARGETS,
        default=DEFAULT_TARGET,
        help="Prediction target to generate for each sample window.",
    )
    parser.add_argument(
        "--horizon-frames",
        type=int,
        default=DEFAULT_HORIZON_FRAMES,
        help=(
            "Tolerance in frames for `completion_within_next_k_frames` and "
            "`completion_within_last_k_frames`. "
            f"Default: {DEFAULT_HORIZON_FRAMES}."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a VisionBeat YAML or TOML configuration file.",
    )
    return parser.parse_args(argv)


def load_frame_feature_rows(
    frame_table_path: str | Path,
) -> tuple[CanonicalFeatureSchema, tuple[FrameFeatureRow, ...]]:
    """Load canonical per-frame rows and verify the feature schema and ordering."""

    csv_path = Path(frame_table_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Frame feature table does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or ())
        _validate_frame_table_columns(csv_path, fieldnames)
        gesture_column = _resolve_required_column(
            fieldnames,
            candidates=_GESTURE_LABEL_COLUMNS,
            csv_path=csv_path,
            column_role="gesture label",
        )
        is_completion_column = _resolve_required_column(
            fieldnames,
            candidates=_IS_COMPLETION_COLUMNS,
            csv_path=csv_path,
            column_role="completion flag",
        )
        is_arm_column = _resolve_optional_column(
            fieldnames,
            candidates=_IS_ARM_COLUMNS,
        )
        feature_columns = tuple(name for name in fieldnames if name in FEATURE_NAMES)
        if feature_columns != FEATURE_NAMES:
            raise FeatureSchemaError(
                "Frame feature table ordering does not match the canonical schema. "
                f"Expected {FEATURE_NAMES}, got {feature_columns}."
            )
        frame_rows = tuple(
            FrameFeatureRow(
                recording_id=row["recording_id"],
                frame_index=int(row["frame_index"]),
                timestamp_seconds=float(row["timestamp_seconds"]),
                gesture_label=row[gesture_column].strip(),
                is_completion_frame=_parse_bool(row[is_completion_column]),
                is_arm_frame=(
                    _parse_bool(row[is_arm_column]) if is_arm_column is not None else False
                ),
                vector=tuple(float(row[name]) for name in FEATURE_NAMES),
            )
            for row in reader
        )
    schema = _load_frame_table_schema(csv_path)
    assert_feature_schemas_match(
        get_canonical_feature_schema(),
        schema,
        context_expected="canonical feature schema",
        context_actual="frame table schema",
    )
    _validate_frame_rows(frame_rows, csv_path)
    return schema, frame_rows


def build_training_samples(
    frame_rows: tuple[FrameFeatureRow, ...],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    target: TrainingTarget = DEFAULT_TARGET,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
    feature_schema: CanonicalFeatureSchema | None = None,
) -> TrainingSampleDataset:
    """Build sliding-window tensors from aligned per-frame feature rows."""

    _validate_window_config(
        window_size=window_size,
        stride=stride,
        target=target,
        horizon_frames=horizon_frames,
    )
    active_schema = feature_schema or get_canonical_feature_schema()
    assert_feature_schemas_match(
        get_canonical_feature_schema(),
        active_schema,
        context_expected="canonical feature schema",
        context_actual="training sample schema",
    )

    rows_by_recording: dict[str, list[FrameFeatureRow]] = defaultdict(list)
    for row in frame_rows:
        rows_by_recording[row.recording_id].append(row)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    recording_ids: list[str] = []
    window_end_frame_indices: list[int] = []
    window_end_timestamps_seconds: list[float] = []
    target_gesture_labels: list[str] = []

    for recording_id in sorted(rows_by_recording):
        recording_rows = sorted(rows_by_recording[recording_id], key=lambda row: row.frame_index)
        for start_index in range(0, len(recording_rows) - window_size + 1, stride):
            end_index = start_index + window_size
            window_rows = recording_rows[start_index:end_index]
            target_value, target_label = _build_target(
                recording_rows,
                window_end_position=end_index - 1,
                target=target,
                horizon_frames=horizon_frames,
            )
            X_rows.append(
                np.asarray([window_row.vector for window_row in window_rows], dtype=np.float32)
            )
            y_rows.append(target_value)
            recording_ids.append(recording_id)
            window_end_frame_indices.append(window_rows[-1].frame_index)
            window_end_timestamps_seconds.append(window_rows[-1].timestamp_seconds)
            target_gesture_labels.append(target_label)

    if X_rows:
        X = np.stack(X_rows).astype(np.float32, copy=False)
    else:
        X = np.zeros((0, window_size, active_schema.feature_count), dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    return TrainingSampleDataset(
        X=X,
        y=y,
        recording_ids=np.asarray(recording_ids, dtype="<U128"),
        window_end_frame_indices=np.asarray(window_end_frame_indices, dtype=np.int64),
        window_end_timestamps_seconds=np.asarray(window_end_timestamps_seconds, dtype=np.float32),
        target_gesture_labels=np.asarray(target_gesture_labels, dtype="<U128"),
        feature_schema=active_schema,
        target_name=target,
        window_size=window_size,
        stride=stride,
        horizon_frames=horizon_frames,
    )


def generate_training_samples(
    frame_table_path: str | Path,
    *,
    output_path: str | Path | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    target: TrainingTarget = DEFAULT_TARGET,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
) -> TrainingSampleGenerationResult:
    """Load a per-frame table, build training samples, and persist them as `.npz`."""

    schema, frame_rows = load_frame_feature_rows(frame_table_path)
    dataset = build_training_samples(
        frame_rows,
        window_size=window_size,
        stride=stride,
        target=target,
        horizon_frames=horizon_frames,
        feature_schema=schema,
    )
    source_path = Path(frame_table_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else _resolve_output_path(source_path)
    )
    saved_path = dataset.save(destination)
    return TrainingSampleGenerationResult(
        frame_table_path=source_path,
        output_path=saved_path,
        target_name=target,
        window_size=window_size,
        stride=stride,
        horizon_frames=horizon_frames,
        sample_count=int(dataset.X.shape[0]),
        feature_schema=schema,
        X_shape=tuple(int(dimension) for dimension in dataset.X.shape),
        y_shape=tuple(int(dimension) for dimension in dataset.y.shape),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for sliding-window training-sample generation."""

    args = parse_args(argv)
    try:
        app_config = load_config(Path(args.config))
        configure_logging(
            app_config.logging.level,
            log_format=app_config.logging.format,
            structured=app_config.logging.structured,
        )
        result = generate_training_samples(
            args.frames,
            output_path=args.out,
            window_size=args.window_size,
            stride=args.stride,
            target=args.target,
            horizon_frames=args.horizon_frames,
        )
    except (ConfigError, FeatureSchemaError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Generated {result.sample_count} samples from {result.frame_table_path} "
        f"to {result.output_path} with X shape {result.X_shape} and y shape {result.y_shape}"
    )


def _resolve_output_path(frame_table_path: Path) -> Path:
    return frame_table_path.with_name(f"{frame_table_path.stem}.samples.npz")


def _load_frame_table_schema(csv_path: Path) -> CanonicalFeatureSchema:
    schema_path = csv_path.with_name(f"{csv_path.name}.schema.json")
    if not schema_path.exists():
        return get_canonical_feature_schema()
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    schema = CanonicalFeatureSchema(
        version=str(payload["schema_version"]),
        feature_names=tuple(payload["feature_names"]),
        feature_count=int(payload["feature_count"]),
    )
    return schema


def _validate_frame_table_columns(csv_path: Path, fieldnames: tuple[str, ...]) -> None:
    if not fieldnames:
        raise ValueError(f"Frame feature table has no header row: {csv_path}")
    missing_columns = [name for name in _REQUIRED_FRAME_TABLE_COLUMNS if name not in fieldnames]
    if missing_columns:
        raise ValueError(
            "Frame feature table is missing required columns: "
            f"{', '.join(missing_columns)}."
        )
    if not any(name in fieldnames for name in _GESTURE_LABEL_COLUMNS):
        raise ValueError(
            "Frame feature table is missing a gesture label column. "
            f"Expected one of {_GESTURE_LABEL_COLUMNS} in {csv_path}."
        )
    if not any(name in fieldnames for name in _IS_COMPLETION_COLUMNS):
        raise ValueError(
            "Frame feature table is missing a completion label column. "
            f"Expected one of {_IS_COMPLETION_COLUMNS} in {csv_path}."
        )
    if len(fieldnames) < FEATURE_COUNT:
        raise FeatureSchemaError(
            f"Frame feature table has too few columns for canonical features: {csv_path}"
        )


def _validate_frame_rows(frame_rows: tuple[FrameFeatureRow, ...], csv_path: Path) -> None:
    seen: set[tuple[str, int]] = set()
    for row in frame_rows:
        key = (row.recording_id, row.frame_index)
        if key in seen:
            raise ValueError(
                "Frame feature table contains duplicate recording/frame rows: "
                f"{row.recording_id}:{row.frame_index} in {csv_path}"
            )
        seen.add(key)


def _validate_window_config(
    *,
    window_size: int,
    stride: int,
    target: TrainingTarget,
    horizon_frames: int,
) -> None:
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero.")
    if stride <= 0:
        raise ValueError("stride must be greater than zero.")
    if _target_requires_horizon(target) and horizon_frames <= 0:
        raise ValueError(
            "horizon_frames must be greater than zero for tolerant frame-horizon targets."
        )


def _build_target(
    recording_rows: list[FrameFeatureRow],
    *,
    window_end_position: int,
    target: TrainingTarget,
    horizon_frames: int,
) -> tuple[int, str]:
    window_end_row = recording_rows[window_end_position]
    if target.endswith("_frame_binary"):
        return _row_target_value(window_end_row, target)

    if target.endswith("_within_last_k_frames"):
        recent_rows = recording_rows[
            max(0, window_end_position - horizon_frames + 1) : window_end_position + 1
        ]
        for row in reversed(recent_rows):
            if _row_matches_target(row, target):
                return _row_target_value(row, target)
        return 0, ""

    future_rows = recording_rows[
        window_end_position + 1 : window_end_position + 1 + horizon_frames
    ]
    for row in future_rows:
        if _row_matches_target(row, target):
            return _row_target_value(row, target)
    return 0, ""


def _row_target_value(row: FrameFeatureRow, target: TrainingTarget) -> tuple[int, str]:
    matches_target = _row_matches_target(row, target)
    return int(matches_target), row.gesture_label if matches_target else ""


def _row_matches_target(row: FrameFeatureRow, target: TrainingTarget) -> bool:
    if target.startswith("completion_"):
        return row.is_completion_frame
    return row.is_arm_frame


def _target_requires_horizon(target: TrainingTarget) -> bool:
    return target.endswith("_within_next_k_frames") or target.endswith("_within_last_k_frames")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _resolve_required_column(
    fieldnames: tuple[str, ...],
    *,
    candidates: tuple[str, ...],
    csv_path: Path,
    column_role: str,
) -> str:
    matches = [name for name in candidates if name in fieldnames]
    if not matches:
        raise ValueError(
            f"Frame feature table is missing a {column_role} column in {csv_path}. "
            f"Expected one of {candidates}."
        )
    return matches[0]


def _resolve_optional_column(
    fieldnames: tuple[str, ...],
    *,
    candidates: tuple[str, ...],
) -> str | None:
    for name in candidates:
        if name in fieldnames:
            return name
    return None


if __name__ == "__main__":
    main()

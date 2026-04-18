"""Prepare a strict offline dataset for gesture completion CNN training."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Any, Final

import numpy as np

from visionbeat.build_training_samples import (
    DEFAULT_HORIZON_FRAMES,
    DEFAULT_STRIDE,
    DEFAULT_TARGET,
    FrameFeatureRow,
    SUPPORTED_TRAINING_TARGETS,
    TrainingSampleDataset,
    TrainingTarget,
    build_training_samples,
    load_frame_feature_rows,
)
from visionbeat.config import ConfigError, TrackerConfig, load_config
from visionbeat.extract_dataset_features import (
    align_dataset_feature_labels,
)
from visionbeat.features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    CanonicalFeatureSchema,
    FeatureSchemaError,
    assert_feature_schemas_match,
    get_canonical_feature_schema,
)
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import PoseBackendError, create_pose_provider
from visionbeat.validate_feature_parity import (
    FeatureParityReport,
    validate_video_feature_parity,
)

DEFAULT_WINDOW_SIZE: Final[int] = 12
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.25
_MAX_SANITY_SAMPLES: Final[int] = 20
_MAX_REASONABLE_VELOCITY: Final[float] = 100.0
_MAX_REASONABLE_WRIST_DISTANCE: Final[float] = math.sqrt(2.0) + 1e-6


@dataclass(frozen=True, slots=True)
class RecordingDatasetInput:
    """One recorded video plus its completion labels."""

    recording_id: str
    video_path: Path
    labels_path: Path


@dataclass(frozen=True, slots=True)
class RecordingDatasetArtifacts:
    """Generated artifacts and checks for one recording."""

    recording_id: str
    video_path: Path
    labels_path: Path
    feature_table_path: Path
    feature_schema_path: Path
    labeled_table_path: Path
    labeled_schema_path: Path
    frames_processed: int
    completion_count: int
    parity_report: FeatureParityReport


@dataclass(frozen=True, slots=True)
class CompletionDatasetPreparationResult:
    """Summary of one completed multi-recording dataset preparation run."""

    output_path: Path
    feature_schema: CanonicalFeatureSchema
    recordings: tuple[RecordingDatasetArtifacts, ...]
    train_sample_count: int
    validation_sample_count: int
    total_frames_processed: int
    total_labeled_gestures: int
    train_shape: tuple[int, ...]
    validation_shape: tuple[int, ...]
    sanity_warnings: tuple[str, ...]
    validation_status: str
    target_name: str
    horizon_frames: int

    def schema_report_lines(self) -> tuple[str, ...]:
        """Return the canonical schema report requested by the dataset workflow."""
        return (
            f"Feature count: {self.feature_schema.feature_count}",
            f"Schema version: {self.feature_schema.version}",
            "Ordered feature names:",
            *[f"- {name}" for name in self.feature_schema.feature_names],
        )

    def summary_lines(self) -> tuple[str, ...]:
        """Return a concise end-of-run summary."""
        return (
            f"Total frames processed: {self.total_frames_processed}",
            f"Total labeled gestures: {self.total_labeled_gestures}",
            f"Total training samples: {self.train_sample_count}",
            f"Total validation samples: {self.validation_sample_count}",
            f"Feature dimension: {self.feature_schema.feature_count}",
            f"Schema version: {self.feature_schema.version}",
            f"Target name: {self.target_name}",
            f"Horizon frames: {self.horizon_frames}",
            f"Validation status: {self.validation_status}",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the completion-dataset preparation workflow."""

    parser = argparse.ArgumentParser(
        description="Prepare VisionBeat gesture completion datasets from recorded videos."
    )
    parser.add_argument(
        "--recording",
        action="append",
        required=True,
        help="Recording spec in the form '<recording_id>=<video_path>'. Repeat per video.",
    )
    parser.add_argument(
        "--labels",
        action="append",
        default=[],
        help="Label spec in the form '<recording_id>=<labels_csv_path>'. Repeat per recording.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where feature tables, labeled tables, and train_dataset.npz are written.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a VisionBeat YAML or TOML configuration file.",
    )
    parser.add_argument(
        "--pose-backend",
        default=None,
        choices=("mediapipe", "movenet"),
        help="Override the configured pose backend for offline extraction.",
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
        "--validation-recording-id",
        default=None,
        help="Recording id reserved for validation. Defaults to the last recording provided.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help=(
            "Fraction of the validation recording tail to reserve, aligned to gesture "
            f"boundaries when possible. Default: {DEFAULT_VALIDATION_FRACTION}."
        ),
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
    return parser.parse_args(argv)


def prepare_completion_dataset(
    recordings: tuple[RecordingDatasetInput, ...],
    *,
    output_dir: str | Path,
    tracker_config: TrackerConfig | None = None,
    cv2_module: Any | None = None,
    pose_provider_factory: Any = create_pose_provider,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    validation_recording_id: str | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    target: TrainingTarget = DEFAULT_TARGET,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
) -> CompletionDatasetPreparationResult:
    """Prepare a split CNN-ready gesture completion dataset with strict parity checks."""

    if len(recordings) < 2:
        raise ValueError("At least two labeled recordings are required for train/validation.")
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero.")
    if stride <= 0:
        raise ValueError("stride must be greater than zero.")
    if not 0.0 < validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be greater than zero and at most one.")
    if target in {
        "completion_within_next_k_frames",
        "completion_within_last_k_frames",
    } and horizon_frames <= 0:
        raise ValueError(
            "horizon_frames must be greater than zero for tolerant completion targets."
        )

    schema = verify_canonical_feature_schema()
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[RecordingDatasetArtifacts] = []
    rows_by_recording: dict[str, tuple[FrameFeatureRow, ...]] = {}
    for recording in recordings:
        feature_table_path = destination_dir / f"{recording.recording_id}_features.csv"
        parity_report = validate_video_feature_parity(
            recording.video_path,
            output_path=feature_table_path,
            tracker_config=tracker_config,
            recording_id=recording.recording_id,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        if not parity_report.passed:
            raise RuntimeError(parity_report.to_text())

        feature_schema_path = feature_table_path.with_name(f"{feature_table_path.name}.schema.json")
        label_alignment = align_dataset_feature_labels(
            feature_table_path,
            labels_path=recording.labels_path,
            output_path=destination_dir / f"{recording.recording_id}_labeled.csv",
        )
        assert_feature_schemas_match(
            schema,
            label_alignment.feature_schema,
            context_expected="canonical feature schema",
            context_actual="aligned feature table schema",
        )
        labeled_schema_path = label_alignment.schema_path
        loaded_schema, frame_rows = load_frame_feature_rows(label_alignment.output_path)
        assert_feature_schemas_match(
            schema,
            loaded_schema,
            context_expected="canonical feature schema",
            context_actual="loaded labeled frame table schema",
        )
        rows_by_recording[recording.recording_id] = frame_rows
        artifacts.append(
            RecordingDatasetArtifacts(
                recording_id=recording.recording_id,
                video_path=recording.video_path,
                labels_path=recording.labels_path,
                feature_table_path=feature_table_path,
                feature_schema_path=feature_schema_path,
                labeled_table_path=label_alignment.output_path,
                labeled_schema_path=labeled_schema_path,
                frames_processed=len(frame_rows),
                completion_count=label_alignment.completion_count,
                parity_report=parity_report,
            )
        )

    warnings = run_dataset_sanity_checks(rows_by_recording, sample_count=_MAX_SANITY_SAMPLES)
    train_rows, validation_rows = _split_recording_rows(
        recordings=recordings,
        rows_by_recording=rows_by_recording,
        validation_recording_id=validation_recording_id or recordings[-1].recording_id,
        validation_fraction=validation_fraction,
        window_size=window_size,
    )
    train_dataset = build_training_samples(
        train_rows,
        window_size=window_size,
        stride=stride,
        target=target,
        horizon_frames=horizon_frames,
        feature_schema=schema,
    )
    validation_dataset = build_training_samples(
        validation_rows,
        window_size=window_size,
        stride=stride,
        target=target,
        horizon_frames=horizon_frames,
        feature_schema=schema,
    )
    if train_dataset.X.shape[0] == 0:
        raise ValueError("Training split produced zero samples.")
    if validation_dataset.X.shape[0] == 0:
        raise ValueError("Validation split produced zero samples.")

    _assert_dataset_is_stable(train_dataset, split_name="train")
    _assert_dataset_is_stable(validation_dataset, split_name="validation")
    output_path = _save_split_dataset(
        destination_dir / "train_dataset.npz",
        schema=schema,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    return CompletionDatasetPreparationResult(
        output_path=output_path,
        feature_schema=schema,
        recordings=tuple(artifacts),
        train_sample_count=int(train_dataset.X.shape[0]),
        validation_sample_count=int(validation_dataset.X.shape[0]),
        total_frames_processed=sum(artifact.frames_processed for artifact in artifacts),
        total_labeled_gestures=sum(artifact.completion_count for artifact in artifacts),
        train_shape=tuple(int(dimension) for dimension in train_dataset.X.shape),
        validation_shape=tuple(int(dimension) for dimension in validation_dataset.X.shape),
        sanity_warnings=warnings,
        validation_status="PASS",
        target_name=train_dataset.target_name,
        horizon_frames=train_dataset.horizon_frames,
    )


def verify_canonical_feature_schema() -> CanonicalFeatureSchema:
    """Validate the single authoritative canonical feature schema."""

    schema = get_canonical_feature_schema()
    if FEATURE_NAMES != CANONICAL_FEATURE_NAMES:
        raise FeatureSchemaError("FEATURE_NAMES and CANONICAL_FEATURE_NAMES diverged.")
    if schema.version != FEATURE_SCHEMA_VERSION:
        raise FeatureSchemaError(
            "Canonical schema version mismatch. "
            f"Expected {FEATURE_SCHEMA_VERSION}, got {schema.version}."
        )
    if schema.feature_names != FEATURE_NAMES:
        raise FeatureSchemaError("Canonical schema feature_names drifted from FEATURE_NAMES.")
    if schema.feature_count != len(FEATURE_NAMES):
        raise FeatureSchemaError(
            "Canonical schema feature_count drifted from FEATURE_NAMES. "
            f"Expected {len(FEATURE_NAMES)}, got {schema.feature_count}."
        )
    return schema


def run_dataset_sanity_checks(
    rows_by_recording: dict[str, tuple[FrameFeatureRow, ...]],
    *,
    sample_count: int = _MAX_SANITY_SAMPLES,
) -> tuple[str, ...]:
    """Inspect a deterministic sample of rows and flag obvious data issues."""

    warnings: list[str] = []
    flat_rows: list[tuple[str, FrameFeatureRow]] = []
    for recording_id, rows in rows_by_recording.items():
        previous_row: FrameFeatureRow | None = None
        for row in rows:
            if previous_row is not None:
                if row.frame_index <= previous_row.frame_index:
                    warnings.append(
                        f"{recording_id}: non-increasing frame_index near frame {row.frame_index}"
                    )
                timestamp_delta = row.timestamp_seconds - previous_row.timestamp_seconds
                if timestamp_delta < 0.0:
                    warnings.append(
                        f"{recording_id}: negative timestamp delta at frame {row.frame_index}"
                    )
                dt_feature = row.vector[FEATURE_NAMES.index("dt_seconds")]
                expected_dt = max(timestamp_delta, 0.0)
                if abs(dt_feature - expected_dt) > 1e-6:
                    warnings.append(
                        f"{recording_id}: dt_seconds mismatch at frame {row.frame_index} "
                        f"(feature={dt_feature:.6f}, expected={expected_dt:.6f})"
                    )
            if row.is_completion_frame and row.gesture_label == "":
                warnings.append(
                    f"{recording_id}: completion frame {row.frame_index} has an empty gesture label"
                )
            flat_rows.append((recording_id, row))
            previous_row = row

    sample_total = min(sample_count, len(flat_rows))
    for sample_index in Random(0).sample(range(len(flat_rows)), sample_total):
        recording_id, row = flat_rows[sample_index]
        warnings.extend(_inspect_sample_row(recording_id=recording_id, row=row))
    return tuple(dict.fromkeys(warnings))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for completion-dataset preparation."""

    args = parse_args(argv)
    try:
        app_config = load_config(Path(args.config))
        tracker_config = app_config.tracker
        if args.pose_backend is not None:
            tracker_config = replace(tracker_config, backend=args.pose_backend.lower())
        configure_logging(
            app_config.logging.level,
            log_format=app_config.logging.format,
            structured=app_config.logging.structured,
        )
        result = prepare_completion_dataset(
            _parse_recording_inputs(args.recording, args.labels),
            output_dir=args.out_dir,
            tracker_config=tracker_config,
            window_size=args.window_size,
            stride=args.stride,
            validation_recording_id=args.validation_recording_id,
            validation_fraction=args.validation_fraction,
            target=args.target,
            horizon_frames=args.horizon_frames,
        )
    except (
        ConfigError,
        FeatureSchemaError,
        PoseBackendError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    for line in result.schema_report_lines():
        print(line)
    for artifact in result.recordings:
        print(artifact.parity_report.to_text())
        print(
            f"Extracted {artifact.frames_processed} frames to {artifact.feature_table_path} "
            f"and {artifact.labeled_table_path}"
        )
    if result.sanity_warnings:
        print("Warnings:")
        for warning in result.sanity_warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")
    for line in result.summary_lines():
        print(line)


def _parse_recording_inputs(
    recording_specs: list[str],
    label_specs: list[str],
) -> tuple[RecordingDatasetInput, ...]:
    labels_by_recording = {
        recording_id: path
        for recording_id, path in (
            _parse_mapping_spec(spec, label="labels") for spec in label_specs
        )
    }
    parsed_recordings: list[RecordingDatasetInput] = []
    for spec in recording_specs:
        recording_id, video_path = _parse_mapping_spec(spec, label="recording")
        if recording_id not in labels_by_recording:
            raise ValueError(f"Missing labels specification for recording {recording_id!r}.")
        parsed_recordings.append(
            RecordingDatasetInput(
                recording_id=recording_id,
                video_path=Path(video_path),
                labels_path=Path(labels_by_recording[recording_id]),
            )
        )
    return tuple(parsed_recordings)


def _parse_mapping_spec(spec: str, *, label: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid {label} spec {spec!r}. Expected the form '<recording_id>=<path>'."
        )
    key, value = spec.split("=", 1)
    normalized_key = key.strip()
    normalized_value = value.strip()
    if normalized_key == "" or normalized_value == "":
        raise ValueError(
            f"Invalid {label} spec {spec!r}. Expected the form '<recording_id>=<path>'."
        )
    return normalized_key, normalized_value


def _split_recording_rows(
    *,
    recordings: tuple[RecordingDatasetInput, ...],
    rows_by_recording: dict[str, tuple[FrameFeatureRow, ...]],
    validation_recording_id: str,
    validation_fraction: float,
    window_size: int,
) -> tuple[tuple[FrameFeatureRow, ...], tuple[FrameFeatureRow, ...]]:
    if validation_recording_id not in rows_by_recording:
        raise ValueError(f"Unknown validation_recording_id: {validation_recording_id}")

    train_rows: list[FrameFeatureRow] = []
    for recording in recordings:
        if recording.recording_id == validation_recording_id:
            continue
        train_rows.extend(rows_by_recording[recording.recording_id])
    validation_rows = _select_validation_rows(
        rows_by_recording[validation_recording_id],
        validation_fraction=validation_fraction,
        window_size=window_size,
    )
    return tuple(train_rows), validation_rows


def _select_validation_rows(
    rows: tuple[FrameFeatureRow, ...],
    *,
    validation_fraction: float,
    window_size: int,
) -> tuple[FrameFeatureRow, ...]:
    if len(rows) < window_size:
        raise ValueError(
            "Validation recording does not contain enough frames for one sample window. "
            f"Need at least {window_size}, got {len(rows)}."
        )

    target_size = max(window_size, int(math.ceil(len(rows) * validation_fraction)))
    target_start = max(0, len(rows) - target_size)
    gesture_boundaries = [
        boundary
        for boundary in _gesture_segment_start_positions(rows)
        if len(rows) - boundary >= window_size
    ]
    if not gesture_boundaries:
        start_position = max(0, len(rows) - window_size)
    else:
        candidate_positions = [
            boundary for boundary in gesture_boundaries if boundary >= target_start
        ]
        start_position = (
            candidate_positions[0] if candidate_positions else gesture_boundaries[-1]
        )
    return rows[start_position:]


def _gesture_segment_start_positions(rows: tuple[FrameFeatureRow, ...]) -> tuple[int, ...]:
    starts = [0]
    for index, row in enumerate(rows[:-1]):
        if row.is_completion_frame:
            starts.append(index + 1)
    return tuple(dict.fromkeys(starts))


def _inspect_sample_row(*, recording_id: str, row: FrameFeatureRow) -> list[str]:
    warnings: list[str] = []
    vector = np.asarray(row.vector, dtype=np.float32)
    if not np.isfinite(vector).all():
        warnings.append(f"{recording_id}: non-finite feature value at frame {row.frame_index}")

    left_wrist_x = row.vector[FEATURE_NAMES.index("left_wrist_x")]
    left_wrist_y = row.vector[FEATURE_NAMES.index("left_wrist_y")]
    right_wrist_x = row.vector[FEATURE_NAMES.index("right_wrist_x")]
    right_wrist_y = row.vector[FEATURE_NAMES.index("right_wrist_y")]
    for name, value in (
        ("left_wrist_x", left_wrist_x),
        ("left_wrist_y", left_wrist_y),
        ("right_wrist_x", right_wrist_x),
        ("right_wrist_y", right_wrist_y),
    ):
        if value < 0.0 or value > 1.0:
            warnings.append(
                f"{recording_id}: {name} out of range at frame {row.frame_index} ({value:.6f})"
            )

    wrist_distance = row.vector[FEATURE_NAMES.index("wrist_distance_xy")]
    if wrist_distance < 0.0 or wrist_distance > _MAX_REASONABLE_WRIST_DISTANCE:
        warnings.append(
            f"{recording_id}: wrist_distance_xy looks unreasonable at frame "
            f"{row.frame_index} ({wrist_distance:.6f})"
        )

    for feature_name in (
        "left_wrist_rel_vx",
        "left_wrist_rel_vy",
        "right_wrist_rel_vx",
        "right_wrist_rel_vy",
        "wrist_delta_x_v",
        "wrist_delta_y_v",
        "wrist_distance_xy_v",
    ):
        value = abs(row.vector[FEATURE_NAMES.index(feature_name)])
        if value > _MAX_REASONABLE_VELOCITY:
            warnings.append(
                f"{recording_id}: {feature_name} looks extreme at frame {row.frame_index} "
                f"({value:.6f})"
            )
    return warnings


def _assert_dataset_is_stable(
    dataset: TrainingSampleDataset,
    *,
    split_name: str,
) -> None:
    schema = get_canonical_feature_schema()
    assert_feature_schemas_match(
        schema,
        dataset.feature_schema,
        context_expected="canonical feature schema",
        context_actual=f"{split_name} dataset schema",
    )
    if dataset.X.shape[2] != schema.feature_count:
        raise FeatureSchemaError(
            f"{split_name} dataset feature dimension mismatch. "
            f"Expected {schema.feature_count}, got {dataset.X.shape[2]}."
        )
    if not np.isfinite(dataset.X).all():
        raise ValueError(f"{split_name} dataset contains non-finite feature values.")
    if not np.isfinite(dataset.y).all():
        raise ValueError(f"{split_name} dataset contains non-finite labels.")


def _save_split_dataset(
    output_path: Path,
    *,
    schema: CanonicalFeatureSchema,
    train_dataset: TrainingSampleDataset,
    validation_dataset: TrainingSampleDataset,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=train_dataset.X,
        y_train=train_dataset.y,
        X_val=validation_dataset.X,
        y_val=validation_dataset.y,
        train_recording_ids=train_dataset.recording_ids,
        validation_recording_ids=validation_dataset.recording_ids,
        train_window_end_frame_indices=train_dataset.window_end_frame_indices,
        validation_window_end_frame_indices=validation_dataset.window_end_frame_indices,
        train_window_end_timestamps_seconds=train_dataset.window_end_timestamps_seconds,
        validation_window_end_timestamps_seconds=validation_dataset.window_end_timestamps_seconds,
        train_target_gesture_labels=train_dataset.target_gesture_labels,
        validation_target_gesture_labels=validation_dataset.target_gesture_labels,
        feature_names=np.asarray(schema.feature_names, dtype="<U64"),
        schema_version=np.asarray(schema.version, dtype="<U64"),
        feature_count=np.asarray(schema.feature_count, dtype=np.int64),
        window_size=np.asarray(train_dataset.window_size, dtype=np.int64),
        stride=np.asarray(train_dataset.stride, dtype=np.int64),
        target_name=np.asarray(train_dataset.target_name, dtype="<U64"),
        horizon_frames=np.asarray(train_dataset.horizon_frames, dtype=np.int64),
        validation_status=np.asarray("PASS", dtype="<U16"),
    )
    return output_path


if __name__ == "__main__":
    main()

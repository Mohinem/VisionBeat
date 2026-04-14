"""Offline dataset feature extraction for recorded videos.

This module runs the same pose-tracking and canonical feature-extraction path used
by the live runtime, but over prerecorded video files for dataset generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from visionbeat.config import ConfigError, TrackerConfig, load_config
from visionbeat.features import (
    FEATURE_NAMES,
    CanonicalFeatureExtractor,
    CanonicalFeatureSchema,
    assert_feature_schemas_match,
    get_canonical_feature_schema,
)
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import PoseBackendError, create_pose_provider

_REQUIRED_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "recording_id",
    "frame_index",
    "timestamp_seconds",
    "person_detected",
    "tracking_status",
)
_COMPLETION_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "gesture_label",
    "is_completion_frame",
)
_PUBLIC_COMPLETION_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "gesture",
    "is_completion",
)
_LABEL_FIELD_ALIASES: Final[dict[str, str]] = {
    "frame_no": "frame_index",
    "gesture_type": "gesture",
}
_RECORDING_LABEL_COLUMNS: Final[tuple[str, ...]] = ("recording_id",)
_GESTURE_LABEL_COLUMNS: Final[tuple[str, ...]] = ("gesture", "gesture_label")
_FRAME_LABEL_COLUMNS: Final[tuple[str, ...]] = ("frame_index", "timestamp_seconds")
_FRAME_RANGE_LABEL_COLUMNS: Final[tuple[str, ...]] = ("start_frame", "end_frame")
_TIME_RANGE_LABEL_COLUMNS: Final[tuple[str, ...]] = ("start_seconds", "end_seconds")


@dataclass(frozen=True, slots=True)
class DatasetExtractionResult:
    """Summary of one completed offline feature extraction job."""

    video_path: Path
    output_path: Path
    schema_path: Path
    recording_id: str
    frames_processed: int
    feature_schema: CanonicalFeatureSchema
    label_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelAlignmentResult:
    """Summary of one completed feature-table label alignment job."""

    input_path: Path
    output_path: Path
    schema_path: Path
    frames_processed: int
    completion_count: int
    feature_schema: CanonicalFeatureSchema
    label_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LabelRule:
    """One label-assignment rule loaded from a CSV row."""

    mode: str
    values: dict[str, str]
    recording_id: str | None = None
    gesture_label: str = ""
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None

    def matches(
        self,
        *,
        recording_id: str,
        frame_index: int,
        timestamp_seconds: float,
    ) -> bool:
        if self.recording_id not in {None, "", recording_id}:
            return False
        if self.mode == "frame_index":
            return self.frame_index == frame_index
        if self.mode == "timestamp_seconds":
            assert self.timestamp_seconds is not None
            return abs(self.timestamp_seconds - timestamp_seconds) <= 1e-6
        if self.mode == "frame_range":
            assert self.start_frame is not None
            assert self.end_frame is not None
            return self.start_frame <= frame_index <= self.end_frame
        if self.mode == "time_range":
            assert self.start_seconds is not None
            assert self.end_seconds is not None
            return self.start_seconds <= timestamp_seconds <= self.end_seconds
        raise ValueError(f"Unsupported label rule mode: {self.mode}")


@dataclass(frozen=True, slots=True)
class _LabelAssigner:
    """Resolve optional label rows for each extracted video frame."""

    rules: tuple[_LabelRule, ...]
    label_columns: tuple[str, ...]

    def labels_for_frame(
        self,
        *,
        recording_id: str,
        frame_index: int,
        timestamp_seconds: float,
    ) -> dict[str, str | bool]:
        assigned: dict[str, str | bool] = {
            "gesture_label": "",
            "is_completion_frame": False,
        }
        for rule in self.rules:
            if rule.matches(
                recording_id=recording_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            ):
                if rule.gesture_label != "":
                    assigned["gesture_label"] = rule.gesture_label
                    assigned["is_completion_frame"] = True
                assigned.update(rule.values)
        return assigned


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for offline dataset feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract VisionBeat canonical per-frame features from a recorded video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the recorded input video.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--out",
        default=None,
        help="Output CSV file path. Defaults to '<video-stem>.features.csv'.",
    )
    output_group.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the derived CSV file.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional labels CSV for completion-frame alignment. Supported selectors: "
            "optional recording_id plus frame_index, timestamp_seconds, "
            "start_frame/end_frame, or start_seconds/end_seconds. Gesture values may "
            "be provided in a gesture or gesture_label column. Common aliases such as "
            "frame_no and gesture_type are also supported."
        ),
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
        "--recording-id",
        default=None,
        help="Optional recording identifier. Defaults to the input video stem.",
    )
    return parser.parse_args(argv)


def extract_dataset_features(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    tracker_config: TrackerConfig | None = None,
    recording_id: str | None = None,
    cv2_module: Any | None = None,
    pose_provider_factory: Any = create_pose_provider,
) -> DatasetExtractionResult:
    """Extract one CSV row of canonical features per video frame.

    This function reuses the exact same canonical feature extractor used by the
    live runtime so offline dataset generation and live inference remain aligned.
    """

    source_video = Path(video_path)
    if not source_video.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_video}")

    resolved_output_path = _resolve_output_path(
        source_video,
        output_path=output_path,
        output_dir=output_dir,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_recording_id = recording_id or source_video.stem
    labels = _load_labels(labels_path)
    tracker = pose_provider_factory(tracker_config or TrackerConfig())
    extractor = CanonicalFeatureExtractor()

    if cv2_module is None:
        import cv2

        cv2_module = cv2

    capture = cv2_module.VideoCapture(source_video.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {source_video}")

    fieldnames = [
        *_REQUIRED_OUTPUT_COLUMNS,
        *labels.label_columns,
        *FEATURE_NAMES,
    ]
    feature_schema = get_canonical_feature_schema()
    assert_feature_schemas_match(
        feature_schema,
        extractor.schema,
        context_expected="offline dataset schema",
        context_actual="feature extractor schema",
    )
    schema_path = _resolve_schema_path(resolved_output_path)
    frames_processed = 0
    try:
        with resolved_output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                timestamp_seconds = _resolve_frame_timestamp_seconds(
                    capture,
                    frame_index=frame_index,
                    cv2_module=cv2_module,
                )
                pose = tracker.process(frame, timestamp_seconds)
                frame_features = extractor.update(pose)
                row = {
                    "recording_id": resolved_recording_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_features.timestamp.seconds,
                    "person_detected": frame_features.person_detected,
                    "tracking_status": frame_features.status,
                    **labels.labels_for_frame(
                        recording_id=resolved_recording_id,
                        frame_index=frame_index,
                        timestamp_seconds=frame_features.timestamp.seconds,
                    ),
                    **frame_features.as_feature_dict(),
                }
                writer.writerow(row)
                frames_processed += 1
                frame_index += 1
        _write_schema_sidecar(
            schema_path,
            feature_schema=feature_schema,
            recording_id=resolved_recording_id,
            output_columns=tuple(fieldnames),
            label_columns=labels.label_columns,
        )
    finally:
        capture.release()
        tracker.close()

    return DatasetExtractionResult(
        video_path=source_video,
        output_path=resolved_output_path,
        schema_path=schema_path,
        recording_id=resolved_recording_id,
        frames_processed=frames_processed,
        feature_schema=feature_schema,
        label_columns=labels.label_columns,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for offline dataset feature extraction."""

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
        result = extract_dataset_features(
            args.video,
            output_path=args.out,
            output_dir=args.out_dir,
            labels_path=args.labels,
            tracker_config=tracker_config,
            recording_id=args.recording_id,
        )
    except (ConfigError, PoseBackendError, ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Extracted {result.frames_processed} frames from {result.video_path} "
        f"to {result.output_path}"
    )


def align_dataset_feature_labels(
    feature_table_path: str | Path,
    *,
    labels_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> LabelAlignmentResult:
    """Apply completion labels to an existing canonical per-frame feature table."""

    source_path = Path(feature_table_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Feature table does not exist: {source_path}")

    resolved_output_path = _resolve_labeled_output_path(
        source_path,
        output_path=output_path,
        output_dir=output_dir,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(labels_path)
    feature_schema = _load_feature_schema_for_table(source_path)
    schema_path = _resolve_schema_path(resolved_output_path)

    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        source_fieldnames = tuple(reader.fieldnames or ())
        _validate_feature_table_for_label_alignment(source_path, source_fieldnames)

        base_columns = tuple(
            name
            for name in source_fieldnames
            if name not in {*_COMPLETION_OUTPUT_COLUMNS, *_PUBLIC_COMPLETION_OUTPUT_COLUMNS}
        )
        feature_columns = tuple(name for name in base_columns if name in FEATURE_NAMES)
        output_label_columns = _public_label_columns(labels.label_columns)
        output_columns = (
            *tuple(name for name in base_columns if name not in FEATURE_NAMES),
            *output_label_columns,
            *feature_columns,
        )

        frames_processed = 0
        completion_count = 0
        with resolved_output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()
            for row in reader:
                gesture_value, is_completion_value, additional_values = _public_labels_for_row(
                    row,
                    labels=labels,
                )
                if is_completion_value:
                    completion_count += 1
                output_row = {
                    **{
                        name: row[name]
                        for name in base_columns
                        if name not in FEATURE_NAMES
                    },
                    "gesture": gesture_value,
                    "is_completion": is_completion_value,
                    **additional_values,
                    **{name: row[name] for name in feature_columns},
                }
                writer.writerow(output_row)
                frames_processed += 1

    _write_schema_sidecar(
        schema_path,
        feature_schema=feature_schema,
        recording_id=_infer_schema_recording_id(source_path),
        output_columns=tuple(output_columns),
        label_columns=output_label_columns,
    )
    return LabelAlignmentResult(
        input_path=source_path,
        output_path=resolved_output_path,
        schema_path=schema_path,
        frames_processed=frames_processed,
        completion_count=completion_count,
        feature_schema=feature_schema,
        label_columns=output_label_columns,
    )


def _resolve_output_path(
    video_path: Path,
    *,
    output_path: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    if output_dir is not None:
        return Path(output_dir) / f"{video_path.stem}.features.csv"
    return video_path.with_name(f"{video_path.stem}.features.csv")


def _resolve_schema_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.schema.json")


def _resolve_labeled_output_path(
    feature_table_path: Path,
    *,
    output_path: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    if output_dir is not None:
        return Path(output_dir) / f"{feature_table_path.stem}.labeled.csv"
    return feature_table_path.with_name(f"{feature_table_path.stem}.labeled.csv")


def _write_schema_sidecar(
    schema_path: Path,
    *,
    feature_schema: CanonicalFeatureSchema,
    recording_id: str,
    output_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> None:
    import json

    payload = {
        **feature_schema.to_dict(),
        "recording_id": recording_id,
        "label_columns": list(label_columns),
        "output_columns": list(output_columns),
    }
    schema_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_frame_timestamp_seconds(
    capture: Any,
    *,
    frame_index: int,
    cv2_module: Any,
) -> float:
    fps_property = getattr(cv2_module, "CAP_PROP_FPS", None)
    pos_msec_property = getattr(cv2_module, "CAP_PROP_POS_MSEC", None)
    timestamp_msec = (
        float(capture.get(pos_msec_property))
        if pos_msec_property is not None
        else 0.0
    )
    if timestamp_msec > 0.0:
        return timestamp_msec / 1000.0
    fps = float(capture.get(fps_property)) if fps_property is not None else 0.0
    if fps > 0.0:
        return frame_index / fps
    return float(frame_index)


def _load_labels(labels_path: str | Path | None) -> _LabelAssigner:
    if labels_path is None:
        return _LabelAssigner(rules=(), label_columns=_COMPLETION_OUTPUT_COLUMNS)

    path = Path(labels_path)
    with path.open("r", encoding="utf-8", newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        raw_fieldnames = tuple(reader.fieldnames or ())
        if not raw_fieldnames:
            raise ValueError(f"Labels CSV has no header row: {path}")
        fieldnames = _canonicalize_label_fieldnames(raw_fieldnames, path=path)
        rules = tuple(
            _parse_label_rule(
                _canonicalize_label_row(row),
                fieldnames=fieldnames,
                path=path,
            )
            for row in reader
        )
        additional_label_columns = _infer_additional_label_columns(fieldnames)
    _validate_additional_label_columns(additional_label_columns)
    return _LabelAssigner(
        rules=rules,
        label_columns=(*_COMPLETION_OUTPUT_COLUMNS, *additional_label_columns),
    )


def _parse_label_rule(
    row: dict[str, str | None],
    *,
    fieldnames: tuple[str, ...],
    path: Path,
) -> _LabelRule:
    normalized = {
        key: "" if value is None else value.strip()
        for key, value in row.items()
        if key is not None
    }
    additional_label_values = {
        key: value
        for key, value in normalized.items()
        if key in _infer_additional_label_columns(fieldnames) and value != ""
    }
    recording_id = normalized.get("recording_id", "") or None
    gesture_label = normalized.get("gesture_label", "") or normalized.get("gesture", "")

    if normalized.get("frame_index", "") != "":
        return _LabelRule(
            mode="frame_index",
            recording_id=recording_id,
            gesture_label=gesture_label,
            frame_index=int(normalized["frame_index"]),
            values=additional_label_values,
        )
    if normalized.get("timestamp_seconds", "") != "":
        return _LabelRule(
            mode="timestamp_seconds",
            recording_id=recording_id,
            gesture_label=gesture_label,
            timestamp_seconds=float(normalized["timestamp_seconds"]),
            values=additional_label_values,
        )
    if normalized.get("start_frame", "") != "" and normalized.get("end_frame", "") != "":
        return _LabelRule(
            mode="frame_range",
            recording_id=recording_id,
            gesture_label=gesture_label,
            start_frame=int(normalized["start_frame"]),
            end_frame=int(normalized["end_frame"]),
            values=additional_label_values,
        )
    if normalized.get("start_seconds", "") != "" and normalized.get("end_seconds", "") != "":
        return _LabelRule(
            mode="time_range",
            recording_id=recording_id,
            gesture_label=gesture_label,
            start_seconds=float(normalized["start_seconds"]),
            end_seconds=float(normalized["end_seconds"]),
            values=additional_label_values,
        )
    raise ValueError(
        "Unsupported labels CSV row in "
        f"{path}. Expected one of: frame_index, timestamp_seconds, "
        "start_frame/end_frame, or start_seconds/end_seconds, with optional "
        "recording_id and gesture columns. Aliases such as frame_no and "
        "gesture_type are also supported."
    )


def _canonicalize_label_fieldnames(fieldnames: tuple[str, ...], *, path: Path) -> tuple[str, ...]:
    canonical_fieldnames = tuple(_canonicalize_label_key(name) for name in fieldnames)
    duplicates = sorted(
        {
            name
            for name in canonical_fieldnames
            if canonical_fieldnames.count(name) > 1
        }
    )
    if duplicates:
        raise ValueError(
            "Labels CSV contains duplicate canonical columns after alias normalization in "
            f"{path}: {', '.join(duplicates)}."
        )
    return canonical_fieldnames


def _canonicalize_label_row(row: dict[str, str | None]) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[_canonicalize_label_key(key)] = value
    return normalized


def _canonicalize_label_key(key: str) -> str:
    return _LABEL_FIELD_ALIASES.get(key.strip(), key.strip())


def _infer_additional_label_columns(fieldnames: tuple[str, ...]) -> tuple[str, ...]:
    reserved = {
        *_RECORDING_LABEL_COLUMNS,
        *_GESTURE_LABEL_COLUMNS,
        *_FRAME_LABEL_COLUMNS,
        *_FRAME_RANGE_LABEL_COLUMNS,
        *_TIME_RANGE_LABEL_COLUMNS,
    }
    return tuple(name for name in fieldnames if name not in reserved)


def _validate_additional_label_columns(label_columns: tuple[str, ...]) -> None:
    reserved_output_columns = {
        *_REQUIRED_OUTPUT_COLUMNS,
        *_COMPLETION_OUTPUT_COLUMNS,
        *_PUBLIC_COMPLETION_OUTPUT_COLUMNS,
        *FEATURE_NAMES,
    }
    collisions = sorted(name for name in label_columns if name in reserved_output_columns)
    if collisions:
        joined = ", ".join(collisions)
        raise ValueError(
            "Labels CSV columns collide with reserved output columns: "
            f"{joined}."
        )


def _public_label_columns(label_columns: tuple[str, ...]) -> tuple[str, ...]:
    return (*_PUBLIC_COMPLETION_OUTPUT_COLUMNS, *label_columns[2:])


def _public_labels_for_row(
    row: dict[str, str],
    *,
    labels: _LabelAssigner,
) -> tuple[str, bool, dict[str, str | bool]]:
    assigned = labels.labels_for_frame(
        recording_id=row["recording_id"],
        frame_index=int(row["frame_index"]),
        timestamp_seconds=float(row["timestamp_seconds"]),
    )
    gesture_value = str(assigned.get("gesture_label", ""))
    is_completion_value = bool(assigned.get("is_completion_frame", False))
    additional_values = {
        name: assigned[name]
        for name in labels.label_columns[2:]
        if name in assigned
    }
    return gesture_value, is_completion_value, additional_values


def _validate_feature_table_for_label_alignment(
    feature_table_path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    missing_columns = [
        name
        for name in ("recording_id", "frame_index", "timestamp_seconds")
        if name not in fieldnames
    ]
    if missing_columns:
        raise ValueError(
            "Feature table is missing required metadata columns for label alignment: "
            f"{', '.join(missing_columns)} in {feature_table_path}"
        )
    feature_columns = tuple(name for name in fieldnames if name in FEATURE_NAMES)
    if feature_columns != FEATURE_NAMES:
        raise ValueError(
            "Feature table feature ordering does not match the canonical schema for label "
            f"alignment. Expected {FEATURE_NAMES}, got {feature_columns}."
        )


def _load_feature_schema_for_table(feature_table_path: Path) -> CanonicalFeatureSchema:
    schema_path = _resolve_schema_path(feature_table_path)
    if not schema_path.exists():
        return get_canonical_feature_schema()
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return CanonicalFeatureSchema(
        version=str(payload["schema_version"]),
        feature_names=tuple(payload["feature_names"]),
        feature_count=int(payload["feature_count"]),
    )


def _infer_schema_recording_id(feature_table_path: Path) -> str:
    schema_path = _resolve_schema_path(feature_table_path)
    if not schema_path.exists():
        return feature_table_path.stem
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return str(payload.get("recording_id", feature_table_path.stem))


if __name__ == "__main__":
    main()

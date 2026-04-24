"""Validate offline and live canonical feature parity for VisionBeat."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

from visionbeat.config import ConfigError, TrackerConfig, load_config
from visionbeat.extract_dataset_features import (
    _resolve_frame_timestamp_seconds,
    extract_dataset_features,
)
from visionbeat.features import (
    CANONICAL_DERIVED_FEATURE_NAMES,
    CANONICAL_TEMPORAL_FEATURE_NAMES,
    FEATURE_COUNT,
    FEATURE_NAMES,
    CanonicalFeatureExtractor,
    CanonicalFeatureSchema,
    compare_feature_schemas,
    get_canonical_feature_schema,
)
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import PoseBackendError, create_pose_provider

_DEFAULT_ABS_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureValueMismatch:
    """One numerical difference between offline and live feature computation."""

    frame_index: int
    timestamp_seconds: float
    feature_name: str
    offline_value: float
    live_value: float
    absolute_difference: float


@dataclass(frozen=True, slots=True)
class FeatureParityReport:
    """Human-readable result of an offline-vs-live feature parity check."""

    passed: bool
    offline_schema: CanonicalFeatureSchema
    live_schema: CanonicalFeatureSchema
    offline_feature_names: tuple[str, ...]
    frame_count_offline: int
    frame_count_live: int
    abs_tolerance: float
    feature_name_mismatches: tuple[str, ...]
    feature_position_mismatches: tuple[str, ...]
    numerical_mismatches: tuple[FeatureValueMismatch, ...]
    max_absolute_difference: float
    likely_source: str | None

    def to_text(self, *, max_differences: int = 20) -> str:
        """Render a concise PASS/FAIL report for CLI and test output."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"{status}: VisionBeat offline/live feature parity",
            (
                "Schema version: "
                f"offline={self.offline_schema.version} live={self.live_schema.version}"
            ),
            (
                "Feature count: "
                f"offline={self.offline_schema.feature_count} live={self.live_schema.feature_count}"
            ),
            f"Compared frames: offline={self.frame_count_offline} live={self.frame_count_live}",
            f"Absolute tolerance: {self.abs_tolerance:g}",
            f"Max absolute difference: {self.max_absolute_difference:g}",
        ]
        if self.feature_name_mismatches:
            lines.append("Mismatched feature names:")
            lines.extend(f"- {message}" for message in self.feature_name_mismatches)
        if self.feature_position_mismatches:
            lines.append("Mismatched feature positions:")
            lines.extend(f"- {message}" for message in self.feature_position_mismatches)
        if self.numerical_mismatches:
            lines.append(
                f"Numerical differences (showing up to {min(max_differences, len(self.numerical_mismatches))}):"
            )
            for mismatch in self.numerical_mismatches[:max_differences]:
                lines.append(
                    "- "
                    f"frame={mismatch.frame_index} "
                    f"timestamp={mismatch.timestamp_seconds:.6f} "
                    f"feature={mismatch.feature_name} "
                    f"offline={mismatch.offline_value:.9f} "
                    f"live={mismatch.live_value:.9f} "
                    f"abs_diff={mismatch.absolute_difference:.9f}"
                )
            remaining = len(self.numerical_mismatches) - max_differences
            if remaining > 0:
                lines.append(f"- ... {remaining} additional numerical differences omitted")
        if self.likely_source is not None:
            lines.append(f"Likely source: {self.likely_source}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _OfflineFeatureRow:
    frame_index: int
    timestamp_seconds: float
    feature_values: dict[str, float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for feature parity validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate that VisionBeat offline dataset extraction and live runtime "
            "feature extraction produce equivalent canonical features."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the recorded video used for both offline and live-equivalent extraction.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--out",
        default=None,
        help="Optional CSV path to keep the offline extraction artifact used during validation.",
    )
    output_group.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for the offline extraction artifact.",
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
        help="Override the configured pose backend.",
    )
    parser.add_argument(
        "--recording-id",
        default=None,
        help="Optional recording identifier override for the offline artifact.",
    )
    parser.add_argument(
        "--abs-tolerance",
        type=float,
        default=_DEFAULT_ABS_TOLERANCE,
        help="Maximum absolute numerical difference allowed per feature value.",
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=20,
        help="Maximum number of numerical mismatches to print in the report.",
    )
    return parser.parse_args(argv)


def extract_live_canonical_features_from_video(
    video_path: str | Path,
    *,
    tracker_config: TrackerConfig | None = None,
    cv2_module: Any | None = None,
    pose_provider_factory: Any = create_pose_provider,
) -> tuple:
    """Run the live-equivalent tracker -> canonical-feature path over a video."""

    source_video = Path(video_path)
    if not source_video.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_video}")

    tracker = pose_provider_factory(tracker_config or TrackerConfig())
    extractor = CanonicalFeatureExtractor()
    if cv2_module is None:
        import cv2

        cv2_module = cv2
    capture = cv2_module.VideoCapture(source_video.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {source_video}")

    features = []
    frame_index = 0
    try:
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
            features.append(extractor.update(pose))
            frame_index += 1
    finally:
        capture.release()
        tracker.close()

    return tuple(features)


def validate_offline_feature_csv_against_live_features(
    offline_csv_path: str | Path,
    *,
    live_frames: Sequence,
    offline_schema: CanonicalFeatureSchema | None = None,
    abs_tolerance: float = _DEFAULT_ABS_TOLERANCE,
) -> FeatureParityReport:
    """Compare an offline feature CSV against live-path canonical features."""

    if abs_tolerance < 0.0:
        raise ValueError("abs_tolerance must be greater than or equal to zero.")

    csv_path = Path(offline_csv_path)
    loaded_offline_feature_names, offline_rows = _load_offline_feature_rows(csv_path)
    active_offline_schema = offline_schema or _load_offline_schema(csv_path)
    live_schema = get_canonical_feature_schema()

    schema_differences = compare_feature_schemas(active_offline_schema, live_schema)
    feature_name_mismatches = tuple(
        [
            *schema_differences,
            *_compare_feature_name_sets(
                offline_feature_names=loaded_offline_feature_names,
                live_feature_names=live_schema.feature_names,
            ),
        ]
    )
    feature_position_mismatches = tuple(
        _compare_feature_positions(
            offline_feature_names=loaded_offline_feature_names,
            live_feature_names=live_schema.feature_names,
        )
    )
    numerical_mismatches = tuple(
        _compare_feature_values(
            offline_rows=offline_rows,
            live_frames=live_frames,
            live_feature_names=live_schema.feature_names,
            abs_tolerance=abs_tolerance,
        )
    )

    max_absolute_difference = max(
        (mismatch.absolute_difference for mismatch in numerical_mismatches),
        default=0.0,
    )
    likely_source = _infer_likely_source(
        feature_name_mismatches=feature_name_mismatches,
        feature_position_mismatches=feature_position_mismatches,
        numerical_mismatches=numerical_mismatches,
        frame_count_offline=len(offline_rows),
        frame_count_live=len(live_frames),
    )
    passed = (
        not feature_name_mismatches
        and not feature_position_mismatches
        and not numerical_mismatches
        and len(offline_rows) == len(live_frames)
    )
    return FeatureParityReport(
        passed=passed,
        offline_schema=active_offline_schema,
        live_schema=live_schema,
        offline_feature_names=loaded_offline_feature_names,
        frame_count_offline=len(offline_rows),
        frame_count_live=len(live_frames),
        abs_tolerance=abs_tolerance,
        feature_name_mismatches=feature_name_mismatches,
        feature_position_mismatches=feature_position_mismatches,
        numerical_mismatches=numerical_mismatches,
        max_absolute_difference=max_absolute_difference,
        likely_source=likely_source,
    )


def validate_video_feature_parity(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    tracker_config: TrackerConfig | None = None,
    recording_id: str | None = None,
    cv2_module: Any | None = None,
    pose_provider_factory: Any = create_pose_provider,
    abs_tolerance: float = _DEFAULT_ABS_TOLERANCE,
) -> FeatureParityReport:
    """Validate offline CSV extraction against the live-equivalent path for one video."""

    if output_path is not None or output_dir is not None:
        offline_result = extract_dataset_features(
            video_path,
            output_path=output_path,
            output_dir=output_dir,
            tracker_config=tracker_config,
            recording_id=recording_id,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        live_frames = extract_live_canonical_features_from_video(
            video_path,
            tracker_config=tracker_config,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        return validate_offline_feature_csv_against_live_features(
            offline_result.output_path,
            live_frames=live_frames,
            offline_schema=offline_result.feature_schema,
            abs_tolerance=abs_tolerance,
        )

    with TemporaryDirectory(prefix="visionbeat-feature-parity-") as temp_dir:
        offline_result = extract_dataset_features(
            video_path,
            output_dir=temp_dir,
            tracker_config=tracker_config,
            recording_id=recording_id,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        live_frames = extract_live_canonical_features_from_video(
            video_path,
            tracker_config=tracker_config,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        return validate_offline_feature_csv_against_live_features(
            offline_result.output_path,
            live_frames=live_frames,
            offline_schema=offline_result.feature_schema,
            abs_tolerance=abs_tolerance,
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for feature parity validation."""

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
        report = validate_video_feature_parity(
            args.video,
            output_path=args.out,
            output_dir=args.out_dir,
            tracker_config=tracker_config,
            recording_id=args.recording_id,
            abs_tolerance=args.abs_tolerance,
        )
    except (ConfigError, PoseBackendError, ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(report.to_text(max_differences=args.max_differences))
    if not report.passed:
        raise SystemExit(1)


def _load_offline_schema(csv_path: Path) -> CanonicalFeatureSchema:
    schema_path = csv_path.with_name(f"{csv_path.name}.schema.json")
    if not schema_path.exists():
        return CanonicalFeatureSchema(
            version="unknown",
            feature_names=FEATURE_NAMES,
            feature_count=FEATURE_COUNT,
        )
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
    feature_count = int(payload.get("feature_count", len(feature_names)))
    version = str(payload.get("schema_version", payload.get("schema", "unknown")))
    return CanonicalFeatureSchema(
        version=version,
        feature_names=feature_names,
        feature_count=feature_count,
    )


def _load_offline_feature_rows(csv_path: Path) -> tuple[tuple[str, ...], tuple[_OfflineFeatureRow, ...]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or ())
        if len(fieldnames) < FEATURE_COUNT:
            raise ValueError(
                f"Offline feature CSV has too few columns for canonical features: {csv_path}"
            )
        feature_names = tuple(fieldnames[-FEATURE_COUNT:])
        rows = []
        for row in reader:
            rows.append(
                _OfflineFeatureRow(
                    frame_index=int(row["frame_index"]),
                    timestamp_seconds=float(row["timestamp_seconds"]),
                    feature_values={
                        name: float(row[name])
                        for name in feature_names
                    },
                )
            )
    return feature_names, tuple(rows)


def _compare_feature_name_sets(
    *,
    offline_feature_names: tuple[str, ...],
    live_feature_names: tuple[str, ...],
) -> list[str]:
    differences: list[str] = []
    missing = [name for name in live_feature_names if name not in offline_feature_names]
    extra = [name for name in offline_feature_names if name not in live_feature_names]
    if missing:
        differences.append(f"missing offline features: {', '.join(missing)}")
    if extra:
        differences.append(f"unexpected offline features: {', '.join(extra)}")
    return differences


def _compare_feature_positions(
    *,
    offline_feature_names: tuple[str, ...],
    live_feature_names: tuple[str, ...],
) -> list[str]:
    differences: list[str] = []
    shared_length = min(len(offline_feature_names), len(live_feature_names))
    for index in range(shared_length):
        expected_name = live_feature_names[index]
        actual_name = offline_feature_names[index]
        if expected_name != actual_name:
            differences.append(
                f"position {index}: offline={actual_name} live={expected_name}"
            )
    if len(offline_feature_names) > shared_length:
        differences.append(
            "offline has additional trailing features starting at position "
            f"{shared_length}: {', '.join(offline_feature_names[shared_length:])}"
        )
    if len(live_feature_names) > shared_length:
        differences.append(
            "live has additional trailing features starting at position "
            f"{shared_length}: {', '.join(live_feature_names[shared_length:])}"
        )
    return differences


def _compare_feature_values(
    *,
    offline_rows: Sequence[_OfflineFeatureRow],
    live_frames: Sequence,
    live_feature_names: tuple[str, ...],
    abs_tolerance: float,
) -> list[FeatureValueMismatch]:
    mismatches: list[FeatureValueMismatch] = []
    for offline_row, live_frame in zip(offline_rows, live_frames, strict=False):
        live_feature_values = live_frame.as_feature_dict()
        for feature_name in live_feature_names:
            if feature_name not in offline_row.feature_values:
                continue
            offline_value = offline_row.feature_values[feature_name]
            live_value = float(live_feature_values[feature_name])
            absolute_difference = abs(offline_value - live_value)
            if absolute_difference > abs_tolerance:
                mismatches.append(
                    FeatureValueMismatch(
                        frame_index=offline_row.frame_index,
                        timestamp_seconds=offline_row.timestamp_seconds,
                        feature_name=feature_name,
                        offline_value=offline_value,
                        live_value=live_value,
                        absolute_difference=absolute_difference,
                    )
                )
    return mismatches


def _infer_likely_source(
    *,
    feature_name_mismatches: Sequence[str],
    feature_position_mismatches: Sequence[str],
    numerical_mismatches: Sequence[FeatureValueMismatch],
    frame_count_offline: int,
    frame_count_live: int,
) -> str | None:
    if feature_name_mismatches or feature_position_mismatches:
        return "Feature schema drift between offline output and the live extractor."
    if frame_count_offline != frame_count_live:
        return "Frame alignment mismatch between offline extraction and live replay."
    mismatch_names = {mismatch.feature_name for mismatch in numerical_mismatches}
    if mismatch_names & set(CANONICAL_TEMPORAL_FEATURE_NAMES):
        return "Temporal feature mismatch in dt or previous-frame velocity handling."
    if mismatch_names & set(CANONICAL_DERIVED_FEATURE_NAMES):
        return "Derived feature mismatch in normalization, relative coordinates, or distances."
    if mismatch_names:
        return "Raw landmark mismatch in normalization or missing-value handling."
    return None


if __name__ == "__main__":
    main()

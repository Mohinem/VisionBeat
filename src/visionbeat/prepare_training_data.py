"""End-to-end offline training-data preparation for VisionBeat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from visionbeat.build_training_samples import (
    DEFAULT_HORIZON_FRAMES,
    DEFAULT_STRIDE,
    DEFAULT_TARGET,
    DEFAULT_WINDOW_SIZE,
    SUPPORTED_TRAINING_TARGETS,
    TrainingTarget,
    generate_training_samples,
)
from visionbeat.config import ConfigError, TrackerConfig, load_config
from visionbeat.extract_dataset_features import extract_dataset_features
from visionbeat.features import CanonicalFeatureSchema
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import PoseBackendError, create_pose_provider


@dataclass(frozen=True, slots=True)
class PreparedTrainingDataResult:
    """Summary of one completed end-to-end training-data preparation job."""

    video_path: Path
    labels_path: Path | None
    output_path: Path
    frame_table_path: Path | None
    frame_schema_path: Path | None
    feature_schema: CanonicalFeatureSchema
    target_name: str
    window_size: int
    stride: int
    horizon_frames: int
    sample_count: int
    X_shape: tuple[int, int, int]
    y_shape: tuple[int, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for end-to-end training-data preparation."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare VisionBeat CNN-ready training data from a recorded video and labels CSV."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the recorded input video.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional labels CSV for completion-frame alignment.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output `.npz` path. Defaults to '<video-stem>.train.npz'.",
    )
    parser.add_argument(
        "--frames-out",
        default=None,
        help=(
            "Optional path to keep the aligned per-frame feature CSV used to build "
            "training samples. Its schema sidecar is written next to it."
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
    return parser.parse_args(argv)


def prepare_training_data(
    video_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    output_path: str | Path | None = None,
    frame_table_path: str | Path | None = None,
    tracker_config: TrackerConfig | None = None,
    recording_id: str | None = None,
    cv2_module: Any | None = None,
    pose_provider_factory: Any = create_pose_provider,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    target: TrainingTarget = DEFAULT_TARGET,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
) -> PreparedTrainingDataResult:
    """Run the full offline pipeline from video to CNN-ready `.npz` training data."""

    source_video = Path(video_path)
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else source_video.with_name(f"{source_video.stem}.train.npz")
    )
    resolved_labels_path = None if labels_path is None else Path(labels_path)

    if frame_table_path is not None:
        extraction_result = extract_dataset_features(
            source_video,
            output_path=frame_table_path,
            labels_path=resolved_labels_path,
            tracker_config=tracker_config,
            recording_id=recording_id,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        training_result = generate_training_samples(
            extraction_result.output_path,
            output_path=resolved_output_path,
            window_size=window_size,
            stride=stride,
            target=target,
            horizon_frames=horizon_frames,
        )
        return PreparedTrainingDataResult(
            video_path=source_video,
            labels_path=resolved_labels_path,
            output_path=training_result.output_path,
            frame_table_path=extraction_result.output_path,
            frame_schema_path=extraction_result.schema_path,
            feature_schema=training_result.feature_schema,
            target_name=training_result.target_name,
            window_size=training_result.window_size,
            stride=training_result.stride,
            horizon_frames=training_result.horizon_frames,
            sample_count=training_result.sample_count,
            X_shape=training_result.X_shape,
            y_shape=training_result.y_shape,
        )

    with TemporaryDirectory(prefix="visionbeat-prepare-training-data-") as temp_dir:
        extraction_result = extract_dataset_features(
            source_video,
            output_dir=temp_dir,
            labels_path=resolved_labels_path,
            tracker_config=tracker_config,
            recording_id=recording_id,
            cv2_module=cv2_module,
            pose_provider_factory=pose_provider_factory,
        )
        training_result = generate_training_samples(
            extraction_result.output_path,
            output_path=resolved_output_path,
            window_size=window_size,
            stride=stride,
            target=target,
            horizon_frames=horizon_frames,
        )
    return PreparedTrainingDataResult(
        video_path=source_video,
        labels_path=resolved_labels_path,
        output_path=training_result.output_path,
        frame_table_path=None,
        frame_schema_path=None,
        feature_schema=training_result.feature_schema,
        target_name=training_result.target_name,
        window_size=training_result.window_size,
        stride=training_result.stride,
        horizon_frames=training_result.horizon_frames,
        sample_count=training_result.sample_count,
        X_shape=training_result.X_shape,
        y_shape=training_result.y_shape,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for end-to-end training-data preparation."""

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
        result = prepare_training_data(
            args.video,
            labels_path=args.labels,
            output_path=args.out,
            frame_table_path=args.frames_out,
            tracker_config=tracker_config,
            recording_id=args.recording_id,
            window_size=args.window_size,
            stride=args.stride,
            target=args.target,
            horizon_frames=args.horizon_frames,
        )
    except (ConfigError, PoseBackendError, ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Prepared {result.sample_count} samples from {result.video_path} to {result.output_path} "
        f"with schema {result.feature_schema.version}, X shape {result.X_shape}, "
        f"and y shape {result.y_shape}"
    )


if __name__ == "__main__":
    main()

"""Prepare an offline dataset for early-arm timing model training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Final

from visionbeat.build_training_samples import (
    DEFAULT_HORIZON_FRAMES,
    DEFAULT_STRIDE,
    TrainingTarget,
)
from visionbeat.config import ConfigError, load_config
from visionbeat.features import FeatureSchemaError
from visionbeat.logging_config import configure_logging
from visionbeat.pose_provider import PoseBackendError
from visionbeat.prepare_completion_dataset import (
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WINDOW_SIZE,
    CompletionDatasetPreparationResult,
    RecordingDatasetInput,
    _parse_recording_inputs,
    prepare_completion_dataset,
)

DEFAULT_EARLY_ARM_TARGET: Final[TrainingTarget] = "arm_frame_binary"
SUPPORTED_EARLY_ARM_TARGETS: Final[tuple[TrainingTarget, ...]] = (
    "arm_frame_binary",
    "arm_within_next_k_frames",
    "arm_within_last_k_frames",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the early-arm dataset preparation workflow."""

    parser = argparse.ArgumentParser(
        description="Prepare VisionBeat early-arm datasets from recorded videos and v2 labels."
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
        help=(
            "Label spec in the form '<recording_id>=<labels_csv_path>'. Labels should use "
            "the early-arm v2 event schema."
        ),
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
        choices=SUPPORTED_EARLY_ARM_TARGETS,
        default=DEFAULT_EARLY_ARM_TARGET,
        help="Early-arm prediction target to generate for each sample window.",
    )
    parser.add_argument(
        "--horizon-frames",
        type=int,
        default=DEFAULT_HORIZON_FRAMES,
        help=(
            "Tolerance in frames for `arm_within_next_k_frames` and "
            "`arm_within_last_k_frames`. "
            f"Default: {DEFAULT_HORIZON_FRAMES}."
        ),
    )
    return parser.parse_args(argv)


def prepare_early_arm_dataset(
    recordings: tuple[RecordingDatasetInput, ...],
    *,
    output_dir: str | Path,
    tracker_config=None,
    cv2_module=None,
    pose_provider_factory=None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    validation_recording_id: str | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    target: TrainingTarget = DEFAULT_EARLY_ARM_TARGET,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
) -> CompletionDatasetPreparationResult:
    """Prepare a split dataset for early-arm timing training."""

    if target not in SUPPORTED_EARLY_ARM_TARGETS:
        raise ValueError(
            f"Unsupported early-arm target {target!r}. Expected one of "
            f"{SUPPORTED_EARLY_ARM_TARGETS}."
        )
    kwargs = {}
    if pose_provider_factory is not None:
        kwargs["pose_provider_factory"] = pose_provider_factory
    if cv2_module is not None:
        kwargs["cv2_module"] = cv2_module
    return prepare_completion_dataset(
        recordings,
        output_dir=output_dir,
        tracker_config=tracker_config,
        window_size=window_size,
        stride=stride,
        validation_recording_id=validation_recording_id,
        validation_fraction=validation_fraction,
        target=target,
        horizon_frames=horizon_frames,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for early-arm dataset preparation."""

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
        result = prepare_early_arm_dataset(
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


if __name__ == "__main__":
    main()

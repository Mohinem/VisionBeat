"""Sweep decoder settings for one fixed CNN checkpoint and prepared dataset."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from visionbeat.cnn_model import (
    load_completion_cnn_from_checkpoint,
    require_torch,
    resolve_device,
)
from visionbeat.cnn_trigger import decode_trigger_events, evaluate_decoded_triggers
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from visionbeat.predict_cnn import (
    evaluate_predictions,
    load_inference_dataset,
    run_inference,
    save_decoded_triggers_csv,
    save_inference_report,
    save_predictions_csv,
)

_DEFAULT_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
_DEFAULT_COOLDOWNS = (6, 8, 10)
_DEFAULT_MAX_GAPS = (1, 2)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """One decoder configuration in the sweep grid."""

    threshold: float
    cooldown_frames: int
    max_gap_frames: int

    @property
    def slug(self) -> str:
        return format_sweep_slug(self)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the decoder sweep helper."""

    parser = argparse.ArgumentParser(
        description="Run a decoder sweep over one prepared dataset and fixed checkpoint."
    )
    parser.add_argument("checkpoint", help="Path to the trained model checkpoint.")
    parser.add_argument("dataset", help="Path to the prepared NPZ dataset.")
    parser.add_argument(
        "--output-dir",
        default="outputs/decoder_sweeps/recording_3_baseline_sweep",
        help="Directory where sweep artifacts and the summary CSV are written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(_DEFAULT_THRESHOLDS),
        help="Threshold values to sweep. Default: 0.55 0.60 0.65 0.70.",
    )
    parser.add_argument(
        "--cooldowns",
        nargs="+",
        type=int,
        default=list(_DEFAULT_COOLDOWNS),
        help="Cooldown frame values to sweep. Default: 6 8 10.",
    )
    parser.add_argument(
        "--max-gaps",
        nargs="+",
        type=int,
        default=list(_DEFAULT_MAX_GAPS),
        help="Max-gap frame values to sweep. Default: 1 2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size. Default: 512.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--trigger-match-tolerance-frames",
        type=int,
        default=0,
        help="Optional extra tolerance when matching decoded triggers to labels.",
    )
    return parser.parse_args(argv)


def build_sweep_configs(
    *,
    thresholds: list[float],
    cooldowns: list[int],
    max_gaps: list[int],
) -> tuple[SweepConfig, ...]:
    """Build the full cartesian sweep grid with validated values."""

    if not thresholds:
        raise ValueError("thresholds must not be empty.")
    if not cooldowns:
        raise ValueError("cooldowns must not be empty.")
    if not max_gaps:
        raise ValueError("max_gaps must not be empty.")

    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")
    for cooldown in cooldowns:
        if cooldown < 0:
            raise ValueError(f"cooldown_frames must be greater than or equal to zero, got {cooldown}.")
    for max_gap in max_gaps:
        if max_gap < 0:
            raise ValueError(f"max_gap_frames must be greater than or equal to zero, got {max_gap}.")

    return tuple(
        SweepConfig(
            threshold=float(threshold),
            cooldown_frames=int(cooldown),
            max_gap_frames=int(max_gap),
        )
        for threshold, cooldown, max_gap in product(thresholds, cooldowns, max_gaps)
    )


def format_sweep_slug(config: SweepConfig) -> str:
    """Render a filesystem-safe label for one sweep configuration."""

    threshold_label = f"{config.threshold:.2f}".replace(".", "p")
    return (
        f"th_{threshold_label}"
        f"_cd_{config.cooldown_frames}"
        f"_gap_{config.max_gap_frames}"
    )


def _build_summary_row(
    *,
    dataset_sample_count: int,
    config: SweepConfig,
    predicted_labels: np.ndarray,
    metrics: dict[str, int | float | None] | None,
    decoded_trigger_metrics: dict[str, int | float] | None,
    predictions_path: Path,
    decoded_triggers_path: Path,
    report_path: Path,
) -> dict[str, int | float | str | None]:
    """Build one summary CSV row for a completed sweep configuration."""

    predicted_positive_count = int(np.sum(predicted_labels))
    predicted_positive_rate = predicted_positive_count / max(dataset_sample_count, 1)
    row: dict[str, int | float | str | None] = {
        "config_slug": config.slug,
        "threshold": config.threshold,
        "cooldown_frames": config.cooldown_frames,
        "max_gap_frames": config.max_gap_frames,
        "sample_count": dataset_sample_count,
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": predicted_positive_rate,
        "predictions_path": str(predictions_path),
        "decoded_triggers_path": str(decoded_triggers_path),
        "report_path": str(report_path),
    }
    if metrics is None:
        row.update(
            {
                "window_accuracy": None,
                "window_precision": None,
                "window_recall": None,
                "window_f1": None,
                "window_roc_auc": None,
                "window_true_negative": None,
                "window_false_positive": None,
                "window_false_negative": None,
                "window_true_positive": None,
            }
        )
    else:
        row.update(
            {
                "window_accuracy": metrics["accuracy"],
                "window_precision": metrics["precision"],
                "window_recall": metrics["recall"],
                "window_f1": metrics["f1"],
                "window_roc_auc": metrics["roc_auc"],
                "window_true_negative": metrics["true_negative"],
                "window_false_positive": metrics["false_positive"],
                "window_false_negative": metrics["false_negative"],
                "window_true_positive": metrics["true_positive"],
            }
        )
    if decoded_trigger_metrics is None:
        row.update(
            {
                "decoded_trigger_count": None,
                "decoded_trigger_precision": None,
                "decoded_trigger_recall": None,
                "decoded_trigger_f1": None,
                "positive_event_count": None,
                "detected_positive_event_count": None,
                "false_positive_trigger_count": None,
                "missed_positive_event_count": None,
            }
        )
    else:
        row.update(
            {
                "decoded_trigger_count": decoded_trigger_metrics["decoded_trigger_count"],
                "decoded_trigger_precision": decoded_trigger_metrics["precision"],
                "decoded_trigger_recall": decoded_trigger_metrics["recall"],
                "decoded_trigger_f1": decoded_trigger_metrics["f1"],
                "positive_event_count": decoded_trigger_metrics["positive_event_count"],
                "detected_positive_event_count": decoded_trigger_metrics[
                    "detected_positive_event_count"
                ],
                "false_positive_trigger_count": decoded_trigger_metrics[
                    "false_positive_trigger_count"
                ],
                "missed_positive_event_count": decoded_trigger_metrics[
                    "missed_positive_event_count"
                ],
            }
        )
    return row


def _save_summary_csv(
    *,
    path: Path,
    rows: list[dict[str, int | float | str | None]],
) -> None:
    """Save the compact one-row-per-config sweep summary."""

    if not rows:
        raise ValueError("rows must not be empty.")
    fieldnames = [
        "config_slug",
        "threshold",
        "cooldown_frames",
        "max_gap_frames",
        "sample_count",
        "predicted_positive_count",
        "predicted_positive_rate",
        "window_accuracy",
        "window_precision",
        "window_recall",
        "window_f1",
        "window_roc_auc",
        "window_true_negative",
        "window_false_positive",
        "window_false_negative",
        "window_true_positive",
        "decoded_trigger_count",
        "decoded_trigger_precision",
        "decoded_trigger_recall",
        "decoded_trigger_f1",
        "positive_event_count",
        "detected_positive_event_count",
        "false_positive_trigger_count",
        "missed_positive_event_count",
        "predictions_path",
        "decoded_triggers_path",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_manifest(
    *,
    path: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    thresholds: list[float],
    cooldowns: list[int],
    max_gaps: list[int],
    config_count: int,
) -> None:
    """Persist the sweep inputs for later reference."""

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "thresholds": thresholds,
        "cooldowns": cooldowns,
        "max_gaps": max_gaps,
        "config_count": config_count,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the decoder sweep helper."""

    args = parse_args(argv)
    try:
        if args.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if args.trigger_match_tolerance_frames < 0:
            raise ValueError("trigger_match_tolerance_frames must be greater than or equal to zero.")

        checkpoint_path = Path(args.checkpoint)
        dataset_path = Path(args.dataset)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        configs = build_sweep_configs(
            thresholds=list(args.thresholds),
            cooldowns=list(args.cooldowns),
            max_gaps=list(args.max_gaps),
        )

        dataset = load_inference_dataset(dataset_path)
        torch, nn, DataLoader, TensorDataset = require_torch()
        resolved_device = resolve_device(args.device, torch)
        runtime_feature_names = dataset.feature_names or FEATURE_NAMES
        runtime_schema_version = dataset.schema_version or FEATURE_SCHEMA_VERSION
        runtime_target_name = dataset.target_name or "completion_frame_binary"
        runtime_horizon_frames = dataset.horizon_frames
        model, model_spec, checkpoint = load_completion_cnn_from_checkpoint(
            checkpoint_path=checkpoint_path,
            torch=torch,
            nn=nn,
            device=resolved_device,
            runtime_feature_names=runtime_feature_names,
            runtime_schema_version=runtime_schema_version,
            runtime_window_size=dataset.input_shape[0],
            runtime_target_name=runtime_target_name,
            runtime_horizon_frames=runtime_horizon_frames,
        )
        probabilities, _ = run_inference(
            model=model,
            X=dataset.X,
            batch_size=args.batch_size,
            threshold=0.5,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            device=resolved_device,
        )

        summary_rows: list[dict[str, int | float | str | None]] = []
        for config in configs:
            predicted_labels = (probabilities >= config.threshold).astype(np.int64, copy=False)
            decoded_triggers = decode_trigger_events(
                recording_ids=dataset.recording_ids,
                window_end_frame_indices=dataset.window_end_frame_indices,
                window_end_timestamps_seconds=dataset.window_end_timestamps_seconds,
                probabilities=probabilities,
                threshold=config.threshold,
                cooldown_frames=config.cooldown_frames,
                max_gap_frames=config.max_gap_frames,
            )
            metrics = evaluate_predictions(
                y_true=dataset.y,
                probabilities=probabilities,
                predicted_labels=predicted_labels,
                threshold=config.threshold,
            )
            decoded_trigger_metrics = None
            if dataset.y is not None:
                decoded_trigger_metrics = evaluate_decoded_triggers(
                    decoded_triggers=decoded_triggers,
                    recording_ids=dataset.recording_ids,
                    window_end_frame_indices=dataset.window_end_frame_indices,
                    labels=dataset.y,
                    match_tolerance_frames=args.trigger_match_tolerance_frames,
                    max_gap_frames=config.max_gap_frames,
                )

            config_dir = output_dir / config.slug
            config_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = config_dir / f"{dataset.path.stem}_predictions.csv"
            decoded_triggers_path = config_dir / f"{dataset.path.stem}_decoded_triggers.csv"
            report_path = config_dir / f"{dataset.path.stem}_report.json"

            save_predictions_csv(
                path=predictions_path,
                dataset=dataset,
                probabilities=probabilities,
                predicted_labels=predicted_labels,
            )
            save_decoded_triggers_csv(
                path=decoded_triggers_path,
                decoded_triggers=decoded_triggers,
            )
            save_inference_report(
                path=report_path,
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                predictions_path=predictions_path,
                decoded_triggers_path=decoded_triggers_path,
                threshold=config.threshold,
                metrics=metrics,
                decoded_trigger_metrics=decoded_trigger_metrics,
                decoded_trigger_count=len(decoded_triggers),
                trigger_cooldown_frames=config.cooldown_frames,
                trigger_max_gap_frames=config.max_gap_frames,
                trigger_match_tolerance_frames=args.trigger_match_tolerance_frames,
                threshold_analysis=None,
                threshold_summary=None,
                threshold_analysis_path=None,
                model_metadata=model_spec.to_checkpoint_metadata(),
                checkpoint_format_version=str(checkpoint.get("checkpoint_format_version", "")),
            )
            summary_rows.append(
                _build_summary_row(
                    dataset_sample_count=dataset.sample_count,
                    config=config,
                    predicted_labels=predicted_labels,
                    metrics=metrics,
                    decoded_trigger_metrics=decoded_trigger_metrics,
                    predictions_path=predictions_path,
                    decoded_triggers_path=decoded_triggers_path,
                    report_path=report_path,
                )
            )

        summary_path = output_dir / f"{dataset.path.stem}_decoder_sweep_summary.csv"
        manifest_path = output_dir / "sweep_manifest.json"
        _save_summary_csv(path=summary_path, rows=summary_rows)
        _save_manifest(
            path=manifest_path,
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            thresholds=list(args.thresholds),
            cooldowns=list(args.cooldowns),
            max_gaps=list(args.max_gaps),
            config_count=len(configs),
        )

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Dataset: {dataset_path}")
        print(f"Output directory: {output_dir}")
        print(f"Configurations evaluated: {len(configs)}")
        print(f"Summary CSV: {summary_path}")
        best_row = max(
            summary_rows,
            key=lambda row: (
                float(row["decoded_trigger_f1"] or 0.0),
                float(row["decoded_trigger_precision"] or 0.0),
                float(row["decoded_trigger_recall"] or 0.0),
                -int(row["false_positive_trigger_count"] or 0),
            ),
        )
        print(
            "Best decoded-trigger F1: "
            f"{best_row['config_slug']} "
            f"(precision={float(best_row['decoded_trigger_precision'] or 0.0):.4f}, "
            f"recall={float(best_row['decoded_trigger_recall'] or 0.0):.4f}, "
            f"f1={float(best_row['decoded_trigger_f1'] or 0.0):.4f}, "
            f"false_triggers={int(best_row['false_positive_trigger_count'] or 0)}, "
            f"missed_events={int(best_row['missed_positive_event_count'] or 0)})"
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

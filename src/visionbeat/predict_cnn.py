"""Run offline inference with a trained VisionBeat CNN on one prepared NPZ dataset."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionbeat.cnn_trigger import (
    DecodedTriggerEvent,
    decode_trigger_events,
    evaluate_decoded_triggers,
)
from visionbeat.cnn_model import (
    binary_classification_metrics,
    format_optional_metric,
    load_completion_cnn_from_checkpoint,
    require_torch,
    resolve_device,
)
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class InferenceDataset:
    """Prepared window dataset used for offline CNN inference."""

    path: Path
    X: np.ndarray
    recording_ids: np.ndarray
    window_end_frame_indices: np.ndarray
    window_end_timestamps_seconds: np.ndarray
    y: np.ndarray | None
    target_gesture_labels: np.ndarray
    feature_names: tuple[str, ...] | None
    schema_version: str | None
    target_name: str | None
    horizon_frames: int
    stride: int

    @property
    def sample_count(self) -> int:
        return int(self.X.shape[0])

    @property
    def input_shape(self) -> tuple[int, int]:
        return (int(self.X.shape[1]), int(self.X.shape[2]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for offline CNN inference."""

    parser = argparse.ArgumentParser(
        description="Run offline inference with a trained VisionBeat CNN on one NPZ dataset."
    )
    parser.add_argument("checkpoint", help="Path to the trained model checkpoint.")
    parser.add_argument("dataset", help="Path to the prepared NPZ window dataset.")
    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory used for the predictions CSV and JSON report.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size. Default: 512.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used to convert probabilities into labels. Default: 0.5.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--threshold-analysis-start",
        type=float,
        default=0.1,
        help="Start of the threshold sweep when labels are available. Default: 0.1.",
    )
    parser.add_argument(
        "--threshold-analysis-stop",
        type=float,
        default=0.9,
        help="End of the threshold sweep when labels are available. Default: 0.9.",
    )
    parser.add_argument(
        "--threshold-analysis-step",
        type=float,
        default=0.1,
        help="Step size of the threshold sweep when labels are available. Default: 0.1.",
    )
    parser.add_argument(
        "--trigger-cooldown-frames",
        type=int,
        default=-1,
        help=(
            "Cooldown in frames applied after decoding one trigger event. "
            "Defaults to half the dataset window size."
        ),
    )
    parser.add_argument(
        "--trigger-max-gap-frames",
        type=int,
        default=-1,
        help=(
            "Maximum frame gap used to merge nearby above-threshold windows into one local peak. "
            "Defaults to the dataset stride."
        ),
    )
    parser.add_argument(
        "--trigger-match-tolerance-frames",
        type=int,
        default=0,
        help=(
            "Extra frame tolerance when matching decoded triggers to labeled positive event neighborhoods. "
            "Default: 0."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for offline inference."""

    args = parse_args(argv)
    try:
        if args.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0].")
        if args.threshold_analysis_step <= 0.0:
            raise ValueError("threshold_analysis_step must be greater than zero.")
        if args.threshold_analysis_start > args.threshold_analysis_stop:
            raise ValueError(
                "threshold_analysis_start must be less than or equal to threshold_analysis_stop."
            )
        if args.trigger_cooldown_frames < -1:
            raise ValueError("trigger_cooldown_frames must be -1 or greater.")
        if args.trigger_max_gap_frames < -1:
            raise ValueError("trigger_max_gap_frames must be -1 or greater.")
        if args.trigger_match_tolerance_frames < 0:
            raise ValueError("trigger_match_tolerance_frames must be greater than or equal to zero.")

        dataset = load_inference_dataset(Path(args.dataset))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch, nn, DataLoader, TensorDataset = require_torch()
        resolved_device = resolve_device(args.device, torch)
        runtime_feature_names = dataset.feature_names or FEATURE_NAMES
        runtime_schema_version = dataset.schema_version or FEATURE_SCHEMA_VERSION
        runtime_target_name = dataset.target_name or "completion_frame_binary"
        runtime_horizon_frames = dataset.horizon_frames
        model, model_spec, checkpoint = load_completion_cnn_from_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            torch=torch,
            nn=nn,
            device=resolved_device,
            runtime_feature_names=runtime_feature_names,
            runtime_schema_version=runtime_schema_version,
            runtime_window_size=dataset.input_shape[0],
            runtime_target_name=runtime_target_name,
            runtime_horizon_frames=runtime_horizon_frames,
        )
        probabilities, predicted_labels = run_inference(
            model=model,
            X=dataset.X,
            batch_size=args.batch_size,
            threshold=args.threshold,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            device=resolved_device,
        )
        resolved_trigger_cooldown_frames = (
            max(dataset.input_shape[0] // 2, 0)
            if args.trigger_cooldown_frames < 0
            else args.trigger_cooldown_frames
        )
        resolved_trigger_max_gap_frames = (
            max(dataset.stride, 1)
            if args.trigger_max_gap_frames < 0
            else args.trigger_max_gap_frames
        )
        decoded_triggers = decode_trigger_events(
            recording_ids=dataset.recording_ids,
            window_end_frame_indices=dataset.window_end_frame_indices,
            window_end_timestamps_seconds=dataset.window_end_timestamps_seconds,
            probabilities=probabilities,
            threshold=args.threshold,
            cooldown_frames=resolved_trigger_cooldown_frames,
            max_gap_frames=resolved_trigger_max_gap_frames,
        )
        metrics = evaluate_predictions(
            y_true=dataset.y,
            probabilities=probabilities,
            predicted_labels=predicted_labels,
            threshold=args.threshold,
        )
        decoded_trigger_metrics = None
        if dataset.y is not None:
            decoded_trigger_metrics = evaluate_decoded_triggers(
                decoded_triggers=decoded_triggers,
                recording_ids=dataset.recording_ids,
                window_end_frame_indices=dataset.window_end_frame_indices,
                labels=dataset.y,
                match_tolerance_frames=args.trigger_match_tolerance_frames,
                max_gap_frames=resolved_trigger_max_gap_frames,
            )
        threshold_analysis = None
        threshold_summary = None
        threshold_analysis_path = None
        if dataset.y is not None:
            thresholds = build_threshold_grid(
                start=args.threshold_analysis_start,
                stop=args.threshold_analysis_stop,
                step=args.threshold_analysis_step,
            )
            threshold_analysis = analyze_thresholds(
                y_true=dataset.y,
                probabilities=probabilities,
                thresholds=thresholds,
            )
            threshold_summary = summarize_threshold_analysis(
                analysis_rows=threshold_analysis,
                selected_threshold=args.threshold,
            )
            threshold_analysis_path = output_dir / f"{dataset.path.stem}_threshold_analysis.csv"
            save_threshold_analysis_csv(
                path=threshold_analysis_path,
                analysis_rows=threshold_analysis,
            )

        predictions_path = output_dir / f"{dataset.path.stem}_predictions.csv"
        decoded_triggers_path = output_dir / f"{dataset.path.stem}_decoded_triggers.csv"
        report_path = output_dir / f"{dataset.path.stem}_report.json"
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
            checkpoint_path=Path(args.checkpoint),
            predictions_path=predictions_path,
            decoded_triggers_path=decoded_triggers_path,
            threshold=args.threshold,
            metrics=metrics,
            decoded_trigger_metrics=decoded_trigger_metrics,
            decoded_trigger_count=len(decoded_triggers),
            trigger_cooldown_frames=resolved_trigger_cooldown_frames,
            trigger_max_gap_frames=resolved_trigger_max_gap_frames,
            trigger_match_tolerance_frames=args.trigger_match_tolerance_frames,
            threshold_analysis=threshold_analysis,
            threshold_summary=threshold_summary,
            threshold_analysis_path=threshold_analysis_path,
            model_metadata=model_spec.to_checkpoint_metadata(),
            checkpoint_format_version=str(checkpoint.get("checkpoint_format_version", "")),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Dataset: {dataset.path}")
    print(f"Samples: {dataset.sample_count}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Decoded triggers saved: {decoded_triggers_path}")
    print(f"Report saved: {report_path}")
    print(
        "Predicted positives: "
        f"{int(predicted_labels.sum())} / {dataset.sample_count} "
        f"({predicted_labels.mean():.2%})"
    )
    print(
        "Decoded triggers: "
        f"{len(decoded_triggers)} "
        f"(cooldown_frames={resolved_trigger_cooldown_frames}, "
        f"max_gap_frames={resolved_trigger_max_gap_frames})"
    )
    if metrics is None:
        print("Evaluation: skipped because ground-truth labels were not found in the dataset.")
        return

    print(
        "Evaluation: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"roc_auc={format_optional_metric(metrics['roc_auc'])}"
    )
    print(
        "  "
        f"confusion=TN:{metrics['true_negative']} "
        f"FP:{metrics['false_positive']} "
        f"FN:{metrics['false_negative']} "
        f"TP:{metrics['true_positive']}"
    )
    if decoded_trigger_metrics is not None:
        print(
            "Decoded trigger evaluation: "
            f"precision={decoded_trigger_metrics['precision']:.4f} "
            f"recall={decoded_trigger_metrics['recall']:.4f} "
            f"f1={decoded_trigger_metrics['f1']:.4f} "
            f"detected_events={decoded_trigger_metrics['detected_positive_event_count']} "
            f"false_triggers={decoded_trigger_metrics['false_positive_trigger_count']} "
            f"missed_events={decoded_trigger_metrics['missed_positive_event_count']}"
        )
    if threshold_summary is not None:
        print("Threshold analysis:")
        print("  threshold  precision  recall  f1      false_pos  false_neg")
        for row in threshold_analysis:
            print(
                "  "
                f"{row['threshold']:.2f}       "
                f"{row['precision']:.4f}     "
                f"{row['recall']:.4f}  "
                f"{row['f1']:.4f}  "
                f"{row['false_positive']:>9d}  "
                f"{row['false_negative']:>9d}"
            )
        best_threshold = threshold_summary["best_f1_threshold"]
        lowest_fp_threshold = threshold_summary["lowest_false_positive_threshold"]
        print(
            "Best threshold by F1: "
            f"{best_threshold['threshold']:.2f} "
            f"(precision={best_threshold['precision']:.4f}, "
            f"recall={best_threshold['recall']:.4f}, "
            f"f1={best_threshold['f1']:.4f}, "
            f"false_pos={best_threshold['false_positive']}, "
            f"false_neg={best_threshold['false_negative']})"
        )
        print(
            "More conservative low-FP threshold: "
            f"{lowest_fp_threshold['threshold']:.2f} "
            f"(precision={lowest_fp_threshold['precision']:.4f}, "
            f"recall={lowest_fp_threshold['recall']:.4f}, "
            f"f1={lowest_fp_threshold['f1']:.4f}, "
            f"false_pos={lowest_fp_threshold['false_positive']}, "
            f"false_neg={lowest_fp_threshold['false_negative']})"
        )
        if threshold_analysis_path is not None:
            print(f"Threshold analysis saved: {threshold_analysis_path}")


def load_inference_dataset(path: Path) -> InferenceDataset:
    """Load one prepared NPZ dataset for inference, allowing optional labels."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if "X" not in archive.files:
            raise ValueError(f"Dataset {path} is missing required key 'X'.")
        X = np.asarray(archive["X"], dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Dataset {path} has invalid X shape {X.shape}; expected rank 3.")
        sample_count = int(X.shape[0])

        recording_ids = _load_optional_array(
            archive=archive,
            key="recording_ids",
            default=np.asarray([""] * sample_count, dtype="<U128"),
        )
        window_end_frame_indices = _load_optional_array(
            archive=archive,
            key="window_end_frame_indices",
            default=np.arange(sample_count, dtype=np.int64),
            dtype=np.int64,
        )
        window_end_timestamps_seconds = _load_optional_array(
            archive=archive,
            key="window_end_timestamps_seconds",
            default=np.full(sample_count, np.nan, dtype=np.float32),
            dtype=np.float32,
        )
        target_gesture_labels = _load_optional_array(
            archive=archive,
            key="target_gesture_labels",
            default=np.asarray([""] * sample_count, dtype="<U128"),
        )

        y = None
        if "y" in archive.files:
            y = np.asarray(archive["y"], dtype=np.int64)
            if y.ndim != 1 or y.shape[0] != sample_count:
                raise ValueError(
                    f"Dataset {path} has invalid y shape {y.shape}; expected ({sample_count},)."
                )
            unique_targets = set(np.unique(y).tolist())
            if not unique_targets <= {0, 1}:
                raise ValueError(
                    f"Dataset {path} has unsupported targets {sorted(unique_targets)}; expected binary 0/1."
                )

        feature_names = None
        if "feature_names" in archive.files:
            feature_names = tuple(str(name) for name in archive["feature_names"].tolist())
            if feature_names != FEATURE_NAMES:
                raise ValueError(
                    f"Dataset {path} feature_names do not match the canonical schema."
                )
        schema_version = (
            str(archive["schema_version"].item()) if "schema_version" in archive.files else None
        )
        if schema_version is not None and schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Dataset {path} schema_version mismatch. Expected {FEATURE_SCHEMA_VERSION}, got {schema_version}."
            )
        target_name = str(archive["target_name"].item()) if "target_name" in archive.files else None
        if target_name is not None and target_name not in {
            "completion_frame_binary",
            "completion_within_next_k_frames",
            "completion_within_last_k_frames",
            "arm_frame_binary",
            "arm_within_next_k_frames",
            "arm_within_last_k_frames",
        }:
            raise ValueError(
                f"Dataset {path} has unsupported target_name {target_name!r}. "
                "Expected a supported VisionBeat timing target."
            )
        horizon_frames = (
            int(archive["horizon_frames"].item()) if "horizon_frames" in archive.files else 0
        )
        if target_name in {
            "completion_within_next_k_frames",
            "completion_within_last_k_frames",
            "arm_within_next_k_frames",
            "arm_within_last_k_frames",
        } and horizon_frames <= 0:
            raise ValueError(
                f"Dataset {path} has invalid horizon_frames {horizon_frames} for tolerant timing targets."
            )
        stride = int(archive["stride"].item()) if "stride" in archive.files else 1
        if stride <= 0:
            raise ValueError(f"Dataset {path} has invalid stride {stride}.")

    if X.shape[2] != len(FEATURE_NAMES):
        raise ValueError(
            f"Dataset {path} feature dimension mismatch. Expected {len(FEATURE_NAMES)}, got {X.shape[2]}."
        )
    if not np.isfinite(X).all():
        raise ValueError(f"Dataset {path} contains non-finite feature values.")

    return InferenceDataset(
        path=path,
        X=X,
        recording_ids=np.asarray(recording_ids),
        window_end_frame_indices=np.asarray(window_end_frame_indices, dtype=np.int64),
        window_end_timestamps_seconds=np.asarray(
            window_end_timestamps_seconds,
            dtype=np.float32,
        ),
        y=y,
        target_gesture_labels=np.asarray(target_gesture_labels),
        feature_names=feature_names,
        schema_version=schema_version,
        target_name=target_name,
        horizon_frames=horizon_frames,
        stride=stride,
    )


def _load_optional_array(
    *,
    archive: Any,
    key: str,
    default: np.ndarray,
    dtype: Any | None = None,
) -> np.ndarray:
    """Load an optional sample-aligned array or use a validated default."""

    values = np.asarray(archive[key], dtype=dtype) if key in archive.files else np.asarray(default)
    if values.ndim != 1 or values.shape[0] != default.shape[0]:
        raise ValueError(
            f"Dataset metadata key {key!r} has invalid shape {values.shape}; expected ({default.shape[0]},)."
        )
    return values


def run_inference(
    *,
    model: Any,
    X: np.ndarray,
    batch_size: int,
    threshold: float,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the checkpoint on all windows and return probabilities and labels."""

    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            logits = model(features)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities.append(batch_probabilities)
    all_probabilities = np.concatenate(probabilities, axis=0).astype(np.float32, copy=False)
    predicted_labels = (all_probabilities >= threshold).astype(np.int64, copy=False)
    return all_probabilities, predicted_labels


def evaluate_predictions(
    *,
    y_true: np.ndarray | None,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    threshold: float,
) -> dict[str, int | float | None] | None:
    """Compute basic evaluation metrics when ground truth is available."""

    if y_true is None:
        return None
    metrics = binary_classification_metrics(y_true, probabilities, threshold=threshold)
    if int(np.sum(predicted_labels)) != int(metrics["predicted_positive_count"]):
        raise ValueError("Predicted labels are inconsistent with the evaluation metrics.")
    return metrics


def build_threshold_grid(*, start: float, stop: float, step: float) -> list[float]:
    """Build a stable inclusive threshold grid."""

    thresholds: list[float] = []
    current = start
    while current <= stop + 1e-8:
        thresholds.append(round(current, 6))
        current += step
    if not thresholds:
        raise ValueError("Threshold analysis grid is empty.")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold analysis value {threshold} is outside [0.0, 1.0].")
    return thresholds


def analyze_thresholds(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, int | float | None]]:
    """Evaluate a set of trigger thresholds on one labeled inference dataset."""

    analysis_rows: list[dict[str, int | float | None]] = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, probabilities, threshold=threshold)
        analysis_rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "predicted_positive_count": metrics["predicted_positive_count"],
            }
        )
    return analysis_rows


def summarize_threshold_analysis(
    *,
    analysis_rows: list[dict[str, int | float | None]],
    selected_threshold: float,
) -> dict[str, dict[str, int | float | None]]:
    """Pick the best-F1 and most conservative low-FP thresholds."""

    if not analysis_rows:
        raise ValueError("threshold analysis rows must not be empty.")
    best_f1_threshold = max(
        analysis_rows,
        key=lambda row: (
            float(row["f1"]),
            -int(row["false_positive"]),
            -int(row["false_negative"]),
            float(row["threshold"]),
        ),
    )
    lower_false_positive_candidates = [
        row for row in analysis_rows if float(row["recall"]) > 0.0
    ]
    if not lower_false_positive_candidates:
        lower_false_positive_candidates = analysis_rows
    lowest_false_positive_threshold = min(
        lower_false_positive_candidates,
        key=lambda row: (
            int(row["false_positive"]),
            int(row["false_negative"]),
            -float(row["precision"]),
            float(row["threshold"]),
        ),
    )
    selected = min(
        analysis_rows,
        key=lambda row: abs(float(row["threshold"]) - selected_threshold),
    )
    return {
        "best_f1_threshold": dict(best_f1_threshold),
        "lowest_false_positive_threshold": dict(lowest_false_positive_threshold),
        "selected_threshold": dict(selected),
    }


def save_threshold_analysis_csv(
    *,
    path: Path,
    analysis_rows: list[dict[str, int | float | None]],
) -> None:
    """Save the threshold sweep table for later inspection."""

    fieldnames = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive",
        "false_negative",
        "predicted_positive_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in analysis_rows:
            writer.writerow(row)


def save_predictions_csv(
    *,
    path: Path,
    dataset: InferenceDataset,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Save one row per window with prediction outputs and optional labels."""

    fieldnames = [
        "window_index",
        "recording_id",
        "window_end_frame_index",
        "window_end_timestamp_seconds",
        "predicted_probability",
        "predicted_label",
        "true_label",
        "target_gesture_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(dataset.sample_count):
            writer.writerow(
                {
                    "window_index": index,
                    "recording_id": str(dataset.recording_ids[index]),
                    "window_end_frame_index": int(dataset.window_end_frame_indices[index]),
                    "window_end_timestamp_seconds": float(
                        dataset.window_end_timestamps_seconds[index]
                    ),
                    "predicted_probability": float(probabilities[index]),
                    "predicted_label": int(predicted_labels[index]),
                    "true_label": (
                        "" if dataset.y is None else int(dataset.y[index])
                    ),
                    "target_gesture_label": str(dataset.target_gesture_labels[index]),
                }
            )


def save_decoded_triggers_csv(
    *,
    path: Path,
    decoded_triggers: tuple[DecodedTriggerEvent, ...],
) -> None:
    """Save one row per decoded trigger event."""

    fieldnames = [
        "trigger_index",
        "recording_id",
        "window_index",
        "window_end_frame_index",
        "window_end_timestamp_seconds",
        "probability",
        "threshold",
        "run_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trigger_index, event in enumerate(decoded_triggers):
            writer.writerow(
                {
                    "trigger_index": trigger_index,
                    "recording_id": event.recording_id,
                    "window_index": event.window_index,
                    "window_end_frame_index": event.window_end_frame_index,
                    "window_end_timestamp_seconds": event.window_end_timestamp_seconds,
                    "probability": event.probability,
                    "threshold": event.threshold,
                    "run_length": event.run_length,
                }
            )


def save_inference_report(
    *,
    path: Path,
    dataset: InferenceDataset,
    checkpoint_path: Path,
    predictions_path: Path,
    decoded_triggers_path: Path,
    threshold: float,
    metrics: dict[str, int | float | None] | None,
    decoded_trigger_metrics: dict[str, int | float] | None,
    decoded_trigger_count: int,
    trigger_cooldown_frames: int,
    trigger_max_gap_frames: int,
    trigger_match_tolerance_frames: int,
    threshold_analysis: list[dict[str, int | float | None]] | None,
    threshold_summary: dict[str, dict[str, int | float | None]] | None,
    threshold_analysis_path: Path | None,
    model_metadata: dict[str, Any],
    checkpoint_format_version: str,
) -> None:
    """Save a compact JSON report for offline inference runs."""

    positive_meaning = _describe_positive_target_meaning(
        target_name=dataset.target_name or "completion_frame_binary",
        horizon_frames=dataset.horizon_frames,
    )
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format_version": checkpoint_format_version,
        "model_metadata": model_metadata,
        "dataset_path": str(dataset.path),
        "predictions_path": str(predictions_path),
        "decoded_triggers_path": str(decoded_triggers_path),
        "sample_count": dataset.sample_count,
        "input_shape": list(dataset.input_shape),
        "schema_version": dataset.schema_version,
        "target_name": dataset.target_name,
        "horizon_frames": dataset.horizon_frames,
        "threshold": threshold,
        "decoded_trigger_count": decoded_trigger_count,
        "decoded_trigger_config": {
            "cooldown_frames": trigger_cooldown_frames,
            "max_gap_frames": trigger_max_gap_frames,
            "match_tolerance_frames": trigger_match_tolerance_frames,
        },
        "report_meaning": (
            "Window-level metrics use the raw model output. "
            f"A positive window means {positive_meaning}. "
            "Decoded trigger metrics collapse above-threshold neighborhoods into one trigger event."
        ),
        "evaluation_available": metrics is not None,
        "metrics": metrics,
        "decoded_trigger_metrics": decoded_trigger_metrics,
        "threshold_analysis_path": (
            None if threshold_analysis_path is None else str(threshold_analysis_path)
        ),
        "threshold_analysis": threshold_analysis,
        "threshold_summary": threshold_summary,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _describe_positive_target_meaning(*, target_name: str, horizon_frames: int) -> str:
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


if __name__ == "__main__":
    main()

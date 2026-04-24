"""Run timing inference, decode triggers, and assign kick/snare labels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionbeat.cnn_model import load_completion_cnn_from_checkpoint
from visionbeat.cnn_trigger import (
    DecodedTriggerEvent,
    decode_trigger_events,
    evaluate_decoded_triggers,
    group_positive_event_spans,
)
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from visionbeat.gesture_classifier import (
    load_gesture_classifier_from_checkpoint,
    multiclass_classification_metrics,
    require_torch,
    resolve_device,
)
from visionbeat.predict_cnn import (
    analyze_thresholds,
    build_threshold_grid,
    evaluate_predictions,
    load_inference_dataset,
    run_inference,
    save_decoded_triggers_csv,
    save_predictions_csv,
    save_threshold_analysis_csv,
    summarize_threshold_analysis,
)


@dataclass(frozen=True, slots=True)
class PositiveGestureSpan:
    """One grouped positive event span with its gesture label."""

    recording_id: str
    start_frame_index: int
    end_frame_index: int
    gesture_label: str


@dataclass(frozen=True, slots=True)
class MatchedTriggerGestureRow:
    """One decoded trigger matched to a labeled positive gesture span."""

    trigger_index: int
    recording_id: str
    window_index: int
    trigger_frame_index: int
    trigger_timestamp_seconds: float
    timing_probability: float
    predicted_gesture_label: str
    predicted_gesture_confidence: float
    true_gesture_label: str
    correct: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the combined timing-plus-gesture pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked timing model, decode trigger events, and assign kick/snare "
            "labels with a second-stage classifier."
        )
    )
    parser.add_argument("timing_checkpoint", help="Path to the trained timing-model checkpoint.")
    parser.add_argument("dataset", help="Path to the prepared NPZ window dataset.")
    parser.add_argument(
        "gesture_checkpoint",
        help="Path to the trained kick/snare classifier checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gesture_classification",
        help="Directory for predictions, decoded triggers, and the combined report.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size for both models. Default: 512.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Timing-model decision threshold. Default: 0.6.",
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
        help="Start of the timing threshold sweep when labels are available. Default: 0.1.",
    )
    parser.add_argument(
        "--threshold-analysis-stop",
        type=float,
        default=0.9,
        help="End of the timing threshold sweep when labels are available. Default: 0.9.",
    )
    parser.add_argument(
        "--threshold-analysis-step",
        type=float,
        default=0.1,
        help="Step size of the timing threshold sweep when labels are available. Default: 0.1.",
    )
    parser.add_argument(
        "--trigger-cooldown-frames",
        type=int,
        default=6,
        help="Cooldown in frames after decoding one trigger event. Default: 6.",
    )
    parser.add_argument(
        "--trigger-max-gap-frames",
        type=int,
        default=1,
        help="Maximum frame gap used to merge nearby timing windows. Default: 1.",
    )
    parser.add_argument(
        "--trigger-match-tolerance-frames",
        type=int,
        default=0,
        help="Extra frame tolerance used when matching decoded triggers to labels. Default: 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for timing-front-end plus gesture classification inference."""
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
        if args.trigger_cooldown_frames < 0:
            raise ValueError("trigger_cooldown_frames must be greater than or equal to zero.")
        if args.trigger_max_gap_frames < 0:
            raise ValueError("trigger_max_gap_frames must be greater than or equal to zero.")
        if args.trigger_match_tolerance_frames < 0:
            raise ValueError(
                "trigger_match_tolerance_frames must be greater than or equal to zero."
            )

        dataset = load_inference_dataset(Path(args.dataset))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch, nn, DataLoader, TensorDataset = require_torch()
        resolved_device = resolve_device(args.device, torch)
        runtime_feature_names = dataset.feature_names or FEATURE_NAMES
        runtime_schema_version = dataset.schema_version or FEATURE_SCHEMA_VERSION
        runtime_target_name = dataset.target_name or "completion_frame_binary"
        timing_model, timing_spec, timing_checkpoint = load_completion_cnn_from_checkpoint(
            checkpoint_path=Path(args.timing_checkpoint),
            torch=torch,
            nn=nn,
            device=resolved_device,
            runtime_feature_names=runtime_feature_names,
            runtime_schema_version=runtime_schema_version,
            runtime_window_size=dataset.input_shape[0],
            runtime_target_name=runtime_target_name,
            runtime_horizon_frames=dataset.horizon_frames,
        )
        timing_probabilities, predicted_labels = run_inference(
            model=timing_model,
            X=dataset.X,
            batch_size=args.batch_size,
            threshold=args.threshold,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            device=resolved_device,
        )
        decoded_triggers = decode_trigger_events(
            recording_ids=dataset.recording_ids,
            window_end_frame_indices=dataset.window_end_frame_indices,
            window_end_timestamps_seconds=dataset.window_end_timestamps_seconds,
            probabilities=timing_probabilities,
            threshold=args.threshold,
            cooldown_frames=args.trigger_cooldown_frames,
            max_gap_frames=args.trigger_max_gap_frames,
        )

        gesture_model, gesture_spec, gesture_checkpoint = load_gesture_classifier_from_checkpoint(
            checkpoint_path=Path(args.gesture_checkpoint),
            torch=torch,
            nn=nn,
            device=resolved_device,
            runtime_feature_names=runtime_feature_names,
            runtime_schema_version=runtime_schema_version,
            runtime_window_size=dataset.input_shape[0],
        )
        classified_triggers = classify_decoded_triggers(
            dataset=dataset,
            decoded_triggers=decoded_triggers,
            model=gesture_model,
            batch_size=args.batch_size,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            device=resolved_device,
            class_labels=gesture_spec.class_labels,
        )

        predictions_path = output_dir / f"{dataset.path.stem}_timing_predictions.csv"
        decoded_triggers_path = output_dir / f"{dataset.path.stem}_decoded_triggers.csv"
        classified_triggers_path = output_dir / f"{dataset.path.stem}_decoded_trigger_classes.csv"
        report_path = output_dir / f"{dataset.path.stem}_gesture_pipeline_report.json"
        save_predictions_csv(
            path=predictions_path,
            dataset=dataset,
            probabilities=timing_probabilities,
            predicted_labels=predicted_labels,
        )
        save_decoded_triggers_csv(
            path=decoded_triggers_path,
            decoded_triggers=decoded_triggers,
        )
        save_classified_triggers_csv(
            path=classified_triggers_path,
            rows=classified_triggers,
        )

        timing_metrics = evaluate_predictions(
            y_true=dataset.y,
            probabilities=timing_probabilities,
            predicted_labels=predicted_labels,
            threshold=args.threshold,
        )
        decoded_trigger_metrics = None
        threshold_analysis = None
        threshold_summary = None
        threshold_analysis_path = None
        gesture_pipeline_metrics = None
        if dataset.y is not None:
            decoded_trigger_metrics = evaluate_decoded_triggers(
                decoded_triggers=decoded_triggers,
                recording_ids=dataset.recording_ids,
                window_end_frame_indices=dataset.window_end_frame_indices,
                labels=dataset.y,
                match_tolerance_frames=args.trigger_match_tolerance_frames,
                max_gap_frames=args.trigger_max_gap_frames,
            )
            positive_spans = build_positive_gesture_spans(
                recording_ids=dataset.recording_ids,
                window_end_frame_indices=dataset.window_end_frame_indices,
                labels=dataset.y,
                target_gesture_labels=dataset.target_gesture_labels,
                max_gap_frames=args.trigger_max_gap_frames,
            )
            matched_rows = match_classified_triggers_to_positive_spans(
                rows=classified_triggers,
                positive_spans=positive_spans,
                match_tolerance_frames=args.trigger_match_tolerance_frames,
            )
            gesture_pipeline_metrics = summarize_matched_gesture_predictions(
                matched_rows=matched_rows,
                false_trigger_count=max(len(classified_triggers) - len(matched_rows), 0),
                positive_event_count=len(positive_spans),
                class_labels=gesture_spec.class_labels,
            )
            matched_rows_path = output_dir / f"{dataset.path.stem}_matched_trigger_gestures.csv"
            save_matched_gesture_rows_csv(path=matched_rows_path, rows=matched_rows)

            thresholds = build_threshold_grid(
                start=args.threshold_analysis_start,
                stop=args.threshold_analysis_stop,
                step=args.threshold_analysis_step,
            )
            threshold_analysis = analyze_thresholds(
                y_true=dataset.y,
                probabilities=timing_probabilities,
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

        save_combined_report(
            path=report_path,
            dataset=dataset,
            timing_checkpoint_path=Path(args.timing_checkpoint),
            gesture_checkpoint_path=Path(args.gesture_checkpoint),
            predictions_path=predictions_path,
            decoded_triggers_path=decoded_triggers_path,
            classified_triggers_path=classified_triggers_path,
            threshold=args.threshold,
            timing_metrics=timing_metrics,
            decoded_trigger_metrics=decoded_trigger_metrics,
            gesture_pipeline_metrics=gesture_pipeline_metrics,
            decoded_trigger_count=len(decoded_triggers),
            trigger_cooldown_frames=args.trigger_cooldown_frames,
            trigger_max_gap_frames=args.trigger_max_gap_frames,
            trigger_match_tolerance_frames=args.trigger_match_tolerance_frames,
            threshold_analysis=threshold_analysis,
            threshold_summary=threshold_summary,
            threshold_analysis_path=threshold_analysis_path,
            timing_model_metadata=timing_spec.to_checkpoint_metadata(),
            timing_checkpoint_format_version=str(
                timing_checkpoint.get("checkpoint_format_version", "")
            ),
            gesture_model_metadata=gesture_spec.to_checkpoint_metadata(),
            gesture_checkpoint_format_version=str(
                gesture_checkpoint.get("checkpoint_format_version", "")
            ),
        )

        print(f"Dataset: {dataset.path}")
        print(f"Decoded triggers: {len(decoded_triggers)}")
        print(f"Timing predictions saved: {predictions_path}")
        print(f"Decoded triggers saved: {decoded_triggers_path}")
        print(f"Classified triggers saved: {classified_triggers_path}")
        print(f"Report saved: {report_path}")
        if gesture_pipeline_metrics is not None:
            print(
                "Matched gesture typing: "
                f"accuracy={float(gesture_pipeline_metrics['matched_accuracy']):.4f} "
                "correctly_typed_events="
                f"{int(gesture_pipeline_metrics['correctly_typed_event_count'])} "
                f"of {int(gesture_pipeline_metrics['positive_event_count'])}"
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def classify_decoded_triggers(
    *,
    dataset: Any,
    decoded_triggers: tuple[DecodedTriggerEvent, ...],
    model: Any,
    batch_size: int,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    device: Any,
    class_labels: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Assign kick/snare labels to decoded timing triggers."""
    if not decoded_triggers:
        return ()
    window_indices = np.asarray(
        [trigger.window_index for trigger in decoded_triggers],
        dtype=np.int64,
    )
    X = dataset.X[window_indices].astype(np.float32, copy=False)
    tensor = torch.from_numpy(X)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    probability_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            logits = model(batch_X.to(device))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(probability_batches, axis=0)

    rows: list[dict[str, Any]] = []
    for trigger_index, (trigger, class_probability) in enumerate(
        zip(decoded_triggers, probabilities, strict=True)
    ):
        predicted_class_index = int(np.argmax(class_probability))
        predicted_label = class_labels[predicted_class_index]
        row = {
            "trigger_index": trigger_index,
            "recording_id": trigger.recording_id,
            "window_index": trigger.window_index,
            "window_end_frame_index": trigger.window_end_frame_index,
            "window_end_timestamp_seconds": trigger.window_end_timestamp_seconds,
            "timing_probability": trigger.probability,
            "timing_threshold": trigger.threshold,
            "run_length": trigger.run_length,
            "predicted_gesture_label": predicted_label,
            "predicted_gesture_confidence": float(class_probability[predicted_class_index]),
        }
        for class_index, class_label in enumerate(class_labels):
            row[f"{class_label}_probability"] = float(class_probability[class_index])
        rows.append(row)
    return tuple(rows)


def build_positive_gesture_spans(
    *,
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
    labels: np.ndarray,
    target_gesture_labels: np.ndarray,
    max_gap_frames: int,
) -> tuple[PositiveGestureSpan, ...]:
    """Attach gesture labels to grouped positive event spans."""
    positive_spans = group_positive_event_spans(
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        labels=labels,
        max_gap_frames=max_gap_frames,
    )
    labeled_spans: list[PositiveGestureSpan] = []
    for span in positive_spans:
        span_mask = (
            (recording_ids == span.recording_id)
            & (window_end_frame_indices >= span.start_frame_index)
            & (window_end_frame_indices <= span.end_frame_index)
            & (labels == 1)
        )
        gesture_values = sorted(
            {
                str(label).strip().lower()
                for label in target_gesture_labels[span_mask].tolist()
                if str(label).strip()
            }
        )
        if len(gesture_values) != 1:
            raise ValueError(
                "Expected exactly one gesture label per positive span, got "
                f"{gesture_values!r} for {span.recording_id} "
                f"{span.start_frame_index}-{span.end_frame_index}."
            )
        labeled_spans.append(
            PositiveGestureSpan(
                recording_id=span.recording_id,
                start_frame_index=span.start_frame_index,
                end_frame_index=span.end_frame_index,
                gesture_label=gesture_values[0],
            )
        )
    return tuple(labeled_spans)


def match_classified_triggers_to_positive_spans(
    *,
    rows: tuple[dict[str, Any], ...],
    positive_spans: tuple[PositiveGestureSpan, ...],
    match_tolerance_frames: int,
) -> tuple[MatchedTriggerGestureRow, ...]:
    """Reuse one-to-one span matching to attach true kick/snare labels."""
    spans_by_recording: dict[str, list[PositiveGestureSpan]] = {}
    for span in positive_spans:
        spans_by_recording.setdefault(span.recording_id, []).append(span)
    matched_flags: dict[str, list[bool]] = {
        recording_id: [False] * len(spans)
        for recording_id, spans in spans_by_recording.items()
    }
    matched_rows: list[MatchedTriggerGestureRow] = []
    for row in rows:
        recording_id = str(row["recording_id"])
        spans = spans_by_recording.get(recording_id, [])
        flags = matched_flags.get(recording_id, [])
        trigger_frame_index = int(row["window_end_frame_index"])
        for index, span in enumerate(spans):
            if flags[index]:
                continue
            if (
                span.start_frame_index - match_tolerance_frames
                <= trigger_frame_index
                <= span.end_frame_index + match_tolerance_frames
            ):
                flags[index] = True
                predicted_label = str(row["predicted_gesture_label"])
                true_label = span.gesture_label
                matched_rows.append(
                    MatchedTriggerGestureRow(
                        trigger_index=int(row["trigger_index"]),
                        recording_id=recording_id,
                        window_index=int(row["window_index"]),
                        trigger_frame_index=trigger_frame_index,
                        trigger_timestamp_seconds=float(row["window_end_timestamp_seconds"]),
                        timing_probability=float(row["timing_probability"]),
                        predicted_gesture_label=predicted_label,
                        predicted_gesture_confidence=float(row["predicted_gesture_confidence"]),
                        true_gesture_label=true_label,
                        correct=predicted_label == true_label,
                    )
                )
                break
    return tuple(matched_rows)


def summarize_matched_gesture_predictions(
    *,
    matched_rows: tuple[MatchedTriggerGestureRow, ...],
    false_trigger_count: int,
    positive_event_count: int,
    class_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize matched trigger typing quality and end-to-end event recall."""
    if positive_event_count < 0:
        raise ValueError("positive_event_count must be greater than or equal to zero.")
    if false_trigger_count < 0:
        raise ValueError("false_trigger_count must be greater than or equal to zero.")
    matched_count = len(matched_rows)
    if matched_count == 0:
        return {
            "matched_trigger_count": 0,
            "matched_accuracy": 0.0,
            "correctly_typed_event_count": 0,
            "false_trigger_count": false_trigger_count,
            "positive_event_count": positive_event_count,
            "correctly_typed_event_recall": 0.0,
            "matched_type_metrics": None,
        }
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    y_true = np.asarray(
        [label_to_index[row.true_gesture_label] for row in matched_rows],
        dtype=np.int64,
    )
    predicted_probabilities = np.zeros((matched_count, len(class_labels)), dtype=np.float32)
    for row_index, row in enumerate(matched_rows):
        predicted_probabilities[row_index, label_to_index[row.predicted_gesture_label]] = 1.0
    type_metrics = multiclass_classification_metrics(
        y_true,
        predicted_probabilities,
        class_labels=class_labels,
    )
    correctly_typed_event_count = int(sum(row.correct for row in matched_rows))
    return {
        "matched_trigger_count": matched_count,
        "matched_accuracy": float(type_metrics["accuracy"]),
        "correctly_typed_event_count": correctly_typed_event_count,
        "false_trigger_count": false_trigger_count,
        "positive_event_count": positive_event_count,
        "correctly_typed_event_recall": correctly_typed_event_count / max(positive_event_count, 1),
        "matched_type_metrics": type_metrics,
    }


def save_classified_triggers_csv(
    *,
    path: Path,
    rows: tuple[dict[str, Any], ...],
) -> None:
    """Save decoded triggers with their predicted kick/snare labels."""
    base_fieldnames = [
        "trigger_index",
        "recording_id",
        "window_index",
        "window_end_frame_index",
        "window_end_timestamp_seconds",
        "timing_probability",
        "timing_threshold",
        "run_length",
        "predicted_gesture_label",
        "predicted_gesture_confidence",
    ]
    probability_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.endswith("_probability") and key not in {"timing_probability"}
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fieldnames + probability_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_matched_gesture_rows_csv(
    *,
    path: Path,
    rows: tuple[MatchedTriggerGestureRow, ...],
) -> None:
    """Save one row per matched decoded trigger with its gesture typing outcome."""
    fieldnames = [
        "trigger_index",
        "recording_id",
        "window_index",
        "trigger_frame_index",
        "trigger_timestamp_seconds",
        "timing_probability",
        "predicted_gesture_label",
        "predicted_gesture_confidence",
        "true_gesture_label",
        "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "trigger_index": row.trigger_index,
                    "recording_id": row.recording_id,
                    "window_index": row.window_index,
                    "trigger_frame_index": row.trigger_frame_index,
                    "trigger_timestamp_seconds": row.trigger_timestamp_seconds,
                    "timing_probability": row.timing_probability,
                    "predicted_gesture_label": row.predicted_gesture_label,
                    "predicted_gesture_confidence": row.predicted_gesture_confidence,
                    "true_gesture_label": row.true_gesture_label,
                    "correct": row.correct,
                }
            )


def save_combined_report(
    *,
    path: Path,
    dataset: Any,
    timing_checkpoint_path: Path,
    gesture_checkpoint_path: Path,
    predictions_path: Path,
    decoded_triggers_path: Path,
    classified_triggers_path: Path,
    threshold: float,
    timing_metrics: dict[str, Any] | None,
    decoded_trigger_metrics: dict[str, Any] | None,
    gesture_pipeline_metrics: dict[str, Any] | None,
    decoded_trigger_count: int,
    trigger_cooldown_frames: int,
    trigger_max_gap_frames: int,
    trigger_match_tolerance_frames: int,
    threshold_analysis: list[dict[str, int | float | None]] | None,
    threshold_summary: dict[str, dict[str, int | float | None]] | None,
    threshold_analysis_path: Path | None,
    timing_model_metadata: dict[str, Any],
    timing_checkpoint_format_version: str,
    gesture_model_metadata: dict[str, Any],
    gesture_checkpoint_format_version: str,
) -> None:
    """Save a compact JSON report for the combined timing-plus-gesture pipeline."""
    report = {
        "timing_checkpoint_path": str(timing_checkpoint_path),
        "gesture_checkpoint_path": str(gesture_checkpoint_path),
        "timing_checkpoint_format_version": timing_checkpoint_format_version,
        "gesture_checkpoint_format_version": gesture_checkpoint_format_version,
        "timing_model_metadata": timing_model_metadata,
        "gesture_model_metadata": gesture_model_metadata,
        "dataset_path": str(dataset.path),
        "predictions_path": str(predictions_path),
        "decoded_triggers_path": str(decoded_triggers_path),
        "classified_triggers_path": str(classified_triggers_path),
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
        "timing_metrics": timing_metrics,
        "decoded_trigger_metrics": decoded_trigger_metrics,
        "gesture_pipeline_metrics": gesture_pipeline_metrics,
        "threshold_analysis": threshold_analysis,
        "threshold_summary": threshold_summary,
        "threshold_analysis_path": (
            None if threshold_analysis_path is None else str(threshold_analysis_path)
        ),
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "MatchedTriggerGestureRow",
    "PositiveGestureSpan",
    "build_positive_gesture_spans",
    "classify_decoded_triggers",
    "match_classified_triggers_to_positive_spans",
    "summarize_matched_gesture_predictions",
]


if __name__ == "__main__":
    main()

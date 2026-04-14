"""Measure decoded-trigger timing relative to labeled completion frames."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Final

import numpy as np

from visionbeat.cnn_trigger import PositiveEventSpan, group_positive_event_spans
from visionbeat.predict_cnn import InferenceDataset, load_inference_dataset

_FRAME_LABEL_COLUMNS: Final[tuple[str, ...]] = ("frame_index", "frame_no")
_GESTURE_LABEL_COLUMNS: Final[tuple[str, ...]] = ("gesture_label", "gesture", "gesture_type")
_RECORDING_ID_COLUMNS: Final[tuple[str, ...]] = ("recording_id",)


@dataclass(frozen=True, slots=True)
class CompletionEvent:
    """One labeled gesture completion event."""

    recording_id: str
    completion_frame_index: int
    gesture_label: str


@dataclass(frozen=True, slots=True)
class DecoderConfigSummary:
    """One ranked decoder configuration from the sweep summary."""

    rank: int
    config_slug: str
    threshold: float
    cooldown_frames: int
    max_gap_frames: int
    decoded_trigger_precision: float
    decoded_trigger_recall: float
    decoded_trigger_f1: float
    false_positive_trigger_count: int
    missed_positive_event_count: int
    decoded_trigger_count: int
    detected_positive_event_count: int
    decoded_triggers_path: Path


@dataclass(frozen=True, slots=True)
class SpanCompletionMapping:
    """One grouped positive span plus the completion event it represents."""

    recording_id: str
    start_frame_index: int
    end_frame_index: int
    completion_frame_index: int
    completion_timestamp_seconds: float
    gesture_label: str


@dataclass(frozen=True, slots=True)
class MatchedTimingRow:
    """One matched decoded trigger with its relative completion timing."""

    config_slug: str
    recording_id: str
    trigger_index: int
    trigger_frame_index: int
    trigger_timestamp_seconds: float
    trigger_probability: float
    completion_frame_index: int
    completion_timestamp_seconds: float
    gesture_label: str
    delta_frames: int
    delta_milliseconds: float
    timing_class: str
    too_early: bool
    too_late: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for decoder timing analysis."""

    parser = argparse.ArgumentParser(
        description="Analyze decoded-trigger timing relative to labeled completion frames."
    )
    parser.add_argument(
        "--sweep-summary",
        required=True,
        help="Path to the ranked or unranked decoder sweep summary CSV.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the prepared NPZ dataset used for the sweep.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to the original completion labels CSV for the recording.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for timing summary and per-event CSV files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-ranked decoder configs to analyze. Default: 3.",
    )
    parser.add_argument(
        "--match-tolerance-frames",
        type=int,
        default=0,
        help="Extra frame tolerance used when matching triggers to positive spans.",
    )
    parser.add_argument(
        "--too-early-frame-threshold",
        type=int,
        default=4,
        help=(
            "Triggers earlier than this many frames before completion are marked too early. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--too-late-frame-threshold",
        type=int,
        default=2,
        help=(
            "Triggers later than this many frames after completion are marked too late. "
            "Default: 2."
        ),
    )
    return parser.parse_args(argv)


def load_ranked_decoder_configs(path: Path) -> tuple[DecoderConfigSummary, ...]:
    """Load the sweep summary and rank configs by decoded-trigger quality."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Sweep summary has no rows: {path}")

    rows.sort(
        key=lambda row: (
            float(row["decoded_trigger_f1"] or 0.0),
            float(row["decoded_trigger_precision"] or 0.0),
            float(row["decoded_trigger_recall"] or 0.0),
            -int(row["false_positive_trigger_count"] or 0),
            -int(row["missed_positive_event_count"] or 0),
        ),
        reverse=True,
    )
    summaries: list[DecoderConfigSummary] = []
    for rank, row in enumerate(rows, start=1):
        summaries.append(
            DecoderConfigSummary(
                rank=rank,
                config_slug=str(row["config_slug"]),
                threshold=float(row["threshold"]),
                cooldown_frames=int(row["cooldown_frames"]),
                max_gap_frames=int(row["max_gap_frames"]),
                decoded_trigger_precision=float(row["decoded_trigger_precision"]),
                decoded_trigger_recall=float(row["decoded_trigger_recall"]),
                decoded_trigger_f1=float(row["decoded_trigger_f1"]),
                false_positive_trigger_count=int(row["false_positive_trigger_count"]),
                missed_positive_event_count=int(row["missed_positive_event_count"]),
                decoded_trigger_count=int(row["decoded_trigger_count"]),
                detected_positive_event_count=int(row["detected_positive_event_count"]),
                decoded_triggers_path=Path(str(row["decoded_triggers_path"])),
            )
        )
    return tuple(summaries)


def load_completion_events(
    path: Path,
    *,
    default_recording_id: str,
) -> tuple[CompletionEvent, ...]:
    """Load exact labeled completion frames from a CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError(f"Labels CSV has no header row: {path}")
        frame_column = _resolve_required_column(
            fieldnames=fieldnames,
            candidates=_FRAME_LABEL_COLUMNS,
            role="frame index",
            csv_path=path,
        )
        gesture_column = _resolve_optional_column(
            fieldnames=fieldnames,
            candidates=_GESTURE_LABEL_COLUMNS,
        )
        recording_column = _resolve_optional_column(
            fieldnames=fieldnames,
            candidates=_RECORDING_ID_COLUMNS,
        )
        events = [
            CompletionEvent(
                recording_id=(
                    default_recording_id
                    if recording_column is None
                    else str(row[recording_column] or default_recording_id)
                ),
                completion_frame_index=int(row[frame_column]),
                gesture_label=(
                    "" if gesture_column is None else str(row[gesture_column]).strip()
                ),
            )
            for row in reader
        ]
    events.sort(key=lambda event: (event.recording_id, event.completion_frame_index))
    return tuple(events)


def map_positive_spans_to_completion_events(
    *,
    dataset: InferenceDataset,
    completion_events: tuple[CompletionEvent, ...],
    max_gap_frames: int,
) -> tuple[SpanCompletionMapping, ...]:
    """Map grouped positive spans back to exact completion frames."""

    if dataset.y is None:
        raise ValueError("Dataset does not include labels required for timing analysis.")
    if dataset.target_name != "completion_within_next_k_frames":
        raise ValueError(
            "Timing analysis currently expects completion_within_next_k_frames datasets."
        )
    if dataset.horizon_frames <= 0:
        raise ValueError("Dataset horizon_frames must be greater than zero.")

    positive_spans = group_positive_event_spans(
        recording_ids=dataset.recording_ids,
        window_end_frame_indices=dataset.window_end_frame_indices,
        labels=dataset.y,
        max_gap_frames=max_gap_frames,
    )
    timestamp_lookup = _build_frame_timestamp_lookup(dataset)
    completion_by_recording: dict[str, list[CompletionEvent]] = {}
    for event in completion_events:
        completion_by_recording.setdefault(event.recording_id, []).append(event)

    mapped: list[SpanCompletionMapping] = []
    used_completion_indices: dict[str, set[int]] = {}
    for span in positive_spans:
        candidates = completion_by_recording.get(span.recording_id, [])
        used = used_completion_indices.setdefault(span.recording_id, set())
        matched_index = None
        for event_index, event in enumerate(candidates):
            if event_index in used:
                continue
            if (
                span.end_frame_index < event.completion_frame_index
                <= span.end_frame_index + dataset.horizon_frames
            ):
                matched_index = event_index
                used.add(event_index)
                completion_timestamp_seconds = _lookup_frame_timestamp(
                    timestamp_lookup=timestamp_lookup,
                    recording_id=span.recording_id,
                    frame_index=event.completion_frame_index,
                )
                mapped.append(
                    SpanCompletionMapping(
                        recording_id=span.recording_id,
                        start_frame_index=span.start_frame_index,
                        end_frame_index=span.end_frame_index,
                        completion_frame_index=event.completion_frame_index,
                        completion_timestamp_seconds=completion_timestamp_seconds,
                        gesture_label=event.gesture_label,
                    )
                )
                break
        if matched_index is None:
            raise ValueError(
                "Unable to map a positive span to a completion event for "
                f"{span.recording_id} span {span.start_frame_index}-{span.end_frame_index}."
            )
    return tuple(mapped)


def load_decoded_triggers(path: Path) -> tuple[dict[str, object], ...]:
    """Load decoded triggers from one sweep config output."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                "trigger_index": int(row["trigger_index"]),
                "recording_id": str(row["recording_id"]),
                "window_index": int(row["window_index"]),
                "window_end_frame_index": int(row["window_end_frame_index"]),
                "window_end_timestamp_seconds": float(row["window_end_timestamp_seconds"]),
                "probability": float(row["probability"]),
                "threshold": float(row["threshold"]),
                "run_length": int(row["run_length"]),
            }
            for row in reader
        ]
    return tuple(rows)


def match_triggers_to_completion_timings(
    *,
    config_slug: str,
    decoded_triggers: tuple[dict[str, object], ...],
    span_mappings: tuple[SpanCompletionMapping, ...],
    match_tolerance_frames: int,
    too_early_frame_threshold: int,
    too_late_frame_threshold: int,
) -> tuple[MatchedTimingRow, ...]:
    """Reuse the span-based matching logic and attach completion timing deltas."""

    spans_by_recording: dict[str, list[SpanCompletionMapping]] = {}
    for span in span_mappings:
        spans_by_recording.setdefault(span.recording_id, []).append(span)
    matched_flags: dict[str, list[bool]] = {
        recording_id: [False] * len(spans)
        for recording_id, spans in spans_by_recording.items()
    }

    matched_rows: list[MatchedTimingRow] = []
    for trigger in decoded_triggers:
        recording_id = str(trigger["recording_id"])
        spans = spans_by_recording.get(recording_id, [])
        flags = matched_flags.get(recording_id, [])
        trigger_frame_index = int(trigger["window_end_frame_index"])
        trigger_timestamp_seconds = float(trigger["window_end_timestamp_seconds"])
        for index, span in enumerate(spans):
            if flags[index]:
                continue
            if (
                span.start_frame_index - match_tolerance_frames
                <= trigger_frame_index
                <= span.end_frame_index + match_tolerance_frames
            ):
                flags[index] = True
                delta_frames = trigger_frame_index - span.completion_frame_index
                delta_milliseconds = (
                    trigger_timestamp_seconds - span.completion_timestamp_seconds
                ) * 1000.0
                matched_rows.append(
                    MatchedTimingRow(
                        config_slug=config_slug,
                        recording_id=recording_id,
                        trigger_index=int(trigger["trigger_index"]),
                        trigger_frame_index=trigger_frame_index,
                        trigger_timestamp_seconds=trigger_timestamp_seconds,
                        trigger_probability=float(trigger["probability"]),
                        completion_frame_index=span.completion_frame_index,
                        completion_timestamp_seconds=span.completion_timestamp_seconds,
                        gesture_label=span.gesture_label,
                        delta_frames=delta_frames,
                        delta_milliseconds=delta_milliseconds,
                        timing_class=_classify_timing(delta_frames),
                        too_early=delta_frames < -too_early_frame_threshold,
                        too_late=delta_frames > too_late_frame_threshold,
                    )
                )
                break
    return tuple(matched_rows)


def summarize_timing_rows(
    *,
    config: DecoderConfigSummary,
    matched_rows: tuple[MatchedTimingRow, ...],
) -> dict[str, int | float | str]:
    """Aggregate timing metrics for one config."""

    if not matched_rows:
        raise ValueError(f"No matched rows found for config {config.config_slug}.")
    delta_frames = [row.delta_frames for row in matched_rows]
    delta_ms = [row.delta_milliseconds for row in matched_rows]
    matched_count = len(matched_rows)
    before_count = sum(row.delta_frames < 0 for row in matched_rows)
    at_count = sum(row.delta_frames == 0 for row in matched_rows)
    after_count = sum(row.delta_frames > 0 for row in matched_rows)
    too_early_count = sum(row.too_early for row in matched_rows)
    too_late_count = sum(row.too_late for row in matched_rows)
    return {
        "rank": config.rank,
        "config_slug": config.config_slug,
        "threshold": config.threshold,
        "cooldown_frames": config.cooldown_frames,
        "max_gap_frames": config.max_gap_frames,
        "decoded_trigger_precision": config.decoded_trigger_precision,
        "decoded_trigger_recall": config.decoded_trigger_recall,
        "decoded_trigger_f1": config.decoded_trigger_f1,
        "false_positive_trigger_count": config.false_positive_trigger_count,
        "missed_positive_event_count": config.missed_positive_event_count,
        "decoded_trigger_count": config.decoded_trigger_count,
        "matched_true_gesture_count": matched_count,
        "mean_delta_frames": float(np.mean(delta_frames)),
        "median_delta_frames": float(median(delta_frames)),
        "mean_delta_milliseconds": float(np.mean(delta_ms)),
        "median_delta_milliseconds": float(median(delta_ms)),
        "percent_before_completion": before_count / matched_count,
        "percent_at_completion": at_count / matched_count,
        "percent_after_completion": after_count / matched_count,
        "percent_too_early": too_early_count / matched_count,
        "percent_too_late": too_late_count / matched_count,
    }


def save_matched_timing_csv(
    *,
    path: Path,
    matched_rows: tuple[MatchedTimingRow, ...],
) -> None:
    """Save per-event matched timing rows for one config."""

    fieldnames = [
        "config_slug",
        "recording_id",
        "trigger_index",
        "trigger_frame_index",
        "trigger_timestamp_seconds",
        "trigger_probability",
        "completion_frame_index",
        "completion_timestamp_seconds",
        "gesture_label",
        "delta_frames",
        "delta_milliseconds",
        "timing_class",
        "too_early",
        "too_late",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in matched_rows:
            writer.writerow(
                {
                    "config_slug": row.config_slug,
                    "recording_id": row.recording_id,
                    "trigger_index": row.trigger_index,
                    "trigger_frame_index": row.trigger_frame_index,
                    "trigger_timestamp_seconds": row.trigger_timestamp_seconds,
                    "trigger_probability": row.trigger_probability,
                    "completion_frame_index": row.completion_frame_index,
                    "completion_timestamp_seconds": row.completion_timestamp_seconds,
                    "gesture_label": row.gesture_label,
                    "delta_frames": row.delta_frames,
                    "delta_milliseconds": row.delta_milliseconds,
                    "timing_class": row.timing_class,
                    "too_early": row.too_early,
                    "too_late": row.too_late,
                }
            )


def save_timing_summary_csv(
    *,
    path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    """Save the aggregate timing summary across configs."""

    fieldnames = [
        "rank",
        "config_slug",
        "threshold",
        "cooldown_frames",
        "max_gap_frames",
        "decoded_trigger_precision",
        "decoded_trigger_recall",
        "decoded_trigger_f1",
        "false_positive_trigger_count",
        "missed_positive_event_count",
        "decoded_trigger_count",
        "matched_true_gesture_count",
        "mean_delta_frames",
        "median_delta_frames",
        "mean_delta_milliseconds",
        "median_delta_milliseconds",
        "percent_before_completion",
        "percent_at_completion",
        "percent_after_completion",
        "percent_too_early",
        "percent_too_late",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for trigger timing analysis."""

    args = parse_args(argv)
    try:
        if args.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if args.match_tolerance_frames < 0:
            raise ValueError("match_tolerance_frames must be greater than or equal to zero.")
        if args.too_early_frame_threshold < 0:
            raise ValueError("too_early_frame_threshold must be greater than or equal to zero.")
        if args.too_late_frame_threshold < 0:
            raise ValueError("too_late_frame_threshold must be greater than or equal to zero.")

        summary_path = Path(args.sweep_summary)
        dataset_path = Path(args.dataset)
        labels_path = Path(args.labels)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_inference_dataset(dataset_path)
        ranked_configs = load_ranked_decoder_configs(summary_path)
        selected_configs = ranked_configs[: args.top_k]
        default_recording_id = _resolve_default_recording_id(dataset)
        completion_events = load_completion_events(
            labels_path,
            default_recording_id=default_recording_id,
        )

        aggregate_rows: list[dict[str, int | float | str]] = []
        for config in selected_configs:
            span_mappings = map_positive_spans_to_completion_events(
                dataset=dataset,
                completion_events=completion_events,
                max_gap_frames=config.max_gap_frames,
            )
            decoded_triggers = load_decoded_triggers(config.decoded_triggers_path)
            matched_rows = match_triggers_to_completion_timings(
                config_slug=config.config_slug,
                decoded_triggers=decoded_triggers,
                span_mappings=span_mappings,
                match_tolerance_frames=args.match_tolerance_frames,
                too_early_frame_threshold=args.too_early_frame_threshold,
                too_late_frame_threshold=args.too_late_frame_threshold,
            )
            per_event_path = output_dir / f"{config.config_slug}_matched_timing.csv"
            save_matched_timing_csv(path=per_event_path, matched_rows=matched_rows)
            aggregate_rows.append(
                summarize_timing_rows(
                    config=config,
                    matched_rows=matched_rows,
                )
            )

        summary_csv_path = output_dir / f"{dataset_path.stem}_timing_summary_top{args.top_k}.csv"
        assumptions_path = output_dir / "timing_analysis_assumptions.json"
        save_timing_summary_csv(path=summary_csv_path, rows=aggregate_rows)
        assumptions = {
            "dataset_path": str(dataset_path),
            "labels_path": str(labels_path),
            "top_k": args.top_k,
            "match_tolerance_frames": args.match_tolerance_frames,
            "too_early_definition": (
                f"delta_frames < -{args.too_early_frame_threshold}"
            ),
            "too_late_definition": (
                f"delta_frames > {args.too_late_frame_threshold}"
            ),
            "note": (
                "delta_frames is trigger_frame - completion_frame, where negative values "
                "mean the trigger fired before the labeled completion."
            ),
        }
        assumptions_path.write_text(
            json.dumps(assumptions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"Timing summary: {summary_csv_path}")
        print(f"Assumptions: {assumptions_path}")
        for row in aggregate_rows:
            print(
                f"{row['config_slug']}: "
                f"mean_delta_frames={float(row['mean_delta_frames']):.3f} "
                f"median_delta_frames={float(row['median_delta_frames']):.3f} "
                f"before={float(row['percent_before_completion']):.2%} "
                f"too_early={float(row['percent_too_early']):.2%} "
                f"too_late={float(row['percent_too_late']):.2%}"
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_required_column(
    *,
    fieldnames: tuple[str, ...],
    candidates: tuple[str, ...],
    role: str,
    csv_path: Path,
) -> str:
    matches = [name for name in candidates if name in fieldnames]
    if not matches:
        raise ValueError(
            f"Labels CSV is missing a {role} column in {csv_path}. Expected one of {candidates}."
        )
    return matches[0]


def _resolve_optional_column(
    *,
    fieldnames: tuple[str, ...],
    candidates: tuple[str, ...],
) -> str | None:
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def _resolve_default_recording_id(dataset: InferenceDataset) -> str:
    recording_ids = np.asarray(dataset.recording_ids)
    unique = list(dict.fromkeys(str(value) for value in recording_ids.tolist()))
    if len(unique) != 1:
        raise ValueError(
            "Timing analysis expects a single-recording dataset or explicit recording_id labels."
        )
    return unique[0]


def _build_frame_timestamp_lookup(
    dataset: InferenceDataset,
) -> dict[str, dict[int, float]]:
    lookup: dict[str, dict[int, float]] = {}
    for recording_id, frame_index, timestamp_seconds in zip(
        dataset.recording_ids.tolist(),
        dataset.window_end_frame_indices.tolist(),
        dataset.window_end_timestamps_seconds.tolist(),
        strict=True,
    ):
        lookup.setdefault(str(recording_id), {})[int(frame_index)] = float(timestamp_seconds)
    return lookup


def _lookup_frame_timestamp(
    *,
    timestamp_lookup: dict[str, dict[int, float]],
    recording_id: str,
    frame_index: int,
) -> float:
    recording_lookup = timestamp_lookup.get(recording_id, {})
    if frame_index not in recording_lookup:
        raise ValueError(
            f"Frame {frame_index} for recording {recording_id!r} was not found in the dataset timestamp lookup."
        )
    return recording_lookup[frame_index]


def _classify_timing(delta_frames: int) -> str:
    if delta_frames < 0:
        return "before_completion"
    if delta_frames == 0:
        return "at_completion"
    return "after_completion"


if __name__ == "__main__":
    main()

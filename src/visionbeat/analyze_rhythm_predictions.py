"""Analyze structured rhythm-prediction logs from live VisionBeat runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Final, Literal, cast

RhythmOutcome = Literal["pending", "matched", "missed", "expired"]
TimingVerdict = Literal["yes", "no", "review"]

_VALID_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"pending", "matched", "missed", "expired"}
)


@dataclass(frozen=True, slots=True)
class RhythmPredictionLogEvent:
    """One structured `rhythm_prediction` payload emitted by the live runtime."""

    timestamp: float
    prediction_id: str
    outcome: RhythmOutcome
    gesture: str
    predicted_time_seconds: float
    actual_time_seconds: float | None
    actual_gesture: str | None
    error_ms: float | None
    last_event_timestamp: float
    interval_seconds: float
    confidence: float
    repetition_count: int
    jitter: float
    active: bool
    shadow_only: bool
    source: str
    trigger_mode: str
    frame_index: int | None = None

    @property
    def prediction_lead_ms(self) -> float:
        """Return how far before the expected beat this prediction was emitted."""
        return max(0.0, (self.predicted_time_seconds - self.timestamp) * 1000.0)

    @property
    def is_false_expectation(self) -> bool:
        """Return whether this outcome represents an unmet rhythm expectation."""
        return self.outcome in {"missed", "expired"}

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> RhythmPredictionLogEvent:
        """Build an event from a structured logging payload."""
        if payload.get("event") != "rhythm_prediction":
            raise ValueError("payload is not a rhythm_prediction event.")

        return cls(
            timestamp=_required_float(payload, "timestamp"),
            frame_index=_optional_int(payload, "frame_index"),
            prediction_id=str(payload.get("prediction_id", "")),
            outcome=_parse_outcome(payload.get("outcome")),
            gesture=str(payload.get("gesture", "")),
            predicted_time_seconds=_required_float(payload, "predicted_time_seconds"),
            actual_time_seconds=_optional_float(payload, "actual_time_seconds"),
            actual_gesture=_optional_string(payload, "actual_gesture"),
            error_ms=_optional_float(payload, "error_ms"),
            last_event_timestamp=_required_float(payload, "last_event_timestamp"),
            interval_seconds=_required_float(payload, "interval_seconds"),
            confidence=_required_float(payload, "confidence"),
            repetition_count=_required_int(payload, "repetition_count"),
            jitter=_required_float(payload, "jitter"),
            active=bool(payload.get("active", False)),
            shadow_only=bool(payload.get("shadow_only", False)),
            source=str(payload.get("source", "")),
            trigger_mode=str(payload.get("trigger_mode", "")),
        )


@dataclass(frozen=True, slots=True)
class RhythmAnalysisConfig:
    """Thresholds and baselines used to summarize rhythm-prediction logs."""

    baseline_hybrid_latency_ms: float = 240.0
    baseline_shadow_latency_ms: float = 210.0
    max_good_error_ms: float = 120.0
    max_false_expectation_rate: float = 0.25


@dataclass(frozen=True, slots=True)
class RhythmAnalysisSummary:
    """Aggregated metrics for one rhythm-continuation evaluation run."""

    label: str
    event_count: int
    pending_count: int
    matched_count: int
    missed_count: int
    expired_count: int
    false_expectation_count: int
    wrong_gesture_count: int
    late_or_early_actual_count: int
    activation_after_hits: int | None
    activation_after_seconds: float | None
    first_prediction_timestamp_seconds: float | None
    median_signed_error_ms: float | None
    median_abs_error_ms: float | None
    mean_abs_error_ms: float | None
    median_prediction_lead_ms: float | None
    median_estimated_effective_latency_ms: float | None
    median_estimated_latency_reduction_ms: float | None
    false_expectation_rate: float | None
    perceived_timing_verdict: TimingVerdict
    perceived_timing_notes: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary into a JSON-friendly dictionary."""
        return {
            "label": self.label,
            "event_count": self.event_count,
            "pending_count": self.pending_count,
            "matched_count": self.matched_count,
            "missed_count": self.missed_count,
            "expired_count": self.expired_count,
            "false_expectation_count": self.false_expectation_count,
            "wrong_gesture_count": self.wrong_gesture_count,
            "late_or_early_actual_count": self.late_or_early_actual_count,
            "activation_after_hits": self.activation_after_hits,
            "activation_after_seconds": self.activation_after_seconds,
            "first_prediction_timestamp_seconds": self.first_prediction_timestamp_seconds,
            "median_signed_error_ms": self.median_signed_error_ms,
            "median_abs_error_ms": self.median_abs_error_ms,
            "mean_abs_error_ms": self.mean_abs_error_ms,
            "median_prediction_lead_ms": self.median_prediction_lead_ms,
            "median_estimated_effective_latency_ms": (
                self.median_estimated_effective_latency_ms
            ),
            "median_estimated_latency_reduction_ms": (
                self.median_estimated_latency_reduction_ms
            ),
            "false_expectation_rate": self.false_expectation_rate,
            "perceived_timing_verdict": self.perceived_timing_verdict,
            "perceived_timing_notes": self.perceived_timing_notes,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for rhythm-prediction log analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze structured rhythm_prediction events from VisionBeat logs."
    )
    parser.add_argument(
        "log_paths",
        nargs="+",
        help="One or more VisionBeat structured log files.",
    )
    parser.add_argument(
        "--label",
        default="rhythm-evaluation",
        help="Human-readable label for this run or pattern.",
    )
    parser.add_argument(
        "--baseline-hybrid-latency-ms",
        type=float,
        default=240.0,
        help="Observed hybrid-mode baseline latency. Default: 240.",
    )
    parser.add_argument(
        "--baseline-shadow-latency-ms",
        type=float,
        default=210.0,
        help="Observed shadow-mode baseline latency. Default: 210.",
    )
    parser.add_argument(
        "--max-good-error-ms",
        type=float,
        default=120.0,
        help="Largest median absolute prediction error considered musically good.",
    )
    parser.add_argument(
        "--max-false-expectation-rate",
        type=float,
        default=0.25,
        help="Largest false-expectation rate considered acceptable.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a JSON summary.",
    )
    parser.add_argument(
        "--by-gesture",
        action="store_true",
        help="Also print and serialize one summary per gesture.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def extract_structured_payload(line: str) -> dict[str, object] | None:
    """Extract the structured JSON payload from one formatted log line."""
    stripped = line.strip()
    if not stripped:
        return None

    candidates = [stripped]
    json_start = stripped.find("{")
    while json_start >= 0:
        candidates.append(stripped[json_start:])
        json_start = stripped.find("{", json_start + 1)

    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return cast(dict[str, object], decoded)
    return None


def iter_rhythm_prediction_events(
    lines: Iterable[str],
) -> Iterable[RhythmPredictionLogEvent]:
    """Yield rhythm-prediction events from formatted structured log lines."""
    for line_number, line in enumerate(lines, start=1):
        payload = extract_structured_payload(line)
        if payload is None or payload.get("event") != "rhythm_prediction":
            continue
        try:
            yield RhythmPredictionLogEvent.from_payload(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rhythm_prediction payload on line {line_number}: {exc}"
            ) from exc


def load_rhythm_prediction_events(paths: Sequence[Path]) -> tuple[RhythmPredictionLogEvent, ...]:
    """Load all rhythm-prediction events from one or more log files."""
    events: list[RhythmPredictionLogEvent] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            events.extend(iter_rhythm_prediction_events(handle))
    events.sort(key=lambda event: (event.timestamp, event.prediction_id, event.outcome))
    return tuple(events)


def summarize_events(
    events: Sequence[RhythmPredictionLogEvent],
    *,
    config: RhythmAnalysisConfig | None = None,
    label: str = "rhythm-evaluation",
) -> RhythmAnalysisSummary:
    """Compute ARJ rhythm-continuation metrics for a set of prediction events."""
    analysis_config = config or RhythmAnalysisConfig()
    pending_events = [event for event in events if event.outcome == "pending"]
    matched_events = [event for event in events if event.outcome == "matched"]
    missed_events = [event for event in events if event.outcome == "missed"]
    expired_events = [event for event in events if event.outcome == "expired"]
    false_events = [event for event in events if event.is_false_expectation]
    wrong_gesture_count = sum(
        1
        for event in false_events
        if event.actual_gesture is not None and event.actual_gesture != event.gesture
    )
    late_or_early_actual_count = sum(
        1
        for event in false_events
        if event.actual_time_seconds is not None
        and (event.actual_gesture is None or event.actual_gesture == event.gesture)
    )

    signed_errors = [event.error_ms for event in matched_events if event.error_ms is not None]
    abs_errors = [abs(error_ms) for error_ms in signed_errors]
    prediction_leads = [event.prediction_lead_ms for event in pending_events]
    effective_latencies = [
        max(0.0, analysis_config.baseline_hybrid_latency_ms - prediction_lead_ms)
        for prediction_lead_ms in prediction_leads
    ]
    latency_reductions = [
        analysis_config.baseline_hybrid_latency_ms - latency
        for latency in effective_latencies
    ]
    first_pending = pending_events[0] if pending_events else None
    activation_after_seconds: float | None = None
    activation_after_hits: int | None = None
    if first_pending is not None:
        estimated_first_hit = (
            first_pending.last_event_timestamp
            - first_pending.interval_seconds * first_pending.repetition_count
        )
        activation_after_seconds = max(0.0, first_pending.timestamp - estimated_first_hit)
        activation_after_hits = first_pending.repetition_count + 1

    evaluated_expectation_count = len(matched_events) + len(false_events)
    false_expectation_rate = (
        None
        if evaluated_expectation_count == 0
        else len(false_events) / evaluated_expectation_count
    )
    median_abs_error_ms = _optional_median(abs_errors)
    median_prediction_lead_ms = _optional_median(prediction_leads)
    verdict, notes = _classify_perceived_timing(
        matched_count=len(matched_events),
        false_expectation_rate=false_expectation_rate,
        median_abs_error_ms=median_abs_error_ms,
        median_prediction_lead_ms=median_prediction_lead_ms,
        config=analysis_config,
    )
    return RhythmAnalysisSummary(
        label=label,
        event_count=len(events),
        pending_count=len(pending_events),
        matched_count=len(matched_events),
        missed_count=len(missed_events),
        expired_count=len(expired_events),
        false_expectation_count=len(false_events),
        wrong_gesture_count=wrong_gesture_count,
        late_or_early_actual_count=late_or_early_actual_count,
        activation_after_hits=activation_after_hits,
        activation_after_seconds=activation_after_seconds,
        first_prediction_timestamp_seconds=(
            None if first_pending is None else first_pending.timestamp
        ),
        median_signed_error_ms=_optional_median(signed_errors),
        median_abs_error_ms=median_abs_error_ms,
        mean_abs_error_ms=None if not abs_errors else mean(abs_errors),
        median_prediction_lead_ms=median_prediction_lead_ms,
        median_estimated_effective_latency_ms=_optional_median(effective_latencies),
        median_estimated_latency_reduction_ms=_optional_median(latency_reductions),
        false_expectation_rate=false_expectation_rate,
        perceived_timing_verdict=verdict,
        perceived_timing_notes=notes,
    )


def summarize_by_gesture(
    events: Sequence[RhythmPredictionLogEvent],
    *,
    config: RhythmAnalysisConfig | None = None,
) -> dict[str, RhythmAnalysisSummary]:
    """Summarize rhythm-prediction behavior separately for each gesture label."""
    gestures = sorted({event.gesture for event in events})
    return {
        gesture: summarize_events(
            [event for event in events if event.gesture == gesture],
            config=config,
            label=gesture,
        )
        for gesture in gestures
    }


def format_summary(summary: RhythmAnalysisSummary, *, config: RhythmAnalysisConfig) -> str:
    """Format a concise human-readable rhythm analysis report."""
    lines = [
        f"Rhythm prediction evaluation: {summary.label}",
        (
            "Events: "
            f"pending={summary.pending_count}, matched={summary.matched_count}, "
            f"missed={summary.missed_count}, expired={summary.expired_count}"
        ),
        (
            "False expectations: "
            f"{summary.false_expectation_count}"
            f"{_format_rate_suffix(summary.false_expectation_rate)}"
        ),
    ]
    if summary.activation_after_hits is None:
        lines.append("Activation: no rhythm prediction activated.")
    else:
        lines.append(
            "Activation: "
            f"after {summary.activation_after_hits} hits "
            f"({_format_optional_ms(summary.activation_after_seconds, scale=1000.0)} "
            "from estimated first hit)"
        )
    lines.append(
        "Prediction timing: "
        f"median_error={_format_optional_ms(summary.median_signed_error_ms)}, "
        f"median_abs_error={_format_optional_ms(summary.median_abs_error_ms)}, "
        f"mean_abs_error={_format_optional_ms(summary.mean_abs_error_ms)}"
    )
    lines.append(
        "Prediction lead: "
        f"median={_format_optional_ms(summary.median_prediction_lead_ms)}, "
        f"baseline_hybrid={config.baseline_hybrid_latency_ms:.1f} ms"
    )
    lines.append(
        "Estimated prediction-side latency: "
        f"median={_format_optional_ms(summary.median_estimated_effective_latency_ms)} "
        f"(reduction={_format_optional_ms(summary.median_estimated_latency_reduction_ms)})"
    )
    lines.append(
        "Perceived timing verdict: "
        f"{summary.perceived_timing_verdict} - {summary.perceived_timing_notes}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Run rhythm-prediction log analysis from the command line."""
    args = parse_args(argv)
    config = RhythmAnalysisConfig(
        baseline_hybrid_latency_ms=args.baseline_hybrid_latency_ms,
        baseline_shadow_latency_ms=args.baseline_shadow_latency_ms,
        max_good_error_ms=args.max_good_error_ms,
        max_false_expectation_rate=args.max_false_expectation_rate,
    )
    paths = tuple(Path(path) for path in args.log_paths)
    try:
        events = load_rhythm_prediction_events(paths)
        summary = summarize_events(events, config=config, label=args.label)
        print(format_summary(summary, config=config))
        result: dict[str, object] = {
            "summary": summary.to_dict(),
            "baselines": {
                "hybrid_latency_ms": config.baseline_hybrid_latency_ms,
                "shadow_latency_ms": config.baseline_shadow_latency_ms,
            },
        }
        if args.by_gesture:
            per_gesture = summarize_by_gesture(events, config=config)
            result["by_gesture"] = {
                gesture: gesture_summary.to_dict()
                for gesture, gesture_summary in per_gesture.items()
            }
            for gesture_summary in per_gesture.values():
                print()
                print(format_summary(gesture_summary, config=config))
        if args.output_json is not None:
            Path(args.output_json).write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _parse_outcome(value: object) -> RhythmOutcome:
    if not isinstance(value, str) or value not in _VALID_OUTCOMES:
        raise ValueError(f"invalid rhythm prediction outcome: {value!r}")
    return cast(RhythmOutcome, value)


def _classify_perceived_timing(
    *,
    matched_count: int,
    false_expectation_rate: float | None,
    median_abs_error_ms: float | None,
    median_prediction_lead_ms: float | None,
    config: RhythmAnalysisConfig,
) -> tuple[TimingVerdict, str]:
    if matched_count == 0:
        return "review", "no matched predictions were observed."
    if median_abs_error_ms is None:
        return "review", "matched predictions did not include timing errors."
    if median_prediction_lead_ms is None:
        return "review", "no pending predictions were available for lead-time analysis."

    good_error = median_abs_error_ms <= config.max_good_error_ms
    good_false_rate = (
        false_expectation_rate is None
        or false_expectation_rate <= config.max_false_expectation_rate
    )
    good_lead = median_prediction_lead_ms >= config.baseline_hybrid_latency_ms
    if good_error and good_false_rate and good_lead:
        return (
            "yes",
            "matched predictions were timely and armed earlier than the hybrid baseline.",
        )
    if (
        median_abs_error_ms > config.max_good_error_ms * 2.0
        or (
            false_expectation_rate is not None
            and false_expectation_rate > max(0.5, config.max_false_expectation_rate * 2.0)
        )
        or median_prediction_lead_ms < config.baseline_hybrid_latency_ms * 0.5
    ):
        return (
            "no",
            "timing error, false expectations, or arm lead were outside evaluation bounds.",
        )
    return (
        "review",
        "some metrics improved, but the run needs listening notes or more repetitions.",
    )


def _required_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required field: {key}")
    return _coerce_float(value, field_name=key)


def _optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else _coerce_float(value, field_name=key)


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required field: {key}")
    return _coerce_int(value, field_name=key)


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    return None if value is None else _coerce_int(value, field_name=key)


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return None if value is None else str(value)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{field_name} must be an integer.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _optional_median(values: Sequence[float]) -> float | None:
    return None if not values else float(median(values))


def _format_rate_suffix(rate: float | None) -> str:
    return "" if rate is None else f" ({rate:.1%})"


def _format_optional_ms(value: float | None, *, scale: float = 1.0) -> str:
    return "n/a" if value is None else f"{value * scale:.1f} ms"


if __name__ == "__main__":
    main()

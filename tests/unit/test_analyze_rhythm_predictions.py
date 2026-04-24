from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionbeat.analyze_rhythm_predictions import (
    RhythmAnalysisConfig,
    extract_structured_payload,
    format_summary,
    load_rhythm_prediction_events,
    summarize_by_gesture,
    summarize_events,
)


def rhythm_payload(
    *,
    outcome: str,
    timestamp: float,
    predicted_time_seconds: float,
    actual_time_seconds: float | None = None,
    actual_gesture: str | None = None,
    error_ms: float | None = None,
    gesture: str = "kick",
    prediction_id: str = "kick:2.000000->2.500000",
    source: str = "heuristic_completion",
    trigger_mode: str = "arm_only",
) -> dict[str, object]:
    return {
        "event": "rhythm_prediction",
        "timestamp": timestamp,
        "frame_index": 7,
        "prediction_id": prediction_id,
        "outcome": outcome,
        "gesture": gesture,
        "predicted_time_seconds": predicted_time_seconds,
        "actual_time_seconds": actual_time_seconds,
        "actual_gesture": actual_gesture,
        "error_ms": error_ms,
        "last_event_timestamp": 2.0,
        "interval_seconds": 0.5,
        "confidence": 0.86,
        "repetition_count": 2,
        "jitter": 0.02,
        "active": outcome == "pending",
        "shadow_only": trigger_mode == "shadow",
        "source": source,
        "trigger_mode": trigger_mode,
    }


def structured_line(payload: dict[str, object]) -> str:
    return (
        "2026-04-24 12:00:00,000 | INFO | visionbeat.observability | "
        f"Rhythm prediction | {json.dumps(payload, sort_keys=True)}\n"
    )


def test_extract_structured_payload_reads_formatted_log_line() -> None:
    payload = rhythm_payload(outcome="pending", timestamp=2.0, predicted_time_seconds=2.5)

    extracted = extract_structured_payload(structured_line(payload))

    assert extracted is not None
    assert extracted["event"] == "rhythm_prediction"
    assert extracted["predicted_time_seconds"] == pytest.approx(2.5)


def test_load_and_summarize_rhythm_prediction_events(tmp_path: Path) -> None:
    log_path = tmp_path / "rhythm.log"
    payloads = [
        rhythm_payload(outcome="pending", timestamp=2.0, predicted_time_seconds=2.5),
        rhythm_payload(
            outcome="matched",
            timestamp=2.48,
            predicted_time_seconds=2.5,
            actual_time_seconds=2.48,
            actual_gesture="kick",
            error_ms=-20.0,
        ),
    ]
    log_path.write_text(
        "".join(structured_line(payload) for payload in payloads),
        encoding="utf-8",
    )

    events = load_rhythm_prediction_events((log_path,))
    summary = summarize_events(
        events,
        config=RhythmAnalysisConfig(baseline_hybrid_latency_ms=240.0),
        label="medium-0.5s",
    )

    assert len(events) == 2
    assert summary.label == "medium-0.5s"
    assert summary.activation_after_hits == 3
    assert summary.activation_after_seconds == pytest.approx(1.0)
    assert summary.matched_count == 1
    assert summary.false_expectation_count == 0
    assert summary.median_signed_error_ms == pytest.approx(-20.0)
    assert summary.median_abs_error_ms == pytest.approx(20.0)
    assert summary.median_prediction_lead_ms == pytest.approx(500.0)
    assert summary.median_estimated_effective_latency_ms == pytest.approx(0.0)
    assert summary.median_estimated_latency_reduction_ms == pytest.approx(240.0)
    assert summary.perceived_timing_verdict == "yes"


def test_summary_counts_false_expectations_and_wrong_gestures() -> None:
    events = tuple(
        load_rhythm_prediction_events_from_payloads(
            [
                rhythm_payload(outcome="pending", timestamp=2.0, predicted_time_seconds=2.5),
                rhythm_payload(
                    outcome="missed",
                    timestamp=2.52,
                    predicted_time_seconds=2.5,
                    actual_time_seconds=2.52,
                    actual_gesture="snare",
                    error_ms=20.0,
                ),
            ]
        )
    )

    summary = summarize_events(events)

    assert summary.matched_count == 0
    assert summary.missed_count == 1
    assert summary.false_expectation_count == 1
    assert summary.wrong_gesture_count == 1
    assert summary.false_expectation_rate == pytest.approx(1.0)
    assert summary.perceived_timing_verdict == "review"


def test_summarize_by_gesture_splits_kick_and_snare() -> None:
    events = tuple(
        load_rhythm_prediction_events_from_payloads(
            [
                rhythm_payload(
                    outcome="pending",
                    timestamp=2.0,
                    predicted_time_seconds=2.5,
                    gesture="kick",
                ),
                rhythm_payload(
                    outcome="pending",
                    timestamp=3.0,
                    predicted_time_seconds=3.5,
                    gesture="snare",
                    prediction_id="snare:3.000000->3.500000",
                ),
            ]
        )
    )

    summaries = summarize_by_gesture(events)

    assert set(summaries) == {"kick", "snare"}
    assert summaries["kick"].pending_count == 1
    assert summaries["snare"].pending_count == 1


def test_format_summary_includes_latency_estimate() -> None:
    events = tuple(
        load_rhythm_prediction_events_from_payloads(
            [
                rhythm_payload(outcome="pending", timestamp=2.0, predicted_time_seconds=2.5),
                rhythm_payload(
                    outcome="matched",
                    timestamp=2.5,
                    predicted_time_seconds=2.5,
                    actual_time_seconds=2.5,
                    actual_gesture="kick",
                    error_ms=0.0,
                ),
            ]
        )
    )
    config = RhythmAnalysisConfig()
    summary = summarize_events(events, config=config, label="steady-medium")

    formatted = format_summary(summary, config=config)

    assert "steady-medium" in formatted
    assert "Estimated prediction-side latency" in formatted
    assert "Perceived timing verdict: yes" in formatted


def load_rhythm_prediction_events_from_payloads(
    payloads: list[dict[str, object]],
):
    lines = [structured_line(payload) for payload in payloads]
    from visionbeat.analyze_rhythm_predictions import iter_rhythm_prediction_events

    return iter_rhythm_prediction_events(lines)

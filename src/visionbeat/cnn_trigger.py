"""Shared decoding utilities for turning CNN window scores into trigger events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class DecodedTriggerEvent:
    """One accepted trigger after local-max decoding and cooldown suppression."""

    recording_id: str
    window_index: int
    window_end_frame_index: int
    window_end_timestamp_seconds: float
    probability: float
    threshold: float
    run_length: int


@dataclass(frozen=True, slots=True)
class PositiveEventSpan:
    """One contiguous positive neighborhood in the labeled window stream."""

    recording_id: str
    start_frame_index: int
    end_frame_index: int
    window_count: int


def decode_trigger_events(
    *,
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
    window_end_timestamps_seconds: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cooldown_frames: int = 0,
    max_gap_frames: int = 1,
) -> tuple[DecodedTriggerEvent, ...]:
    """Collapse noisy window probabilities into one trigger per local peak."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")
    if cooldown_frames < 0:
        raise ValueError("cooldown_frames must be greater than or equal to zero.")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be greater than or equal to zero.")

    sample_count = _validate_sample_aligned_arrays(
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        window_end_timestamps_seconds=window_end_timestamps_seconds,
        probabilities=probabilities,
    )
    if sample_count == 0:
        return ()

    accepted: list[DecodedTriggerEvent] = []
    for recording_id in _ordered_unique_strings(recording_ids):
        ordered_positions = np.flatnonzero(recording_ids == recording_id)
        if ordered_positions.size == 0:
            continue
        ordered_positions = ordered_positions[
            np.argsort(window_end_frame_indices[ordered_positions], kind="mergesort")
        ]
        current_run: list[int] = []
        previous_frame_index: int | None = None
        for window_index in ordered_positions.tolist():
            frame_index = int(window_end_frame_indices[window_index])
            is_positive = float(probabilities[window_index]) >= threshold
            run_continues = (
                current_run
                and previous_frame_index is not None
                and frame_index - previous_frame_index <= max_gap_frames
            )
            if is_positive:
                if current_run and not run_continues:
                    _accept_or_replace_trigger(
                        accepted=accepted,
                        candidate=_build_trigger_event(
                            current_run,
                            recording_ids=recording_ids,
                            window_end_frame_indices=window_end_frame_indices,
                            window_end_timestamps_seconds=window_end_timestamps_seconds,
                            probabilities=probabilities,
                            threshold=threshold,
                        ),
                        cooldown_frames=cooldown_frames,
                    )
                    current_run = []
                current_run.append(window_index)
            elif current_run:
                _accept_or_replace_trigger(
                    accepted=accepted,
                    candidate=_build_trigger_event(
                        current_run,
                        recording_ids=recording_ids,
                        window_end_frame_indices=window_end_frame_indices,
                        window_end_timestamps_seconds=window_end_timestamps_seconds,
                        probabilities=probabilities,
                        threshold=threshold,
                    ),
                    cooldown_frames=cooldown_frames,
                )
                current_run = []
            previous_frame_index = frame_index
        if current_run:
            _accept_or_replace_trigger(
                accepted=accepted,
                candidate=_build_trigger_event(
                    current_run,
                    recording_ids=recording_ids,
                    window_end_frame_indices=window_end_frame_indices,
                    window_end_timestamps_seconds=window_end_timestamps_seconds,
                    probabilities=probabilities,
                    threshold=threshold,
                ),
                cooldown_frames=cooldown_frames,
            )
    return tuple(accepted)


def group_positive_event_spans(
    *,
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
    labels: np.ndarray,
    max_gap_frames: int = 1,
) -> tuple[PositiveEventSpan, ...]:
    """Group contiguous positive windows into event neighborhoods."""

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be greater than or equal to zero.")
    sample_count = _validate_sample_aligned_arrays(
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        labels=labels,
    )
    if sample_count == 0:
        return ()
    values = np.asarray(labels, dtype=np.int64)
    unique_targets = set(np.unique(values).tolist())
    if not unique_targets <= {0, 1}:
        raise ValueError(
            f"Expected binary labels containing only 0/1, got {sorted(unique_targets)}."
        )

    spans: list[PositiveEventSpan] = []
    for recording_id in _ordered_unique_strings(recording_ids):
        ordered_positions = np.flatnonzero(recording_ids == recording_id)
        if ordered_positions.size == 0:
            continue
        ordered_positions = ordered_positions[
            np.argsort(window_end_frame_indices[ordered_positions], kind="mergesort")
        ]
        current_positions: list[int] = []
        previous_frame_index: int | None = None
        for window_index in ordered_positions.tolist():
            frame_index = int(window_end_frame_indices[window_index])
            is_positive = int(values[window_index]) == 1
            run_continues = (
                current_positions
                and previous_frame_index is not None
                and frame_index - previous_frame_index <= max_gap_frames
            )
            if is_positive:
                if current_positions and not run_continues:
                    spans.append(
                        _build_positive_event_span(
                            current_positions,
                            recording_ids=recording_ids,
                            window_end_frame_indices=window_end_frame_indices,
                        )
                    )
                    current_positions = []
                current_positions.append(window_index)
            elif current_positions:
                spans.append(
                    _build_positive_event_span(
                        current_positions,
                        recording_ids=recording_ids,
                        window_end_frame_indices=window_end_frame_indices,
                    )
                )
                current_positions = []
            previous_frame_index = frame_index
        if current_positions:
            spans.append(
                _build_positive_event_span(
                    current_positions,
                    recording_ids=recording_ids,
                    window_end_frame_indices=window_end_frame_indices,
                )
            )
    return tuple(spans)


def evaluate_decoded_triggers(
    *,
    decoded_triggers: tuple[DecodedTriggerEvent, ...],
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
    labels: np.ndarray,
    match_tolerance_frames: int = 0,
    max_gap_frames: int = 1,
) -> dict[str, int | float]:
    """Evaluate decoded trigger events against grouped positive neighborhoods."""

    if match_tolerance_frames < 0:
        raise ValueError("match_tolerance_frames must be greater than or equal to zero.")
    positive_spans = group_positive_event_spans(
        recording_ids=recording_ids,
        window_end_frame_indices=window_end_frame_indices,
        labels=labels,
        max_gap_frames=max_gap_frames,
    )
    matched_positive_count = 0
    false_positive_count = 0
    positive_by_recording: dict[str, list[PositiveEventSpan]] = {}
    for span in positive_spans:
        positive_by_recording.setdefault(span.recording_id, []).append(span)

    matched_flags: dict[str, list[bool]] = {
        recording_id: [False] * len(spans)
        for recording_id, spans in positive_by_recording.items()
    }
    for trigger in decoded_triggers:
        spans = positive_by_recording.get(trigger.recording_id, [])
        flags = matched_flags.get(trigger.recording_id, [])
        matched = False
        for index, span in enumerate(spans):
            if flags[index]:
                continue
            if (
                span.start_frame_index - match_tolerance_frames
                <= trigger.window_end_frame_index
                <= span.end_frame_index + match_tolerance_frames
            ):
                flags[index] = True
                matched = True
                matched_positive_count += 1
                break
        if not matched:
            false_positive_count += 1

    total_positive_events = len(positive_spans)
    missed_positive_count = total_positive_events - matched_positive_count
    precision = matched_positive_count / max(len(decoded_triggers), 1)
    recall = matched_positive_count / max(total_positive_events, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, _EPSILON)
    return {
        "decoded_trigger_count": len(decoded_triggers),
        "positive_event_count": total_positive_events,
        "detected_positive_event_count": matched_positive_count,
        "false_positive_trigger_count": false_positive_count,
        "missed_positive_event_count": missed_positive_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "match_tolerance_frames": match_tolerance_frames,
    }


def _build_trigger_event(
    run_indices: list[int],
    *,
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
    window_end_timestamps_seconds: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> DecodedTriggerEvent:
    run_array = np.asarray(run_indices, dtype=np.int64)
    peak_local_index = int(np.argmax(probabilities[run_array]))
    peak_window_index = int(run_array[peak_local_index])
    return DecodedTriggerEvent(
        recording_id=str(recording_ids[peak_window_index]),
        window_index=peak_window_index,
        window_end_frame_index=int(window_end_frame_indices[peak_window_index]),
        window_end_timestamp_seconds=float(window_end_timestamps_seconds[peak_window_index]),
        probability=float(probabilities[peak_window_index]),
        threshold=float(threshold),
        run_length=int(run_array.size),
    )


def _build_positive_event_span(
    run_indices: list[int],
    *,
    recording_ids: np.ndarray,
    window_end_frame_indices: np.ndarray,
) -> PositiveEventSpan:
    run_array = np.asarray(run_indices, dtype=np.int64)
    return PositiveEventSpan(
        recording_id=str(recording_ids[int(run_array[0])]),
        start_frame_index=int(window_end_frame_indices[int(run_array[0])]),
        end_frame_index=int(window_end_frame_indices[int(run_array[-1])]),
        window_count=int(run_array.size),
    )


def _accept_or_replace_trigger(
    *,
    accepted: list[DecodedTriggerEvent],
    candidate: DecodedTriggerEvent,
    cooldown_frames: int,
) -> None:
    if not accepted or accepted[-1].recording_id != candidate.recording_id:
        accepted.append(candidate)
        return
    if candidate.window_end_frame_index - accepted[-1].window_end_frame_index > cooldown_frames:
        accepted.append(candidate)
        return
    if candidate.probability > accepted[-1].probability:
        accepted[-1] = candidate


def _ordered_unique_strings(values: np.ndarray) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values.tolist():
        normalized = str(value)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def _validate_sample_aligned_arrays(**arrays: np.ndarray) -> int:
    sample_count: int | None = None
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be rank 1, got shape {array.shape}.")
        if sample_count is None:
            sample_count = int(array.shape[0])
            continue
        if int(array.shape[0]) != sample_count:
            raise ValueError(
                f"{name} has sample count {array.shape[0]}, expected {sample_count}."
            )
    return 0 if sample_count is None else sample_count

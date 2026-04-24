"""Timestamp-based rhythm continuation prediction for accepted gesture events."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Literal

from visionbeat.models import GestureEvent, GestureType

_EPSILON = 1.0e-9


@dataclass(frozen=True, slots=True)
class RhythmPredictionConfig:
    """Configuration for inferring a repeated pulse from gesture timestamps."""

    min_hits: int = 3
    history_size: int = 8
    min_interval_seconds: float = 0.25
    max_interval_seconds: float = 2.0
    jitter_tolerance: float = 0.18
    expiry_ratio: float = 1.75
    max_horizon_seconds: float = 2.0
    match_tolerance_seconds: float = 0.12

    def __post_init__(self) -> None:
        """Validate rhythm-tracker thresholds."""
        if self.min_hits < 3:
            raise ValueError("min_hits must be greater than or equal to 3.")
        if self.history_size < self.min_hits:
            raise ValueError("history_size must be greater than or equal to min_hits.")
        _validate_positive_number(
            self.min_interval_seconds,
            field_name="min_interval_seconds",
        )
        _validate_positive_number(
            self.max_interval_seconds,
            field_name="max_interval_seconds",
        )
        if self.max_interval_seconds <= self.min_interval_seconds:
            raise ValueError("max_interval_seconds must be greater than min_interval_seconds.")
        _validate_non_negative_number(self.jitter_tolerance, field_name="jitter_tolerance")
        _validate_positive_number(self.expiry_ratio, field_name="expiry_ratio")
        if self.expiry_ratio <= 1.0:
            raise ValueError("expiry_ratio must be greater than 1.0.")
        _validate_positive_number(self.max_horizon_seconds, field_name="max_horizon_seconds")
        _validate_non_negative_number(
            self.match_tolerance_seconds,
            field_name="match_tolerance_seconds",
        )


@dataclass(frozen=True, slots=True)
class RhythmObservation:
    """One accepted gesture timestamp that can contribute to a pulse estimate."""

    gesture: GestureType
    timestamp_seconds: float
    confidence: float = 1.0
    source: str = "confirmed"
    frame_index: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the observation payload."""
        object.__setattr__(self, "gesture", GestureType(self.gesture))
        timestamp_seconds = _validate_non_negative_number(
            self.timestamp_seconds,
            field_name="timestamp_seconds",
        )
        confidence = _validate_non_negative_number(self.confidence, field_name="confidence")
        if confidence > 1.0:
            raise ValueError("confidence must be less than or equal to 1.0.")
        source = self.source.strip()
        if not source:
            raise ValueError("source must not be empty.")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("frame_index must be greater than or equal to 0.")
        object.__setattr__(self, "timestamp_seconds", timestamp_seconds)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", source)

    @classmethod
    def from_event(
        cls,
        event: GestureEvent,
        *,
        source: str = "confirmed",
        frame_index: int | None = None,
    ) -> RhythmObservation:
        """Build a rhythm observation from a confirmed gesture event."""
        return cls(
            gesture=event.gesture,
            timestamp_seconds=event.timestamp.seconds,
            confidence=event.confidence,
            source=source,
            frame_index=frame_index,
        )


@dataclass(frozen=True, slots=True)
class RhythmPrediction:
    """Predicted continuation of a stable repeated gesture pulse."""

    prediction_id: str
    gesture: GestureType
    expected_timestamp_seconds: float
    interval_seconds: float
    confidence: float
    jitter_ratio: float
    observation_count: int
    repetition_count: int
    last_observed_timestamp_seconds: float
    expires_after_seconds: float

    def seconds_until_expected(self, *, timestamp_seconds: float) -> float:
        """Return signed seconds until the expected beat time."""
        return self.expected_timestamp_seconds - timestamp_seconds

    def is_expired(self, *, timestamp_seconds: float) -> bool:
        """Return whether this prediction is too stale to be musically useful."""
        return timestamp_seconds > self.expires_after_seconds


RhythmPredictionOutcomeType = Literal["pending", "matched", "missed", "expired"]


@dataclass(frozen=True, slots=True)
class RhythmPredictionOutcome:
    """Evaluation result for one rhythm expectation."""

    prediction_id: str
    outcome: RhythmPredictionOutcomeType
    gesture: GestureType
    predicted_time_seconds: float
    actual_time_seconds: float | None
    actual_gesture: GestureType | None
    error_ms: float | None
    confidence: float
    interval_seconds: float
    repetition_count: int
    jitter_ratio: float
    last_observed_timestamp_seconds: float
    expires_after_seconds: float


@dataclass(frozen=True, slots=True)
class RhythmTrackerUpdate:
    """Result of advancing or observing rhythm state."""

    prediction: RhythmPrediction | None
    outcomes: tuple[RhythmPredictionOutcome, ...]


class RhythmPredictionTracker:
    """Infer next-beat expectations from repeated accepted gesture timestamps."""

    def __init__(self, config: RhythmPredictionConfig | None = None) -> None:
        """Create an empty rhythm tracker."""
        self.config = config or RhythmPredictionConfig()
        self._histories: dict[GestureType, deque[RhythmObservation]] = {}
        self._pending_predictions: dict[str, RhythmPrediction] = {}
        self._completed_prediction_ids: set[str] = set()

    def observe(self, observation: RhythmObservation) -> RhythmPrediction | None:
        """Record one accepted gesture observation and return its next prediction, if any."""
        return self.observe_with_outcomes(observation).prediction

    def observe_with_outcomes(self, observation: RhythmObservation) -> RhythmTrackerUpdate:
        """Record one accepted gesture and classify pending rhythm expectations."""
        observation = RhythmObservation(
            gesture=observation.gesture,
            timestamp_seconds=observation.timestamp_seconds,
            confidence=observation.confidence,
            source=observation.source,
            frame_index=observation.frame_index,
        )
        outcomes = list(self._classify_pending_for_observation(observation))
        self._expire_stale_histories(timestamp_seconds=observation.timestamp_seconds)
        history = self._history_for_mutation(observation.gesture)
        if history and observation.timestamp_seconds <= history[-1].timestamp_seconds:
            raise ValueError(
                "timestamp_seconds must be strictly increasing for each gesture history."
            )
        history.append(observation)
        prediction = self.active_prediction(
            observation.gesture,
            timestamp_seconds=observation.timestamp_seconds,
        )
        pending = self._register_pending_prediction(prediction)
        if pending is not None:
            outcomes.append(pending)
        return RhythmTrackerUpdate(prediction=prediction, outcomes=tuple(outcomes))

    def observe_event(
        self,
        event: GestureEvent,
        *,
        source: str = "confirmed",
        frame_index: int | None = None,
    ) -> RhythmPrediction | None:
        """Record one confirmed gesture event and return its next prediction, if any."""
        return self.observe_event_with_outcomes(
            event,
            source=source,
            frame_index=frame_index,
        ).prediction

    def observe_event_with_outcomes(
        self,
        event: GestureEvent,
        *,
        source: str = "confirmed",
        frame_index: int | None = None,
    ) -> RhythmTrackerUpdate:
        """Record one confirmed gesture event and classify rhythm expectations."""
        return self.observe_with_outcomes(
            RhythmObservation.from_event(
                event,
                source=source,
                frame_index=frame_index,
            )
        )

    def advance(self, *, timestamp_seconds: float) -> tuple[RhythmPrediction, ...]:
        """Expire stale rhythm histories and return the predictions that expired."""
        return self._expire_stale_histories(timestamp_seconds=timestamp_seconds)

    def advance_with_outcomes(self, *, timestamp_seconds: float) -> RhythmTrackerUpdate:
        """Advance time and classify pending expectations without a new event."""
        timestamp_seconds = _validate_non_negative_number(
            timestamp_seconds,
            field_name="timestamp_seconds",
        )
        outcomes = list(self._advance_pending_predictions(timestamp_seconds=timestamp_seconds))
        for prediction in self._expire_stale_histories(timestamp_seconds=timestamp_seconds):
            if prediction.prediction_id in self._completed_prediction_ids:
                continue
            outcome = self._outcome_from_prediction(
                prediction,
                outcome="expired",
                actual_observation=None,
            )
            self._completed_prediction_ids.add(prediction.prediction_id)
            outcomes.append(outcome)
        return RhythmTrackerUpdate(prediction=None, outcomes=tuple(outcomes))

    def _expire_stale_histories(
        self,
        *,
        timestamp_seconds: float,
    ) -> tuple[RhythmPrediction, ...]:
        """Expire stale rhythm histories and return the predictions that expired."""
        timestamp_seconds = _validate_non_negative_number(
            timestamp_seconds,
            field_name="timestamp_seconds",
        )
        expired: list[RhythmPrediction] = []
        for gesture in tuple(self._histories):
            prediction = self._build_prediction(gesture, timestamp_seconds=None)
            if prediction is None or not prediction.is_expired(timestamp_seconds=timestamp_seconds):
                continue
            expired.append(prediction)
            del self._histories[gesture]
        return tuple(expired)

    def _classify_pending_for_observation(
        self,
        observation: RhythmObservation,
    ) -> tuple[RhythmPredictionOutcome, ...]:
        outcomes: list[RhythmPredictionOutcome] = []
        for prediction in sorted(
            self._pending_predictions.values(),
            key=lambda item: item.expected_timestamp_seconds,
        ):
            if prediction.prediction_id in self._completed_prediction_ids:
                continue
            error_seconds = observation.timestamp_seconds - prediction.expected_timestamp_seconds
            same_gesture = observation.gesture is prediction.gesture
            inside_match_window = abs(error_seconds) <= self.config.match_tolerance_seconds
            if same_gesture and inside_match_window:
                outcomes.append(
                    self._complete_pending_prediction(
                        prediction,
                        outcome="matched",
                        actual_observation=observation,
                    )
                )
                continue
            if (
                same_gesture
                or inside_match_window
                or error_seconds > self.config.match_tolerance_seconds
            ):
                outcomes.append(
                    self._complete_pending_prediction(
                        prediction,
                        outcome="missed",
                        actual_observation=observation,
                    )
                )
        return tuple(outcomes)

    def _advance_pending_predictions(
        self,
        *,
        timestamp_seconds: float,
    ) -> tuple[RhythmPredictionOutcome, ...]:
        outcomes: list[RhythmPredictionOutcome] = []
        for prediction in sorted(
            self._pending_predictions.values(),
            key=lambda item: item.expected_timestamp_seconds,
        ):
            if prediction.prediction_id in self._completed_prediction_ids:
                continue
            if (
                timestamp_seconds - prediction.expected_timestamp_seconds
                <= self.config.match_tolerance_seconds
            ):
                continue
            outcomes.append(
                self._complete_pending_prediction(
                    prediction,
                    outcome="missed",
                    actual_observation=None,
                )
            )
        return tuple(outcomes)

    def _register_pending_prediction(
        self,
        prediction: RhythmPrediction | None,
    ) -> RhythmPredictionOutcome | None:
        if prediction is None:
            return None
        if prediction.prediction_id in self._completed_prediction_ids:
            return None
        if prediction.prediction_id in self._pending_predictions:
            return None
        self._pending_predictions[prediction.prediction_id] = prediction
        return self._outcome_from_prediction(
            prediction,
            outcome="pending",
            actual_observation=None,
        )

    def _complete_pending_prediction(
        self,
        prediction: RhythmPrediction,
        *,
        outcome: RhythmPredictionOutcomeType,
        actual_observation: RhythmObservation | None,
    ) -> RhythmPredictionOutcome:
        self._pending_predictions.pop(prediction.prediction_id, None)
        self._completed_prediction_ids.add(prediction.prediction_id)
        return self._outcome_from_prediction(
            prediction,
            outcome=outcome,
            actual_observation=actual_observation,
        )

    def _outcome_from_prediction(
        self,
        prediction: RhythmPrediction,
        *,
        outcome: RhythmPredictionOutcomeType,
        actual_observation: RhythmObservation | None,
    ) -> RhythmPredictionOutcome:
        actual_time = (
            None if actual_observation is None else actual_observation.timestamp_seconds
        )
        error_ms = (
            None
            if actual_time is None
            else (actual_time - prediction.expected_timestamp_seconds) * 1000.0
        )
        return RhythmPredictionOutcome(
            prediction_id=prediction.prediction_id,
            outcome=outcome,
            gesture=prediction.gesture,
            predicted_time_seconds=prediction.expected_timestamp_seconds,
            actual_time_seconds=actual_time,
            actual_gesture=None if actual_observation is None else actual_observation.gesture,
            error_ms=error_ms,
            confidence=prediction.confidence,
            interval_seconds=prediction.interval_seconds,
            repetition_count=prediction.repetition_count,
            jitter_ratio=prediction.jitter_ratio,
            last_observed_timestamp_seconds=prediction.last_observed_timestamp_seconds,
            expires_after_seconds=prediction.expires_after_seconds,
        )

    def active_prediction(
        self,
        gesture: GestureType,
        *,
        timestamp_seconds: float | None = None,
    ) -> RhythmPrediction | None:
        """Return the active prediction for one gesture, if its pulse is stable."""
        if timestamp_seconds is not None:
            timestamp_seconds = _validate_non_negative_number(
                timestamp_seconds,
                field_name="timestamp_seconds",
            )
        return self._build_prediction(GestureType(gesture), timestamp_seconds=timestamp_seconds)

    def active_predictions(
        self,
        *,
        timestamp_seconds: float | None = None,
    ) -> tuple[RhythmPrediction, ...]:
        """Return active predictions for all independently tracked gesture histories."""
        predictions: list[RhythmPrediction] = []
        for gesture in sorted(self._histories, key=lambda item: item.value):
            prediction = self.active_prediction(gesture, timestamp_seconds=timestamp_seconds)
            if prediction is not None:
                predictions.append(prediction)
        return tuple(predictions)

    def history_for(self, gesture: GestureType) -> tuple[RhythmObservation, ...]:
        """Return the retained observation history for a gesture."""
        return tuple(self._histories.get(GestureType(gesture), ()))

    def _history_for_mutation(self, gesture: GestureType) -> deque[RhythmObservation]:
        if gesture not in self._histories:
            self._histories[gesture] = deque(maxlen=self.config.history_size)
        return self._histories[gesture]

    def _build_prediction(
        self,
        gesture: GestureType,
        *,
        timestamp_seconds: float | None,
    ) -> RhythmPrediction | None:
        observations = tuple(self._histories.get(gesture, ()))
        if len(observations) < self.config.min_hits:
            return None
        intervals = _intervals(observations)
        if not intervals:
            return None
        if any(
            interval < self.config.min_interval_seconds
            or interval > self.config.max_interval_seconds
            for interval in intervals
        ):
            return None
        interval_seconds = float(median(intervals))
        if interval_seconds > self.config.max_horizon_seconds:
            return None
        jitter_ratio = _normalized_median_absolute_deviation(
            intervals,
            center=interval_seconds,
        )
        if jitter_ratio > self.config.jitter_tolerance:
            return None
        last_observed = observations[-1].timestamp_seconds
        expected = last_observed + interval_seconds
        expires_after = last_observed + (interval_seconds * self.config.expiry_ratio)
        if timestamp_seconds is not None and timestamp_seconds > expires_after:
            return None
        confidence = self._confidence_for(
            observations=observations,
            jitter_ratio=jitter_ratio,
            timestamp_seconds=timestamp_seconds,
            expected_timestamp_seconds=expected,
            expires_after_seconds=expires_after,
        )
        return RhythmPrediction(
            prediction_id=_prediction_id(
                gesture=gesture,
                last_observed_timestamp_seconds=last_observed,
                expected_timestamp_seconds=expected,
            ),
            gesture=gesture,
            expected_timestamp_seconds=expected,
            interval_seconds=interval_seconds,
            confidence=confidence,
            jitter_ratio=jitter_ratio,
            observation_count=len(observations),
            repetition_count=len(intervals),
            last_observed_timestamp_seconds=last_observed,
            expires_after_seconds=expires_after,
        )

    def _confidence_for(
        self,
        *,
        observations: tuple[RhythmObservation, ...],
        jitter_ratio: float,
        timestamp_seconds: float | None,
        expected_timestamp_seconds: float,
        expires_after_seconds: float,
    ) -> float:
        observation_confidence = float(
            median(observation.confidence for observation in observations)
        )
        if self.config.jitter_tolerance <= _EPSILON:
            stability_score = 1.0 if jitter_ratio <= _EPSILON else 0.0
        else:
            stability_score = 1.0 - (jitter_ratio / self.config.jitter_tolerance)
        confidence = observation_confidence * (0.5 + 0.5 * _clamp(stability_score))
        if timestamp_seconds is None or timestamp_seconds <= expected_timestamp_seconds:
            return _clamp(confidence)
        decay_span = max(expires_after_seconds - expected_timestamp_seconds, _EPSILON)
        late_ratio = (timestamp_seconds - expected_timestamp_seconds) / decay_span
        return _clamp(confidence * (1.0 - late_ratio))


def _intervals(observations: tuple[RhythmObservation, ...]) -> tuple[float, ...]:
    return tuple(
        current.timestamp_seconds - previous.timestamp_seconds
        for previous, current in zip(observations, observations[1:], strict=False)
    )


def _normalized_median_absolute_deviation(
    values: Iterable[float],
    *,
    center: float,
) -> float:
    deviations = tuple(abs(value - center) for value in values)
    if not deviations:
        return 0.0
    return float(median(deviations)) / max(center, _EPSILON)


def _prediction_id(
    *,
    gesture: GestureType,
    last_observed_timestamp_seconds: float,
    expected_timestamp_seconds: float,
) -> str:
    return (
        f"{gesture.value}:"
        f"{last_observed_timestamp_seconds:.6f}->{expected_timestamp_seconds:.6f}"
    )


def _validate_positive_number(value: float, *, field_name: str) -> float:
    numeric = _validate_non_negative_number(value, field_name=field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be greater than 0.0.")
    return numeric


def _validate_non_negative_number(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    if numeric < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.0.")
    return numeric


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "RhythmObservation",
    "RhythmPrediction",
    "RhythmPredictionConfig",
    "RhythmPredictionOutcome",
    "RhythmPredictionOutcomeType",
    "RhythmPredictionTracker",
    "RhythmTrackerUpdate",
]

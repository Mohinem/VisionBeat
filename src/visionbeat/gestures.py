"""Gesture recognition logic isolated from camera and MediaPipe dependencies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from visionbeat.config import GestureConfig
from visionbeat.math_utils import l1_velocity
from visionbeat.models import (
    DetectionCandidate,
    FrameTimestamp,
    GestureEvent,
    GestureType,
    LandmarkPoint,
    TrackerOutput,
)


@dataclass(frozen=True, slots=True)
class MotionSample:
    """Temporal sample of a tracked wrist position."""

    timestamp: float
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class MotionMetrics:
    """Aggregate motion statistics computed across a recent temporal window."""

    elapsed: float
    delta_x: float
    delta_y: float
    delta_z: float
    net_velocity: float
    peak_x_velocity: float
    peak_y_velocity: float
    peak_z_velocity: float

    @property
    def forward_velocity(self) -> float:
        """Return the strongest forward wrist velocity in normalized z-space."""
        return max(0.0, -self.peak_z_velocity)

    @property
    def downward_velocity(self) -> float:
        """Return the strongest downward wrist velocity in normalized y-space."""
        return max(0.0, self.peak_y_velocity)

    @property
    def punch_axis_ratio(self) -> float:
        """Return forward motion dominance over lateral and vertical drift."""
        return abs(self.delta_z) / max(abs(self.delta_x) + abs(self.delta_y), 1e-6)

    @property
    def strike_axis_ratio(self) -> float:
        """Return downward motion dominance over lateral and depth drift."""
        return abs(self.delta_y) / max(abs(self.delta_x) + abs(self.delta_z), 1e-6)


@dataclass(frozen=True, slots=True)
class PendingGesture:
    """Candidate gesture state awaiting final confirmation."""

    gesture: GestureType
    started_at: float
    label: str


@dataclass(slots=True)
class HandHistory:
    """Recent wrist samples plus cooldown and candidate state for a tracked hand."""

    samples: deque[MotionSample] = field(default_factory=deque)
    last_trigger_time: float = -1_000.0
    pending: PendingGesture | None = None


COMPARISON_EPSILON = 1e-9


class GestureDetector:
    """Detect percussion gestures from normalized wrist trajectories."""

    def __init__(self, config: GestureConfig) -> None:
        """Store gesture thresholds and initialize state."""
        self.config = config
        self._histories: dict[str, HandHistory] = {
            "left": HandHistory(deque(maxlen=config.history_size)),
            "right": HandHistory(deque(maxlen=config.history_size)),
        }
        self._last_candidates: tuple[DetectionCandidate, ...] = ()

    @property
    def candidates(self) -> tuple[DetectionCandidate, ...]:
        """Return the currently active pending gesture candidates."""
        return self._last_candidates

    def update(self, frame: TrackerOutput) -> list[GestureEvent]:
        """Consume tracker output and emit any newly detected gestures."""
        events: list[GestureEvent] = []
        candidates: list[DetectionCandidate] = []
        for hand in ("left", "right"):
            history = self._histories[hand]
            wrist = frame.get(f"{hand}_wrist")
            if wrist is None or wrist.visibility < 0.5:
                history.pending = None
                continue
            self._append_sample(history, frame.timestamp.seconds, wrist)
            candidate, event = self._evaluate_hand(hand, history, frame.timestamp)
            if candidate is not None:
                candidates.append(candidate)
            if event is not None:
                events.append(event)
        self._last_candidates = tuple(candidates)
        return events

    def _append_sample(self, history: HandHistory, timestamp: float, wrist: LandmarkPoint) -> None:
        """Append a wrist sample and discard samples outside the configured time window."""
        history.samples.append(MotionSample(timestamp, wrist.x, wrist.y, wrist.z))
        min_timestamp = timestamp - self.config.analysis_window_seconds
        while len(history.samples) > 1 and history.samples[0].timestamp < min_timestamp:
            history.samples.popleft()

    def _evaluate_hand(
        self,
        hand: str,
        history: HandHistory,
        timestamp: FrameTimestamp,
    ) -> tuple[DetectionCandidate | None, GestureEvent | None]:
        """Update candidate state and return any confirmed gesture for one hand."""
        if len(history.samples) < 2 or hand != self.config.active_hand:
            history.pending = None
            return None, None

        if timestamp.seconds - history.last_trigger_time < self.config.cooldown_seconds:
            history.pending = None
            return None, None

        metrics = self._compute_metrics(history.samples)
        if metrics is None:
            history.pending = None
            return None, None

        if history.pending is not None and (
            timestamp.seconds - history.pending.started_at > self.config.confirmation_window_seconds
            or not self._candidate_is_still_valid(history.pending.gesture, metrics)
        ):
            history.pending = None

        if history.pending is None:
            history.pending = self._start_candidate(timestamp.seconds, metrics)
            if history.pending is None:
                return None, None
            return self._build_candidate(history.pending.gesture, hand, metrics), None

        candidate = self._build_candidate(history.pending.gesture, hand, metrics)
        if not self._is_confirmed(history.pending.gesture, metrics):
            return candidate, None

        gesture = history.pending.gesture
        history.pending = None
        history.last_trigger_time = timestamp.seconds
        return None, self._build_event(gesture, hand, timestamp, metrics)

    def _start_candidate(self, timestamp: float, metrics: MotionMetrics) -> PendingGesture | None:
        """Create a pending gesture if the current window has valid onset motion."""
        if self._meets_punch_candidate(metrics):
            return PendingGesture(
                gesture=GestureType.KICK,
                started_at=timestamp,
                label="Forward punch candidate",
            )
        if self._meets_strike_candidate(metrics):
            return PendingGesture(
                gesture=GestureType.SNARE,
                started_at=timestamp,
                label="Downward strike candidate",
            )
        return None

    def _candidate_is_still_valid(self, gesture: GestureType, metrics: MotionMetrics) -> bool:
        """Return whether a pending gesture still matches its expected motion direction."""
        if gesture is GestureType.KICK:
            return self._meets_punch_candidate(metrics)
        return self._meets_strike_candidate(metrics)

    def _build_candidate(
        self,
        gesture: GestureType,
        hand: str,
        metrics: MotionMetrics,
    ) -> DetectionCandidate:
        """Convert pending state into a public candidate payload."""
        if gesture is GestureType.KICK:
            confidence = min(
                1.0,
                max(
                    abs(metrics.delta_z) / self.config.punch_forward_delta_z,
                    metrics.forward_velocity / self.config.min_velocity,
                )
                * 0.5,
            )
            return DetectionCandidate(
                gesture=gesture,
                confidence=confidence,
                hand=hand,
                label="Forward punch candidate",
            )
        confidence = min(
            1.0,
            max(
                metrics.delta_y / self.config.strike_down_delta_y,
                metrics.downward_velocity / self.config.min_velocity,
            )
            * 0.5,
        )
        return DetectionCandidate(
            gesture=gesture,
            confidence=confidence,
            hand=hand,
            label="Downward strike candidate",
        )

    def _build_event(
        self,
        gesture: GestureType,
        hand: str,
        timestamp: FrameTimestamp,
        metrics: MotionMetrics,
    ) -> GestureEvent:
        """Create a confirmed gesture event for the provided motion window."""
        if gesture is GestureType.KICK:
            confidence = min(
                1.0,
                abs(metrics.delta_z) / (self.config.punch_forward_delta_z * 1.25),
            )
            label = "Forward punch → kick"
        else:
            confidence = min(
                1.0,
                metrics.delta_y / (self.config.strike_down_delta_y * 1.25),
            )
            label = "Downward strike → snare"
        return GestureEvent(
            gesture=gesture,
            confidence=confidence,
            hand=hand,
            timestamp=timestamp,
            label=label,
        )

    def _compute_metrics(self, samples: Iterable[MotionSample]) -> MotionMetrics | None:
        """Summarize displacement and per-frame velocities for a temporal wrist window."""
        sample_list = list(samples)
        if len(sample_list) < 2:
            return None

        oldest = sample_list[0]
        newest = sample_list[-1]
        elapsed = newest.timestamp - oldest.timestamp
        if elapsed <= 0.0:
            return None

        deltas_x: list[float] = []
        deltas_y: list[float] = []
        deltas_z: list[float] = []
        for previous, current in zip(sample_list, sample_list[1:], strict=False):
            frame_elapsed = current.timestamp - previous.timestamp
            if frame_elapsed <= 0.0:
                continue
            deltas_x.append((current.x - previous.x) / frame_elapsed)
            deltas_y.append((current.y - previous.y) / frame_elapsed)
            deltas_z.append((current.z - previous.z) / frame_elapsed)

        if not deltas_x or not deltas_y or not deltas_z:
            return None

        delta_x = newest.x - oldest.x
        delta_y = newest.y - oldest.y
        delta_z = newest.z - oldest.z
        return MotionMetrics(
            elapsed=elapsed,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_z=delta_z,
            net_velocity=l1_velocity((delta_x, delta_y, delta_z), elapsed),
            peak_x_velocity=max(deltas_x, key=abs),
            peak_y_velocity=max(deltas_y, key=abs),
            peak_z_velocity=max(deltas_z, key=abs),
        )

    def _meets_punch_candidate(self, metrics: MotionMetrics) -> bool:
        """Return whether current motion is a viable punch candidate."""
        return (
            metrics.delta_z
            <= -(self.config.punch_forward_delta_z * self.config.candidate_ratio)
            + COMPARISON_EPSILON
            and abs(metrics.delta_y)
            <= self.config.punch_max_vertical_drift + COMPARISON_EPSILON
            and metrics.forward_velocity
            >= self.config.min_velocity * self.config.candidate_ratio - COMPARISON_EPSILON
            and metrics.net_velocity
            >= self.config.min_velocity * self.config.candidate_ratio - COMPARISON_EPSILON
            and metrics.punch_axis_ratio
            >= self.config.axis_dominance_ratio - COMPARISON_EPSILON
        )

    def _meets_strike_candidate(self, metrics: MotionMetrics) -> bool:
        """Return whether current motion is a viable downward-strike candidate."""
        return (
            metrics.delta_y
            >= self.config.strike_down_delta_y * self.config.candidate_ratio
            - COMPARISON_EPSILON
            and abs(metrics.delta_z)
            <= self.config.strike_max_depth_drift + COMPARISON_EPSILON
            and metrics.downward_velocity
            >= self.config.min_velocity * self.config.candidate_ratio - COMPARISON_EPSILON
            and metrics.net_velocity
            >= self.config.min_velocity * self.config.candidate_ratio - COMPARISON_EPSILON
            and metrics.strike_axis_ratio
            >= self.config.axis_dominance_ratio - COMPARISON_EPSILON
        )

    def _is_confirmed(self, gesture: GestureType, metrics: MotionMetrics) -> bool:
        """Return whether a pending gesture has crossed its final trigger threshold."""
        if gesture is GestureType.KICK:
            return (
                metrics.delta_z
                <= -self.config.punch_forward_delta_z + COMPARISON_EPSILON
                and abs(metrics.delta_y)
                <= self.config.punch_max_vertical_drift + COMPARISON_EPSILON
                and metrics.forward_velocity
                >= self.config.min_velocity - COMPARISON_EPSILON
                and metrics.net_velocity >= self.config.min_velocity - COMPARISON_EPSILON
                and metrics.punch_axis_ratio
                >= self.config.axis_dominance_ratio - COMPARISON_EPSILON
            )
        return (
            metrics.delta_y >= self.config.strike_down_delta_y - COMPARISON_EPSILON
            and abs(metrics.delta_z)
            <= self.config.strike_max_depth_drift + COMPARISON_EPSILON
            and metrics.downward_velocity
            >= self.config.min_velocity - COMPARISON_EPSILON
            and metrics.net_velocity >= self.config.min_velocity - COMPARISON_EPSILON
            and metrics.strike_axis_ratio
            >= self.config.axis_dominance_ratio - COMPARISON_EPSILON
        )

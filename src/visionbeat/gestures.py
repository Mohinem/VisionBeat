"""Gesture recognition logic isolated from camera and MediaPipe dependencies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

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
from visionbeat.observability import GestureObservationEvent, VelocityStats


class GestureObserver(Protocol):
    """Observer interface used to emit structured gesture-analysis events."""

    def log_gesture_candidate(self, event: GestureObservationEvent) -> None:
        """Record a pending gesture candidate."""

    def log_confirmed_trigger(self, event: GestureObservationEvent) -> None:
        """Record a confirmed trigger."""

    def log_cooldown_suppression(self, event: GestureObservationEvent) -> None:
        """Record a gesture that was suppressed by cooldown."""


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

    def to_velocity_stats(self) -> VelocityStats:
        """Convert gesture metrics into the observability velocity schema."""
        return VelocityStats(
            elapsed=self.elapsed,
            delta_x=self.delta_x,
            delta_y=self.delta_y,
            delta_z=self.delta_z,
            net_velocity=self.net_velocity,
            peak_x_velocity=self.peak_x_velocity,
            peak_y_velocity=self.peak_y_velocity,
            peak_z_velocity=self.peak_z_velocity,
        )


@dataclass(frozen=True, slots=True)
class PendingGesture:
    """Candidate gesture state awaiting final confirmation."""

    gesture: GestureType
    started_at: float
    label: str


@dataclass(frozen=True, slots=True)
class RecoveryState:
    """Post-trigger recovery gate that prevents repeated hits from one motion."""

    gesture: GestureType
    anchor: MotionSample


@dataclass(slots=True)
class HandHistory:
    """Recent wrist samples plus cooldown and candidate state for a tracked hand."""

    samples: deque[MotionSample] = field(default_factory=deque)
    last_trigger_time: float = -1_000.0
    pending: PendingGesture | None = None
    recovery: RecoveryState | None = None


COMPARISON_EPSILON = 1e-9
SHOULDER_DEPTH_COMPENSATION = 0.7


class GestureDetector:
    """Detect percussion gestures from normalized wrist trajectories."""

    def __init__(self, config: GestureConfig, *, observer: GestureObserver | None = None) -> None:
        """Store gesture thresholds, initialize state, and configure observability."""
        self.config = config
        self.observer = observer
        self._histories: dict[str, HandHistory] = {
            "left": HandHistory(deque(maxlen=config.history_size)),
            "right": HandHistory(deque(maxlen=config.history_size)),
        }
        self._last_candidates: tuple[DetectionCandidate, ...] = ()

    @property
    def candidates(self) -> tuple[DetectionCandidate, ...]:
        """Return the currently active pending gesture candidates."""
        return self._last_candidates

    def cooldown_remaining(self, timestamp: FrameTimestamp | float) -> float:
        """Return the remaining detector cooldown for the configured active hand."""
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        history = self._histories[self.config.active_hand]
        elapsed = seconds - history.last_trigger_time
        return max(0.0, self.config.cooldown_seconds - elapsed)

    def status_summary(self, timestamp: FrameTimestamp | float | None = None) -> str:
        """Return a short human-readable summary of the active hand detector state."""
        history = self._histories[self.config.active_hand]
        if history.recovery is not None:
            return f"recovering {history.recovery.gesture.value}"
        if history.pending is not None:
            return history.pending.label
        if timestamp is not None:
            cooldown = self.cooldown_remaining(timestamp)
            if cooldown > 0.0:
                return f"cooldown {cooldown:.2f}s"
        return "armed"

    def update(self, frame: TrackerOutput) -> list[GestureEvent]:
        """Consume tracker output and emit any newly detected gestures."""
        events: list[GestureEvent] = []
        candidates: list[DetectionCandidate] = []
        for hand in ("left", "right"):
            history = self._histories[hand]
            wrist = frame.get(f"{hand}_wrist")
            if wrist is None or wrist.visibility < 0.5:
                self._reset_history_state(history)
                continue
            self._append_sample(
                history,
                self._resolve_motion_sample(frame, hand, wrist),
            )
            candidate, event = self._evaluate_hand(hand, history, frame.timestamp)
            if candidate is not None:
                candidates.append(candidate)
            if event is not None:
                events.append(event)
        self._last_candidates = tuple(candidates)
        return events

    def _append_sample(self, history: HandHistory, sample: MotionSample) -> None:
        """Append a wrist sample and discard samples outside the configured time window."""
        history.samples.append(sample)
        min_timestamp = sample.timestamp - self.config.analysis_window_seconds
        while len(history.samples) > 1 and history.samples[0].timestamp < min_timestamp:
            history.samples.popleft()

    def _resolve_motion_sample(
        self,
        frame: TrackerOutput,
        hand: str,
        wrist: LandmarkPoint,
    ) -> MotionSample:
        """Return a wrist sample with partial shoulder compensation for body sway."""
        shoulder = frame.get(f"{hand}_shoulder")
        if shoulder is None or shoulder.visibility < 0.5:
            return MotionSample(
                timestamp=frame.timestamp.seconds,
                x=wrist.x,
                y=wrist.y,
                z=wrist.z,
            )
        return MotionSample(
            timestamp=frame.timestamp.seconds,
            x=wrist.x - shoulder.x,
            y=wrist.y - shoulder.y,
            z=wrist.z - (shoulder.z * SHOULDER_DEPTH_COMPENSATION),
        )

    def _reset_history_state(self, history: HandHistory) -> None:
        """Clear transient state when a wrist cannot be tracked reliably."""
        history.samples.clear()
        history.pending = None
        history.recovery = None

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

        metrics = self._compute_metrics(history.samples)
        if metrics is None:
            history.pending = None
            return None, None

        if history.recovery is not None:
            if not self._recovery_complete(history.recovery, history.samples[-1], metrics):
                history.pending = None
                return None, None
            history.recovery = None
            history.pending = None
            history.samples = deque((history.samples[-1],), maxlen=self.config.history_size)
            return None, None

        if timestamp.seconds - history.last_trigger_time < self.config.cooldown_seconds:
            history.pending = None
            self._emit_cooldown_suppressed(timestamp, hand, metrics)
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
            candidate = self._build_candidate(history.pending.gesture, hand, metrics)
            self._emit_candidate(timestamp, candidate, metrics)
            if self._is_confirmed(history.pending.gesture, metrics):
                event = self._confirm_pending(history, hand, timestamp, metrics)
                return None, event
            return candidate, None

        candidate = self._build_candidate(history.pending.gesture, hand, metrics)
        if not self._is_confirmed(history.pending.gesture, metrics):
            return candidate, None

        event = self._confirm_pending(history, hand, timestamp, metrics)
        return None, event

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

    def _confirm_pending(
        self,
        history: HandHistory,
        hand: str,
        timestamp: FrameTimestamp,
        metrics: MotionMetrics,
    ) -> GestureEvent:
        """Finalize the active pending gesture into a confirmed event."""
        assert history.pending is not None
        gesture = history.pending.gesture
        anchor = history.samples[-1]
        history.pending = None
        history.last_trigger_time = timestamp.seconds
        history.recovery = RecoveryState(gesture=gesture, anchor=anchor)
        history.samples = deque((anchor,), maxlen=self.config.history_size)
        event = self._build_event(gesture, hand, timestamp, metrics)
        self._emit_trigger(event, metrics)
        return event

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

        filtered_samples = self._smooth_samples(sample_list)

        deltas_x: list[float] = []
        deltas_y: list[float] = []
        deltas_z: list[float] = []
        for previous, current in zip(filtered_samples, filtered_samples[1:], strict=False):
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

    def _smooth_samples(self, samples: list[MotionSample]) -> list[MotionSample]:
        """Smooth sample positions before computing per-frame peak velocities."""
        if len(samples) < 2:
            return samples

        alpha = self.config.velocity_smoothing_alpha
        if alpha >= 1.0 - COMPARISON_EPSILON:
            return samples

        smoothed = [samples[0]]
        for sample in samples[1:]:
            previous = smoothed[-1]
            smoothed.append(
                MotionSample(
                    timestamp=sample.timestamp,
                    x=previous.x + ((sample.x - previous.x) * alpha),
                    y=previous.y + ((sample.y - previous.y) * alpha),
                    z=previous.z + ((sample.z - previous.z) * alpha),
                )
            )
        return smoothed

    def _recovery_complete(
        self,
        recovery: RecoveryState,
        current: MotionSample,
        metrics: MotionMetrics,
    ) -> bool:
        """Return whether the hand has reset enough to arm a new gesture."""
        required_distance = self._rearm_distance(recovery.gesture)
        partial_distance = required_distance * 0.5
        required_velocity = self.config.min_velocity * self.config.rearm_threshold_ratio
        if recovery.gesture is GestureType.KICK:
            reverse_distance = current.z - recovery.anchor.z
            reverse_velocity = max(0.0, metrics.peak_z_velocity)
        else:
            reverse_distance = recovery.anchor.y - current.y
            reverse_velocity = max(0.0, -metrics.peak_y_velocity)
        return (
            reverse_distance >= required_distance - COMPARISON_EPSILON
            or (
                reverse_distance >= partial_distance - COMPARISON_EPSILON
                and reverse_velocity >= required_velocity - COMPARISON_EPSILON
            )
        )

    def _rearm_distance(self, gesture: GestureType) -> float:
        """Return the minimum opposite-direction travel needed after a trigger."""
        if gesture is GestureType.KICK:
            return self.config.punch_forward_delta_z * self.config.rearm_threshold_ratio
        return self.config.strike_down_delta_y * self.config.rearm_threshold_ratio

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
        """Return whether current motion is a viable strike candidate."""
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
        """Return whether a pending gesture now exceeds the final trigger thresholds."""
        if gesture is GestureType.KICK:
            return (
                metrics.delta_z <= -self.config.punch_forward_delta_z + COMPARISON_EPSILON
                and abs(metrics.delta_y)
                <= self.config.punch_max_vertical_drift + COMPARISON_EPSILON
                and metrics.forward_velocity >= self.config.min_velocity - COMPARISON_EPSILON
                and metrics.net_velocity >= self.config.min_velocity - COMPARISON_EPSILON
                and metrics.punch_axis_ratio
                >= self.config.axis_dominance_ratio - COMPARISON_EPSILON
            )
        return (
            metrics.delta_y >= self.config.strike_down_delta_y - COMPARISON_EPSILON
            and abs(metrics.delta_z) <= self.config.strike_max_depth_drift + COMPARISON_EPSILON
            and metrics.downward_velocity >= self.config.min_velocity - COMPARISON_EPSILON
            and metrics.net_velocity >= self.config.min_velocity - COMPARISON_EPSILON
            and metrics.strike_axis_ratio
            >= self.config.axis_dominance_ratio - COMPARISON_EPSILON
        )

    def _emit_candidate(
        self,
        timestamp: FrameTimestamp,
        candidate: DetectionCandidate,
        metrics: MotionMetrics,
    ) -> None:
        """Send a gesture-candidate observation to the configured observer."""
        if self.observer is None:
            return
        self.observer.log_gesture_candidate(
            GestureObservationEvent(
                timestamp=timestamp.seconds,
                event_kind="candidate",
                gesture_type=candidate.gesture,
                accepted=False,
                reason=candidate.label,
                velocity_stats=metrics.to_velocity_stats(),
                confidence=candidate.confidence,
                hand=candidate.hand,
            )
        )

    def _emit_trigger(self, event: GestureEvent, metrics: MotionMetrics) -> None:
        """Send a confirmed-trigger observation to the configured observer."""
        if self.observer is None:
            return
        self.observer.log_confirmed_trigger(
            GestureObservationEvent(
                timestamp=event.timestamp.seconds,
                event_kind="trigger",
                gesture_type=event.gesture,
                accepted=True,
                reason=event.label,
                velocity_stats=metrics.to_velocity_stats(),
                confidence=event.confidence,
                hand=event.hand,
            )
        )

    def _emit_cooldown_suppressed(
        self,
        timestamp: FrameTimestamp,
        hand: str,
        metrics: MotionMetrics,
    ) -> None:
        """Send a cooldown-suppression observation to the configured observer."""
        if self.observer is None:
            return
        self.observer.log_cooldown_suppression(
            GestureObservationEvent(
                timestamp=timestamp.seconds,
                event_kind="cooldown_suppressed",
                gesture_type=None,
                accepted=False,
                reason="cooldown_active",
                velocity_stats=metrics.to_velocity_stats(),
                confidence=None,
                hand=hand,
            )
        )

"""Gesture recognition logic isolated from camera and MediaPipe dependencies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import hypot
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
    shoulder_relative: bool = False


@dataclass(frozen=True, slots=True)
class MotionMetrics:
    """Aggregate motion statistics computed across a recent temporal window."""

    elapsed: float
    delta_x: float
    delta_abs_x: float
    delta_y: float
    delta_z: float
    net_velocity: float
    peak_x_velocity: float
    peak_abs_x_velocity: float
    peak_y_velocity: float
    peak_z_velocity: float

    @property
    def downward_velocity(self) -> float:
        """Return the strongest downward wrist velocity in normalized y-space."""
        return max(0.0, self.peak_y_velocity)

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
class WristCollisionSample:
    """Temporal sample of bilateral wrist separation."""

    timestamp: float
    x_gap: float
    y_gap: float
    z_gap: float
    distance_xy: float


@dataclass(frozen=True, slots=True)
class WristCollisionMetrics:
    """Aggregate motion statistics for bilateral wrist convergence."""

    elapsed: float
    delta_x_gap: float
    delta_y_gap: float
    delta_z_gap: float
    delta_distance_xy: float
    current_distance_xy: float
    current_depth_gap: float
    net_velocity: float
    peak_x_velocity: float
    peak_y_velocity: float
    peak_z_velocity: float
    peak_distance_velocity: float
    peak_closing_velocity: float | None = None
    peak_opening_velocity: float | None = None

    @property
    def closing_velocity(self) -> float:
        """Return the strongest wrist-closing velocity in image space."""
        if self.peak_closing_velocity is not None:
            return max(0.0, -self.peak_closing_velocity)
        return max(0.0, -self.peak_distance_velocity)

    @property
    def opening_velocity(self) -> float:
        """Return the strongest wrist-opening velocity in image space."""
        if self.peak_opening_velocity is not None:
            return max(0.0, self.peak_opening_velocity)
        return max(0.0, self.peak_distance_velocity)

    def to_velocity_stats(self) -> VelocityStats:
        """Convert pair metrics into the observability velocity schema."""
        return VelocityStats(
            elapsed=self.elapsed,
            delta_x=self.delta_x_gap,
            delta_y=self.delta_y_gap,
            delta_z=self.delta_z_gap,
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
    hand: str
    anchor: MotionSample


@dataclass(frozen=True, slots=True)
class CollisionRecoveryState:
    """Post-trigger recovery gate for bilateral snare collisions."""

    anchor_distance_xy: float


@dataclass(slots=True)
class HandHistory:
    """Recent wrist samples plus cooldown and candidate state for a tracked hand."""

    samples: deque[MotionSample] = field(default_factory=deque)
    last_trigger_time: float = -1_000.0
    pending: PendingGesture | None = None
    recovery: RecoveryState | None = None


@dataclass(slots=True)
class CollisionHistory:
    """Recent bilateral wrist-separation samples plus candidate state."""

    samples: deque[WristCollisionSample] = field(default_factory=deque)
    last_trigger_time: float = -1_000.0
    pending: PendingGesture | None = None
    recovery: CollisionRecoveryState | None = None


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
        self._collision_history = CollisionHistory(deque(maxlen=config.history_size))
        self._last_candidates: tuple[DetectionCandidate, ...] = ()

    @property
    def candidates(self) -> tuple[DetectionCandidate, ...]:
        """Return the currently active pending gesture candidates."""
        return self._last_candidates

    def cooldown_remaining(self, timestamp: FrameTimestamp | float) -> float:
        """Return the remaining detector cooldown across active gesture paths."""
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        active_history = self._histories[self.config.active_hand]
        return max(
            self._cooldown_remaining_for(active_history.last_trigger_time, seconds),
            self._cooldown_remaining_for(self._collision_history.last_trigger_time, seconds),
        )

    def status_summary(self, timestamp: FrameTimestamp | float | None = None) -> str:
        """Return a short human-readable summary of the detector state."""
        active_history = self._histories[self.config.active_hand]
        if self._collision_history.recovery is not None:
            return "recovering snare"
        if active_history.recovery is not None:
            return f"recovering {active_history.recovery.gesture.value}"
        if self._collision_history.pending is not None:
            return self._collision_history.pending.label
        if active_history.pending is not None:
            return active_history.pending.label
        if timestamp is not None:
            cooldown = self.cooldown_remaining(timestamp)
            if cooldown > 0.0:
                return f"cooldown {cooldown:.2f}s"
        return "armed"

    def update(self, frame: TrackerOutput) -> list[GestureEvent]:
        """Consume tracker output and emit any newly detected gestures."""
        events: list[GestureEvent] = []
        candidates: list[DetectionCandidate] = []
        tracked_wrists: dict[str, LandmarkPoint | None] = {"left": None, "right": None}

        for hand in ("left", "right"):
            history = self._histories[hand]
            wrist = frame.get(f"{hand}_wrist")
            if wrist is None or wrist.visibility < 0.5:
                self._reset_history_state(history)
                continue
            tracked_wrists[hand] = wrist
            self._append_sample(history, self._resolve_motion_sample(frame, hand, wrist))

        collision_candidate, collision_event = self._evaluate_collision(
            frame,
            left_wrist=tracked_wrists["left"],
            right_wrist=tracked_wrists["right"],
        )
        if collision_candidate is not None:
            candidates.append(collision_candidate)
        if collision_event is not None:
            events.append(collision_event)

        active_hand = self.config.active_hand
        if tracked_wrists[active_hand] is not None:
            kick_candidate, kick_event = self._evaluate_hand(
                active_hand,
                self._histories[active_hand],
                frame.timestamp,
                kick_blocked=collision_candidate is not None or collision_event is not None,
            )
            if kick_candidate is not None:
                candidates.append(kick_candidate)
            if kick_event is not None:
                events.append(kick_event)
        else:
            self._histories[active_hand].pending = None

        self._last_candidates = tuple(candidates)
        return events

    def _cooldown_remaining_for(self, last_trigger_time: float, seconds: float) -> float:
        """Return remaining cooldown for one gesture path."""
        elapsed = seconds - last_trigger_time
        return max(0.0, self.config.cooldown_seconds - elapsed)

    def _append_sample(self, history: HandHistory, sample: MotionSample) -> None:
        """Append a wrist sample and discard samples outside the configured time window."""
        if history.samples and history.samples[-1].shoulder_relative != sample.shoulder_relative:
            self._reset_history_state(history)
        history.samples.append(sample)
        min_timestamp = sample.timestamp - self.config.analysis_window_seconds
        while len(history.samples) > 1 and history.samples[0].timestamp < min_timestamp:
            history.samples.popleft()

    def _append_collision_sample(
        self,
        frame: TrackerOutput,
        left_wrist: LandmarkPoint,
        right_wrist: LandmarkPoint,
    ) -> None:
        """Append a wrist-separation sample for bilateral collision detection."""
        history = self._collision_history
        sample = WristCollisionSample(
            timestamp=frame.timestamp.seconds,
            x_gap=abs(right_wrist.x - left_wrist.x),
            y_gap=abs(right_wrist.y - left_wrist.y),
            z_gap=abs(right_wrist.z - left_wrist.z),
            distance_xy=hypot(right_wrist.x - left_wrist.x, right_wrist.y - left_wrist.y),
        )
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
                shoulder_relative=False,
            )
        return MotionSample(
            timestamp=frame.timestamp.seconds,
            x=wrist.x - shoulder.x,
            y=wrist.y - shoulder.y,
            z=wrist.z - (shoulder.z * SHOULDER_DEPTH_COMPENSATION),
            shoulder_relative=True,
        )

    def _reset_history_state(self, history: HandHistory) -> None:
        """Clear transient state when a wrist cannot be tracked reliably."""
        history.samples.clear()
        history.pending = None
        history.recovery = None

    def _reset_collision_state(self) -> None:
        """Clear transient bilateral state when both wrists are not reliably tracked."""
        self._collision_history.samples.clear()
        self._collision_history.pending = None
        self._collision_history.recovery = None

    def _evaluate_hand(
        self,
        hand: str,
        history: HandHistory,
        timestamp: FrameTimestamp,
        *,
        kick_blocked: bool,
    ) -> tuple[DetectionCandidate | None, GestureEvent | None]:
        """Update candidate state and return any confirmed kick for the active hand."""
        if len(history.samples) < 2:
            history.pending = None
            return None, None

        metrics = self._compute_metrics(history.samples)
        if metrics is None:
            history.pending = None
            return None, None

        if history.recovery is not None:
            if not self._hand_recovery_complete(history.recovery, history.samples[-1], metrics):
                history.pending = None
                return None, None
            history.recovery = None
            history.pending = None
            history.samples = deque((history.samples[-1],), maxlen=self.config.history_size)
            return None, None

        if kick_blocked:
            history.pending = None
            return None, None

        if timestamp.seconds - history.last_trigger_time < self.config.cooldown_seconds:
            history.pending = None
            if self._meets_kick_candidate(metrics):
                self._emit_cooldown_suppressed(timestamp, hand, metrics.to_velocity_stats())
            return None, None

        if history.pending is not None and (
            timestamp.seconds - history.pending.started_at > self.config.confirmation_window_seconds
            or not self._meets_kick_candidate(metrics)
        ):
            history.pending = None

        if history.pending is None:
            if not self._meets_kick_candidate(metrics):
                return None, None
            history.pending = PendingGesture(
                gesture=GestureType.KICK,
                started_at=timestamp.seconds,
                label="Downward strike candidate",
            )
            candidate = self._build_kick_candidate(hand, metrics)
            self._emit_candidate(timestamp, candidate, metrics.to_velocity_stats())
            if self._is_kick_confirmed(metrics):
                event = self._confirm_hand_pending(history, hand, timestamp, metrics)
                return None, event
            return candidate, None

        candidate = self._build_kick_candidate(hand, metrics)
        if not self._is_kick_confirmed(metrics):
            return candidate, None

        event = self._confirm_hand_pending(history, hand, timestamp, metrics)
        return None, event

    def _evaluate_collision(
        self,
        frame: TrackerOutput,
        *,
        left_wrist: LandmarkPoint | None,
        right_wrist: LandmarkPoint | None,
    ) -> tuple[DetectionCandidate | None, GestureEvent | None]:
        """Update candidate state and return any confirmed bilateral snare."""
        if left_wrist is None or right_wrist is None:
            self._reset_collision_state()
            return None, None

        self._append_collision_sample(frame, left_wrist, right_wrist)
        history = self._collision_history
        if len(history.samples) < 2:
            history.pending = None
            return None, None

        metrics = self._compute_collision_metrics(history.samples)
        if metrics is None:
            history.pending = None
            return None, None

        if history.recovery is not None:
            if not self._collision_recovery_complete(history.recovery, metrics):
                history.pending = None
                return None, None
            history.recovery = None
            history.pending = None
            history.samples = deque((history.samples[-1],), maxlen=self.config.history_size)
            return None, None

        if frame.timestamp.seconds - history.last_trigger_time < self.config.cooldown_seconds:
            history.pending = None
            if self._meets_snare_candidate(metrics):
                self._emit_cooldown_suppressed(
                    frame.timestamp,
                    self.config.active_hand,
                    metrics.to_velocity_stats(),
                )
            return None, None

        if history.pending is not None and (
            frame.timestamp.seconds - history.pending.started_at > self.config.confirmation_window_seconds
            or not self._collision_candidate_is_still_valid(metrics)
        ):
            history.pending = None

        if history.pending is None:
            if not self._meets_snare_candidate(metrics):
                return None, None
            history.pending = PendingGesture(
                gesture=GestureType.SNARE,
                started_at=frame.timestamp.seconds,
                label="Wrist collision candidate",
            )
            candidate = self._build_snare_candidate(metrics)
            self._emit_candidate(frame.timestamp, candidate, metrics.to_velocity_stats())
            if self._is_snare_confirmed(metrics):
                event = self._confirm_collision_pending(frame.timestamp, metrics)
                return None, event
            return candidate, None

        candidate = self._build_snare_candidate(metrics)
        if not self._is_snare_confirmed(metrics):
            return candidate, None

        event = self._confirm_collision_pending(frame.timestamp, metrics)
        return None, event

    def _confirm_hand_pending(
        self,
        history: HandHistory,
        hand: str,
        timestamp: FrameTimestamp,
        metrics: MotionMetrics,
    ) -> GestureEvent:
        """Finalize the active hand candidate into a confirmed kick."""
        anchor = history.samples[-1]
        history.pending = None
        history.last_trigger_time = timestamp.seconds
        history.recovery = RecoveryState(gesture=GestureType.KICK, anchor=anchor, hand=hand)
        history.samples = deque((anchor,), maxlen=self.config.history_size)
        event = self._build_kick_event(hand, timestamp, metrics)
        self._emit_trigger(event, metrics.to_velocity_stats())
        return event

    def _confirm_collision_pending(
        self,
        timestamp: FrameTimestamp,
        metrics: WristCollisionMetrics,
    ) -> GestureEvent:
        """Finalize the bilateral collision candidate into a confirmed snare."""
        history = self._collision_history
        history.pending = None
        history.last_trigger_time = timestamp.seconds
        history.recovery = CollisionRecoveryState(anchor_distance_xy=metrics.current_distance_xy)
        history.samples = deque((history.samples[-1],), maxlen=self.config.history_size)
        event = self._build_snare_event(timestamp, metrics)
        self._emit_trigger(event, metrics.to_velocity_stats())
        return event

    def _build_kick_candidate(
        self,
        hand: str,
        metrics: MotionMetrics,
    ) -> DetectionCandidate:
        """Convert a downward-strike onset into a public kick candidate payload."""
        confidence = min(
            1.0,
            max(
                metrics.delta_y / self.config.strike_down_delta_y,
                metrics.downward_velocity / self.config.min_velocity,
            )
            * 0.5,
        )
        return DetectionCandidate(
            gesture=GestureType.KICK,
            confidence=confidence,
            hand=hand,
            label="Downward strike candidate",
        )

    def _build_snare_candidate(self, metrics: WristCollisionMetrics) -> DetectionCandidate:
        """Convert bilateral wrist convergence into a public snare candidate payload."""
        candidate_velocity_floor = self.config.min_velocity * self.config.candidate_ratio
        confidence = min(
            1.0,
            max(
                self._snare_candidate_distance_threshold() / max(metrics.current_distance_xy, 1e-6),
                metrics.closing_velocity / max(candidate_velocity_floor, 1e-6),
            )
            * 0.5,
        )
        return DetectionCandidate(
            gesture=GestureType.SNARE,
            confidence=confidence,
            hand=self.config.active_hand,
            label="Wrist collision candidate",
        )

    def _build_kick_event(
        self,
        hand: str,
        timestamp: FrameTimestamp,
        metrics: MotionMetrics,
    ) -> GestureEvent:
        """Create a confirmed kick event for the active hand."""
        confidence = min(
            1.0,
            metrics.delta_y / (self.config.strike_down_delta_y * 1.25),
        )
        return GestureEvent(
            gesture=GestureType.KICK,
            confidence=confidence,
            hand=hand,
            timestamp=timestamp,
            label="Downward strike → kick",
        )

    def _build_snare_event(
        self,
        timestamp: FrameTimestamp,
        metrics: WristCollisionMetrics,
    ) -> GestureEvent:
        """Create a confirmed snare event for a bilateral wrist collision."""
        confidence = min(
            1.0,
            max(
                self.config.snare_collision_distance / max(metrics.current_distance_xy, 1e-6),
                metrics.closing_velocity / max(self._snare_confirmation_velocity_threshold(), 1e-6),
            )
            * 0.5,
        )
        return GestureEvent(
            gesture=GestureType.SNARE,
            confidence=confidence,
            hand=self.config.active_hand,
            timestamp=timestamp,
            label="Wrist collision → snare",
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
        deltas_abs_x: list[float] = []
        deltas_y: list[float] = []
        deltas_z: list[float] = []
        for previous, current in zip(filtered_samples, filtered_samples[1:], strict=False):
            frame_elapsed = current.timestamp - previous.timestamp
            if frame_elapsed <= 0.0:
                continue
            deltas_x.append((current.x - previous.x) / frame_elapsed)
            deltas_abs_x.append((abs(current.x) - abs(previous.x)) / frame_elapsed)
            deltas_y.append((current.y - previous.y) / frame_elapsed)
            deltas_z.append((current.z - previous.z) / frame_elapsed)

        if not deltas_x or not deltas_abs_x or not deltas_y or not deltas_z:
            return None

        delta_x = newest.x - oldest.x
        delta_abs_x = abs(newest.x) - abs(oldest.x)
        delta_y = newest.y - oldest.y
        delta_z = newest.z - oldest.z
        return MotionMetrics(
            elapsed=elapsed,
            delta_x=delta_x,
            delta_abs_x=delta_abs_x,
            delta_y=delta_y,
            delta_z=delta_z,
            net_velocity=l1_velocity((delta_x, delta_y, delta_z), elapsed),
            peak_x_velocity=max(deltas_x, key=abs),
            peak_abs_x_velocity=max(deltas_abs_x, key=abs),
            peak_y_velocity=max(deltas_y, key=abs),
            peak_z_velocity=max(deltas_z, key=abs),
        )

    def _compute_collision_metrics(
        self,
        samples: Iterable[WristCollisionSample],
    ) -> WristCollisionMetrics | None:
        """Summarize bilateral wrist convergence over a recent temporal window."""
        sample_list = list(samples)
        if len(sample_list) < 2:
            return None

        oldest = sample_list[0]
        newest = sample_list[-1]
        elapsed = newest.timestamp - oldest.timestamp
        if elapsed <= 0.0:
            return None

        filtered_samples = self._smooth_collision_samples(sample_list)

        deltas_x: list[float] = []
        deltas_y: list[float] = []
        deltas_z: list[float] = []
        deltas_distance: list[float] = []
        for previous, current in zip(filtered_samples, filtered_samples[1:], strict=False):
            frame_elapsed = current.timestamp - previous.timestamp
            if frame_elapsed <= 0.0:
                continue
            deltas_x.append((current.x_gap - previous.x_gap) / frame_elapsed)
            deltas_y.append((current.y_gap - previous.y_gap) / frame_elapsed)
            deltas_z.append((current.z_gap - previous.z_gap) / frame_elapsed)
            deltas_distance.append((current.distance_xy - previous.distance_xy) / frame_elapsed)

        if not deltas_x or not deltas_y or not deltas_z or not deltas_distance:
            return None

        delta_x = newest.x_gap - oldest.x_gap
        delta_y = newest.y_gap - oldest.y_gap
        delta_z = newest.z_gap - oldest.z_gap
        return WristCollisionMetrics(
            elapsed=elapsed,
            delta_x_gap=delta_x,
            delta_y_gap=delta_y,
            delta_z_gap=delta_z,
            delta_distance_xy=newest.distance_xy - oldest.distance_xy,
            current_distance_xy=newest.distance_xy,
            current_depth_gap=newest.z_gap,
            net_velocity=l1_velocity((delta_x, delta_y, delta_z), elapsed),
            peak_x_velocity=max(deltas_x, key=abs),
            peak_y_velocity=max(deltas_y, key=abs),
            peak_z_velocity=max(deltas_z, key=abs),
            peak_distance_velocity=max(deltas_distance, key=abs),
            peak_closing_velocity=min(deltas_distance),
            peak_opening_velocity=max(deltas_distance),
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
                    shoulder_relative=sample.shoulder_relative,
                )
            )
        return smoothed

    def _smooth_collision_samples(
        self,
        samples: list[WristCollisionSample],
    ) -> list[WristCollisionSample]:
        """Smooth bilateral wrist-separation samples before peak-velocity analysis."""
        if len(samples) < 2:
            return samples

        alpha = self.config.velocity_smoothing_alpha
        if alpha >= 1.0 - COMPARISON_EPSILON:
            return samples

        smoothed = [samples[0]]
        for sample in samples[1:]:
            previous = smoothed[-1]
            x_gap = previous.x_gap + ((sample.x_gap - previous.x_gap) * alpha)
            y_gap = previous.y_gap + ((sample.y_gap - previous.y_gap) * alpha)
            z_gap = previous.z_gap + ((sample.z_gap - previous.z_gap) * alpha)
            smoothed.append(
                WristCollisionSample(
                    timestamp=sample.timestamp,
                    x_gap=x_gap,
                    y_gap=y_gap,
                    z_gap=z_gap,
                    distance_xy=hypot(x_gap, y_gap),
                )
            )
        return smoothed

    def _hand_recovery_complete(
        self,
        recovery: RecoveryState,
        current: MotionSample,
        metrics: MotionMetrics,
    ) -> bool:
        """Return whether the active hand has reset after a downward-strike kick."""
        required_distance = self.config.strike_down_delta_y * self.config.rearm_threshold_ratio
        partial_distance = required_distance * 0.5
        required_velocity = self.config.min_velocity * self.config.rearm_threshold_ratio
        reverse_distance = recovery.anchor.y - current.y
        reverse_velocity = max(0.0, -metrics.peak_y_velocity)
        return (
            reverse_distance >= required_distance - COMPARISON_EPSILON
            or (
                reverse_distance >= partial_distance - COMPARISON_EPSILON
                and reverse_velocity >= required_velocity - COMPARISON_EPSILON
            )
        )

    def _collision_recovery_complete(
        self,
        recovery: CollisionRecoveryState,
        metrics: WristCollisionMetrics,
    ) -> bool:
        """Return whether the wrists have separated enough to re-arm the snare."""
        required_distance = self.config.snare_collision_distance * self.config.rearm_threshold_ratio
        partial_distance = required_distance * 0.5
        required_velocity = self.config.min_velocity * self.config.rearm_threshold_ratio
        reverse_distance = metrics.current_distance_xy - recovery.anchor_distance_xy
        reverse_velocity = metrics.opening_velocity
        return (
            reverse_distance >= required_distance - COMPARISON_EPSILON
            or (
                reverse_distance >= partial_distance - COMPARISON_EPSILON
                and reverse_velocity >= required_velocity - COMPARISON_EPSILON
            )
        )

    def _kick_candidate_velocity_threshold(self) -> float:
        """Return the relaxed downward-speed floor used while arming kick."""
        return self.config.min_velocity * self.config.candidate_ratio * 0.65

    def _kick_confirmation_velocity_threshold(self) -> float:
        """Return the downward-speed floor used while confirming kick."""
        return self.config.min_velocity * 0.7

    def _kick_axis_threshold(self) -> float:
        """Return the relaxed downward-dominance floor used for kick detection."""
        return max(0.55, self.config.axis_dominance_ratio * 0.6)

    def _kick_depth_threshold(self) -> float:
        """Return the relaxed depth-drift ceiling used for kick detection."""
        return max(0.24, self.config.strike_max_depth_drift * 1.35)

    def _meets_kick_candidate(self, metrics: MotionMetrics) -> bool:
        """Return whether current motion is a viable downward-strike kick candidate."""
        return (
            metrics.delta_y
            >= self.config.strike_down_delta_y * self.config.candidate_ratio - COMPARISON_EPSILON
            and abs(metrics.delta_z)
            <= self._kick_depth_threshold() + COMPARISON_EPSILON
            and metrics.downward_velocity
            >= self._kick_candidate_velocity_threshold() - COMPARISON_EPSILON
            and metrics.strike_axis_ratio
            >= self._kick_axis_threshold() - COMPARISON_EPSILON
        )

    def _is_kick_confirmed(self, metrics: MotionMetrics) -> bool:
        """Return whether a kick candidate now exceeds the final trigger thresholds."""
        return (
            metrics.delta_y
            >= (
                self.config.strike_down_delta_y * self.config.strike_confirmation_ratio
            )
            - COMPARISON_EPSILON
            and abs(metrics.delta_z) <= self._kick_depth_threshold() + COMPARISON_EPSILON
            and metrics.downward_velocity
            >= self._kick_confirmation_velocity_threshold() - COMPARISON_EPSILON
            and metrics.strike_axis_ratio
            >= self._kick_axis_threshold() - COMPARISON_EPSILON
        )

    def _snare_candidate_distance_threshold(self) -> float:
        """Return the relaxed wrist-distance threshold used while arming snare."""
        return self.config.snare_collision_distance / self.config.candidate_ratio

    def _snare_confirmation_velocity_threshold(self) -> float:
        """Return the minimum closing speed required to confirm a snare."""
        return self.config.min_velocity * self.config.snare_confirmation_velocity_ratio

    def _meets_snare_candidate(self, metrics: WristCollisionMetrics) -> bool:
        """Return whether current bilateral motion is a viable snare candidate."""
        return (
            metrics.current_distance_xy
            <= self._snare_candidate_distance_threshold() + COMPARISON_EPSILON
            and metrics.current_depth_gap
            <= self.config.snare_collision_max_depth_gap + COMPARISON_EPSILON
            and metrics.closing_velocity
            >= self.config.min_velocity * self.config.candidate_ratio - COMPARISON_EPSILON
            and metrics.net_velocity
            >= self.config.min_velocity * self.config.candidate_ratio - COMPARISON_EPSILON
        )

    def _collision_candidate_is_still_valid(self, metrics: WristCollisionMetrics) -> bool:
        """Return whether a pending collision candidate still looks snare-like."""
        return (
            metrics.current_distance_xy
            <= self._snare_candidate_distance_threshold() + COMPARISON_EPSILON
            and metrics.current_depth_gap
            <= self.config.snare_collision_max_depth_gap + COMPARISON_EPSILON
            and (
                metrics.closing_velocity
                >= self.config.min_velocity * self.config.candidate_ratio * 0.5
                - COMPARISON_EPSILON
                or metrics.current_distance_xy
                <= self.config.snare_collision_distance + COMPARISON_EPSILON
            )
        )

    def _is_snare_confirmed(self, metrics: WristCollisionMetrics) -> bool:
        """Return whether a snare candidate now exceeds the final trigger thresholds."""
        return (
            metrics.current_distance_xy <= self.config.snare_collision_distance + COMPARISON_EPSILON
            and metrics.current_depth_gap
            <= self.config.snare_collision_max_depth_gap + COMPARISON_EPSILON
            and metrics.closing_velocity
            >= self._snare_confirmation_velocity_threshold() - COMPARISON_EPSILON
        )

    def _emit_candidate(
        self,
        timestamp: FrameTimestamp,
        candidate: DetectionCandidate,
        velocity_stats: VelocityStats,
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
                velocity_stats=velocity_stats,
                confidence=candidate.confidence,
                hand=candidate.hand,
            )
        )

    def _emit_trigger(self, event: GestureEvent, velocity_stats: VelocityStats) -> None:
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
                velocity_stats=velocity_stats,
                confidence=event.confidence,
                hand=event.hand,
            )
        )

    def _emit_cooldown_suppressed(
        self,
        timestamp: FrameTimestamp,
        hand: str,
        velocity_stats: VelocityStats,
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
                velocity_stats=velocity_stats,
                confidence=None,
                hand=hand,
            )
        )

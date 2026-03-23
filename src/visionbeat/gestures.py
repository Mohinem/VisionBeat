"""Gesture recognition logic isolated from camera and MediaPipe dependencies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from visionbeat.config import GestureConfig
from visionbeat.math_utils import l1_velocity
from visionbeat.models import GestureEvent, GestureType, PoseFrame


@dataclass(slots=True)
class MotionSample:
    """Temporal sample of a tracked wrist position."""

    timestamp: float
    x: float
    y: float
    z: float


@dataclass(slots=True)
class HandHistory:
    """Recent wrist samples plus cooldown state for a tracked hand."""

    samples: deque[MotionSample] = field(default_factory=deque)
    last_trigger_time: float = -1_000.0


class GestureDetector:
    """Detect percussion gestures from normalized wrist trajectories."""

    def __init__(self, config: GestureConfig) -> None:
        """Store gesture thresholds and initialize state."""
        self.config = config
        self._histories: dict[str, HandHistory] = {
            "left": HandHistory(deque(maxlen=config.history_size)),
            "right": HandHistory(deque(maxlen=config.history_size)),
        }

    def update(self, frame: PoseFrame) -> list[GestureEvent]:
        """Consume a pose frame and emit any newly detected gestures."""
        events: list[GestureEvent] = []
        for hand in ("left", "right"):
            wrist = frame.get(f"{hand}_wrist")
            if wrist is None or wrist.visibility < 0.5:
                continue
            history = self._histories[hand]
            history.samples.append(MotionSample(frame.timestamp, wrist.x, wrist.y, wrist.z))
            event = self._detect_for_hand(hand, history)
            if event is not None:
                events.append(event)
        return events

    def _detect_for_hand(self, hand: str, history: HandHistory) -> GestureEvent | None:
        """Evaluate current motion history for one hand."""
        if len(history.samples) < 2:
            return None

        newest = history.samples[-1]
        oldest = history.samples[0]
        elapsed = newest.timestamp - oldest.timestamp
        delta_x = newest.x - oldest.x
        delta_y = newest.y - oldest.y
        delta_z = newest.z - oldest.z
        velocity = l1_velocity((delta_x, delta_y, delta_z), elapsed)

        if newest.timestamp - history.last_trigger_time < self.config.cooldown_seconds:
            return None

        if hand != self.config.active_hand:
            return None

        if (
            delta_z <= -self.config.punch_forward_delta_z
            and abs(delta_y) <= self.config.punch_max_vertical_drift
            and velocity >= self.config.min_velocity
        ):
            history.last_trigger_time = newest.timestamp
            return GestureEvent(
                gesture=GestureType.KICK,
                confidence=min(1.0, abs(delta_z) / (self.config.punch_forward_delta_z * 1.5)),
                hand=hand,
                timestamp=newest.timestamp,
                label="Forward punch → kick",
            )

        if (
            delta_y >= self.config.strike_down_delta_y
            and abs(delta_z) <= self.config.strike_max_depth_drift
            and velocity >= self.config.min_velocity
        ):
            history.last_trigger_time = newest.timestamp
            return GestureEvent(
                gesture=GestureType.SNARE,
                confidence=min(1.0, delta_y / (self.config.strike_down_delta_y * 1.5)),
                hand=hand,
                timestamp=newest.timestamp,
                label="Downward strike → snare",
            )

        return None

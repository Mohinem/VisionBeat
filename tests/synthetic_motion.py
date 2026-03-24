from __future__ import annotations

from dataclasses import dataclass
from random import Random

from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput

MotionPoint = tuple[float, tuple[float, float, float]]
MotionSequence = tuple[MotionPoint, ...]


@dataclass(slots=True)
class SyntheticMotionGenerator:
    """Build deterministic synthetic wrist trajectories for gesture tests."""

    frame_interval: float = 0.05
    base_position: tuple[float, float, float] = (0.50, 0.40, -0.10)
    seed: int = 7

    def stationary_hand(
        self,
        *,
        duration: float = 0.15,
        noise: float = 0.0,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        start_time: float = 0.0,
    ) -> MotionSequence:
        return self._linear_motion(
            duration=duration,
            velocity=velocity,
            noise=noise,
            start_time=start_time,
        )

    def forward_punch(
        self,
        *,
        velocity: float = 2.5,
        duration: float = 0.10,
        noise: float = 0.0,
        start_time: float = 0.0,
    ) -> MotionSequence:
        return self._linear_motion(
            duration=duration,
            velocity=(0.1, 0.2, -velocity),
            noise=noise,
            start_time=start_time,
            easing_exponent=0.5,
        )

    def downward_strike(
        self,
        *,
        velocity: float = 2.8,
        duration: float = 0.10,
        noise: float = 0.0,
        start_time: float = 0.0,
    ) -> MotionSequence:
        return self._linear_motion(
            duration=duration,
            velocity=(0.1, velocity, -0.1),
            noise=noise,
            start_time=start_time,
            base_position=(0.55, 0.20, -0.05),
            easing_exponent=0.5,
        )

    def jitter_noise(
        self,
        *,
        duration: float = 0.15,
        noise: float = 0.03,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        start_time: float = 0.0,
    ) -> MotionSequence:
        return self._linear_motion(
            duration=duration,
            velocity=velocity,
            noise=noise,
            start_time=start_time,
            base_position=(0.52, 0.32, -0.12),
        )

    def non_trigger_movement(
        self,
        *,
        velocity: float = 1.3,
        duration: float = 0.10,
        noise: float = 0.0,
        start_time: float = 0.0,
    ) -> MotionSequence:
        return self._linear_motion(
            duration=duration,
            velocity=(velocity, velocity, -velocity),
            noise=noise,
            start_time=start_time,
        )

    def to_tracker_outputs(
        self,
        sequence: MotionSequence,
        *,
        hand: str = "right",
        visibility: float = 1.0,
    ) -> list[TrackerOutput]:
        frames: list[TrackerOutput] = []
        for timestamp, (x, y, z) in sequence:
            landmark = LandmarkPoint(x=x, y=y, z=z, visibility=visibility)
            frames.append(
                TrackerOutput(
                    timestamp=FrameTimestamp(seconds=timestamp),
                    landmarks={f"{hand}_wrist": landmark},
                    status="tracking",
                    person_detected=True,
                )
            )
        return frames

    def _linear_motion(
        self,
        *,
        duration: float,
        velocity: tuple[float, float, float],
        noise: float,
        start_time: float,
        base_position: tuple[float, float, float] | None = None,
        easing_exponent: float = 1.0,
    ) -> MotionSequence:
        if duration <= 0.0:
            raise ValueError("duration must be greater than zero.")
        if self.frame_interval <= 0.0:
            raise ValueError("frame_interval must be greater than zero.")

        frames = max(2, int(round(duration / self.frame_interval)) + 1)
        rng = Random(self.seed)
        base_x, base_y, base_z = base_position or self.base_position
        vel_x, vel_y, vel_z = velocity

        output: list[MotionPoint] = []
        for index in range(frames):
            elapsed = min(duration, index * self.frame_interval)
            timestamp = start_time + elapsed
            progress = (elapsed / duration) ** easing_exponent
            x = base_x + vel_x * duration * progress + rng.gauss(0.0, noise)
            y = base_y + vel_y * duration * progress + rng.gauss(0.0, noise)
            z = base_z + vel_z * duration * progress + rng.gauss(0.0, noise)
            output.append((timestamp, (x, y, z)))
        return tuple(output)

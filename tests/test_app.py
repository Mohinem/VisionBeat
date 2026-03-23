from __future__ import annotations

from dataclasses import dataclass

import pytest

from visionbeat.app import VisionBeatRuntime
from visionbeat.camera import CameraFrame
from visionbeat.config import AppConfig
from visionbeat.models import (
    DetectionCandidate,
    FrameTimestamp,
    GestureEvent,
    GestureType,
    LandmarkPoint,
    RenderState,
    TrackerOutput,
)


@dataclass
class FakeFrame:
    name: str

    def copy(self) -> FakeFrame:
        return FakeFrame(name=f"{self.name}-copy")


class FakeCamera:
    def __init__(self, frames: list[CameraFrame]) -> None:
        self.frames = list(frames)
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1

    def read_frame(self) -> CameraFrame:
        return self.frames.pop(0)

    def close(self) -> None:
        self.close_calls += 1


class FakeTracker:
    def __init__(self, outputs: list[TrackerOutput]) -> None:
        self.outputs = list(outputs)
        self.process_calls: list[tuple[object, FrameTimestamp]] = []
        self.close_calls = 0

    def process(self, frame: object, timestamp: FrameTimestamp) -> TrackerOutput:
        self.process_calls.append((frame, timestamp))
        return self.outputs.pop(0)

    def close(self) -> None:
        self.close_calls += 1


class FakeDetector:
    def __init__(
        self,
        *,
        events_by_frame: list[list[GestureEvent]],
        candidates_by_frame: list[tuple[DetectionCandidate, ...]],
        cooldowns: list[float],
    ) -> None:
        self._events_by_frame = list(events_by_frame)
        self._candidates_by_frame = list(candidates_by_frame)
        self._cooldowns = list(cooldowns)
        self._candidates: tuple[DetectionCandidate, ...] = ()
        self.update_calls: list[TrackerOutput] = []

    @property
    def candidates(self) -> tuple[DetectionCandidate, ...]:
        return self._candidates

    def update(self, frame: TrackerOutput) -> list[GestureEvent]:
        self.update_calls.append(frame)
        self._candidates = self._candidates_by_frame.pop(0)
        return self._events_by_frame.pop(0)

    def cooldown_remaining(self, timestamp: FrameTimestamp) -> float:
        assert timestamp.seconds >= 0.0
        return self._cooldowns.pop(0)


class FakeAudio:
    def __init__(self) -> None:
        self.triggers: list[object] = []
        self.close_calls = 0

    def trigger(self, trigger: object) -> None:
        self.triggers.append(trigger)

    def close(self) -> None:
        self.close_calls += 1


class FakeOverlay:
    def __init__(self) -> None:
        self.calls: list[tuple[object, RenderState]] = []

    def render(self, frame: object, state: RenderState) -> str:
        self.calls.append((frame, state))
        return f"rendered:{state.frame_index}"


class FakePreview:
    def __init__(self, should_close_sequence: list[bool]) -> None:
        self.should_close_sequence = list(should_close_sequence)
        self.show_calls: list[tuple[str, object]] = []
        self.close_calls = 0

    def show(self, window_name: str, frame: object) -> None:
        self.show_calls.append((window_name, frame))

    def should_close(self) -> bool:
        return self.should_close_sequence.pop(0)

    def close(self) -> None:
        self.close_calls += 1


def make_pose(timestamp: float) -> TrackerOutput:
    return TrackerOutput(
        timestamp=FrameTimestamp(seconds=timestamp),
        landmarks={
            "right_wrist": LandmarkPoint(x=0.6, y=0.4, z=-0.3, visibility=0.9),
        },
        person_detected=True,
        status="tracking",
    )


def test_runtime_orchestrates_tracking_detection_audio_and_overlay() -> None:
    frame = FakeFrame("frame-0")
    camera = FakeCamera([CameraFrame(image=frame, captured_at=1.0, frame_index=7)])
    tracker = FakeTracker([make_pose(1.0)])
    candidate = DetectionCandidate(
        gesture=GestureType.KICK,
        confidence=0.72,
        hand="right",
        label="Forward punch candidate",
    )
    event = GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.91,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.0),
        label="Forward punch → kick",
    )
    detector = FakeDetector(
        events_by_frame=[[event]],
        candidates_by_frame=[(candidate,)],
        cooldowns=[0.08],
    )
    audio = FakeAudio()
    overlay = FakeOverlay()
    preview = FakePreview([False])
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=camera,
        tracker=tracker,
        detector=detector,
        audio=audio,
        overlay=overlay,
        preview=preview,
    )

    should_continue = runtime.process_next_frame()

    assert should_continue is True
    assert tracker.process_calls[0][0] is frame
    assert detector.update_calls[0].status == "tracking"
    assert len(audio.triggers) == 1
    state = overlay.calls[0][1]
    assert state.pose.status == "tracking"
    assert state.current_candidate == candidate
    assert state.confirmed_gesture == event
    assert state.cooldown_remaining_seconds == 0.08
    assert state.frame_index == 7
    assert preview.show_calls == [("VisionBeat", "rendered:7")]


def test_runtime_run_opens_and_closes_resources_on_keypress() -> None:
    frames = [
        CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=0),
        CameraFrame(image=FakeFrame("frame-1"), captured_at=1.1, frame_index=1),
    ]
    camera = FakeCamera(frames)
    tracker = FakeTracker([make_pose(1.0), make_pose(1.1)])
    detector = FakeDetector(
        events_by_frame=[[], []],
        candidates_by_frame=[(), ()],
        cooldowns=[0.0, 0.0],
    )
    audio = FakeAudio()
    overlay = FakeOverlay()
    preview = FakePreview([False, True])
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=camera,
        tracker=tracker,
        detector=detector,
        audio=audio,
        overlay=overlay,
        preview=preview,
    )

    runtime.run()

    assert camera.open_calls == 1
    assert camera.close_calls == 1
    assert tracker.close_calls == 1
    assert audio.close_calls == 1
    assert preview.close_calls == 1
    assert len(overlay.calls) == 2
    assert overlay.calls[1][1].fps == pytest.approx(10.0)


def test_runtime_keeps_last_confirmed_gesture_visible_until_replaced() -> None:
    frames = [
        CameraFrame(image=FakeFrame("frame-0"), captured_at=2.0, frame_index=0),
        CameraFrame(image=FakeFrame("frame-1"), captured_at=2.05, frame_index=1),
    ]
    tracker = FakeTracker([make_pose(2.0), make_pose(2.05)])
    event = GestureEvent(
        gesture=GestureType.SNARE,
        confidence=0.88,
        hand="right",
        timestamp=FrameTimestamp(seconds=2.0),
        label="Downward strike → snare",
    )
    detector = FakeDetector(
        events_by_frame=[[event], []],
        candidates_by_frame=[(), ()],
        cooldowns=[0.12, 0.04],
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=FakeCamera(frames),
        tracker=tracker,
        detector=detector,
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, True]),
    )

    runtime.run()

    first_state = runtime.overlay.calls[0][1]
    second_state = runtime.overlay.calls[1][1]
    assert first_state.confirmed_gesture == event
    assert second_state.confirmed_gesture == event
    assert second_state.cooldown_remaining_seconds == 0.04

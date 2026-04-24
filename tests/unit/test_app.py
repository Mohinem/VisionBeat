from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from visionbeat.app import VisionBeatApp, VisionBeatRuntime
from visionbeat.camera import CameraFrame
from visionbeat.config import AppConfig, PredictiveConfig, RuntimeConfig
from visionbeat.features import CANONICAL_FEATURE_NAMES, get_canonical_feature_schema
from visionbeat.models import (
    AudioTrigger,
    DetectionCandidate,
    FrameTimestamp,
    GestureEvent,
    GestureType,
    LandmarkPoint,
    RenderState,
    TrackerOutput,
)
from visionbeat.predictive_shadow import PredictiveStatus, ShadowPredictionEvent


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
        self.triggers: list[AudioTrigger] = []
        self.close_calls = 0

    def trigger(self, trigger: AudioTrigger) -> None:
        self.triggers.append(trigger)

    def close(self) -> None:
        self.close_calls += 1


class FakeOverlay:
    def __init__(self) -> None:
        self.calls: list[tuple[object, RenderState]] = []
        self.overlay_enabled: list[bool] = []
        self.debug_enabled: list[bool] = []

    def render(self, frame: object, state: RenderState) -> str:
        self.calls.append((frame, state))
        return f"rendered:{state.frame_index}"

    def set_overlay_enabled(self, enabled: bool) -> None:
        self.overlay_enabled.append(enabled)

    def set_debug_enabled(self, enabled: bool) -> None:
        self.debug_enabled.append(enabled)


class FakeTransport:
    def __init__(self) -> None:
        self.events: list[GestureEvent] = []
        self.close_calls = 0

    def emit(self, event: GestureEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.close_calls += 1


class FakePreview:
    def __init__(
        self,
        should_close_sequence: list[bool],
        key_sequence: list[int | None] | None = None,
    ) -> None:
        self.should_close_sequence = list(should_close_sequence)
        self.key_sequence = list(key_sequence or [None] * len(should_close_sequence))
        self.show_calls: list[tuple[str, object]] = []
        self.close_calls = 0

    def show(self, window_name: str, frame: object) -> None:
        self.show_calls.append((window_name, frame))

    def poll_key(self) -> int | None:
        return self.key_sequence.pop(0)

    def should_close(self, key_code: int | None = None) -> bool:
        return self.should_close_sequence.pop(0)

    def close(self) -> None:
        self.close_calls += 1


class FakeRecorder:
    def __init__(self) -> None:
        self.runtime_started: list[str] = []
        self.runtime_stopped: list[str] = []
        self.tracking_failures: list[tuple[float, str]] = []
        self.predictive_shadow_triggers: list[dict[str, object]] = []
        self.predictive_live_triggers: list[dict[str, object]] = []
        self.rhythm_predictions: list[dict[str, object]] = []
        self.rhythm_live_triggers: list[dict[str, object]] = []
        self.shutdown_calls = 0
        self.close_calls = 0
        self.app_startups: list[dict[str, object]] = []

    def log_app_startup(self, *, config_summary: dict[str, object]) -> None:
        self.app_startups.append(config_summary)

    def log_app_shutdown(self) -> None:
        self.shutdown_calls += 1

    def log_runtime_started(self, *, window_name: str) -> None:
        self.runtime_started.append(window_name)

    def log_runtime_stopped(self, *, reason: str) -> None:
        self.runtime_stopped.append(reason)

    def log_tracking_failure(self, *, timestamp: float, status: str) -> None:
        self.tracking_failures.append((timestamp, status))

    def log_predictive_shadow_trigger(
        self,
        *,
        timestamp: float,
        frame_index: int,
        timing_probability: float,
        predicted_gesture: GestureType,
        predicted_gesture_confidence: float,
        heuristic_gesture_types: tuple[str, ...],
        class_probabilities: dict[str, float],
    ) -> None:
        self.predictive_shadow_triggers.append(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "timing_probability": timing_probability,
                "predicted_gesture": predicted_gesture,
                "predicted_gesture_confidence": predicted_gesture_confidence,
                "heuristic_gesture_types": heuristic_gesture_types,
                "class_probabilities": class_probabilities,
            }
        )

    def log_predictive_live_trigger(
        self,
        *,
        timestamp: float,
        frame_index: int,
        timing_probability: float,
        predicted_gesture: GestureType,
        predicted_gesture_confidence: float,
        hand: str,
        class_probabilities: dict[str, float],
        source: str = "cnn",
    ) -> None:
        self.predictive_live_triggers.append(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "timing_probability": timing_probability,
                "predicted_gesture": predicted_gesture,
                "predicted_gesture_confidence": predicted_gesture_confidence,
                "hand": hand,
                "class_probabilities": class_probabilities,
                "source": source,
            }
        )

    def log_rhythm_prediction(
        self,
        *,
        timestamp: float,
        frame_index: int | None,
        prediction_id: str,
        outcome: str,
        gesture: GestureType,
        predicted_time_seconds: float,
        actual_time_seconds: float | None,
        actual_gesture: GestureType | None,
        error_ms: float | None,
        last_event_timestamp: float,
        interval_seconds: float,
        expires_after_seconds: float,
        seconds_until_prediction: float,
        seconds_until_expiry: float,
        match_tolerance_seconds: float,
        confidence: float,
        repetition_count: int,
        jitter: float,
        active: bool,
        shadow_only: bool,
        source: str,
        trigger_mode: str,
        status_description: str,
    ) -> None:
        self.rhythm_predictions.append(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "prediction_id": prediction_id,
                "outcome": outcome,
                "gesture": gesture,
                "predicted_time_seconds": predicted_time_seconds,
                "actual_time_seconds": actual_time_seconds,
                "actual_gesture": actual_gesture,
                "error_ms": error_ms,
                "last_event_timestamp": last_event_timestamp,
                "estimated_interval_seconds": interval_seconds,
                "next_expected_timestamp": predicted_time_seconds,
                "interval_seconds": interval_seconds,
                "expires_after_seconds": expires_after_seconds,
                "seconds_until_prediction": seconds_until_prediction,
                "seconds_until_expiry": seconds_until_expiry,
                "match_tolerance_seconds": match_tolerance_seconds,
                "confidence": confidence,
                "repetition_count": repetition_count,
                "jitter": jitter,
                "active": active,
                "shadow_only": shadow_only,
                "source": source,
                "trigger_mode": trigger_mode,
                "status_description": status_description,
            }
        )

    def log_rhythm_live_trigger(
        self,
        *,
        timestamp: float,
        frame_index: int,
        prediction_id: str,
        predicted_time_seconds: float,
        scheduling_error_ms: float,
        gesture: GestureType,
        confidence: float,
        interval_seconds: float,
        repetition_count: int,
        jitter: float,
        source: str,
    ) -> None:
        self.rhythm_live_triggers.append(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "prediction_id": prediction_id,
                "predicted_time_seconds": predicted_time_seconds,
                "scheduling_error_ms": scheduling_error_ms,
                "gesture": gesture,
                "confidence": confidence,
                "interval_seconds": interval_seconds,
                "repetition_count": repetition_count,
                "jitter": jitter,
                "source": source,
            }
        )

    def close(self) -> None:
        self.close_calls += 1


class StreamingFakeCamera:
    def __init__(self, *, interval_seconds: float = 1.0 / 30.0) -> None:
        self.interval_seconds = interval_seconds
        self.open_calls = 0
        self.close_calls = 0
        self.frame_index = 0
        self.started_at = 1.0

    def open(self) -> None:
        self.open_calls += 1

    def read_frame(self) -> CameraFrame:
        time.sleep(self.interval_seconds)
        frame_index = self.frame_index
        self.frame_index += 1
        captured_at = self.started_at + (frame_index * self.interval_seconds)
        return CameraFrame(
            image=FakeFrame(f"frame-{frame_index}"),
            captured_at=captured_at,
            frame_index=frame_index,
        )

    def close(self) -> None:
        self.close_calls += 1


class StreamingFakeTracker:
    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        self.delay_seconds = delay_seconds
        self.process_calls: list[tuple[object, FrameTimestamp]] = []
        self.close_calls = 0

    def process(self, frame: object, timestamp: FrameTimestamp) -> TrackerOutput:
        self.process_calls.append((frame, timestamp))
        time.sleep(self.delay_seconds)
        return make_pose(timestamp.seconds)

    def close(self) -> None:
        self.close_calls += 1


class StreamingFakeDetector:
    def __init__(self) -> None:
        self.update_calls: list[TrackerOutput] = []

    @property
    def candidates(self) -> tuple[DetectionCandidate, ...]:
        return ()

    def update(self, frame: TrackerOutput) -> list[GestureEvent]:
        self.update_calls.append(frame)
        return []

    def cooldown_remaining(self, timestamp: FrameTimestamp) -> float:
        assert timestamp.seconds >= 0.0
        return 0.0


class FakeSessionRecorder:
    def __init__(self) -> None:
        self.session_dir = Path("/tmp/fake-session")
        self.camera_frames: list[CameraFrame] = []
        self.tracker_outputs: list[tuple[CameraFrame, TrackerOutput]] = []
        self.triggers: list[GestureEvent] = []
        self.predictive_shadow_triggers: list[dict[str, object]] = []
        self.close_calls = 0

    def record_camera_frame(self, camera_frame: CameraFrame) -> None:
        self.camera_frames.append(camera_frame)

    def record_tracker_output(
        self,
        camera_frame: CameraFrame,
        tracker_output: TrackerOutput,
    ) -> None:
        self.tracker_outputs.append((camera_frame, tracker_output))

    def record_trigger(self, event: GestureEvent) -> None:
        self.triggers.append(event)

    def record_predictive_shadow_trigger(self, payload: dict[str, object]) -> None:
        self.predictive_shadow_triggers.append(payload)

    def close(self) -> None:
        self.close_calls += 1


def make_pose(
    timestamp: float,
    *,
    status: str = "tracking",
    person_detected: bool = True,
) -> TrackerOutput:
    landmarks = (
        {"right_wrist": LandmarkPoint(x=0.6, y=0.4, z=-0.3, visibility=0.9)}
        if person_detected
        else {}
    )
    return TrackerOutput(
        timestamp=FrameTimestamp(seconds=timestamp),
        landmarks=landmarks,
        person_detected=person_detected,
        status=status,
    )


def test_runtime_orchestrates_tracking_detection_audio_and_overlay() -> None:
    frame = FakeFrame("frame-0")
    camera = FakeCamera([CameraFrame(image=frame, captured_at=1.0, frame_index=7)])
    tracker = FakeTracker([make_pose(1.0)])
    candidate = DetectionCandidate(
        gesture=GestureType.KICK,
        confidence=0.72,
        hand="right",
        label="Inward jab candidate",
    )
    event = GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.91,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.0),
        label="Inward jab → kick",
    )
    detector = FakeDetector(
        events_by_frame=[[event]],
        candidates_by_frame=[(candidate,)],
        cooldowns=[0.08],
    )
    audio = FakeAudio()
    transport = FakeTransport()
    overlay = FakeOverlay()
    preview = FakePreview([False])
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=camera,
        tracker=tracker,
        detector=detector,
        audio=audio,
        transport=transport,
        overlay=overlay,
        preview=preview,
    )

    should_continue = runtime.process_next_frame()

    assert should_continue is True
    assert tracker.process_calls[0][0] is frame
    assert detector.update_calls[0].status == "tracking"
    assert len(audio.triggers) == 1
    assert transport.events == [event]
    assert audio.triggers[0].gesture is GestureType.KICK
    assert audio.triggers[0].intensity == pytest.approx(0.91)
    state = overlay.calls[0][1]
    assert state.pose.status == "tracking"
    assert state.current_candidate == candidate
    assert state.confirmed_gesture == event
    assert state.cooldown_remaining_seconds == 0.08
    assert state.frame_index == 7
    assert preview.show_calls == [("VisionBeat", "rendered:7")]


def test_runtime_records_session_data_for_replay() -> None:
    frame = FakeFrame("frame-0")
    camera_frame = CameraFrame(image=frame, captured_at=1.0, frame_index=7)
    camera = FakeCamera([camera_frame])
    pose = make_pose(1.0)
    tracker = FakeTracker([pose])
    event = GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.91,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.0),
        label="Inward jab → kick",
    )
    detector = FakeDetector(
        events_by_frame=[[event]],
        candidates_by_frame=[()],
        cooldowns=[0.08],
    )
    session_recorder = FakeSessionRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=camera,
        tracker=tracker,
        detector=detector,
        audio=FakeAudio(),
        transport=FakeTransport(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        session_recorder=session_recorder,
    )

    runtime.process_next_frame()

    assert session_recorder.camera_frames == [camera_frame]
    assert session_recorder.tracker_outputs == [(camera_frame, pose)]
    assert session_recorder.triggers == [event]


def test_runtime_tracks_on_raw_frame_and_renders_mirrored_preview() -> None:
    raw_frame = FakeFrame("raw-frame")
    mirrored_frame = FakeFrame("mirrored-frame")
    pose = TrackerOutput(
        timestamp=FrameTimestamp(seconds=1.0),
        landmarks={"right_wrist": LandmarkPoint(x=0.2, y=0.4, z=-0.3, visibility=0.9)},
        person_detected=True,
        status="tracking",
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=raw_frame,
                    captured_at=1.0,
                    frame_index=0,
                    display_image=mirrored_frame,
                    mirrored_for_display=True,
                )
            ]
        ),
        tracker=FakeTracker([pose]),
        detector=FakeDetector(events_by_frame=[[]], candidates_by_frame=[()], cooldowns=[0.0]),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
    )

    runtime.process_next_frame()

    assert runtime.tracker.process_calls[0][0] is raw_frame
    assert runtime.overlay.calls[0][0] is mirrored_frame
    assert runtime.overlay.calls[0][1].pose.get("right_wrist").x == pytest.approx(0.8)


def test_runtime_extracts_canonical_live_features_and_maintains_window() -> None:
    frames = [
        CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=0),
        CameraFrame(image=FakeFrame("frame-1"), captured_at=1.5, frame_index=1),
    ]
    poses = [
        TrackerOutput(
            timestamp=FrameTimestamp(seconds=1.0),
            landmarks={
                "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
                "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
                "left_wrist": LandmarkPoint(x=0.25, y=0.50, z=-0.2, visibility=0.95),
                "right_wrist": LandmarkPoint(x=0.75, y=0.50, z=-0.2, visibility=0.95),
            },
            person_detected=True,
            status="tracking",
        ),
        TrackerOutput(
            timestamp=FrameTimestamp(seconds=1.5),
            landmarks={
                "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
                "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
                "left_wrist": LandmarkPoint(x=0.50, y=0.75, z=-0.2, visibility=0.95),
                "right_wrist": LandmarkPoint(x=0.75, y=0.75, z=-0.2, visibility=0.95),
            },
            person_detected=True,
            status="tracking",
        ),
    ]
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=FakeCamera(frames),
        tracker=FakeTracker(poses),
        detector=FakeDetector(
            events_by_frame=[[], []],
            candidates_by_frame=[(), ()],
            cooldowns=[0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False]),
    )

    runtime.process_next_frame()
    runtime.process_next_frame()

    assert runtime.latest_frame_features is not None
    assert runtime.latest_feature_vector is not None
    assert runtime.live_feature_schema == get_canonical_feature_schema()
    assert len(runtime.latest_feature_vector) == len(CANONICAL_FEATURE_NAMES)
    assert runtime.latest_frame_features.temporal_features["dt_seconds"] == pytest.approx(0.5)
    assert runtime.latest_frame_features.temporal_features["left_wrist_rel_vx"] == pytest.approx(
        1.0
    )
    assert runtime.latest_frame_features.temporal_features["left_wrist_rel_vy"] == pytest.approx(
        1.0
    )

    window = runtime.build_live_feature_window(window_size=4)
    assert window.schema_version == runtime.live_feature_schema.version
    assert window.feature_count == runtime.live_feature_schema.feature_count
    assert len(window.frames) == 2
    assert len(window.matrix) == 4
    assert window.matrix[-1] == runtime.latest_feature_vector


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
    transport = FakeTransport()
    preview = FakePreview([False, True])
    recorder = FakeRecorder()
    session_recorder = FakeSessionRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=camera,
        tracker=tracker,
        detector=detector,
        audio=audio,
        transport=transport,
        overlay=overlay,
        preview=preview,
        recorder=recorder,
        session_recorder=session_recorder,
    )

    runtime.run()

    assert recorder.runtime_started == ["VisionBeat"]
    assert recorder.runtime_stopped == ["user_request"]
    assert recorder.shutdown_calls == 1
    assert recorder.close_calls == 1
    assert camera.open_calls == 1
    assert camera.close_calls == 1
    assert tracker.close_calls == 1
    assert audio.close_calls == 1
    assert transport.close_calls == 1
    assert preview.close_calls == 1
    assert session_recorder.close_calls == 1
    assert len(overlay.calls) == 2
    assert overlay.calls[1][1].fps == pytest.approx(10.0)
    assert overlay.overlay_enabled == [True]
    assert overlay.debug_enabled == [True]


def test_runtime_supports_overlay_toggle_shortcuts() -> None:
    camera = FakeCamera(
        [
            CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=0),
            CameraFrame(image=FakeFrame("frame-1"), captured_at=1.1, frame_index=1),
            CameraFrame(image=FakeFrame("frame-2"), captured_at=1.2, frame_index=2),
        ]
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=camera,
        tracker=FakeTracker([make_pose(1.0), make_pose(1.1), make_pose(1.2)]),
        detector=FakeDetector(
            events_by_frame=[[], [], []],
            candidates_by_frame=[(), (), ()],
            cooldowns=[0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, True], key_sequence=[ord("o"), ord("d"), None]),
    )

    runtime.run()

    assert runtime.overlay.overlay_enabled == [True, False]
    assert runtime.overlay.debug_enabled == [True, False]


def test_runtime_ignores_debug_toggle_when_debug_panel_disabled() -> None:
    config = AppConfig()
    config = replace(
        config,
        debug=replace(
            config.debug,
            overlays=replace(config.debug.overlays, show_debug_panel=False),
        ),
    )
    runtime = VisionBeatRuntime(
        config=config,
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=0)]
        ),
        tracker=FakeTracker([make_pose(1.0)]),
        detector=FakeDetector(events_by_frame=[[]], candidates_by_frame=[()], cooldowns=[0.0]),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([True], key_sequence=[ord("d")]),
    )

    runtime.run()

    assert runtime.overlay.debug_enabled == [False]


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


def test_runtime_records_tracking_failures_for_non_tracking_status() -> None:
    recorder = FakeRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=3.0, frame_index=2)]
        ),
        tracker=FakeTracker(
            [make_pose(3.0, status="no_person_detected", person_detected=False)]
        ),
        detector=FakeDetector(
            events_by_frame=[[]],
            candidates_by_frame=[()],
            cooldowns=[0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        recorder=recorder,
    )

    runtime.process_next_frame()

    assert recorder.tracking_failures == [(3.0, "no_person_detected")]


def test_runtime_logs_rhythm_predictions_without_touching_audio() -> None:
    recorder = FakeRecorder()
    timestamps = (1.0, 1.5, 2.0)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.9,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in timestamps
    ]
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "shadow",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in timestamps]),
        detector=FakeDetector(
            events_by_frame=[[event] for event in events],
            candidates_by_frame=[(), (), ()],
            cooldowns=[0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False]),
        recorder=recorder,
    )

    runtime.process_next_frame()
    runtime.process_next_frame()
    runtime.process_next_frame()

    assert [trigger.gesture for trigger in runtime.audio.triggers] == [
        GestureType.KICK,
        GestureType.KICK,
        GestureType.KICK,
    ]
    assert len(recorder.rhythm_predictions) == 1
    prediction = recorder.rhythm_predictions[0]
    assert prediction["gesture"] is GestureType.KICK
    assert prediction["outcome"] == "pending"
    assert prediction["timestamp"] == pytest.approx(2.0)
    assert prediction["frame_index"] == 2
    assert prediction["predicted_time_seconds"] == pytest.approx(2.5)
    assert prediction["actual_time_seconds"] is None
    assert prediction["actual_gesture"] is None
    assert prediction["error_ms"] is None
    assert prediction["last_event_timestamp"] == pytest.approx(2.0)
    assert prediction["estimated_interval_seconds"] == pytest.approx(0.5)
    assert prediction["next_expected_timestamp"] == pytest.approx(2.5)
    assert prediction["interval_seconds"] == pytest.approx(0.5)
    assert prediction["expires_after_seconds"] == pytest.approx(2.875)
    assert prediction["seconds_until_prediction"] == pytest.approx(0.5)
    assert prediction["seconds_until_expiry"] == pytest.approx(0.875)
    assert prediction["match_tolerance_seconds"] == pytest.approx(0.12)
    assert prediction["confidence"] == pytest.approx(0.9)
    assert prediction["repetition_count"] == 2
    assert prediction["jitter"] == pytest.approx(0.0)
    assert prediction["active"] is True
    assert prediction["shadow_only"] is True
    assert prediction["source"] == "heuristic"
    assert prediction["trigger_mode"] == "shadow"
    assert "predicted next kick" in prediction["status_description"]
    assert runtime.overlay.calls[-1][1].rhythm_status is not None
    assert "kick next @2.500s" in runtime.overlay.calls[-1][1].rhythm_status
    assert "conf=0.90/0.70" in runtime.overlay.calls[-1][1].rhythm_status


def test_runtime_logs_missed_rhythm_prediction_after_skipped_beat() -> None:
    recorder = FakeRecorder()
    event_timestamps = (1.0, 1.5, 2.0)
    frame_timestamps = (1.0, 1.5, 2.0, 2.9)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.9,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in event_timestamps
    ]
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "shadow",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(frame_timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in frame_timestamps]),
        detector=FakeDetector(
            events_by_frame=[[events[0]], [events[1]], [events[2]], []],
            candidates_by_frame=[(), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False]),
        recorder=recorder,
    )

    for _ in frame_timestamps:
        runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 3
    assert len(recorder.rhythm_predictions) == 2
    active_prediction, missed_prediction = recorder.rhythm_predictions
    assert active_prediction["active"] is True
    assert missed_prediction["active"] is False
    assert missed_prediction["outcome"] == "missed"
    assert missed_prediction["gesture"] is GestureType.KICK
    assert missed_prediction["timestamp"] == pytest.approx(2.9)
    assert missed_prediction["predicted_time_seconds"] == pytest.approx(2.5)
    assert missed_prediction["actual_time_seconds"] is None
    assert missed_prediction["error_ms"] is None
    assert missed_prediction["next_expected_timestamp"] == pytest.approx(2.5)
    assert missed_prediction["expires_after_seconds"] == pytest.approx(2.875)
    assert missed_prediction["seconds_until_prediction"] == pytest.approx(-0.4)
    assert missed_prediction["seconds_until_expiry"] == pytest.approx(-0.025)
    assert missed_prediction["source"] == "advance"
    assert missed_prediction["shadow_only"] is True
    assert "prediction state has expired" in missed_prediction["status_description"]
    assert runtime.overlay.calls[-1][1].rhythm_status is not None
    assert "last=missed/expired kick" in runtime.overlay.calls[-1][1].rhythm_status


def test_runtime_logs_matched_rhythm_prediction_without_extra_audio() -> None:
    recorder = FakeRecorder()
    timestamps = (1.0, 1.5, 2.0, 2.55)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.9,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in timestamps
    ]
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "shadow",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in timestamps]),
        detector=FakeDetector(
            events_by_frame=[[event] for event in events],
            candidates_by_frame=[(), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False]),
        recorder=recorder,
    )

    for _ in timestamps:
        runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 4
    matched = [
        prediction
        for prediction in recorder.rhythm_predictions
        if prediction["outcome"] == "matched"
    ]
    assert len(matched) == 1
    assert matched[0]["predicted_time_seconds"] == pytest.approx(2.5)
    assert matched[0]["actual_time_seconds"] == pytest.approx(2.55)
    assert matched[0]["error_ms"] == pytest.approx(50.0)
    assert matched[0]["active"] is False


def test_runtime_rhythm_direct_trigger_plays_expected_beat_without_self_training() -> None:
    event_timestamps = (1.0, 1.5, 2.0)
    frame_timestamps = (1.0, 1.5, 2.0, 2.5, 3.0)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.82,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in event_timestamps
    ]
    recorder = FakeRecorder()
    session_recorder = FakeSessionRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "direct",
                    "rhythm_confidence_threshold": 0.7,
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(frame_timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in frame_timestamps]),
        detector=FakeDetector(
            events_by_frame=[[events[0]], [events[1]], [events[2]], [], []],
            candidates_by_frame=[(), (), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False, False]),
        recorder=recorder,
        session_recorder=session_recorder,
    )

    for _ in frame_timestamps:
        runtime.process_next_frame()

    assert [trigger.gesture for trigger in runtime.audio.triggers] == [
        GestureType.KICK,
        GestureType.KICK,
        GestureType.KICK,
        GestureType.KICK,
    ]
    assert runtime.audio.triggers[-1].intensity == pytest.approx(0.82)
    assert session_recorder.triggers[-1].label == "Kick (rhythm predictor)"
    assert len(recorder.rhythm_live_triggers) == 1
    rhythm_trigger = recorder.rhythm_live_triggers[0]
    assert rhythm_trigger["gesture"] is GestureType.KICK
    assert rhythm_trigger["predicted_time_seconds"] == pytest.approx(2.5)
    assert rhythm_trigger["timestamp"] == pytest.approx(2.5)
    assert rhythm_trigger["scheduling_error_ms"] == pytest.approx(0.0)
    assert rhythm_trigger["source"] == "rhythm_direct"
    assert [trigger.label for trigger in session_recorder.triggers].count(
        "Kick (rhythm predictor)"
    ) == 1


def test_runtime_rhythm_direct_does_not_duplicate_matching_real_hit() -> None:
    timestamps = (1.0, 1.5, 2.0)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.82,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in timestamps
    ]
    matching_event = GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.84,
        hand="right",
        timestamp=FrameTimestamp(seconds=2.5),
        label="Downward strike trigger",
    )
    recorder = FakeRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "direct",
                    "rhythm_confidence_threshold": 0.7,
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate((*timestamps, 2.5))
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in (*timestamps, 2.5)]),
        detector=FakeDetector(
            events_by_frame=[[events[0]], [events[1]], [events[2]], [matching_event]],
            candidates_by_frame=[(), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False]),
        recorder=recorder,
    )

    for _ in (*timestamps, 2.5):
        runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 4
    assert runtime.audio.triggers[-1].intensity == pytest.approx(0.84)
    assert recorder.rhythm_live_triggers == []
    matched = [
        prediction
        for prediction in recorder.rhythm_predictions
        if prediction["outcome"] == "matched"
    ]
    assert len(matched) == 1
    assert matched[0]["actual_time_seconds"] == pytest.approx(2.5)


def test_runtime_rhythm_direct_suppresses_late_matching_real_hit_duplicate() -> None:
    event_timestamps = (1.0, 1.5, 2.0)
    frame_timestamps = (1.0, 1.5, 2.0, 2.5, 2.55)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.82,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in event_timestamps
    ]
    late_matching_event = GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.84,
        hand="right",
        timestamp=FrameTimestamp(seconds=2.55),
        label="Downward strike trigger",
    )
    recorder = FakeRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "direct",
                    "rhythm_confidence_threshold": 0.7,
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(frame_timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in frame_timestamps]),
        detector=FakeDetector(
            events_by_frame=[
                [events[0]],
                [events[1]],
                [events[2]],
                [],
                [late_matching_event],
            ],
            candidates_by_frame=[(), (), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False, False]),
        recorder=recorder,
    )

    for _ in frame_timestamps:
        runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 4
    assert len(recorder.rhythm_live_triggers) == 1
    matched = [
        prediction
        for prediction in recorder.rhythm_predictions
        if prediction["outcome"] == "matched"
    ]
    assert len(matched) == 1
    assert matched[0]["actual_time_seconds"] == pytest.approx(2.55)
    assert matched[0]["source"] == "heuristic_after_rhythm_direct"


def test_runtime_predictive_primary_events_feed_rhythm_direct_trigger() -> None:
    timestamps = (1.0, 1.5, 2.0, 2.5)
    predictive_event = ShadowPredictionEvent(
        frame_index=7,
        timestamp=FrameTimestamp(seconds=1.0),
        timing_probability=0.86,
        threshold=0.6,
        run_length=2,
        gesture=GestureType.KICK,
        gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
        heuristic_triggered_on_peak_frame=False,
        heuristic_gesture_types_on_peak_frame=(),
    )
    recorder = FakeRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "primary",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "direct",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in timestamps]),
        detector=FakeDetector(
            events_by_frame=[[], [], [], []],
            candidates_by_frame=[(), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False]),
        recorder=recorder,
        session_recorder=FakeSessionRecorder(),
        predictive_shadow_runner=FakePredictiveShadowRunner((predictive_event,)),
    )

    runtime.process_next_frame()
    runtime.process_next_frame()
    runtime.process_next_frame()
    runtime.predictive_shadow_runner.events = ()
    runtime.process_next_frame()

    assert [trigger.gesture for trigger in runtime.audio.triggers] == [
        GestureType.KICK,
        GestureType.KICK,
        GestureType.KICK,
        GestureType.KICK,
    ]
    assert len(recorder.predictive_live_triggers) == 3
    assert len(recorder.rhythm_live_triggers) == 1
    assert recorder.rhythm_live_triggers[0]["predicted_time_seconds"] == pytest.approx(2.5)
    assert runtime.session_recorder.triggers[-1].label == "Kick (rhythm predictor)"


def test_runtime_predictive_hybrid_silent_completions_do_not_train_rhythm() -> None:
    timestamps = (1.0, 1.5, 2.0)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.82,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in timestamps
    ]
    recorder = FakeRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "arm_only",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in timestamps]),
        detector=FakeDetector(
            events_by_frame=[[event] for event in events],
            candidates_by_frame=[(), (), ()],
            cooldowns=[0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False]),
        recorder=recorder,
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.10/0.60 top=kick 0.51",
            latest_status=PredictiveStatus(
                available_window_frames=24,
                required_window_size=24,
                threshold=0.6,
                timing_probability=0.10,
                predicted_gesture=GestureType.KICK,
                predicted_gesture_confidence=0.51,
                class_probabilities={"kick": 0.51, "snare": 0.49},
            ),
        ),
    )

    for _ in timestamps:
        runtime.process_next_frame()

    assert runtime.audio.triggers == []
    assert recorder.rhythm_predictions == []
    assert runtime.overlay.calls[-1][1].predictive_status == (
        "p=0.10/0.60 top=kick 0.51 arm=--"
    )


def test_runtime_predictive_hybrid_rhythm_arm_learns_from_confirmed_audio() -> None:
    timestamps = (1.0, 1.5, 2.0)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.82,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in timestamps
    ]
    recorder = FakeRecorder()
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "arm_only",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in timestamps]),
        detector=FakeDetector(
            events_by_frame=[[event] for event in events],
            candidates_by_frame=[(), (), ()],
            cooldowns=[0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False]),
        recorder=recorder,
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.90/0.60 top=kick 0.93",
            latest_status=PredictiveStatus(
                available_window_frames=24,
                required_window_size=24,
                threshold=0.6,
                timing_probability=0.90,
                predicted_gesture=GestureType.KICK,
                predicted_gesture_confidence=0.93,
                class_probabilities={"kick": 0.93, "snare": 0.07},
            ),
        ),
    )

    for _ in timestamps:
        runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 3
    assert len(recorder.predictive_live_triggers) == 3
    pending_predictions = [
        event for event in recorder.rhythm_predictions if event["outcome"] == "pending"
    ]
    assert len(pending_predictions) == 1
    assert runtime.overlay.calls[-1][1].predictive_status == (
        "p=0.90/0.60 top=kick 0.93 arm=rhythm:kick 0.82 ttl=28"
    )


def test_runtime_predictive_hybrid_rhythm_arm_expires_safely() -> None:
    event_timestamps = (1.0, 1.5, 2.0)
    frame_timestamps = (1.0, 1.5, 2.0, 2.9)
    events = [
        GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.82,
            hand="right",
            timestamp=FrameTimestamp(seconds=timestamp),
            label="Downward strike trigger",
        )
        for timestamp in event_timestamps
    ]
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                    "rhythm_prediction_enabled": True,
                    "rhythm_trigger_mode": "arm_only",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(
                    image=FakeFrame(f"frame-{index}"),
                    captured_at=timestamp,
                    frame_index=index,
                )
                for index, timestamp in enumerate(frame_timestamps)
            ]
        ),
        tracker=FakeTracker([make_pose(timestamp) for timestamp in frame_timestamps]),
        detector=FakeDetector(
            events_by_frame=[[events[0]], [events[1]], [events[2]], []],
            candidates_by_frame=[(), (), (), ()],
            cooldowns=[0.0, 0.0, 0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False, False, False]),
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.90/0.60 top=kick 0.93",
            latest_status=PredictiveStatus(
                available_window_frames=24,
                required_window_size=24,
                threshold=0.6,
                timing_probability=0.90,
                predicted_gesture=GestureType.KICK,
                predicted_gesture_confidence=0.93,
                class_probabilities={"kick": 0.93, "snare": 0.07},
            ),
        ),
    )

    runtime.process_next_frame()
    runtime.process_next_frame()
    runtime.process_next_frame()
    runtime.predictive_shadow_runner.latest_status = PredictiveStatus(
        available_window_frames=24,
        required_window_size=24,
        threshold=0.6,
        timing_probability=0.10,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.51,
        class_probabilities={"kick": 0.51, "snare": 0.49},
    )
    runtime.predictive_shadow_runner._status_summary = "p=0.10/0.60 top=kick 0.51"
    runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 3
    assert runtime.overlay.calls[-1][1].predictive_status == (
        "p=0.10/0.60 top=kick 0.51 arm=--"
    )


def test_runtime_async_pipeline_decouples_render_loop_from_inference() -> None:
    config = replace(
        AppConfig(),
        runtime=RuntimeConfig(
            async_pipeline=True,
            target_render_fps=30,
            idle_sleep_seconds=0.001,
        ),
    )
    camera = StreamingFakeCamera(interval_seconds=1.0 / 30.0)
    tracker = StreamingFakeTracker(delay_seconds=0.05)
    overlay = FakeOverlay()
    preview = FakePreview([False, False, False, False, False, True])
    runtime = VisionBeatRuntime(
        config=config,
        camera=camera,
        tracker=tracker,
        detector=StreamingFakeDetector(),
        audio=FakeAudio(),
        transport=FakeTransport(),
        overlay=overlay,
        preview=preview,
    )

    runtime.run()

    assert camera.open_calls == 1
    assert camera.close_calls == 1
    assert tracker.close_calls == 1
    assert len(preview.show_calls) >= 3
    states = [state for _, state in overlay.calls]
    assert any(state.pose.status == "warming_up" for state in states)
    tracked_states = [state for state in states if state.pose.status == "tracking"]
    assert tracked_states
    assert any(state.capture_fps is not None for state in tracked_states)
    assert any(state.inference_fps is not None for state in tracked_states)
    assert any(state.render_fps is not None for state in tracked_states)
    assert any(state.pipeline_latency_ms is not None for state in tracked_states)


class FakePredictiveShadowRunner:
    def __init__(
        self,
        events: tuple[ShadowPredictionEvent, ...],
        *,
        status_summary: str = "p=0.23/0.30 top=kick 0.71",
        latest_status: PredictiveStatus | None = None,
        prediction_horizon_frames: int = 8,
    ) -> None:
        self.required_window_size = 2
        self.prediction_horizon_frames = prediction_horizon_frames
        self.events = events
        self._status_summary = status_summary
        self.latest_status = latest_status or PredictiveStatus(
            available_window_frames=2,
            required_window_size=2,
            threshold=0.3,
            timing_probability=0.23,
            predicted_gesture=GestureType.KICK,
            predicted_gesture_confidence=0.71,
            class_probabilities={"kick": 0.71, "snare": 0.29},
        )
        self.update_calls: list[dict[str, object]] = []
        self.flush_calls = 0

    def update(
        self,
        *,
        feature_window,
        frame_index: int,
        timestamp: FrameTimestamp,
        heuristic_events: tuple[GestureEvent, ...],
    ) -> tuple[ShadowPredictionEvent, ...]:
        self.update_calls.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "heuristic_events": heuristic_events,
                "window_rows": len(feature_window.matrix),
                "frame_count": len(feature_window.frames),
            }
        )
        return self.events

    def flush(self) -> tuple[ShadowPredictionEvent, ...]:
        self.flush_calls += 1
        return ()

    def status_summary(self) -> str:
        return self._status_summary


def test_runtime_logs_predictive_shadow_triggers_without_touching_audio() -> None:
    recorder = FakeRecorder()
    session_recorder = FakeSessionRecorder()
    shadow_event = ShadowPredictionEvent(
        frame_index=7,
        timestamp=FrameTimestamp(seconds=1.0),
        timing_probability=0.81,
        threshold=0.6,
        run_length=2,
        gesture=GestureType.SNARE,
        gesture_confidence=0.92,
        class_probabilities={"kick": 0.08, "snare": 0.92},
        heuristic_triggered_on_peak_frame=False,
        heuristic_gesture_types_on_peak_frame=(),
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "shadow",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                }
            )
        ),
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=7)]
        ),
        tracker=FakeTracker([make_pose(1.0)]),
        detector=FakeDetector(events_by_frame=[[]], candidates_by_frame=[()], cooldowns=[0.0]),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        recorder=recorder,
        session_recorder=session_recorder,
        predictive_shadow_runner=FakePredictiveShadowRunner((shadow_event,)),
    )

    runtime.process_next_frame()

    assert runtime.audio.triggers == []
    assert recorder.predictive_shadow_triggers[0]["predicted_gesture"] is GestureType.SNARE
    assert session_recorder.predictive_shadow_triggers[0]["gesture"] == "snare"
    assert runtime.overlay.calls[0][1].predictive_status == "p=0.23/0.30 top=kick 0.71"


def test_runtime_uses_predictive_primary_mode_to_drive_audio() -> None:
    recorder = FakeRecorder()
    session_recorder = FakeSessionRecorder()
    predictive_event = ShadowPredictionEvent(
        frame_index=9,
        timestamp=FrameTimestamp(seconds=1.10),
        timing_probability=0.84,
        threshold=0.6,
        run_length=2,
        gesture=GestureType.KICK,
        gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
        heuristic_triggered_on_peak_frame=True,
        heuristic_gesture_types_on_peak_frame=("snare",),
    )
    heuristic_event = GestureEvent(
        gesture=GestureType.SNARE,
        confidence=0.77,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.25),
        label="Wrist collision trigger",
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "primary",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                }
            )
        ),
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=1.25, frame_index=10)]
        ),
        tracker=FakeTracker([make_pose(1.25)]),
        detector=FakeDetector(
            events_by_frame=[[heuristic_event]],
            candidates_by_frame=[()],
            cooldowns=[0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        recorder=recorder,
        session_recorder=session_recorder,
        predictive_shadow_runner=FakePredictiveShadowRunner((predictive_event,)),
    )

    runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 1
    assert runtime.audio.triggers[0].gesture is GestureType.KICK
    assert runtime.audio.triggers[0].intensity == pytest.approx(0.84)
    assert recorder.predictive_shadow_triggers == []
    assert recorder.predictive_live_triggers[0]["predicted_gesture"] is GestureType.KICK
    assert recorder.predictive_live_triggers[0]["timestamp"] == pytest.approx(1.25)
    assert recorder.predictive_live_triggers[0]["frame_index"] == 10
    assert session_recorder.predictive_shadow_triggers == []
    assert session_recorder.triggers[0].gesture is GestureType.KICK
    assert session_recorder.triggers[0].timestamp.seconds == pytest.approx(1.25)
    assert session_recorder.triggers[0].label == "Kick (CNN)"
    assert runtime.overlay.calls[0][1].confirmed_gesture is not None
    assert runtime.overlay.calls[0][1].confirmed_gesture.timestamp.seconds == pytest.approx(1.25)


def test_runtime_predictive_hybrid_mode_shows_active_arm_while_waiting_for_completion() -> None:
    latest_status = PredictiveStatus(
        available_window_frames=24,
        required_window_size=24,
        threshold=0.6,
        timing_probability=0.84,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                }
            )
        ),
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=9)]
        ),
        tracker=FakeTracker([make_pose(1.0)]),
        detector=FakeDetector(events_by_frame=[[]], candidates_by_frame=[()], cooldowns=[0.0]),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.84/0.60 top=kick 0.91",
            latest_status=latest_status,
            prediction_horizon_frames=3,
        ),
    )

    runtime.process_next_frame()

    assert runtime.audio.triggers == []
    assert runtime.overlay.calls[0][1].predictive_status == (
        "p=0.84/0.60 top=kick 0.91 arm=kick 0.91 ttl=4"
    )


def test_runtime_predictive_hybrid_mode_fires_on_matching_completion() -> None:
    recorder = FakeRecorder()
    session_recorder = FakeSessionRecorder()
    latest_status = PredictiveStatus(
        available_window_frames=24,
        required_window_size=24,
        threshold=0.6,
        timing_probability=0.84,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    )
    heuristic_event = GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.77,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.25),
        label="Downward strike trigger",
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                }
            )
        ),
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=1.25, frame_index=9)]
        ),
        tracker=FakeTracker([make_pose(1.25)]),
        detector=FakeDetector(
            events_by_frame=[[heuristic_event]],
            candidates_by_frame=[()],
            cooldowns=[0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        recorder=recorder,
        session_recorder=session_recorder,
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.84/0.60 top=kick 0.91",
            latest_status=latest_status,
        ),
    )

    runtime.process_next_frame()

    assert len(runtime.audio.triggers) == 1
    assert runtime.audio.triggers[0].gesture is GestureType.KICK
    assert runtime.audio.triggers[0].intensity == pytest.approx(0.77)
    assert recorder.predictive_shadow_triggers == []
    assert recorder.predictive_live_triggers[0]["predicted_gesture"] is GestureType.KICK
    assert recorder.predictive_live_triggers[0]["timing_probability"] == pytest.approx(0.84)
    assert session_recorder.predictive_shadow_triggers == []
    assert session_recorder.triggers[0].gesture is GestureType.KICK
    assert session_recorder.triggers[0].label == "Kick (CNN)"
    assert runtime.overlay.calls[0][1].predictive_status == "p=0.84/0.60 top=kick 0.91 arm=--"


def test_runtime_predictive_hybrid_mode_latches_arm_across_conflicting_status() -> None:
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                }
            )
        ),
        camera=FakeCamera(
            [
                CameraFrame(image=FakeFrame("frame-0"), captured_at=1.00, frame_index=9),
                CameraFrame(image=FakeFrame("frame-1"), captured_at=1.03, frame_index=10),
            ]
        ),
        tracker=FakeTracker([make_pose(1.00), make_pose(1.03)]),
        detector=FakeDetector(
            events_by_frame=[[], []],
            candidates_by_frame=[(), ()],
            cooldowns=[0.0, 0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False, False]),
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.84/0.60 top=snare 0.88",
            latest_status=PredictiveStatus(
                available_window_frames=24,
                required_window_size=24,
                threshold=0.6,
                timing_probability=0.84,
                predicted_gesture=GestureType.SNARE,
                predicted_gesture_confidence=0.88,
                class_probabilities={"kick": 0.12, "snare": 0.88},
            ),
            prediction_horizon_frames=3,
        ),
    )

    runtime.process_next_frame()
    runtime.predictive_shadow_runner.latest_status = PredictiveStatus(
        available_window_frames=24,
        required_window_size=24,
        threshold=0.6,
        timing_probability=0.93,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.92,
        class_probabilities={"kick": 0.92, "snare": 0.08},
    )
    runtime.predictive_shadow_runner._status_summary = "p=0.93/0.60 top=kick 0.92"

    runtime.process_next_frame()

    assert runtime.audio.triggers == []
    assert runtime.overlay.calls[0][1].predictive_status == (
        "p=0.84/0.60 top=snare 0.88 arm=snare 0.88 ttl=4"
    )
    assert runtime.overlay.calls[1][1].predictive_status == (
        "p=0.93/0.60 top=kick 0.92 arm=snare 0.88 ttl=3"
    )


def test_runtime_predictive_hybrid_mode_rejects_mismatched_completion() -> None:
    recorder = FakeRecorder()
    latest_status = PredictiveStatus(
        available_window_frames=24,
        required_window_size=24,
        threshold=0.6,
        timing_probability=0.84,
        predicted_gesture=GestureType.KICK,
        predicted_gesture_confidence=0.91,
        class_probabilities={"kick": 0.91, "snare": 0.09},
    )
    heuristic_event = GestureEvent(
        gesture=GestureType.SNARE,
        confidence=0.77,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.25),
        label="Wrist collision trigger",
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(
            predictive=PredictiveConfig.from_mapping(
                {
                    "mode": "hybrid",
                    "timing_checkpoint_path": "models/timing.pt",
                    "gesture_checkpoint_path": "models/gesture.pt",
                }
            )
        ),
        camera=FakeCamera(
            [CameraFrame(image=FakeFrame("frame-0"), captured_at=1.25, frame_index=9)]
        ),
        tracker=FakeTracker([make_pose(1.25)]),
        detector=FakeDetector(
            events_by_frame=[[heuristic_event]],
            candidates_by_frame=[()],
            cooldowns=[0.0],
        ),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([False]),
        recorder=recorder,
        predictive_shadow_runner=FakePredictiveShadowRunner(
            (),
            status_summary="p=0.84/0.60 top=kick 0.91",
            latest_status=latest_status,
        ),
    )

    runtime.process_next_frame()

    assert runtime.audio.triggers == []
    assert recorder.predictive_live_triggers == []
    assert runtime.overlay.calls[0][1].predictive_status == (
        "p=0.84/0.60 top=kick 0.91 arm=kick 0.91 ttl=9"
    )


def test_compute_fps_handles_initial_non_increasing_and_increasing_timestamps() -> None:
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=FakeCamera([]),
        tracker=FakeTracker([]),
        detector=FakeDetector(events_by_frame=[], candidates_by_frame=[], cooldowns=[]),
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([]),
    )

    first = runtime._compute_fps(
        CameraFrame(image=FakeFrame("frame-0"), captured_at=1.0, frame_index=0)
    )
    second = runtime._compute_fps(
        CameraFrame(image=FakeFrame("frame-1"), captured_at=1.0, frame_index=1)
    )
    third = runtime._compute_fps(
        CameraFrame(image=FakeFrame("frame-2"), captured_at=1.25, frame_index=2)
    )

    assert first is None
    assert second is None
    assert third == pytest.approx(4.0)


def test_select_candidate_prefers_highest_confidence() -> None:
    best = DetectionCandidate(
        gesture=GestureType.KICK,
        confidence=0.9,
        hand="right",
        label="best",
    )
    detector = FakeDetector(
        events_by_frame=[],
        candidates_by_frame=[],
        cooldowns=[],
    )
    detector._candidates = (
        DetectionCandidate(gesture=GestureType.SNARE, confidence=0.4, hand="right", label="weak"),
        best,
    )
    runtime = VisionBeatRuntime(
        config=AppConfig(),
        camera=FakeCamera([]),
        tracker=FakeTracker([]),
        detector=detector,
        audio=FakeAudio(),
        overlay=FakeOverlay(),
        preview=FakePreview([]),
    )

    assert runtime._select_candidate() == best


def test_visionbeat_app_builds_runtime_from_default_dependency_factories(monkeypatch) -> None:
    recorder = FakeRecorder()
    created: dict[str, object] = {}

    class StubCamera:
        def __init__(self, config, recorder=None) -> None:
            created["camera"] = (config, recorder)

    class StubTracker:
        def __init__(self, config) -> None:
            created["tracker"] = config

    class StubDetector:
        def __init__(self, config, observer=None) -> None:
            created["detector"] = (config, observer)

    class StubAudio:
        def close(self) -> None:
            return None

    class StubOverlay:
        def __init__(self, config) -> None:
            created["overlay"] = config

    class StubPreview:
        def close(self) -> None:
            return None

    monkeypatch.setattr("visionbeat.app.build_observability_recorder", lambda config: recorder)
    monkeypatch.setattr("visionbeat.app.CameraSource", StubCamera)
    monkeypatch.setattr("visionbeat.app.create_pose_provider", StubTracker)
    monkeypatch.setattr("visionbeat.app.GestureDetector", StubDetector)
    monkeypatch.setattr("visionbeat.app.create_audio_engine", lambda config: StubAudio())
    monkeypatch.setattr("visionbeat.app.OverlayRenderer", StubOverlay)
    monkeypatch.setattr("visionbeat.app.OpenCVPreviewWindow", StubPreview)

    app = VisionBeatApp(AppConfig())

    assert isinstance(app.runtime, VisionBeatRuntime)
    assert created["camera"][0] == app.config.camera
    assert created["camera"][1] is recorder
    assert created["tracker"] == app.config.tracker
    assert created["detector"] == (app.config.gestures, recorder)
    assert created["overlay"] == app.config.overlay
    assert recorder.app_startups[0]["camera_resolution"] == "1280x720"
    assert recorder.app_startups[0]["pose_backend"] == "mediapipe"
    assert app.runtime.overlay_toggle_key == ord("o")
    assert app.runtime.debug_toggle_key == ord("d")


def test_visionbeat_app_accepts_custom_toggle_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "visionbeat.app.build_observability_recorder",
        lambda config: FakeRecorder(),
    )
    monkeypatch.setattr("visionbeat.app.CameraSource", lambda config, recorder=None: object())
    monkeypatch.setattr("visionbeat.app.create_pose_provider", lambda config: object())
    monkeypatch.setattr("visionbeat.app.GestureDetector", lambda config, observer=None: object())
    monkeypatch.setattr("visionbeat.app.create_audio_engine", lambda config: FakeAudio())
    monkeypatch.setattr("visionbeat.app.OverlayRenderer", lambda config: FakeOverlay())
    monkeypatch.setattr("visionbeat.app.OpenCVPreviewWindow", lambda: FakePreview([True]))

    app = VisionBeatApp(AppConfig(), overlay_toggle_key="x", debug_toggle_key="z")

    assert app.runtime.overlay_toggle_key == ord("x")
    assert app.runtime.debug_toggle_key == ord("z")

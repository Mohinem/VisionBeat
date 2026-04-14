from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from visionbeat.app import VisionBeatApp, VisionBeatRuntime
from visionbeat.camera import CameraFrame
from visionbeat.config import AppConfig
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

    def close(self) -> None:
        self.close_calls += 1


class FakeSessionRecorder:
    def __init__(self) -> None:
        self.session_dir = Path("/tmp/fake-session")
        self.camera_frames: list[CameraFrame] = []
        self.tracker_outputs: list[tuple[CameraFrame, TrackerOutput]] = []
        self.triggers: list[GestureEvent] = []
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
    assert runtime.latest_frame_features.temporal_features["left_wrist_rel_vx"] == pytest.approx(1.0)
    assert runtime.latest_frame_features.temporal_features["left_wrist_rel_vy"] == pytest.approx(1.0)

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

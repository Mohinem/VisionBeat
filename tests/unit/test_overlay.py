from __future__ import annotations

from dataclasses import dataclass

from visionbeat.config import OverlayConfig
from visionbeat.models import (
    FrameTimestamp,
    GestureEvent,
    GestureType,
    LandmarkPoint,
    RenderState,
    TrackerOutput,
)
from visionbeat.overlay import OverlayRenderer, draw_labels, draw_pose_landmarks, draw_trigger_flash


@dataclass
class FakeFrame:
    height: int
    width: int
    channels: int = 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def copy(self) -> FakeFrame:
        return FakeFrame(self.height, self.width, self.channels)


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def line(self, *args: object) -> None:
        self.calls.append(("line", args))

    def circle(self, *args: object) -> None:
        self.calls.append(("circle", args))

    def putText(self, *args: object) -> None:  # noqa: N802
        self.calls.append(("putText", args))

    def rectangle(self, *args: object) -> None:
        self.calls.append(("rectangle", args))


def make_pose() -> TrackerOutput:
    return TrackerOutput(
        timestamp=FrameTimestamp(seconds=1.0),
        landmarks={
            "left_shoulder": LandmarkPoint(x=0.2, y=0.2, z=-0.1, visibility=0.9),
            "left_elbow": LandmarkPoint(x=0.25, y=0.35, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.3, y=0.5, z=-0.1, visibility=0.9),
        },
        person_detected=True,
        status="tracking",
    )


def test_draw_pose_landmarks_uses_helper_primitives() -> None:
    frame = FakeFrame(100, 200)
    fake_cv2 = FakeCV2()

    result = draw_pose_landmarks(frame, make_pose(), cv2_module=fake_cv2)

    assert result is frame
    assert any(name == "circle" for name, _ in fake_cv2.calls)
    assert any(name == "line" for name, _ in fake_cv2.calls)


def test_draw_pose_landmarks_can_hide_landmark_labels() -> None:
    frame = FakeFrame(100, 200)
    fake_cv2 = FakeCV2()

    result = draw_pose_landmarks(frame, make_pose(), cv2_module=fake_cv2, show_labels=False)

    assert result is frame
    assert any(name == "circle" for name, _ in fake_cv2.calls)
    assert any(name == "line" for name, _ in fake_cv2.calls)
    assert not any(name == "putText" for name, _ in fake_cv2.calls)


def test_draw_labels_skips_empty_entries() -> None:
    frame = FakeFrame(50, 50)
    fake_cv2 = FakeCV2()

    draw_labels(frame, ["Header", "", "Status ok"], cv2_module=fake_cv2)

    assert sum(1 for name, _ in fake_cv2.calls if name == "putText") == 2


def test_overlay_renderer_renders_debug_panel_without_events() -> None:
    renderer = OverlayRenderer(OverlayConfig(), cv2_module=FakeCV2())
    frame = FakeFrame(120, 160)
    pose = TrackerOutput(timestamp=FrameTimestamp(seconds=1.0), status="no_person_detected")

    output = renderer.render(frame, RenderState(pose=pose, frame_index=0))

    assert output.shape == frame.shape
    assert output is not frame


def test_overlay_renderer_skeleton_only_hud_preserves_landmark_labels_without_panels() -> None:
    fake_cv2 = FakeCV2()
    renderer = OverlayRenderer(
        OverlayConfig(
            show_landmark_labels=True,
            show_debug_panel=False,
            show_trigger_flash=False,
        ),
        cv2_module=fake_cv2,
    )
    renderer.set_debug_enabled(True)
    frame = FakeFrame(120, 160)
    pose = make_pose()
    state = RenderState(
        pose=pose,
        frame_index=0,
        confirmed_gesture=GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.9,
            hand="right",
            timestamp=FrameTimestamp(seconds=0.95),
            label="Downward strike -> kick",
        ),
    )

    output = renderer.render(frame, state)

    assert output is not frame
    assert any(name == "line" for name, _ in fake_cv2.calls)
    assert any(name == "circle" for name, _ in fake_cv2.calls)
    assert not any(name == "rectangle" for name, _ in fake_cv2.calls)
    assert any(
        name == "putText" and args[1] == "left shoulder"
        for name, args in fake_cv2.calls
    )


def test_overlay_renderer_debug_panel_includes_predictive_status() -> None:
    fake_cv2 = FakeCV2()
    renderer = OverlayRenderer(OverlayConfig(), cv2_module=fake_cv2)
    frame = FakeFrame(120, 160)
    state = RenderState(
        pose=make_pose(),
        frame_index=4,
        predictive_status="p=0.23/0.30 top=kick 0.71",
        rhythm_status="mode=direct kick next @2.500s (+500ms)",
    )

    renderer.render(frame, state)

    assert any(
        name == "putText" and args[1] == "Predictive: p=0.23/0.30 top=kick 0.71"
        for name, args in fake_cv2.calls
    )
    assert any(
        name == "putText"
        and args[1] == "Rhythm: mode=direct kick next @2.500s (+500ms)"
        for name, args in fake_cv2.calls
    )


def test_draw_trigger_flash_renders_centered_red_block_with_sound_name() -> None:
    frame = FakeFrame(120, 160)
    fake_cv2 = FakeCV2()
    pose = make_pose()
    state = RenderState(
        pose=pose,
        frame_index=0,
        confirmed_gesture=GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.9,
            hand="right",
            timestamp=FrameTimestamp(seconds=0.95),
            label="Downward strike → kick",
        ),
    )

    result = draw_trigger_flash(frame, state, cv2_module=fake_cv2)

    assert result is frame
    assert ("rectangle", (frame, (32, 24), (128, 96), (0, 0, 255), -1)) in fake_cv2.calls
    assert any(
        name == "putText" and args[1] == "KICK"
        for name, args in fake_cv2.calls
    )


def test_draw_trigger_flash_labels_rhythm_predictor_triggers() -> None:
    frame = FakeFrame(360, 640)
    fake_cv2 = FakeCV2()
    pose = make_pose()
    state = RenderState(
        pose=pose,
        frame_index=0,
        confirmed_gesture=GestureEvent(
            gesture=GestureType.SNARE,
            confidence=0.88,
            hand="right",
            timestamp=FrameTimestamp(seconds=0.95),
            label="Snare (rhythm predictor)",
        ),
    )

    result = draw_trigger_flash(frame, state, cv2_module=fake_cv2)

    assert result is frame
    assert any(
        name == "putText" and args[1] == "Snare (rhythm predictor)"
        for name, args in fake_cv2.calls
    )


def test_draw_trigger_flash_labels_cnn_predictor_triggers() -> None:
    frame = FakeFrame(360, 640)
    fake_cv2 = FakeCV2()
    pose = make_pose()
    state = RenderState(
        pose=pose,
        frame_index=0,
        confirmed_gesture=GestureEvent(
            gesture=GestureType.KICK,
            confidence=0.88,
            hand="right",
            timestamp=FrameTimestamp(seconds=0.95),
            label="Kick (CNN)",
        ),
    )

    result = draw_trigger_flash(frame, state, cv2_module=fake_cv2)

    assert result is frame
    assert any(
        name == "putText" and args[1] == "Kick (CNN)"
        for name, args in fake_cv2.calls
    )


def test_draw_trigger_flash_skips_stale_confirmed_gesture() -> None:
    frame = FakeFrame(120, 160)
    fake_cv2 = FakeCV2()
    pose = make_pose()
    state = RenderState(
        pose=pose,
        frame_index=0,
        confirmed_gesture=GestureEvent(
            gesture=GestureType.SNARE,
            confidence=0.9,
            hand="right",
            timestamp=FrameTimestamp(seconds=0.70),
            label="Wrist collision → snare",
        ),
    )

    result = draw_trigger_flash(frame, state, cv2_module=fake_cv2)

    assert result is frame
    assert fake_cv2.calls == []

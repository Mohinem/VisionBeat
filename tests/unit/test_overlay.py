from __future__ import annotations

from dataclasses import dataclass

from visionbeat.config import OverlayConfig
from visionbeat.models import FrameTimestamp, LandmarkPoint, RenderState, TrackerOutput
from visionbeat.overlay import OverlayRenderer, draw_labels, draw_pose_landmarks


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

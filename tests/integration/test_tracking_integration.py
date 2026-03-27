from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from visionbeat.camera import CameraSource
from visionbeat.config import CameraConfig, TrackerConfig
from visionbeat.models import FrameTimestamp
from visionbeat.tracking import PoseTracker

pytestmark = pytest.mark.integration


class FakeCapture:
    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)
        self.properties: list[tuple[int, int]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: int) -> None:
        self.properties.append((prop, value))

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 99

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.flip_calls: list[tuple[object, int]] = []
        self.cvt_calls: list[tuple[object, int]] = []

    def VideoCapture(self, _: int) -> FakeCapture:  # noqa: N802
        return self.capture

    def flip(self, frame: object, flip_code: int) -> object:
        self.flip_calls.append((frame, flip_code))
        return ("flipped", frame, flip_code)

    def cvtColor(self, frame: object, code: int) -> object:  # noqa: N802
        self.cvt_calls.append((frame, code))
        return ("rgb", frame, code)


class FakePose:
    def __init__(self) -> None:
        self.received_frames: list[object] = []
        landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.1) for _ in range(33)]
        landmarks[15] = SimpleNamespace(x=0.4, y=0.5, z=-0.3, visibility=0.95)
        self.result = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))
        self.closed = False

    def process(self, frame: object) -> object:
        self.received_frames.append(frame)
        return self.result

    def close(self) -> None:
        self.closed = True


def test_webcam_capture_scaffolding_allows_camera_source_without_hardware(monkeypatch) -> None:
    fake_capture = FakeCapture(frames=["frame-0"])
    fake_cv2 = FakeCV2(fake_capture)
    camera = CameraSource(CameraConfig(width=640, height=480, fps=24), _cv2=fake_cv2)

    camera.open()
    frame = camera.read_frame()
    camera.close()

    assert frame.image == ("flipped", "frame-0", 1)
    assert frame.frame_index == 0
    assert fake_capture.properties == [(3, 640), (4, 480), (5, 24)]
    assert fake_capture.released is True


def test_pose_tracker_scaffolding_exercises_mediapipe_and_cv2_paths(monkeypatch) -> None:
    fake_pose = FakePose()
    fake_cv2 = FakeCV2(FakeCapture([]))
    fake_mp = SimpleNamespace(
        solutions=SimpleNamespace(
            pose=SimpleNamespace(
                Pose=lambda **_: fake_pose,
            )
        )
    )

    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    tracker = PoseTracker(TrackerConfig(min_tracking_confidence=0.5))
    output = tracker.process("frame-bgr", FrameTimestamp(seconds=1.5))
    tracker.close()

    assert output.status == "tracking"
    assert output.person_detected is True
    assert set(output.landmarks) == {"left_wrist"}
    assert fake_cv2.cvt_calls == [("frame-bgr", fake_cv2.COLOR_BGR2RGB)]
    assert fake_pose.received_frames == [("rgb", "frame-bgr", fake_cv2.COLOR_BGR2RGB)]
    assert fake_pose.closed is True


def test_pose_tracker_uses_python_solutions_when_top_level_solutions_missing(
    monkeypatch,
) -> None:
    fake_pose = FakePose()
    fake_cv2 = FakeCV2(FakeCapture([]))
    fake_mp = SimpleNamespace()
    fake_python_solutions = SimpleNamespace(
        pose=SimpleNamespace(
            Pose=lambda **_: fake_pose,
        )
    )

    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe.python",
        SimpleNamespace(solutions=fake_python_solutions),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    tracker = PoseTracker(TrackerConfig(min_tracking_confidence=0.5))
    output = tracker.process("frame-bgr", FrameTimestamp(seconds=2.0))
    tracker.close()

    assert output.status == "tracking"
    assert output.person_detected is True
    assert set(output.landmarks) == {"left_wrist"}
    assert fake_cv2.cvt_calls == [("frame-bgr", fake_cv2.COLOR_BGR2RGB)]
    assert fake_pose.received_frames == [("rgb", "frame-bgr", fake_cv2.COLOR_BGR2RGB)]
    assert fake_pose.closed is True


def test_pose_tracker_imports_pose_module_when_solutions_namespace_lacks_pose(
    monkeypatch,
) -> None:
    fake_pose = FakePose()
    fake_cv2 = FakeCV2(FakeCapture([]))
    fake_mp = SimpleNamespace(solutions=SimpleNamespace())
    fake_pose_module = SimpleNamespace(Pose=lambda **_: fake_pose)

    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(sys.modules, "mediapipe.solutions.pose", fake_pose_module)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    tracker = PoseTracker(TrackerConfig(min_tracking_confidence=0.5))
    output = tracker.process("frame-bgr", FrameTimestamp(seconds=2.0))
    tracker.close()

    assert output.status == "tracking"
    assert output.person_detected is True
    assert set(output.landmarks) == {"left_wrist"}
    assert fake_cv2.cvt_calls == [("frame-bgr", fake_cv2.COLOR_BGR2RGB)]
    assert fake_pose.received_frames == [("rgb", "frame-bgr", fake_cv2.COLOR_BGR2RGB)]
    assert fake_pose.closed is True


def test_pose_tracker_raises_clear_error_when_pose_api_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace())
    monkeypatch.delitem(sys.modules, "mediapipe.python", raising=False)

    with pytest.raises(RuntimeError, match="Unable to locate MediaPipe Pose API"):
        PoseTracker(TrackerConfig(min_tracking_confidence=0.5))


@pytest.mark.webcam
def test_default_webcam_can_capture_frame() -> None:
    pytest.importorskip("cv2", exc_type=ImportError)

    camera = CameraSource(CameraConfig())
    try:
        camera.open()
        frame = camera.read_frame()
    except RuntimeError as exc:
        pytest.skip(f"Webcam integration test skipped: {exc}")
    else:
        assert frame.image is not None
        assert frame.frame_index == 0
    finally:
        camera.close()

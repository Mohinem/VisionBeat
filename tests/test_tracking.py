from types import SimpleNamespace

from visionbeat.config import TrackerConfig
from visionbeat.models import FrameTimestamp
from visionbeat.tracking import PoseTracker


class FakeCV2:
    COLOR_BGR2RGB = 99

    def __init__(self) -> None:
        self.cvt_calls: list[tuple[object, int]] = []

    def cvtColor(self, frame: object, code: int) -> object:  # noqa: N802
        self.cvt_calls.append((frame, code))
        return ("rgb", frame)


class FakePose:
    def __init__(self, result: object) -> None:
        self.result = result
        self.closed = False
        self.inputs: list[object] = []

    def process(self, frame: object) -> object:
        self.inputs.append(frame)
        return self.result

    def close(self) -> None:
        self.closed = True


def make_tracker(result: object, *, min_tracking_confidence: float = 0.5) -> PoseTracker:
    tracker = object.__new__(PoseTracker)
    tracker.config = TrackerConfig(min_tracking_confidence=min_tracking_confidence)
    tracker._cv2 = FakeCV2()
    tracker._pose = FakePose(result)
    return tracker


def test_pose_tracker_returns_structured_landmarks() -> None:
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.1) for _ in range(33)]
    landmarks[11] = SimpleNamespace(x=0.2, y=0.3, z=-0.1, visibility=0.95)
    landmarks[15] = SimpleNamespace(x=0.4, y=0.5, z=-0.3, visibility=0.91)
    tracker = make_tracker(SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks)))

    output = tracker.process("frame", FrameTimestamp(seconds=1.0))

    assert output.person_detected is True
    assert output.status == "tracking"
    assert set(output.landmarks) == {"left_shoulder", "left_wrist"}
    assert output.get("left_wrist") is not None


def test_pose_tracker_handles_missing_person_gracefully() -> None:
    tracker = make_tracker(SimpleNamespace(pose_landmarks=None))

    output = tracker.process("frame", FrameTimestamp(seconds=2.0))

    assert output.person_detected is False
    assert output.status == "no_person_detected"
    assert output.landmarks == {}


def test_pose_tracker_filters_landmarks_below_threshold() -> None:
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.1) for _ in range(33)]
    landmarks[15] = SimpleNamespace(x=0.4, y=0.5, z=-0.3, visibility=0.49)
    tracker = make_tracker(
        SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks)),
        min_tracking_confidence=0.5,
    )

    output = tracker.process("frame", 2.5)

    assert output.person_detected is True
    assert output.status == "landmarks_below_confidence_threshold"
    assert output.landmarks == {}


def test_pose_tracker_close_releases_resources() -> None:
    tracker = make_tracker(SimpleNamespace(pose_landmarks=None))

    tracker.close()

    assert tracker._pose.closed is True

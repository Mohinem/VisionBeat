from visionbeat.config import GestureConfig
from visionbeat.gestures import GestureDetector
from visionbeat.models import GestureType, LandmarkPoint, PoseFrame


def make_frame(timestamp: float, *, right_wrist: tuple[float, float, float]) -> PoseFrame:
    x, y, z = right_wrist
    return PoseFrame(
        timestamp=timestamp,
        landmarks={"right_wrist": LandmarkPoint(x=x, y=y, z=z, visibility=1.0)},
    )


def test_forward_punch_triggers_kick() -> None:
    detector = GestureDetector(GestureConfig(history_size=4, cooldown_seconds=0.05))

    frames = [
        make_frame(0.00, right_wrist=(0.50, 0.40, -0.10)),
        make_frame(0.05, right_wrist=(0.51, 0.42, -0.18)),
        make_frame(0.10, right_wrist=(0.52, 0.43, -0.34)),
    ]

    events = []
    for frame in frames:
        events.extend(detector.update(frame))

    assert len(events) == 1
    assert events[0].gesture is GestureType.KICK


def test_downward_strike_triggers_snare() -> None:
    detector = GestureDetector(GestureConfig(history_size=4, cooldown_seconds=0.05))

    frames = [
        make_frame(0.00, right_wrist=(0.55, 0.25, -0.05)),
        make_frame(0.05, right_wrist=(0.56, 0.34, -0.06)),
        make_frame(0.10, right_wrist=(0.57, 0.50, -0.08)),
    ]

    events = []
    for frame in frames:
        events.extend(detector.update(frame))

    assert len(events) == 1
    assert events[0].gesture is GestureType.SNARE


def test_cooldown_prevents_duplicate_triggers() -> None:
    detector = GestureDetector(GestureConfig(history_size=4, cooldown_seconds=0.25))

    frames = [
        make_frame(0.00, right_wrist=(0.50, 0.40, -0.10)),
        make_frame(0.05, right_wrist=(0.50, 0.42, -0.22)),
        make_frame(0.10, right_wrist=(0.50, 0.42, -0.34)),
        make_frame(0.12, right_wrist=(0.50, 0.43, -0.40)),
    ]

    events = []
    for frame in frames:
        events.extend(detector.update(frame))

    assert len(events) == 1


def test_inactive_hand_is_ignored() -> None:
    detector = GestureDetector(GestureConfig(active_hand="left"))
    frame = PoseFrame(
        timestamp=0.1,
        landmarks={"right_wrist": LandmarkPoint(x=0.5, y=0.5, z=-0.5, visibility=1.0)},
    )

    assert detector.update(frame) == []

from collections.abc import Iterable
from typing import Any

from visionbeat.config import GestureConfig
from visionbeat.gestures import GestureDetector
from visionbeat.models import FrameTimestamp, GestureType, LandmarkPoint, TrackerOutput


def make_frame(
    timestamp: float,
    *,
    right_wrist: tuple[float, float, float] | None = None,
    left_wrist: tuple[float, float, float] | None = None,
) -> TrackerOutput:
    landmarks: dict[str, LandmarkPoint] = {}
    if right_wrist is not None:
        x, y, z = right_wrist
        landmarks["right_wrist"] = LandmarkPoint(x=x, y=y, z=z, visibility=1.0)
    if left_wrist is not None:
        x, y, z = left_wrist
        landmarks["left_wrist"] = LandmarkPoint(x=x, y=y, z=z, visibility=1.0)
    return TrackerOutput(timestamp=FrameTimestamp(seconds=timestamp), landmarks=landmarks)


def collect_events(
    detector: GestureDetector,
    positions: Iterable[tuple[float, tuple[float, float, float]]],
) -> list[Any]:
    events: list[Any] = []
    for timestamp, wrist in positions:
        events.extend(detector.update(make_frame(timestamp, right_wrist=wrist)))
    return events


def test_no_trigger_when_stationary() -> None:
    detector = GestureDetector(GestureConfig(history_size=6, analysis_window_seconds=0.2))

    events = collect_events(
        detector,
        [
            (0.00, (0.50, 0.40, -0.10)),
            (0.05, (0.50, 0.40, -0.10)),
            (0.10, (0.50, 0.40, -0.10)),
            (0.15, (0.50, 0.40, -0.10)),
        ],
    )

    assert events == []
    assert detector.candidates == ()


def test_punch_trigger_under_valid_motion() -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            analysis_window_seconds=0.2,
            confirmation_window_seconds=0.15,
            cooldown_seconds=0.05,
        )
    )

    first_update = detector.update(make_frame(0.00, right_wrist=(0.50, 0.40, -0.08)))
    second_update = detector.update(make_frame(0.05, right_wrist=(0.51, 0.42, -0.22)))

    assert first_update == []
    assert second_update == []
    assert len(detector.candidates) == 1
    assert detector.candidates[0].gesture is GestureType.KICK
    assert detector.candidates[0].label == "Forward punch candidate"

    third_update = detector.update(make_frame(0.10, right_wrist=(0.52, 0.43, -0.33)))

    assert len(third_update) == 1
    assert third_update[0].gesture is GestureType.KICK
    assert third_update[0].label == "Forward punch → kick"


def test_downward_strike_trigger_under_valid_motion() -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            analysis_window_seconds=0.2,
            confirmation_window_seconds=0.15,
            cooldown_seconds=0.05,
        )
    )

    detector.update(make_frame(0.00, right_wrist=(0.55, 0.20, -0.05)))
    detector.update(make_frame(0.05, right_wrist=(0.56, 0.36, -0.06)))

    assert len(detector.candidates) == 1
    assert detector.candidates[0].gesture is GestureType.SNARE
    assert detector.candidates[0].label == "Downward strike candidate"

    events = detector.update(make_frame(0.10, right_wrist=(0.57, 0.45, -0.07)))

    assert len(events) == 1
    assert events[0].gesture is GestureType.SNARE
    assert events[0].label == "Downward strike → snare"


def test_cooldown_suppresses_duplicates() -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldown_seconds=0.30,
            analysis_window_seconds=0.25,
            confirmation_window_seconds=0.15,
        )
    )

    events = collect_events(
        detector,
        [
            (0.00, (0.50, 0.40, -0.08)),
            (0.05, (0.50, 0.41, -0.22)),
            (0.10, (0.50, 0.42, -0.34)),
            (0.12, (0.50, 0.42, -0.42)),
            (0.20, (0.50, 0.40, -0.10)),
            (0.25, (0.50, 0.41, -0.22)),
            (0.28, (0.50, 0.42, -0.34)),
        ],
    )

    assert len(events) == 1
    assert events[0].gesture is GestureType.KICK


def test_noisy_motion_does_not_trigger_falsely() -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=8,
            analysis_window_seconds=0.2,
            confirmation_window_seconds=0.12,
            axis_dominance_ratio=1.7,
        )
    )

    events = collect_events(
        detector,
        [
            (0.00, (0.50, 0.30, -0.10)),
            (0.03, (0.58, 0.36, -0.18)),
            (0.06, (0.46, 0.25, -0.07)),
            (0.09, (0.59, 0.40, -0.21)),
            (0.12, (0.48, 0.31, -0.12)),
            (0.15, (0.60, 0.43, -0.19)),
        ],
    )

    assert events == []
    assert detector.candidates == ()


def test_edge_cases_near_thresholds() -> None:
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            analysis_window_seconds=0.2,
            confirmation_window_seconds=0.15,
            punch_forward_delta_z=0.18,
            strike_down_delta_y=0.22,
            min_velocity=0.5,
        )
    )

    below_threshold = collect_events(
        detector,
        [
            (0.00, (0.50, 0.40, -0.10)),
            (0.10, (0.50, 0.40, -0.279)),
            (0.12, (0.50, 0.40, -0.279)),
        ],
    )
    at_threshold = collect_events(
        detector,
        [
            (0.50, (0.55, 0.20, -0.05)),
            (0.56, (0.55, 0.36, -0.05)),
            (0.60, (0.55, 0.42, -0.05)),
        ],
    )

    assert below_threshold == []
    assert len(at_threshold) == 1
    assert at_threshold[0].gesture is GestureType.SNARE


def test_inactive_hand_is_ignored() -> None:
    detector = GestureDetector(GestureConfig(active_hand="left"))
    frame = make_frame(0.1, right_wrist=(0.5, 0.5, -0.5))

    assert detector.update(frame) == []

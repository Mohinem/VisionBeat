from __future__ import annotations

import math

import pytest

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


@pytest.mark.parametrize("seconds", [0, 1, 1.25])
def test_frame_timestamp_round_trips_and_coerces_numeric_values(seconds: int | float) -> None:
    timestamp = FrameTimestamp(seconds=seconds)

    restored = FrameTimestamp.from_dict(timestamp.to_dict())

    assert restored == timestamp
    assert isinstance(restored.seconds, float)


@pytest.mark.parametrize("seconds", [-0.01, math.inf, -math.inf, math.nan])
def test_frame_timestamp_rejects_invalid_values(seconds: float) -> None:
    with pytest.raises(ValueError, match="seconds"):
        FrameTimestamp(seconds=seconds)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"x": 0.1, "y": 0.2, "z": -0.3}, LandmarkPoint(0.1, 0.2, -0.3, 1.0)),
        (
            {"x": 0.1, "y": 0.2, "z": -0.3, "visibility": 0.75},
            LandmarkPoint(0.1, 0.2, -0.3, 0.75),
        ),
    ],
)
def test_landmark_point_round_trip_and_defaults(
    payload: dict[str, float],
    expected: LandmarkPoint,
) -> None:
    assert LandmarkPoint.from_dict(payload) == expected
    assert LandmarkPoint.from_dict(expected.to_dict()) == expected


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("x", math.nan),
        ("y", math.inf),
        ("z", -math.inf),
    ],
)
def test_landmark_point_rejects_non_finite_coordinates(field_name: str, value: float) -> None:
    payload = {"x": 0.0, "y": 0.0, "z": 0.0, field_name: value}

    with pytest.raises(ValueError, match=field_name):
        LandmarkPoint(**payload)


@pytest.mark.parametrize("visibility", [-0.1, 1.1])
def test_landmark_point_rejects_invalid_visibility(visibility: float) -> None:
    with pytest.raises(ValueError, match="visibility"):
        LandmarkPoint(x=0.0, y=0.0, z=0.0, visibility=visibility)


@pytest.mark.parametrize(
    ("hand", "label", "expected_hand", "expected_label"),
    [
        (" RIGHT ", " forward motion ", "right", "forward motion"),
        ("left", "", "left", ""),
    ],
)
def test_detection_candidate_normalizes_payload(
    hand: str,
    label: str,
    expected_hand: str,
    expected_label: str,
) -> None:
    candidate = DetectionCandidate(
        gesture=GestureType.KICK,
        confidence=0.9,
        hand=hand,
        label=label,
    )

    assert DetectionCandidate.from_dict(candidate.to_dict()) == candidate
    assert candidate.hand == expected_hand
    assert candidate.label == expected_label


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_detection_candidate_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DetectionCandidate(gesture=GestureType.KICK, confidence=confidence, hand="right")


@pytest.mark.parametrize("hand", ["", "center", "dominant"])
def test_detection_candidate_rejects_invalid_hand(hand: str) -> None:
    with pytest.raises(ValueError, match="hand"):
        DetectionCandidate(gesture=GestureType.KICK, confidence=0.5, hand=hand)


@pytest.mark.parametrize("gesture", [GestureType.KICK, GestureType.SNARE])
def test_gesture_event_round_trip_for_all_supported_gestures(gesture: GestureType) -> None:
    event = GestureEvent(
        gesture=gesture,
        confidence=0.8,
        hand="left",
        timestamp=FrameTimestamp(seconds=3.2),
        label=f"{gesture.value} hit",
    )

    assert GestureEvent.from_dict(event.to_dict()) == event


@pytest.mark.parametrize("intensity", [0.0, 0.5, 1.0])
def test_audio_trigger_round_trip_and_boundary_intensities(intensity: float) -> None:
    trigger = AudioTrigger(
        gesture=GestureType.KICK,
        timestamp=FrameTimestamp(seconds=4.0),
        intensity=intensity,
    )

    assert AudioTrigger.from_dict(trigger.to_dict()) == trigger


@pytest.mark.parametrize("intensity", [-0.01, 1.01])
def test_audio_trigger_rejects_invalid_intensity(intensity: float) -> None:
    with pytest.raises(ValueError, match="intensity"):
        AudioTrigger(
            gesture=GestureType.KICK,
            timestamp=FrameTimestamp(seconds=0.0),
            intensity=intensity,
        )


def test_tracker_output_normalizes_landmarks_candidates_and_status() -> None:
    output = TrackerOutput(
        timestamp=FrameTimestamp(seconds=1.25),
        landmarks={"right_wrist": {"x": 0.4, "y": 0.5, "z": -0.2, "visibility": 1.0}},
        candidates=(
            {
                "gesture": "kick",
                "confidence": 0.77,
                "hand": "right",
                "label": "possible kick",
            },
        ),
        person_detected=False,
        status="   ",
    )

    assert output.person_detected is True
    assert output.status == "unknown"
    assert isinstance(output.get("right_wrist"), LandmarkPoint)
    assert isinstance(output.candidates[0], DetectionCandidate)
    assert TrackerOutput.from_dict(output.to_dict()) == output


@pytest.mark.parametrize(
    ("fps", "cooldown_remaining_seconds"),
    [(None, 0.0), (60.0, 0.25)],
)
def test_render_state_accepts_valid_runtime_values(
    fps: float | None,
    cooldown_remaining_seconds: float,
) -> None:
    state = RenderState(
        pose=TrackerOutput(timestamp=FrameTimestamp(seconds=1.0)),
        frame_index=3,
        fps=fps,
        cooldown_remaining_seconds=cooldown_remaining_seconds,
    )

    assert state.frame_index == 3
    assert state.fps == fps
    assert state.cooldown_remaining_seconds == cooldown_remaining_seconds


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_index", -1, "frame_index"),
        ("fps", 0.0, "fps"),
        ("cooldown_remaining_seconds", -0.1, "cooldown_remaining_seconds"),
    ],
)
def test_render_state_rejects_invalid_runtime_values(
    field: str,
    value: float,
    message: str,
) -> None:
    kwargs = {
        "pose": TrackerOutput(timestamp=FrameTimestamp(seconds=1.0)),
        "frame_index": 0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        RenderState(**kwargs)

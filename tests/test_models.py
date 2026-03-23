import pytest

from visionbeat.config import AppConfig, AudioConfig, CameraConfig, GestureConfig, TrackerConfig
from visionbeat.models import (
    AudioTrigger,
    DetectionCandidate,
    FrameTimestamp,
    GestureEvent,
    GestureType,
    LandmarkPoint,
    TrackerOutput,
)


def test_landmark_point_serializes_round_trip() -> None:
    point = LandmarkPoint(x=0.1, y=0.2, z=-0.3, visibility=0.75)

    assert LandmarkPoint.from_dict(point.to_dict()) == point


@pytest.mark.parametrize("visibility", [-0.1, 1.1])
def test_landmark_point_rejects_invalid_visibility(visibility: float) -> None:
    with pytest.raises(ValueError, match="visibility"):
        LandmarkPoint(x=0.0, y=0.0, z=0.0, visibility=visibility)


def test_frame_timestamp_serializes_round_trip() -> None:
    timestamp = FrameTimestamp(seconds=12.5)

    assert FrameTimestamp.from_dict(timestamp.to_dict()) == timestamp


def test_frame_timestamp_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        FrameTimestamp(seconds=-0.01)


def test_detection_candidate_round_trip() -> None:
    candidate = DetectionCandidate(
        gesture=GestureType.KICK,
        confidence=0.9,
        hand="RIGHT",
        label="forward motion",
    )

    assert DetectionCandidate.from_dict(candidate.to_dict()) == candidate
    assert candidate.hand == "right"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_detection_candidate_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DetectionCandidate(gesture=GestureType.KICK, confidence=confidence, hand="right")


def test_gesture_event_serializes_round_trip() -> None:
    event = GestureEvent(
        gesture=GestureType.SNARE,
        confidence=0.8,
        hand="left",
        timestamp=FrameTimestamp(seconds=3.2),
        label="snare hit",
    )

    assert GestureEvent.from_dict(event.to_dict()) == event


def test_audio_trigger_serializes_round_trip() -> None:
    trigger = AudioTrigger(
        gesture=GestureType.KICK,
        timestamp=FrameTimestamp(seconds=4.0),
        intensity=0.5,
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


def test_tracker_output_serializes_round_trip() -> None:
    output = TrackerOutput(
        timestamp=FrameTimestamp(seconds=1.25),
        landmarks={"right_wrist": LandmarkPoint(x=0.4, y=0.5, z=-0.2, visibility=1.0)},
        candidates=(
            DetectionCandidate(
                gesture=GestureType.KICK,
                confidence=0.77,
                hand="right",
                label="possible kick",
            ),
        ),
    )

    restored = TrackerOutput.from_dict(output.to_dict())

    assert restored == output
    assert restored.get("right_wrist") == output.landmarks["right_wrist"]


def test_camera_config_serializes_round_trip() -> None:
    config = CameraConfig(width=1920, height=1080, fps=60, window_name=" VisionBeat ")

    assert CameraConfig.from_dict(config.to_dict()) == CameraConfig(
        width=1920,
        height=1080,
        fps=60,
        window_name="VisionBeat",
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [({"width": 0}, "width"), ({"device_index": -1}, "device_index")],
)
def test_camera_config_rejects_invalid_values(payload: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CameraConfig(**payload)


def test_gesture_config_serializes_round_trip() -> None:
    config = GestureConfig(active_hand="LEFT", history_size=8, cooldown_seconds=0.25)

    assert GestureConfig.from_dict(config.to_dict()) == GestureConfig(
        active_hand="left",
        history_size=8,
        cooldown_seconds=0.25,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [({"history_size": 0}, "history_size"), ({"active_hand": "both"}, "active_hand")],
)
def test_gesture_config_rejects_invalid_values(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GestureConfig(**payload)


def test_audio_config_serializes_round_trip() -> None:
    config = AudioConfig(
        sample_rate=48_000,
        buffer_size=512,
        kick_sample="kick.wav",
        snare_sample="snare.wav",
        volume=0.4,
    )

    assert AudioConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("payload", "message"),
    [({"volume": 1.1}, "volume"), ({"kick_sample": " "}, "kick_sample")],
)
def test_audio_config_rejects_invalid_values(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AudioConfig(**payload)


def test_app_config_serializes_round_trip() -> None:
    config = AppConfig(
        camera=CameraConfig(width=640, height=480, fps=24),
        tracker=TrackerConfig(model_complexity=2),
        gestures=GestureConfig(active_hand="right"),
        audio=AudioConfig(kick_sample="kick.wav", snare_sample="snare.wav"),
        log_level="debug",
    )

    restored = AppConfig.from_dict(config.to_dict())

    assert restored == AppConfig(
        camera=CameraConfig(width=640, height=480, fps=24),
        tracker=TrackerConfig(model_complexity=2),
        gestures=GestureConfig(active_hand="right"),
        audio=AudioConfig(kick_sample="kick.wav", snare_sample="snare.wav"),
        log_level="DEBUG",
    )

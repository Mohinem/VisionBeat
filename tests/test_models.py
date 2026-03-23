import pytest

from visionbeat.config import (
    AppConfig,
    AudioConfig,
    CameraConfig,
    GestureConfig,
    GestureCooldownsConfig,
    GestureThresholdsConfig,
    LoggingConfig,
    TrackerConfig,
)
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
    config = CameraConfig(width=1920, height=1080, fps=60, window_name="VisionBeat")

    assert CameraConfig.from_dict(config.to_dict()) == config


def test_gesture_config_serializes_round_trip() -> None:
    config = GestureConfig(
        thresholds=GestureThresholdsConfig(punch_forward_delta_z=0.24),
        cooldowns=GestureCooldownsConfig(trigger_seconds=0.25),
        active_hand="left",
        history_size=8,
    )

    assert GestureConfig.from_dict(config.to_dict()) == config


def test_audio_config_serializes_round_trip() -> None:
    config = AudioConfig(
        backend="pygame",
        sample_rate=48_000,
        buffer_size=512,
        output_channels=2,
        simultaneous_voices=24,
        output_device_name="Interface 1",
        sample_mapping={"kick": "kick.wav", "snare": "snare.wav"},
        volume=0.4,
    )

    assert AudioConfig.from_dict(config.to_dict()) == config


def test_app_config_serializes_round_trip() -> None:
    config = AppConfig(
        camera=CameraConfig(width=640, height=480, fps=24),
        tracker=TrackerConfig(model_complexity=2),
        gestures=GestureConfig(active_hand="right"),
        audio=AudioConfig(sample_mapping={"kick": "kick.wav", "snare": "snare.wav"}),
        logging=LoggingConfig(level="DEBUG"),
    )

    restored = AppConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.log_level == "DEBUG"

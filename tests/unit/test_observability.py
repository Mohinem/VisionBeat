from __future__ import annotations

import json
import logging
from pathlib import Path

from visionbeat.config import GestureConfig, GestureCooldownsConfig, LoggingConfig
from visionbeat.gestures import GestureDetector
from visionbeat.models import FrameTimestamp, GestureType, LandmarkPoint, TrackerOutput
from visionbeat.observability import (
    GestureEventLogger,
    GestureObservationEvent,
    ObservabilityRecorder,
    VelocityStats,
)


class RecordingObserver:
    def __init__(self) -> None:
        self.candidates: list[GestureObservationEvent] = []
        self.triggers: list[GestureObservationEvent] = []
        self.cooldowns: list[GestureObservationEvent] = []

    def log_gesture_candidate(self, event: GestureObservationEvent) -> None:
        self.candidates.append(event)

    def log_confirmed_trigger(self, event: GestureObservationEvent) -> None:
        self.triggers.append(event)

    def log_cooldown_suppression(self, event: GestureObservationEvent) -> None:
        self.cooldowns.append(event)


def make_frame(timestamp: float, right_wrist: tuple[float, float, float]) -> TrackerOutput:
    x, y, z = right_wrist
    return TrackerOutput(
        timestamp=FrameTimestamp(seconds=timestamp),
        landmarks={
            "right_wrist": LandmarkPoint(x=x, y=y, z=z, visibility=1.0),
            "right_shoulder": LandmarkPoint(x=0.40, y=0.20, z=-0.02, visibility=1.0),
        },
    )


def test_gesture_observation_round_trips_through_dict_and_jsonl(tmp_path: Path) -> None:
    event = GestureObservationEvent(
        timestamp=12.5,
        event_kind="trigger",
        gesture_type=GestureType.KICK,
        accepted=True,
        reason="Outward jab → kick",
        velocity_stats=VelocityStats(
            elapsed=0.08,
            delta_x=0.01,
            delta_y=0.02,
            delta_z=-0.27,
            net_velocity=3.75,
            peak_x_velocity=0.4,
            peak_y_velocity=0.6,
            peak_z_velocity=-3.9,
        ),
        confidence=0.94,
        hand="right",
    )

    payload = event.to_dict()
    restored = GestureObservationEvent.from_dict(payload)
    assert restored == event

    log_path = tmp_path / "events.jsonl"
    writer = GestureEventLogger(log_path, fmt="jsonl")
    writer.write(event)
    writer.close()

    persisted = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert GestureObservationEvent.from_dict(persisted) == event


def test_gesture_observation_writes_csv_rows(tmp_path: Path) -> None:
    event = GestureObservationEvent(
        timestamp=1.25,
        event_kind="candidate",
        gesture_type=GestureType.SNARE,
        accepted=False,
        reason="Downward strike candidate",
        velocity_stats=VelocityStats(
            elapsed=0.1,
            delta_x=0.01,
            delta_y=0.24,
            delta_z=0.02,
            net_velocity=2.7,
            peak_x_velocity=0.15,
            peak_y_velocity=2.4,
            peak_z_velocity=0.2,
        ),
        confidence=0.73,
        hand="right",
    )

    log_path = tmp_path / "events.csv"
    writer = GestureEventLogger(log_path, fmt="csv")
    writer.write(event)
    writer.close()

    contents = log_path.read_text(encoding="utf-8")
    assert "event_kind,gesture_type,accepted,reason,confidence,hand" in contents
    assert "candidate,snare,false,Downward strike candidate,0.730000,right" in contents


def test_detector_emits_candidate_trigger_and_cooldown_observations() -> None:
    observer = RecordingObserver()
    detector = GestureDetector(
        GestureConfig(
            history_size=6,
            cooldowns=GestureCooldownsConfig(
                trigger_seconds=0.3,
                analysis_window_seconds=0.25,
                confirmation_window_seconds=0.15,
            ),
        ),
        observer=observer,
    )

    first_events = detector.update(make_frame(0.00, (0.55, 0.20, -0.05)))
    second_events = detector.update(make_frame(0.05, (0.56, 0.36, -0.06)))
    third_events = detector.update(make_frame(0.10, (0.57, 0.45, -0.07)))
    detector.update(make_frame(0.20, (0.56, 0.24, -0.05)))
    detector.update(make_frame(0.24, (0.57, 0.39, -0.06)))
    events = first_events + second_events + third_events

    assert len(events) == 1
    assert [event.event_kind for event in observer.candidates] == ["candidate"]
    assert [event.event_kind for event in observer.triggers] == ["trigger"]
    assert observer.triggers[0].accepted is True
    assert observer.triggers[0].gesture_type is GestureType.KICK
    assert observer.candidates[0].velocity_stats is not None


def test_observability_recorder_builds_event_logger_from_config(tmp_path: Path) -> None:
    event_path = tmp_path / "analysis.jsonl"
    recorder = ObservabilityRecorder(
        event_logger=GestureEventLogger(event_path, fmt=LoggingConfig().event_log_format)
    )
    recorder.log_tracking_failure(timestamp=3.0, status="no_person_detected")
    recorder.close()

    persisted = json.loads(event_path.read_text(encoding="utf-8").strip())
    assert persisted["event_kind"] == "tracking_failure"
    assert persisted["accepted"] is False
    assert persisted["reason"] == "no_person_detected"


def test_observability_recorder_emits_structured_rhythm_prediction(caplog) -> None:
    recorder = ObservabilityRecorder()

    with caplog.at_level(logging.INFO, logger="visionbeat.observability"):
        recorder.log_rhythm_prediction(
            timestamp=2.0,
            frame_index=7,
            prediction_id="kick:1.500000->2.000000",
            outcome="pending",
            gesture=GestureType.KICK,
            predicted_time_seconds=2.0,
            actual_time_seconds=None,
            actual_gesture=None,
            error_ms=None,
            last_event_timestamp=1.5,
            interval_seconds=0.5,
            expires_after_seconds=2.375,
            seconds_until_prediction=0.0,
            seconds_until_expiry=0.375,
            match_tolerance_seconds=0.12,
            confidence=0.86,
            repetition_count=3,
            jitter=0.02,
            active=True,
            shadow_only=True,
            source="heuristic",
            trigger_mode="shadow",
            status_description="predicted next kick at 2.000s",
        )

    structured = caplog.records[0].structured
    assert structured["event"] == "rhythm_prediction"
    assert structured["gesture"] == "kick"
    assert structured["outcome"] == "pending"
    assert structured["predicted_time_seconds"] == 2.0
    assert structured["actual_time_seconds"] is None
    assert structured["active"] is True
    assert structured["shadow_only"] is True
    assert structured["interval_seconds"] == 0.5
    assert structured["expires_after_seconds"] == 2.375
    assert structured["seconds_until_prediction"] == 0.0
    assert structured["seconds_until_expiry"] == 0.375
    assert structured["match_tolerance_seconds"] == 0.12
    assert structured["status_description"] == "predicted next kick at 2.000s"


def test_observability_recorder_emits_structured_rhythm_live_trigger(caplog) -> None:
    recorder = ObservabilityRecorder()

    with caplog.at_level(logging.INFO, logger="visionbeat.observability"):
        recorder.log_rhythm_live_trigger(
            timestamp=2.5,
            frame_index=9,
            prediction_id="kick:2.000000->2.500000",
            predicted_time_seconds=2.5,
            scheduling_error_ms=0.0,
            gesture=GestureType.KICK,
            confidence=0.86,
            interval_seconds=0.5,
            repetition_count=2,
            jitter=0.02,
            source="rhythm_direct",
        )

    structured = caplog.records[0].structured
    assert structured["event"] == "rhythm_live_trigger"
    assert structured["gesture"] == "kick"
    assert structured["prediction_id"] == "kick:2.000000->2.500000"
    assert structured["predicted_time_seconds"] == 2.5
    assert structured["scheduling_error_ms"] == 0.0
    assert structured["source"] == "rhythm_direct"

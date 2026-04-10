from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionbeat.camera import CameraFrame
from visionbeat.models import (
    FrameTimestamp,
    GestureEvent,
    GestureType,
    LandmarkPoint,
    TrackerOutput,
)
from visionbeat.session_recording import SessionRecorder, build_session_recorder


def make_pose(timestamp: float) -> TrackerOutput:
    return TrackerOutput(
        timestamp=FrameTimestamp(seconds=timestamp),
        landmarks={
            "right_wrist": LandmarkPoint(x=0.6, y=0.4, z=-0.3, visibility=0.9),
        },
        person_detected=True,
        status="tracking",
    )


def test_build_session_recorder_returns_none_when_disabled() -> None:
    recorder = build_session_recorder(
        type("LoggingConfigStub", (), {"session_recording_path": None})(),
        config_payload={"camera": {"fps": 30}},
    )

    assert recorder is None


def test_session_recorder_writes_tracker_outputs_and_triggers(tmp_path: Path) -> None:
    recorder = SessionRecorder(
        tmp_path,
        mode="tracker_outputs",
        config_payload={"camera": {"fps": 30}},
    )
    frame = CameraFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        captured_at=1.25,
        frame_index=3,
    )
    pose = make_pose(1.25)
    event = GestureEvent(
        gesture=GestureType.SNARE,
        confidence=0.82,
        hand="right",
        timestamp=FrameTimestamp(seconds=1.25),
        label="Wrist collision → snare",
    )

    recorder.record_camera_frame(frame)
    recorder.record_tracker_output(frame, pose)
    recorder.record_trigger(event)
    recorder.close()

    session_dir = recorder.session_dir
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    tracker_rows = [
        json.loads(line)
        for line in (session_dir / "tracker_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trigger_rows = [
        json.loads(line)
        for line in (session_dir / "triggers.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["recording_mode"] == "tracker_outputs"
    assert manifest["config"]["camera"]["fps"] == 30
    assert manifest["counts"]["camera_frames"] == 0
    assert manifest["counts"]["tracker_outputs"] == 1
    assert manifest["counts"]["triggers"] == 1
    assert tracker_rows == [
        {
            "captured_at": 1.25,
            "frame_index": 3,
            "tracker_output": pose.to_dict(),
        }
    ]
    assert trigger_rows == [event.to_dict()]


def test_session_recorder_writes_lossless_raw_frames_for_replay(tmp_path: Path) -> None:
    recorder = SessionRecorder(
        tmp_path,
        mode="both",
        config_payload={"tracker": {"backend": "mediapipe"}},
    )
    raw_frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    camera_frame = CameraFrame(
        image=raw_frame,
        captured_at=2.5,
        frame_index=12,
        mirrored_for_display=True,
    )
    pose = make_pose(2.5)

    recorder.record_camera_frame(camera_frame)
    recorder.record_tracker_output(camera_frame, pose)
    recorder.close()

    session_dir = recorder.session_dir
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    frame_row = json.loads(
        (session_dir / "camera_frames.jsonl").read_text(encoding="utf-8").strip()
    )
    persisted_frame = np.load(session_dir / frame_row["frame_path"])

    assert manifest["recording_mode"] == "both"
    assert manifest["artifacts"]["frames_directory"] == "frames"
    assert frame_row["frame_index"] == 12
    assert frame_row["captured_at"] == 2.5
    assert frame_row["mirrored_for_display"] is True
    assert frame_row["shape"] == [2, 3, 3]
    assert frame_row["dtype"] == "uint8"
    assert np.array_equal(persisted_frame, raw_frame)

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from visionbeat.config import AppConfig, TrackerConfig
from visionbeat.extract_dataset_features import (
    DatasetExtractionResult,
    extract_dataset_features,
    main,
)
from visionbeat.features import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_SCHEMA,
    CanonicalFeatureExtractor,
    get_canonical_feature_schema,
)
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, fps: float) -> None:
        self.frames = list(frames)
        self.fps = fps
        self.released = False
        self._last_frame_index = -1

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        self._last_frame_index += 1
        return True, self.frames.pop(0)

    def get(self, prop: int) -> float:  # noqa: N802
        if prop == FakeCV2.CAP_PROP_FPS:
            return self.fps
        if prop == FakeCV2.CAP_PROP_POS_MSEC:
            if self._last_frame_index < 0:
                return 0.0
            return (self._last_frame_index / self.fps) * 1000.0
        return 0.0

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 6

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _: str) -> FakeCapture:  # noqa: N802
        return self.capture


class FakePoseProvider:
    def __init__(self, landmarks_by_frame: list[dict[str, LandmarkPoint]]) -> None:
        self.landmarks_by_frame = list(landmarks_by_frame)
        self.timestamps: list[float] = []
        self.closed = False

    def process(self, frame: object, timestamp: FrameTimestamp | float) -> TrackerOutput:
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        self.timestamps.append(seconds)
        landmarks = self.landmarks_by_frame.pop(0)
        return TrackerOutput(
            timestamp=FrameTimestamp(seconds=seconds),
            landmarks=landmarks,
            person_detected=bool(landmarks),
            status="tracking" if landmarks else "no_person_detected",
        )

    def close(self) -> None:
        self.closed = True


def test_extract_dataset_features_writes_csv_using_shared_canonical_features(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "recording_id,frame_index,gesture,split\n"
        "other-recording,0,hihat,eval\n"
        "session,0,kick,train\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features.csv"

    landmarks_by_frame = [
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.25, y=0.50, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.75, y=0.50, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.50, y=0.75, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.75, y=0.75, z=-0.2, visibility=0.95),
        },
    ]
    provider = FakePoseProvider(landmarks_by_frame)
    capture = FakeCapture(
        [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)],
        fps=20.0,
    )

    result = extract_dataset_features(
        video_path,
        output_path=output_path,
        labels_path=labels_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(capture),
        pose_provider_factory=lambda _: provider,
    )

    assert result.video_path == video_path
    assert result.output_path == output_path
    assert result.schema_path == tmp_path / "features.csv.schema.json"
    assert result.recording_id == "session"
    assert result.frames_processed == 2
    assert result.feature_schema == get_canonical_feature_schema()
    assert result.label_columns == ("gesture_label", "is_completion_frame", "split")
    assert provider.closed is True
    assert capture.released is True
    assert provider.timestamps == [0.0, 0.05]

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    schema_sidecar = json.loads(result.schema_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["recording_id"] == "session"
    assert rows[0]["frame_index"] == "0"
    assert rows[0]["timestamp_seconds"] == "0.0"
    assert rows[0]["gesture_label"] == "kick"
    assert rows[0]["is_completion_frame"] == "True"
    assert rows[0]["split"] == "train"
    assert rows[1]["gesture_label"] == ""
    assert rows[1]["is_completion_frame"] == "False"
    assert rows[1]["split"] == ""
    assert schema_sidecar["schema"] == CANONICAL_FEATURE_SCHEMA
    assert schema_sidecar["schema_version"] == CANONICAL_FEATURE_SCHEMA
    assert schema_sidecar["feature_count"] == len(CANONICAL_FEATURE_NAMES)
    assert schema_sidecar["feature_names"] == list(CANONICAL_FEATURE_NAMES)
    assert schema_sidecar["recording_id"] == "session"
    assert schema_sidecar["label_columns"] == ["gesture_label", "is_completion_frame", "split"]
    assert schema_sidecar["output_columns"][:5] == [
        "recording_id",
        "frame_index",
        "timestamp_seconds",
        "person_detected",
        "tracking_status",
    ]
    assert schema_sidecar["output_columns"][5:8] == [
        "gesture_label",
        "is_completion_frame",
        "split",
    ]
    assert schema_sidecar["output_columns"][-len(CANONICAL_FEATURE_NAMES) :] == list(
        CANONICAL_FEATURE_NAMES
    )

    expected_extractor = CanonicalFeatureExtractor()
    expected_frames = [
        expected_extractor.update(
            TrackerOutput(
                timestamp=FrameTimestamp(seconds=0.0),
                landmarks=landmarks_by_frame[0],
                person_detected=True,
                status="tracking",
            )
        ),
        expected_extractor.update(
            TrackerOutput(
                timestamp=FrameTimestamp(seconds=0.05),
                landmarks=landmarks_by_frame[1],
                person_detected=True,
                status="tracking",
            )
        ),
    ]
    for row, expected in zip(rows, expected_frames, strict=True):
        for feature_name in CANONICAL_FEATURE_NAMES:
            assert float(row[feature_name]) == pytest.approx(expected.as_feature_dict()[feature_name])


def test_extract_dataset_features_accepts_frame_no_and_gesture_type_aliases(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "frame_no,gesture_type\n"
        "0,kick\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features.csv"
    provider = FakePoseProvider(
        [
            {
                "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
                "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
                "left_wrist": LandmarkPoint(x=0.25, y=0.50, z=-0.2, visibility=0.95),
                "right_wrist": LandmarkPoint(x=0.75, y=0.50, z=-0.2, visibility=0.95),
            }
        ]
    )
    capture = FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)], fps=20.0)

    result = extract_dataset_features(
        video_path,
        output_path=output_path,
        labels_path=labels_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(capture),
        pose_provider_factory=lambda _: provider,
    )

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    schema_sidecar = json.loads(result.schema_path.read_text(encoding="utf-8"))

    assert rows[0]["gesture_label"] == "kick"
    assert rows[0]["is_completion_frame"] == "True"
    assert schema_sidecar["label_columns"] == ["gesture_label", "is_completion_frame"]


def test_main_invokes_extraction_cli(monkeypatch, capsys) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "visionbeat.extract_dataset_features.load_config",
        lambda _: AppConfig(),
    )
    monkeypatch.setattr(
        "visionbeat.extract_dataset_features.configure_logging",
        lambda *args, **kwargs: None,
    )

    def fake_extract_dataset_features(*args, **kwargs) -> DatasetExtractionResult:
        called["args"] = args
        called["kwargs"] = kwargs
        return DatasetExtractionResult(
            video_path=Path("clip.mp4"),
            output_path=Path("features.csv"),
            schema_path=Path("features.csv.schema.json"),
            recording_id="clip",
            frames_processed=12,
            feature_schema=get_canonical_feature_schema(),
            label_columns=(),
        )

    monkeypatch.setattr(
        "visionbeat.extract_dataset_features.extract_dataset_features",
        fake_extract_dataset_features,
    )

    main(["--video", "clip.mp4", "--out", "features.csv", "--pose-backend", "movenet"])

    assert called["args"] == ("clip.mp4",)
    assert called["kwargs"]["output_path"] == "features.csv"
    assert called["kwargs"]["labels_path"] is None
    assert called["kwargs"]["recording_id"] is None
    assert called["kwargs"]["tracker_config"].backend == "movenet"
    assert "Extracted 12 frames from clip.mp4 to features.csv" in capsys.readouterr().out

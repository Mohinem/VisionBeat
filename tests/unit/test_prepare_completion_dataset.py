from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from visionbeat.config import TrackerConfig
from visionbeat.features import CANONICAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput
from visionbeat.prepare_completion_dataset import (
    CompletionDatasetPreparationResult,
    RecordingDatasetInput,
    prepare_completion_dataset,
    verify_canonical_feature_schema,
)


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, fps: float) -> None:
        self.frames = list(frames)
        self.fps = fps
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
        return None


class FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 6

    def __init__(self, frames_by_path: dict[str, list[np.ndarray]], *, fps: float) -> None:
        self.frames_by_path = {path: list(frames) for path, frames in frames_by_path.items()}
        self.fps = fps

    def VideoCapture(self, path: str) -> FakeCapture:  # noqa: N802
        return FakeCapture(self.frames_by_path[path], fps=self.fps)


class FakePoseProvider:
    def __init__(self, landmarks_by_marker: dict[int, list[dict[str, LandmarkPoint]]]) -> None:
        self.landmarks_by_marker = {
            marker: list(frames) for marker, frames in landmarks_by_marker.items()
        }

    def process(self, frame: np.ndarray, timestamp: FrameTimestamp | float) -> TrackerOutput:
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        marker = int(frame[0, 0, 0])
        landmarks = self.landmarks_by_marker[marker].pop(0)
        return TrackerOutput(
            timestamp=FrameTimestamp(seconds=seconds),
            landmarks=landmarks,
            person_detected=bool(landmarks),
            status="tracking" if landmarks else "no_person_detected",
        )

    def close(self) -> None:
        return None


def test_prepare_completion_dataset_builds_split_npz_and_labeled_tables(
    tmp_path: Path,
) -> None:
    schema = verify_canonical_feature_schema()
    assert schema.feature_names == CANONICAL_FEATURE_NAMES
    assert schema.version == FEATURE_SCHEMA_VERSION

    rec1_video = tmp_path / "rec1.mp4"
    rec2_video = tmp_path / "rec2.mp4"
    rec1_video.write_bytes(b"")
    rec2_video.write_bytes(b"")

    rec1_labels = tmp_path / "rec1_labels.csv"
    rec2_labels = tmp_path / "rec2_labels.csv"
    rec1_labels.write_text(
        "recording_id,frame_index,gesture\n"
        "rec1,2,kick\n"
        "rec1,5,snare\n",
        encoding="utf-8",
    )
    rec2_labels.write_text(
        "recording_id,frame_index,gesture\n"
        "rec2,1,kick\n"
        "rec2,4,snare\n"
        "rec2,7,kick\n",
        encoding="utf-8",
    )

    frames_by_path = {
        rec1_video.as_posix(): build_frames(marker=1, frame_count=6),
        rec2_video.as_posix(): build_frames(marker=2, frame_count=8),
    }
    landmarks_by_marker = {
        1: build_landmarks_by_frame(
            left_wrist_positions=(
                (0.20, 0.52),
                (0.24, 0.56),
                (0.28, 0.60),
                (0.32, 0.64),
                (0.36, 0.68),
                (0.40, 0.72),
            )
        ),
        2: build_landmarks_by_frame(
            left_wrist_positions=(
                (0.18, 0.48),
                (0.20, 0.52),
                (0.22, 0.56),
                (0.24, 0.60),
                (0.26, 0.64),
                (0.28, 0.68),
                (0.30, 0.72),
                (0.32, 0.76),
            )
        ),
    }

    result = prepare_completion_dataset(
        (
            RecordingDatasetInput("rec1", rec1_video, rec1_labels),
            RecordingDatasetInput("rec2", rec2_video, rec2_labels),
        ),
        output_dir=tmp_path / "prepared",
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(frames_by_path, fps=20.0),
        pose_provider_factory=lambda _: FakePoseProvider(landmarks_by_marker),
        window_size=3,
        stride=1,
        validation_recording_id="rec2",
        validation_fraction=0.5,
    )

    assert isinstance(result, CompletionDatasetPreparationResult)
    assert result.output_path == tmp_path / "prepared" / "train_dataset.npz"
    assert result.output_path.exists()
    assert result.validation_status == "PASS"
    assert result.total_frames_processed == 14
    assert result.total_labeled_gestures == 5
    assert result.train_sample_count == 4
    assert result.validation_sample_count == 1
    assert result.train_shape == (4, 3, len(CANONICAL_FEATURE_NAMES))
    assert result.validation_shape == (1, 3, len(CANONICAL_FEATURE_NAMES))
    assert result.sanity_warnings == ()

    rec1_labeled = tmp_path / "prepared" / "rec1_labeled.csv"
    rec2_labeled = tmp_path / "prepared" / "rec2_labeled.csv"
    rec1_features = tmp_path / "prepared" / "rec1_features.csv"
    rec2_features = tmp_path / "prepared" / "rec2_features.csv"
    assert rec1_features.exists()
    assert rec2_features.exists()
    assert rec1_labeled.exists()
    assert rec2_labeled.exists()

    rec1_rows = list(csv.DictReader(rec1_labeled.open("r", encoding="utf-8", newline="")))
    assert rec1_rows[2]["gesture"] == "kick"
    assert rec1_rows[2]["is_completion"] == "True"
    assert "gesture_label" not in rec1_rows[0]
    assert "is_completion_frame" not in rec1_rows[0]

    rec1_schema = json.loads(
        (tmp_path / "prepared" / "rec1_labeled.csv.schema.json").read_text(encoding="utf-8")
    )
    assert rec1_schema["feature_names"] == list(CANONICAL_FEATURE_NAMES)
    assert rec1_schema["label_columns"] == ["gesture", "is_completion"]

    archive = np.load(result.output_path)
    assert archive["feature_names"].tolist() == list(CANONICAL_FEATURE_NAMES)
    assert archive["schema_version"].item() == FEATURE_SCHEMA_VERSION
    assert archive["X_train"].shape == (4, 3, len(CANONICAL_FEATURE_NAMES))
    assert archive["y_train"].tolist() == [1, 0, 0, 1]
    assert archive["train_recording_ids"].tolist() == ["rec1", "rec1", "rec1", "rec1"]
    assert archive["validation_recording_ids"].tolist() == ["rec2"]
    assert archive["validation_window_end_frame_indices"].tolist() == [7]
    assert archive["y_val"].tolist() == [1]
    assert archive["validation_status"].item() == "PASS"


def test_prepare_completion_dataset_supports_future_completion_targets(tmp_path: Path) -> None:
    rec1_video = tmp_path / "rec1.mp4"
    rec2_video = tmp_path / "rec2.mp4"
    rec1_video.write_bytes(b"")
    rec2_video.write_bytes(b"")

    rec1_labels = tmp_path / "rec1_labels.csv"
    rec2_labels = tmp_path / "rec2_labels.csv"
    rec1_labels.write_text(
        "recording_id,frame_index,gesture\n"
        "rec1,2,kick\n"
        "rec1,5,snare\n",
        encoding="utf-8",
    )
    rec2_labels.write_text(
        "recording_id,frame_index,gesture\n"
        "rec2,3,kick\n"
        "rec2,6,snare\n",
        encoding="utf-8",
    )

    frames_by_path = {
        rec1_video.as_posix(): build_frames(marker=1, frame_count=6),
        rec2_video.as_posix(): build_frames(marker=2, frame_count=7),
    }
    landmarks_by_marker = {
        1: build_landmarks_by_frame(
            left_wrist_positions=(
                (0.20, 0.52),
                (0.24, 0.56),
                (0.28, 0.60),
                (0.32, 0.64),
                (0.36, 0.68),
                (0.40, 0.72),
            )
        ),
        2: build_landmarks_by_frame(
            left_wrist_positions=(
                (0.18, 0.48),
                (0.20, 0.52),
                (0.22, 0.56),
                (0.24, 0.60),
                (0.26, 0.64),
                (0.28, 0.68),
                (0.30, 0.72),
            )
        ),
    }

    result = prepare_completion_dataset(
        (
            RecordingDatasetInput("rec1", rec1_video, rec1_labels),
            RecordingDatasetInput("rec2", rec2_video, rec2_labels),
        ),
        output_dir=tmp_path / "prepared_future",
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(frames_by_path, fps=20.0),
        pose_provider_factory=lambda _: FakePoseProvider(landmarks_by_marker),
        window_size=3,
        stride=1,
        validation_recording_id="rec2",
        validation_fraction=0.5,
        target="completion_within_next_k_frames",
        horizon_frames=2,
    )

    archive = np.load(result.output_path)
    assert result.target_name == "completion_within_next_k_frames"
    assert result.horizon_frames == 2
    assert archive["target_name"].item() == "completion_within_next_k_frames"
    assert int(archive["horizon_frames"].item()) == 2
    assert archive["y_train"].tolist() == [0, 1, 1, 0]


def test_prepare_completion_dataset_supports_recent_completion_targets(tmp_path: Path) -> None:
    rec1_video = tmp_path / "rec1.mp4"
    rec2_video = tmp_path / "rec2.mp4"
    rec1_video.write_bytes(b"")
    rec2_video.write_bytes(b"")

    rec1_labels = tmp_path / "rec1_labels.csv"
    rec2_labels = tmp_path / "rec2_labels.csv"
    rec1_labels.write_text(
        "recording_id,frame_index,gesture\n"
        "rec1,2,kick\n"
        "rec1,5,snare\n",
        encoding="utf-8",
    )
    rec2_labels.write_text(
        "recording_id,frame_index,gesture\n"
        "rec2,3,kick\n"
        "rec2,6,snare\n",
        encoding="utf-8",
    )

    frames_by_path = {
        rec1_video.as_posix(): build_frames(marker=1, frame_count=6),
        rec2_video.as_posix(): build_frames(marker=2, frame_count=7),
    }
    landmarks_by_marker = {
        1: build_landmarks_by_frame(
            left_wrist_positions=((0.20, 0.52), (0.24, 0.56), (0.28, 0.60), (0.32, 0.64), (0.36, 0.68), (0.40, 0.72))
        ),
        2: build_landmarks_by_frame(
            left_wrist_positions=((0.18, 0.48), (0.20, 0.52), (0.22, 0.56), (0.24, 0.60), (0.26, 0.64), (0.28, 0.68), (0.30, 0.72))
        ),
    }

    result = prepare_completion_dataset(
        (
            RecordingDatasetInput("rec1", rec1_video, rec1_labels),
            RecordingDatasetInput("rec2", rec2_video, rec2_labels),
        ),
        output_dir=tmp_path / "prepared_recent",
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(frames_by_path, fps=20.0),
        pose_provider_factory=lambda _: FakePoseProvider(landmarks_by_marker),
        window_size=3,
        stride=1,
        validation_recording_id="rec2",
        validation_fraction=0.5,
        target="completion_within_last_k_frames",
        horizon_frames=2,
    )

    archive = np.load(result.output_path)
    assert result.target_name == "completion_within_last_k_frames"
    assert result.horizon_frames == 2
    assert archive["target_name"].item() == "completion_within_last_k_frames"
    assert int(archive["horizon_frames"].item()) == 2
    assert archive["y_train"].tolist() == [1, 1, 0, 1]


def test_prepare_completion_dataset_supports_arm_targets_with_v2_labels(
    tmp_path: Path,
) -> None:
    rec1_video = tmp_path / "rec1_v2.mp4"
    rec2_video = tmp_path / "rec2_v2.mp4"
    rec1_video.write_bytes(b"")
    rec2_video.write_bytes(b"")

    rec1_labels = tmp_path / "rec1_v2_labels.csv"
    rec2_labels = tmp_path / "rec2_v2_labels.csv"
    rec1_labels.write_text(
        "recording_id,event_id,gesture_label,arm_start_frame,completion_frame\n"
        "rec1,evt-001,kick,1,1\n"
        "rec1,evt-002,snare,4,5\n",
        encoding="utf-8",
    )
    rec2_labels.write_text(
        "recording_id,event_id,gesture_label,arm_start_frame,completion_frame\n"
        "rec2,evt-101,kick,2,3\n"
        "rec2,evt-102,snare,5,6\n",
        encoding="utf-8",
    )

    frames_by_path = {
        rec1_video.as_posix(): build_frames(marker=1, frame_count=6),
        rec2_video.as_posix(): build_frames(marker=2, frame_count=7),
    }
    landmarks_by_marker = {
        1: build_landmarks_by_frame(
            left_wrist_positions=(
                (0.20, 0.52),
                (0.24, 0.56),
                (0.28, 0.60),
                (0.32, 0.64),
                (0.36, 0.68),
                (0.40, 0.72),
            )
        ),
        2: build_landmarks_by_frame(
            left_wrist_positions=(
                (0.18, 0.48),
                (0.20, 0.52),
                (0.22, 0.56),
                (0.24, 0.60),
                (0.26, 0.64),
                (0.28, 0.68),
                (0.30, 0.72),
            )
        ),
    }

    result = prepare_completion_dataset(
        (
            RecordingDatasetInput("rec1", rec1_video, rec1_labels),
            RecordingDatasetInput("rec2", rec2_video, rec2_labels),
        ),
        output_dir=tmp_path / "prepared_arm",
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(frames_by_path, fps=20.0),
        pose_provider_factory=lambda _: FakePoseProvider(landmarks_by_marker),
        window_size=2,
        stride=1,
        validation_recording_id="rec2",
        validation_fraction=0.5,
        target="arm_frame_binary",
    )

    archive = np.load(result.output_path)
    labeled_rows = list(
        csv.DictReader(
            (tmp_path / "prepared_arm" / "rec1_labeled.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )

    assert result.target_name == "arm_frame_binary"
    assert archive["target_name"].item() == "arm_frame_binary"
    assert archive["y_train"].tolist() == [1, 0, 0, 1, 1]
    assert archive["train_target_gesture_labels"].tolist() == [
        "kick",
        "",
        "",
        "snare",
        "snare",
    ]
    assert labeled_rows[1]["event_id"] == "evt-001"
    assert labeled_rows[1]["is_arm_frame"] == "True"
    assert labeled_rows[1]["is_completion"] == "True"
    assert labeled_rows[4]["event_id"] == "evt-002"
    assert labeled_rows[4]["is_arm_frame"] == "True"


def build_frames(*, marker: int, frame_count: int) -> list[np.ndarray]:
    return [
        np.full((2, 2, 3), marker, dtype=np.uint8)
        for _ in range(frame_count)
    ]


def build_landmarks_by_frame(
    *,
    left_wrist_positions: tuple[tuple[float, float], ...],
) -> list[dict[str, LandmarkPoint]]:
    frames: list[dict[str, LandmarkPoint]] = []
    for left_wrist_x, left_wrist_y in left_wrist_positions:
        frames.append(
            {
                "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
                "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
                "left_elbow": LandmarkPoint(
                    x=min(left_wrist_x + 0.05, 1.0),
                    y=max(left_wrist_y - 0.10, 0.0),
                    z=-0.15,
                    visibility=0.9,
                ),
                "right_elbow": LandmarkPoint(x=0.70, y=0.55, z=-0.15, visibility=0.9),
                "left_wrist": LandmarkPoint(
                    x=left_wrist_x,
                    y=left_wrist_y,
                    z=-0.2,
                    visibility=0.95,
                ),
                "right_wrist": LandmarkPoint(
                    x=0.80 - left_wrist_x * 0.1,
                    y=0.45,
                    z=-0.2,
                    visibility=0.95,
                ),
            }
        )
    return frames

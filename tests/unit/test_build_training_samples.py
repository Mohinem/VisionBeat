from __future__ import annotations

import numpy as np
from pathlib import Path

from visionbeat.build_training_samples import (
    build_training_samples,
    generate_training_samples,
    load_frame_feature_rows,
)
from visionbeat.config import TrackerConfig
from visionbeat.extract_dataset_features import extract_dataset_features
from visionbeat.features import CANONICAL_FEATURE_NAMES, get_canonical_feature_schema
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput


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

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _: str) -> FakeCapture:  # noqa: N802
        return self.capture


class FakePoseProvider:
    def __init__(self, landmarks_by_frame: list[dict[str, LandmarkPoint]]) -> None:
        self.landmarks_by_frame = list(landmarks_by_frame)

    def process(self, frame: object, timestamp: FrameTimestamp | float) -> TrackerOutput:
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        landmarks = self.landmarks_by_frame.pop(0)
        return TrackerOutput(
            timestamp=FrameTimestamp(seconds=seconds),
            landmarks=landmarks,
            person_detected=bool(landmarks),
            status="tracking" if landmarks else "no_person_detected",
        )

    def close(self) -> None:
        return None


def test_generate_training_samples_writes_binary_completion_windows(tmp_path: Path) -> None:
    frame_table_path = build_feature_table(tmp_path)
    output_path = tmp_path / "samples.npz"

    result = generate_training_samples(
        frame_table_path,
        output_path=output_path,
        window_size=2,
        stride=1,
        target="completion_frame_binary",
    )

    assert result.output_path == output_path
    assert result.sample_count == 4
    assert result.feature_schema == get_canonical_feature_schema()
    assert result.X_shape == (4, 2, len(CANONICAL_FEATURE_NAMES))
    assert result.y_shape == (4,)

    archive = np.load(output_path)
    assert archive["X"].shape == (4, 2, len(CANONICAL_FEATURE_NAMES))
    assert archive["y"].tolist() == [1, 0, 1, 0]
    assert archive["recording_ids"].tolist() == ["session", "session", "session", "session"]
    assert archive["window_end_frame_indices"].tolist() == [1, 2, 3, 4]
    assert archive["target_gesture_labels"].tolist() == ["kick", "", "snare", ""]
    assert archive["feature_names"].tolist() == list(CANONICAL_FEATURE_NAMES)


def test_build_training_samples_supports_future_completion_targets(tmp_path: Path) -> None:
    frame_table_path = build_feature_table(tmp_path)
    schema, frame_rows = load_frame_feature_rows(frame_table_path)

    dataset = build_training_samples(
        frame_rows,
        window_size=2,
        stride=1,
        target="completion_within_next_k_frames",
        horizon_frames=2,
        feature_schema=schema,
    )

    assert dataset.X.shape == (4, 2, len(CANONICAL_FEATURE_NAMES))
    assert dataset.y.tolist() == [1, 1, 0, 0]
    assert dataset.target_gesture_labels.tolist() == ["snare", "snare", "", ""]
    assert dataset.window_end_frame_indices.tolist() == [1, 2, 3, 4]


def test_build_training_samples_supports_recent_completion_targets(tmp_path: Path) -> None:
    frame_table_path = build_feature_table(tmp_path)
    schema, frame_rows = load_frame_feature_rows(frame_table_path)

    dataset = build_training_samples(
        frame_rows,
        window_size=2,
        stride=1,
        target="completion_within_last_k_frames",
        horizon_frames=2,
        feature_schema=schema,
    )

    assert dataset.X.shape == (4, 2, len(CANONICAL_FEATURE_NAMES))
    assert dataset.y.tolist() == [1, 1, 1, 1]
    assert dataset.target_gesture_labels.tolist() == ["kick", "kick", "snare", "snare"]
    assert dataset.window_end_frame_indices.tolist() == [1, 2, 3, 4]


def test_build_training_samples_supports_arm_frame_targets(tmp_path: Path) -> None:
    frame_table_path = build_v2_feature_table(tmp_path)
    schema, frame_rows = load_frame_feature_rows(frame_table_path)

    dataset = build_training_samples(
        frame_rows,
        window_size=2,
        stride=1,
        target="arm_frame_binary",
        feature_schema=schema,
    )

    assert dataset.X.shape == (5, 2, len(CANONICAL_FEATURE_NAMES))
    assert dataset.y.tolist() == [1, 0, 0, 1, 1]
    assert dataset.target_gesture_labels.tolist() == ["kick", "", "", "snare", "snare"]
    assert dataset.window_end_frame_indices.tolist() == [1, 2, 3, 4, 5]


def test_build_training_samples_supports_future_arm_targets(tmp_path: Path) -> None:
    frame_table_path = build_v2_feature_table(tmp_path)
    schema, frame_rows = load_frame_feature_rows(frame_table_path)

    dataset = build_training_samples(
        frame_rows,
        window_size=2,
        stride=1,
        target="arm_within_next_k_frames",
        horizon_frames=2,
        feature_schema=schema,
    )

    assert dataset.X.shape == (5, 2, len(CANONICAL_FEATURE_NAMES))
    assert dataset.y.tolist() == [0, 1, 1, 1, 0]
    assert dataset.target_gesture_labels.tolist() == ["", "snare", "snare", "snare", ""]
    assert dataset.window_end_frame_indices.tolist() == [1, 2, 3, 4, 5]


def build_feature_table(tmp_path: Path) -> Path:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "recording_id,frame_index,gesture\n"
        "session,1,kick\n"
        "session,3,snare\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features.csv"
    capture = FakeCapture(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)],
        fps=20.0,
    )
    extract_dataset_features(
        video_path,
        output_path=output_path,
        labels_path=labels_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(capture),
        pose_provider_factory=lambda _: FakePoseProvider(build_landmarks_by_frame()),
    )
    return output_path


def build_v2_feature_table(tmp_path: Path) -> Path:
    video_path = tmp_path / "session_v2.mp4"
    video_path.write_bytes(b"")
    labels_path = tmp_path / "labels_v2.csv"
    labels_path.write_text(
        "recording_id,event_id,gesture_label,arm_start_frame,completion_frame\n"
        "session_v2,evt-001,kick,1,1\n"
        "session_v2,evt-002,snare,4,5\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features_v2.csv"
    capture = FakeCapture(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(6)],
        fps=20.0,
    )
    extract_dataset_features(
        video_path,
        output_path=output_path,
        labels_path=labels_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(capture),
        pose_provider_factory=lambda _: FakePoseProvider(
            [*build_landmarks_by_frame(), build_landmarks_by_frame()[-1]]
        ),
    )
    return output_path


def build_landmarks_by_frame() -> list[dict[str, LandmarkPoint]]:
    return [
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.20, y=0.50, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.80, y=0.50, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.22, y=0.55, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.78, y=0.48, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.30, y=0.60, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.70, y=0.45, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.35, y=0.65, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.65, y=0.40, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.40, y=0.70, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.60, y=0.35, z=-0.2, visibility=0.95),
        },
    ]

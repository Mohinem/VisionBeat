from __future__ import annotations

import numpy as np
from pathlib import Path

from visionbeat.config import TrackerConfig
from visionbeat.features import CANONICAL_FEATURE_NAMES, CANONICAL_FEATURE_SCHEMA
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput
from visionbeat.prepare_training_data import prepare_training_data


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


def test_prepare_training_data_builds_npz_and_discards_temp_frame_table(tmp_path: Path) -> None:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "recording_id,frame_index,gesture\n"
        "session,1,kick\n"
        "session,3,snare\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "session.train.npz"

    result = prepare_training_data(
        video_path,
        labels_path=labels_path,
        output_path=output_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(
            FakeCapture(
                [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)],
                fps=20.0,
            )
        ),
        pose_provider_factory=lambda _: FakePoseProvider(build_landmarks_by_frame()),
        window_size=2,
        stride=1,
    )

    assert result.output_path == output_path
    assert result.frame_table_path is None
    assert result.frame_schema_path is None
    assert result.feature_schema.version == CANONICAL_FEATURE_SCHEMA
    assert result.sample_count == 4
    assert result.X_shape == (4, 2, len(CANONICAL_FEATURE_NAMES))
    assert result.y_shape == (4,)

    archive = np.load(output_path)
    assert archive["schema_version"].item() == CANONICAL_FEATURE_SCHEMA
    assert archive["feature_names"].tolist() == list(CANONICAL_FEATURE_NAMES)
    assert archive["X"].shape == (4, 2, len(CANONICAL_FEATURE_NAMES))
    assert archive["y"].tolist() == [1, 0, 1, 0]


def test_prepare_training_data_can_keep_intermediate_frame_table(tmp_path: Path) -> None:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "recording_id,frame_index,gesture\n"
        "session,1,kick\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "session.train.npz"
    frames_out = tmp_path / "session.features.csv"

    result = prepare_training_data(
        video_path,
        labels_path=labels_path,
        output_path=output_path,
        frame_table_path=frames_out,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(
            FakeCapture(
                [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)],
                fps=20.0,
            )
        ),
        pose_provider_factory=lambda _: FakePoseProvider(build_landmarks_by_frame()[:3]),
        window_size=2,
        stride=1,
    )

    assert result.frame_table_path == frames_out
    assert result.frame_schema_path == tmp_path / "session.features.csv.schema.json"
    assert result.frame_table_path.exists()
    assert result.frame_schema_path.exists()
    assert result.output_path.exists()


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

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from visionbeat.config import TrackerConfig
from visionbeat.extract_dataset_features import extract_dataset_features
from visionbeat.features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    CanonicalFeatureExtractor,
    assert_feature_vectors_match,
)
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput

_EXPECTED_V1_FEATURE_NAMES = (
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_visibility",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_visibility",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_visibility",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_visibility",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_visibility",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_visibility",
    "shoulder_center_x",
    "shoulder_center_y",
    "shoulder_width",
    "left_elbow_rel_x",
    "left_elbow_rel_y",
    "right_elbow_rel_x",
    "right_elbow_rel_y",
    "left_wrist_rel_x",
    "left_wrist_rel_y",
    "right_wrist_rel_x",
    "right_wrist_rel_y",
    "wrist_delta_x",
    "wrist_delta_y",
    "wrist_distance_xy",
    "dt_seconds",
    "left_wrist_rel_vx",
    "left_wrist_rel_vy",
    "right_wrist_rel_vx",
    "right_wrist_rel_vy",
    "wrist_delta_x_v",
    "wrist_delta_y_v",
    "wrist_distance_xy_v",
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


def test_feature_order_changes_require_schema_version_bump() -> None:
    if FEATURE_SCHEMA_VERSION == "visionbeat.features.v1":
        assert CANONICAL_FEATURE_NAMES == _EXPECTED_V1_FEATURE_NAMES
        return
    assert FEATURE_SCHEMA_VERSION != "visionbeat.features.v1"


def test_same_landmark_input_produces_same_vector_in_offline_and_live_paths(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"")
    output_path = tmp_path / "features.csv"

    landmarks_by_frame = build_landmarks_by_frame()
    extract_dataset_features(
        video_path,
        output_path=output_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(
            FakeCapture(
                [np.zeros((2, 2, 3), dtype=np.uint8) for _ in landmarks_by_frame],
                fps=20.0,
            )
        ),
        pose_provider_factory=lambda _: FakePoseProvider(build_landmarks_by_frame()),
    )

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))
    live_extractor = CanonicalFeatureExtractor()
    live_frames = [
        live_extractor.update(
            TrackerOutput(
                timestamp=FrameTimestamp(seconds=index * 0.05),
                landmarks=landmarks,
                person_detected=bool(landmarks),
                status="tracking" if landmarks else "no_person_detected",
            )
        )
        for index, landmarks in enumerate(build_landmarks_by_frame())
    ]

    assert len(rows) == len(live_frames)
    for row, live_frame in zip(rows, live_frames, strict=True):
        offline_vector = tuple(float(row[name]) for name in CANONICAL_FEATURE_NAMES)
        assert_feature_vectors_match(offline_vector, live_frame.vector)


def build_landmarks_by_frame() -> list[dict[str, LandmarkPoint]]:
    return [
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.3, visibility=0.90),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.2, visibility=0.80),
            "right_elbow": LandmarkPoint(x=0.75, y=0.75, z=0.0, visibility=0.70),
            "left_wrist": LandmarkPoint(x=-0.20, y=1.20, z=0.2, visibility=1.00),
            "right_wrist": LandmarkPoint(x=1.20, y=0.00, z=-0.4, visibility=0.95),
            "nose": LandmarkPoint(x=0.40, y=0.40, z=0.0, visibility=0.99),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=0.0, visibility=0.90),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=0.0, visibility=0.90),
            "left_wrist": LandmarkPoint(x=0.50, y=0.75, z=0.0, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.75, y=0.75, z=0.0, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=0.0, visibility=0.90),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=0.0, visibility=0.90),
            "right_wrist": LandmarkPoint(x=0.75, y=0.75, z=0.0, visibility=0.95),
        },
    ]

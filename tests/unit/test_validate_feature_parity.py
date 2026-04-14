from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from visionbeat.config import TrackerConfig
from visionbeat.extract_dataset_features import extract_dataset_features
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput
from visionbeat.validate_feature_parity import (
    extract_live_canonical_features_from_video,
    validate_offline_feature_csv_against_live_features,
    validate_video_feature_parity,
)


class FakeCapture:
    def __init__(self, *, frame_count: int, fps: float) -> None:
        self._frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(frame_count)]
        self._fps = fps
        self._last_frame_index = -1

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        self._last_frame_index += 1
        return True, self._frames.pop(0)

    def get(self, prop: int) -> float:  # noqa: N802
        if prop == FakeCV2.CAP_PROP_FPS:
            return self._fps
        if prop == FakeCV2.CAP_PROP_POS_MSEC:
            if self._last_frame_index < 0:
                return 0.0
            return (self._last_frame_index / self._fps) * 1000.0
        return 0.0

    def release(self) -> None:
        return None


class FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 6

    def __init__(self, *, frame_count: int, fps: float) -> None:
        self._frame_count = frame_count
        self._fps = fps

    def VideoCapture(self, _: str) -> FakeCapture:  # noqa: N802
        return FakeCapture(frame_count=self._frame_count, fps=self._fps)


class FakePoseProvider:
    def __init__(self, landmarks_by_frame: list[dict[str, LandmarkPoint]]) -> None:
        self._landmarks_by_frame = list(landmarks_by_frame)

    def process(self, frame: object, timestamp: FrameTimestamp | float) -> TrackerOutput:
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        landmarks = self._landmarks_by_frame.pop(0)
        return TrackerOutput(
            timestamp=FrameTimestamp(seconds=seconds),
            landmarks=landmarks,
            person_detected=bool(landmarks),
            status="tracking" if landmarks else "no_person_detected",
        )

    def close(self) -> None:
        return None


def build_landmarks_by_frame() -> list[dict[str, LandmarkPoint]]:
    return [
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "left_wrist": LandmarkPoint(x=0.25, y=0.50, z=-0.2, visibility=0.95),
            "right_wrist": LandmarkPoint(x=0.75, y=0.50, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.25, y=0.50, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.75, y=0.50, z=-0.1, visibility=0.9),
            "right_wrist": LandmarkPoint(x=0.75, y=0.75, z=-0.2, visibility=0.95),
        },
        {
            "left_shoulder": LandmarkPoint(x=0.20, y=0.55, z=-0.1, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.80, y=0.55, z=-0.1, visibility=0.9),
            "left_elbow": LandmarkPoint(x=0.15, y=0.70, z=-0.2, visibility=0.8),
            "right_elbow": LandmarkPoint(x=0.85, y=0.70, z=-0.2, visibility=0.8),
            "left_wrist": LandmarkPoint(x=-0.10, y=1.10, z=-0.2, visibility=1.0),
            "right_wrist": LandmarkPoint(x=1.20, y=0.10, z=-0.2, visibility=0.9),
        },
    ]


def make_pose_provider_factory():
    return lambda _: FakePoseProvider(build_landmarks_by_frame())


def test_validate_video_feature_parity_reports_pass_for_equivalent_paths(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")

    report = validate_video_feature_parity(
        video_path,
        tracker_config=TrackerConfig(),
        cv2_module=FakeCV2(frame_count=3, fps=20.0),
        pose_provider_factory=make_pose_provider_factory(),
        abs_tolerance=1e-9,
    )

    assert report.passed is True
    assert report.feature_name_mismatches == ()
    assert report.feature_position_mismatches == ()
    assert report.numerical_mismatches == ()
    assert "PASS: VisionBeat offline/live feature parity" in report.to_text()


def test_validate_offline_feature_csv_against_live_features_reports_failure_details(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    output_path = tmp_path / "features.csv"
    cv2_module = FakeCV2(frame_count=3, fps=20.0)
    pose_provider_factory = make_pose_provider_factory()

    extraction_result = extract_dataset_features(
        video_path,
        output_path=output_path,
        tracker_config=TrackerConfig(),
        cv2_module=cv2_module,
        pose_provider_factory=pose_provider_factory,
    )
    live_frames = extract_live_canonical_features_from_video(
        video_path,
        tracker_config=TrackerConfig(),
        cv2_module=cv2_module,
        pose_provider_factory=make_pose_provider_factory(),
    )

    with output_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    rows[1]["wrist_delta_y"] = "0.125"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = validate_offline_feature_csv_against_live_features(
        extraction_result.output_path,
        live_frames=live_frames,
        offline_schema=extraction_result.feature_schema,
        abs_tolerance=1e-9,
    )

    assert report.passed is False
    assert report.numerical_mismatches
    assert report.numerical_mismatches[0].feature_name == "wrist_delta_y"
    assert report.likely_source == (
        "Derived feature mismatch in normalization, relative coordinates, or distances."
    )
    assert "FAIL: VisionBeat offline/live feature parity" in report.to_text()
    assert "feature=wrist_delta_y" in report.to_text()

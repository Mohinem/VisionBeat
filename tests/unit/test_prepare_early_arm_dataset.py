from __future__ import annotations

from pathlib import Path

import pytest

from visionbeat.prepare_completion_dataset import CompletionDatasetPreparationResult, RecordingDatasetInput
from visionbeat.prepare_early_arm_dataset import prepare_early_arm_dataset
from visionbeat.features import get_canonical_feature_schema


def test_prepare_early_arm_dataset_defaults_to_arm_target(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_prepare_completion_dataset(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return CompletionDatasetPreparationResult(
            output_path=Path("train_dataset.npz"),
            feature_schema=get_canonical_feature_schema(),
            recordings=(),
            train_sample_count=0,
            validation_sample_count=0,
            total_frames_processed=0,
            total_labeled_gestures=0,
            train_shape=(0, 0, 0),
            validation_shape=(0, 0, 0),
            sanity_warnings=(),
            validation_status="PASS",
            target_name="arm_frame_binary",
            horizon_frames=4,
        )

    monkeypatch.setattr(
        "visionbeat.prepare_early_arm_dataset.prepare_completion_dataset",
        fake_prepare_completion_dataset,
    )

    prepare_early_arm_dataset(
        (RecordingDatasetInput("rec1", Path("rec1.mp4"), Path("labels.csv")),),
        output_dir="prepared",
    )

    assert called["kwargs"]["target"] == "arm_frame_binary"


def test_prepare_early_arm_dataset_rejects_completion_targets() -> None:
    with pytest.raises(ValueError, match="Unsupported early-arm target"):
        prepare_early_arm_dataset(
            (RecordingDatasetInput("rec1", Path("rec1.mp4"), Path("labels.csv")),),
            output_dir="prepared",
            target="completion_frame_binary",
        )

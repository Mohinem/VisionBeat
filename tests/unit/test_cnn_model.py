from __future__ import annotations

import pytest

from visionbeat.cnn_model import (
    VisionBeatCnnSpec,
    build_checkpoint_payload,
    validate_runtime_compatibility,
)
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


def test_checkpoint_payload_round_trips_model_metadata() -> None:
    spec = VisionBeatCnnSpec(
        feature_count=len(FEATURE_NAMES),
        window_size=32,
        hidden_channels=64,
        dropout=0.2,
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
        target_name="completion_within_next_k_frames",
        horizon_frames=4,
    )

    checkpoint = build_checkpoint_payload(
        spec=spec,
        model_state_dict={"features.0.weight": object()},
        extra={"epoch": 3},
    )

    loaded_spec = VisionBeatCnnSpec.from_checkpoint(checkpoint)
    assert loaded_spec == spec
    assert checkpoint["model_metadata"]["feature_schema"]["feature_names"] == list(FEATURE_NAMES)
    assert checkpoint["window_size"] == 32
    assert checkpoint["feature_count"] == len(FEATURE_NAMES)
    assert checkpoint["horizon_frames"] == 4


def test_validate_runtime_compatibility_rejects_schema_mismatch() -> None:
    spec = VisionBeatCnnSpec(
        feature_count=len(FEATURE_NAMES),
        window_size=32,
        hidden_channels=64,
        dropout=0.2,
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
    )

    with pytest.raises(ValueError, match="feature_names"):
        validate_runtime_compatibility(
            spec,
            feature_names=FEATURE_NAMES[:-1] + ("wrong_feature",),
            schema_version=FEATURE_SCHEMA_VERSION,
            window_size=32,
            target_name="completion_frame_binary",
        )


def test_validate_runtime_compatibility_rejects_future_target_horizon_mismatch() -> None:
    spec = VisionBeatCnnSpec(
        feature_count=len(FEATURE_NAMES),
        window_size=32,
        hidden_channels=64,
        dropout=0.2,
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
        target_name="completion_within_next_k_frames",
        horizon_frames=4,
    )

    with pytest.raises(ValueError, match="horizon_frames"):
        validate_runtime_compatibility(
            spec,
            feature_names=FEATURE_NAMES,
            schema_version=FEATURE_SCHEMA_VERSION,
            window_size=32,
            target_name="completion_within_next_k_frames",
            horizon_frames=2,
        )

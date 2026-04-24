from __future__ import annotations

from math import hypot

import pytest

from visionbeat.features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    CanonicalFeatureExtractor,
    CanonicalFeatureSchema,
    FeatureSchemaError,
    assert_feature_schemas_match,
    assert_feature_vectors_match,
    build_sequence_window,
    build_feature_vector,
    compare_feature_schemas,
    compare_feature_vectors,
    extract_canonical_frame_features,
    get_canonical_feature_schema,
)
from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput


def make_frame(
    timestamp: float,
    **landmarks: tuple[float, float, float, float],
) -> TrackerOutput:
    return TrackerOutput(
        timestamp=FrameTimestamp(seconds=timestamp),
        landmarks={
            name: LandmarkPoint(x=x, y=y, z=z, visibility=visibility)
            for name, (x, y, z, visibility) in landmarks.items()
        },
        person_detected=bool(landmarks),
        status="tracking" if landmarks else "no_person_detected",
    )


def test_canonical_feature_extractor_clamps_common_subset_and_fills_missing_values() -> None:
    extractor = CanonicalFeatureExtractor()
    frame = make_frame(
        1.0,
        left_shoulder=(0.25, 0.50, -0.3, 0.90),
        right_shoulder=(0.75, 0.50, -0.2, 0.80),
        right_elbow=(0.75, 0.75, 0.0, 0.70),
        left_wrist=(-0.20, 1.20, 0.2, 1.00),
        right_wrist=(1.20, 0.00, -0.4, 0.95),
        nose=(0.40, 0.40, 0.0, 0.99),
    )

    features = extractor.update(frame).as_feature_dict()

    assert tuple(features) == CANONICAL_FEATURE_NAMES
    assert "nose_x" not in features
    assert features["left_shoulder_x"] == pytest.approx(0.25)
    assert features["left_shoulder_y"] == pytest.approx(0.50)
    assert features["right_shoulder_visibility"] == pytest.approx(0.80)
    assert features["left_elbow_x"] == pytest.approx(0.0)
    assert features["left_elbow_rel_x"] == pytest.approx(0.0)
    assert features["left_wrist_x"] == pytest.approx(0.0)
    assert features["left_wrist_y"] == pytest.approx(1.0)
    assert features["left_wrist_visibility"] == pytest.approx(1.0)
    assert features["right_wrist_x"] == pytest.approx(1.0)
    assert features["right_wrist_y"] == pytest.approx(0.0)
    assert features["shoulder_center_x"] == pytest.approx(0.50)
    assert features["shoulder_center_y"] == pytest.approx(0.50)
    assert features["shoulder_width"] == pytest.approx(0.50)
    assert features["right_elbow_rel_x"] == pytest.approx(0.50)
    assert features["right_elbow_rel_y"] == pytest.approx(0.50)
    assert features["left_wrist_rel_x"] == pytest.approx(-1.0)
    assert features["left_wrist_rel_y"] == pytest.approx(1.0)
    assert features["right_wrist_rel_x"] == pytest.approx(1.0)
    assert features["right_wrist_rel_y"] == pytest.approx(-1.0)
    assert features["wrist_delta_x"] == pytest.approx(1.0)
    assert features["wrist_delta_y"] == pytest.approx(-1.0)
    assert features["wrist_distance_xy"] == pytest.approx(hypot(1.0, -1.0))
    assert features["dt_seconds"] == pytest.approx(0.0)
    assert features["left_wrist_rel_vx"] == pytest.approx(0.0)

    direct = extract_canonical_frame_features(frame)
    assert direct.raw_features["left_shoulder_x"] == pytest.approx(0.25)
    assert direct.derived_features["shoulder_width"] == pytest.approx(0.50)
    assert direct.temporal_features["dt_seconds"] == pytest.approx(0.0)


def test_canonical_feature_extractor_emits_causal_temporal_features_and_can_reset() -> None:
    extractor = CanonicalFeatureExtractor()
    first = make_frame(
        1.0,
        left_shoulder=(0.25, 0.50, 0.0, 0.90),
        right_shoulder=(0.75, 0.50, 0.0, 0.90),
        left_wrist=(0.25, 0.50, 0.0, 0.95),
        right_wrist=(0.75, 0.50, 0.0, 0.95),
    )
    second = make_frame(
        1.5,
        left_shoulder=(0.25, 0.50, 0.0, 0.90),
        right_shoulder=(0.75, 0.50, 0.0, 0.90),
        left_wrist=(0.50, 0.75, 0.0, 0.95),
        right_wrist=(0.75, 0.75, 0.0, 0.95),
    )

    first_features = extractor.update(first).as_feature_dict()
    second_features = extractor.update(second).as_feature_dict()

    assert first_features["dt_seconds"] == pytest.approx(0.0)
    assert second_features["dt_seconds"] == pytest.approx(0.5)
    assert second_features["left_wrist_rel_vx"] == pytest.approx(1.0)
    assert second_features["left_wrist_rel_vy"] == pytest.approx(1.0)
    assert second_features["right_wrist_rel_vx"] == pytest.approx(0.0)
    assert second_features["right_wrist_rel_vy"] == pytest.approx(1.0)
    assert second_features["wrist_delta_x_v"] == pytest.approx(-0.5)
    assert second_features["wrist_delta_y_v"] == pytest.approx(0.0)
    assert second_features["wrist_distance_xy_v"] == pytest.approx(-0.5)

    extractor.reset()
    reset_features = extractor.update(second).as_feature_dict()
    assert reset_features["dt_seconds"] == pytest.approx(0.0)
    assert reset_features["left_wrist_rel_vx"] == pytest.approx(0.0)


def test_build_sequence_window_uses_same_shared_path_and_left_pads() -> None:
    frames = [
        make_frame(
            1.0,
            left_shoulder=(0.25, 0.50, 0.0, 0.90),
            right_shoulder=(0.75, 0.50, 0.0, 0.90),
            left_wrist=(0.25, 0.50, 0.0, 0.95),
            right_wrist=(0.75, 0.50, 0.0, 0.95),
        ),
        make_frame(
            1.5,
            left_shoulder=(0.25, 0.50, 0.0, 0.90),
            right_shoulder=(0.75, 0.50, 0.0, 0.90),
            left_wrist=(0.50, 0.75, 0.0, 0.95),
            right_wrist=(0.75, 0.75, 0.0, 0.95),
        ),
    ]

    window = build_sequence_window(frames, window_size=4)

    assert window.feature_names == CANONICAL_FEATURE_NAMES
    assert window.schema_version == get_canonical_feature_schema().version
    assert window.feature_count == len(CANONICAL_FEATURE_NAMES)
    assert len(window.frames) == 2
    assert len(window.matrix) == 4
    assert window.matrix[0] == tuple(0.0 for _ in CANONICAL_FEATURE_NAMES)
    assert window.matrix[1] == tuple(0.0 for _ in CANONICAL_FEATURE_NAMES)
    assert window.matrix[-1][CANONICAL_FEATURE_NAMES.index("dt_seconds")] == pytest.approx(0.5)


def test_schema_helpers_report_version_and_order_differences() -> None:
    schema = get_canonical_feature_schema()
    reordered_schema = CanonicalFeatureSchema(
        version=schema.version,
        feature_names=(schema.feature_names[1], schema.feature_names[0], *schema.feature_names[2:]),
        feature_count=schema.feature_count,
    )
    version_mismatch = CanonicalFeatureSchema(
        version="visionbeat.features.v999",
        feature_names=schema.feature_names,
        feature_count=schema.feature_count,
    )

    assert compare_feature_schemas(schema, reordered_schema) == ["feature_names/order differs"]
    assert compare_feature_schemas(schema, version_mismatch) == [
        "schema_version differs ('visionbeat.features.v1' vs 'visionbeat.features.v999')"
    ]
    with pytest.raises(FeatureSchemaError, match="feature_names/order differs"):
        assert_feature_schemas_match(schema, reordered_schema)


def test_canonical_feature_schema_version_and_names_are_consistent() -> None:
    schema = get_canonical_feature_schema()

    assert FEATURE_NAMES == CANONICAL_FEATURE_NAMES
    assert schema.version == FEATURE_SCHEMA_VERSION
    assert schema.feature_names == FEATURE_NAMES
    assert schema.feature_count == len(FEATURE_NAMES)


def test_build_feature_vector_raises_on_missing_or_reordered_features() -> None:
    frame = extract_canonical_frame_features(
        make_frame(
            1.0,
            left_shoulder=(0.25, 0.50, 0.0, 0.90),
            right_shoulder=(0.75, 0.50, 0.0, 0.90),
            left_wrist=(0.25, 0.50, 0.0, 0.95),
            right_wrist=(0.75, 0.50, 0.0, 0.95),
        )
    )

    missing_raw = dict(frame.raw_features)
    missing_raw.pop("left_shoulder_x")
    with pytest.raises(FeatureSchemaError, match="missing required features: left_shoulder_x"):
        build_feature_vector(
            missing_raw,
            frame.derived_features,
            frame.temporal_features,
        )

    reordered_raw = {
        "left_shoulder_y": frame.raw_features["left_shoulder_y"],
        "left_shoulder_x": frame.raw_features["left_shoulder_x"],
        **{
            name: value
            for name, value in frame.raw_features.items()
            if name not in {"left_shoulder_x", "left_shoulder_y"}
        },
    }
    with pytest.raises(FeatureSchemaError, match="ordering changed unexpectedly"):
        build_feature_vector(
            reordered_raw,
            frame.derived_features,
            frame.temporal_features,
        )


def test_feature_vector_helpers_compare_vectors_by_named_feature() -> None:
    frame = extract_canonical_frame_features(
        make_frame(
            1.0,
            left_shoulder=(0.25, 0.50, 0.0, 0.90),
            right_shoulder=(0.75, 0.50, 0.0, 0.90),
            left_wrist=(0.25, 0.50, 0.0, 0.95),
            right_wrist=(0.75, 0.50, 0.0, 0.95),
        )
    )
    mutated = list(frame.vector)
    mutated[CANONICAL_FEATURE_NAMES.index("wrist_delta_x")] += 0.25

    assert compare_feature_vectors(frame.vector, mutated) == ["wrist_delta_x: 0.5 != 0.75"]
    with pytest.raises(FeatureSchemaError, match="wrist_delta_x: 0.5 != 0.75"):
        assert_feature_vectors_match(frame.vector, mutated)

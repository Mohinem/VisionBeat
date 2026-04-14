"""Canonical pose feature extraction shared by live and offline pipelines.

This module defines the only supported feature formulas for VisionBeat model
input. Offline dataset generation and live runtime inference must call into this
module, not reimplement any of the formulas elsewhere.

Conventions
- Landmark subset: the six upper-body landmarks common to MediaPipe and MoveNet.
- Coordinates: absolute `x`/`y` are image-normalized and clamped into `[0.0, 1.0]`.
- Depth: excluded from the canonical schema because provider semantics differ.
- Shoulder-relative coordinates: `(point - shoulder_center) / shoulder_width`.
- Velocities: causal first-order finite differences using the immediately previous frame.
- Wrist distances: Euclidean distance in the normalized 2D image plane.
- Visibility/confidence: clamped into `[0.0, 1.0]` and stored as `visibility`.
- Missing landmarks: replaced with a deterministic default fill value.
- Default fill value: `0.0` for raw, derived, temporal, and padded sequence features.
- Feature ordering: fixed by `FEATURE_NAMES` and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isclose
from typing import Any, Final

from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput

FEATURE_SCHEMA_VERSION: Final[str] = "visionbeat.features.v1"
DEFAULT_FILL_VALUE: Final[float] = 0.0
CANONICAL_LANDMARKS: Final[tuple[str, ...]] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
CANONICAL_LABEL_FIELDS: Final[tuple[str, ...]] = (
    "sequence_id",
    "frame_index",
    "timestamp_seconds",
    "gesture_label",
    "label_source",
    "source_path",
)
CANONICAL_RAW_FEATURE_NAMES: Final[tuple[str, ...]] = (
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
)
CANONICAL_DERIVED_FEATURE_NAMES: Final[tuple[str, ...]] = (
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
)
CANONICAL_TEMPORAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "dt_seconds",
    "left_wrist_rel_vx",
    "left_wrist_rel_vy",
    "right_wrist_rel_vx",
    "right_wrist_rel_vy",
    "wrist_delta_x_v",
    "wrist_delta_y_v",
    "wrist_distance_xy_v",
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *CANONICAL_RAW_FEATURE_NAMES,
    *CANONICAL_DERIVED_FEATURE_NAMES,
    *CANONICAL_TEMPORAL_FEATURE_NAMES,
)
FEATURE_COUNT: Final[int] = len(FEATURE_NAMES)

# Compatibility aliases for existing callers.
CANONICAL_FEATURE_SCHEMA: Final[str] = FEATURE_SCHEMA_VERSION
CANONICAL_FEATURE_NAMES: Final[tuple[str, ...]] = FEATURE_NAMES

_SCALE_EPSILON: Final[float] = 1e-6


class FeatureSchemaError(ValueError):
    """Raised when canonical feature schema or vector invariants are violated."""


@dataclass(frozen=True, slots=True)
class CanonicalFeatureSchema:
    """Authoritative feature schema contract shared by offline and live paths."""

    version: str
    feature_names: tuple[str, ...]
    feature_count: int

    def __post_init__(self) -> None:
        if self.feature_count != len(self.feature_names):
            raise FeatureSchemaError(
                "feature_count must match the number of feature_names. "
                f"Expected {len(self.feature_names)}, got {self.feature_count}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature schema for manifests, logs, and sidecars."""
        return {
            "schema": self.version,
            "schema_version": self.version,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True, slots=True)
class CanonicalFrameFeatures:
    """Canonical per-frame features plus the final ordered vector.

    This container is the single per-frame feature contract for both offline
    dataset generation and live runtime inference.
    """

    timestamp: FrameTimestamp
    person_detected: bool
    status: str
    raw_features: dict[str, float]
    derived_features: dict[str, float]
    temporal_features: dict[str, float]
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        _ensure_exact_feature_order(
            self.raw_features,
            expected_names=CANONICAL_RAW_FEATURE_NAMES,
            context="raw_features",
        )
        _ensure_exact_feature_order(
            self.derived_features,
            expected_names=CANONICAL_DERIVED_FEATURE_NAMES,
            context="derived_features",
        )
        _ensure_exact_feature_order(
            self.temporal_features,
            expected_names=CANONICAL_TEMPORAL_FEATURE_NAMES,
            context="temporal_features",
        )
        assert_feature_vector_matches_schema(self.vector)

    def as_feature_dict(self) -> dict[str, float]:
        """Return the complete ordered feature mapping for one frame."""
        return {
            **self.raw_features,
            **self.derived_features,
            **self.temporal_features,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical per-frame features into a JSON-friendly dictionary."""
        schema = get_canonical_feature_schema()
        return {
            "schema": schema.version,
            "schema_version": schema.version,
            "feature_count": schema.feature_count,
            "feature_names": list(schema.feature_names),
            "timestamp": self.timestamp.to_dict(),
            "person_detected": self.person_detected,
            "status": self.status,
            "raw_features": dict(self.raw_features),
            "derived_features": dict(self.derived_features),
            "temporal_features": dict(self.temporal_features),
            "vector": list(self.vector),
        }


@dataclass(frozen=True, slots=True)
class CanonicalSequenceWindow:
    """Fixed-order sequence matrix built from canonical per-frame features."""

    frames: tuple[CanonicalFrameFeatures, ...]
    matrix: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    fill_value: float = DEFAULT_FILL_VALUE
    schema_version: str = FEATURE_SCHEMA_VERSION
    feature_count: int = FEATURE_COUNT

    def __post_init__(self) -> None:
        schema = CanonicalFeatureSchema(
            version=self.schema_version,
            feature_names=self.feature_names,
            feature_count=self.feature_count,
        )
        assert_feature_schemas_match(get_canonical_feature_schema(), schema)
        for row in self.matrix:
            assert_feature_vector_matches_schema(row, schema=schema, context="sequence row")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical sequence window into a JSON-friendly dictionary."""
        schema = get_canonical_feature_schema()
        return {
            "schema": schema.version,
            "schema_version": schema.version,
            "feature_count": schema.feature_count,
            "feature_names": list(schema.feature_names),
            "fill_value": self.fill_value,
            "matrix": [list(row) for row in self.matrix],
            "frame_timestamps": [frame.timestamp.seconds for frame in self.frames],
        }


class CanonicalFeatureExtractor:
    """Thin stateful wrapper over the pure canonical feature functions."""

    def __init__(self, *, fill_value: float = DEFAULT_FILL_VALUE) -> None:
        self._fill_value = float(fill_value)
        self._previous: CanonicalFrameFeatures | None = None

    @property
    def schema(self) -> CanonicalFeatureSchema:
        """Return the authoritative schema used by this extractor."""
        return get_canonical_feature_schema()

    def reset(self) -> None:
        """Discard temporal state before starting a new independent sequence."""
        self._previous = None

    def update(self, frame: TrackerOutput) -> CanonicalFrameFeatures:
        """Extract one canonical frame using the same formulas as offline processing."""
        features = extract_canonical_frame_features(
            frame,
            previous=self._previous,
            fill_value=self._fill_value,
        )
        assert_feature_schemas_match(self.schema, get_canonical_feature_schema())
        self._previous = features
        return features


def get_canonical_feature_schema() -> CanonicalFeatureSchema:
    """Return the single authoritative CNN input schema for VisionBeat."""
    return CanonicalFeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
        feature_count=FEATURE_COUNT,
    )


def extract_canonical_frame_features(
    frame: TrackerOutput,
    previous: CanonicalFrameFeatures | None = None,
    *,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> CanonicalFrameFeatures:
    """Extract one canonical per-frame feature payload from a `TrackerOutput`.

    This function must be used by both offline dataset generation and live runtime
    inference to guarantee train/inference feature parity.
    """

    normalized_landmarks = _normalize_canonical_landmarks(frame)
    raw_features = _extract_raw_features(normalized_landmarks, fill_value=fill_value)
    derived_features = compute_derived_features(
        normalized_landmarks,
        fill_value=fill_value,
    )
    temporal_features = compute_temporal_features(
        frame.timestamp,
        normalized_landmarks,
        derived_features,
        previous=previous,
        fill_value=fill_value,
    )
    vector = build_feature_vector(
        raw_features,
        derived_features,
        temporal_features,
    )
    return CanonicalFrameFeatures(
        timestamp=frame.timestamp,
        person_detected=frame.person_detected,
        status=frame.status,
        raw_features=raw_features,
        derived_features=derived_features,
        temporal_features=temporal_features,
        vector=vector,
    )


def compute_derived_features(
    landmarks: dict[str, LandmarkPoint | None],
    *,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> dict[str, float]:
    """Compute all non-temporal derived features from canonical landmarks.

    This function must be used by both offline dataset generation and live runtime
    inference to guarantee train/inference feature parity.
    """

    fill = float(fill_value)
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    shoulder_center_x = fill
    shoulder_center_y = fill
    shoulder_width = fill
    shoulder_valid = left_shoulder is not None and right_shoulder is not None
    if shoulder_valid:
        assert left_shoulder is not None
        assert right_shoulder is not None
        candidate_width = hypot(
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
        )
        if candidate_width > _SCALE_EPSILON:
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) * 0.5
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) * 0.5
            shoulder_width = candidate_width
        else:
            shoulder_valid = False

    features = {
        "shoulder_center_x": shoulder_center_x,
        "shoulder_center_y": shoulder_center_y,
        "shoulder_width": shoulder_width,
        "left_elbow_rel_x": fill,
        "left_elbow_rel_y": fill,
        "right_elbow_rel_x": fill,
        "right_elbow_rel_y": fill,
        "left_wrist_rel_x": fill,
        "left_wrist_rel_y": fill,
        "right_wrist_rel_x": fill,
        "right_wrist_rel_y": fill,
        "wrist_delta_x": fill,
        "wrist_delta_y": fill,
        "wrist_distance_xy": fill,
    }

    if shoulder_valid:
        for landmark_name, feature_prefix in (
            ("left_elbow", "left_elbow"),
            ("right_elbow", "right_elbow"),
            ("left_wrist", "left_wrist"),
            ("right_wrist", "right_wrist"),
        ):
            point = landmarks[landmark_name]
            if point is None:
                continue
            features[f"{feature_prefix}_rel_x"] = (
                (point.x - shoulder_center_x) / shoulder_width
            )
            features[f"{feature_prefix}_rel_y"] = (
                (point.y - shoulder_center_y) / shoulder_width
            )

    left_wrist = landmarks["left_wrist"]
    right_wrist = landmarks["right_wrist"]
    if left_wrist is not None and right_wrist is not None:
        wrist_delta_x = right_wrist.x - left_wrist.x
        wrist_delta_y = right_wrist.y - left_wrist.y
        features["wrist_delta_x"] = wrist_delta_x
        features["wrist_delta_y"] = wrist_delta_y
        features["wrist_distance_xy"] = hypot(wrist_delta_x, wrist_delta_y)

    _ensure_exact_feature_order(
        features,
        expected_names=CANONICAL_DERIVED_FEATURE_NAMES,
        context="derived_features",
    )
    return features


def compute_temporal_features(
    timestamp: FrameTimestamp,
    landmarks: dict[str, LandmarkPoint | None],
    derived_features: dict[str, float],
    *,
    previous: CanonicalFrameFeatures | None = None,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> dict[str, float]:
    """Compute causal temporal features using the immediately previous frame only.

    This function must be used by both offline dataset generation and live runtime
    inference to guarantee train/inference feature parity.
    """

    fill = float(fill_value)
    features = {
        "dt_seconds": fill,
        "left_wrist_rel_vx": fill,
        "left_wrist_rel_vy": fill,
        "right_wrist_rel_vx": fill,
        "right_wrist_rel_vy": fill,
        "wrist_delta_x_v": fill,
        "wrist_delta_y_v": fill,
        "wrist_distance_xy_v": fill,
    }
    if previous is None:
        _ensure_exact_feature_order(
            features,
            expected_names=CANONICAL_TEMPORAL_FEATURE_NAMES,
            context="temporal_features",
        )
        return features

    dt_seconds = timestamp.seconds - previous.timestamp.seconds
    if dt_seconds <= 0.0:
        _ensure_exact_feature_order(
            features,
            expected_names=CANONICAL_TEMPORAL_FEATURE_NAMES,
            context="temporal_features",
        )
        return features
    features["dt_seconds"] = dt_seconds

    current_relative_valid = _relative_positions_available(landmarks, derived_features)
    previous_relative_valid = _relative_positions_available_from_features(previous)
    current_wrist_pair_valid = _wrist_pair_available(landmarks)
    previous_wrist_pair_valid = _wrist_pair_available_from_features(previous)

    if current_relative_valid["left_wrist"] and previous_relative_valid["left_wrist"]:
        features["left_wrist_rel_vx"] = _velocity(
            previous.derived_features["left_wrist_rel_x"],
            derived_features["left_wrist_rel_x"],
            dt_seconds,
        )
        features["left_wrist_rel_vy"] = _velocity(
            previous.derived_features["left_wrist_rel_y"],
            derived_features["left_wrist_rel_y"],
            dt_seconds,
        )

    if current_relative_valid["right_wrist"] and previous_relative_valid["right_wrist"]:
        features["right_wrist_rel_vx"] = _velocity(
            previous.derived_features["right_wrist_rel_x"],
            derived_features["right_wrist_rel_x"],
            dt_seconds,
        )
        features["right_wrist_rel_vy"] = _velocity(
            previous.derived_features["right_wrist_rel_y"],
            derived_features["right_wrist_rel_y"],
            dt_seconds,
        )

    if current_wrist_pair_valid and previous_wrist_pair_valid:
        features["wrist_delta_x_v"] = _velocity(
            previous.derived_features["wrist_delta_x"],
            derived_features["wrist_delta_x"],
            dt_seconds,
        )
        features["wrist_delta_y_v"] = _velocity(
            previous.derived_features["wrist_delta_y"],
            derived_features["wrist_delta_y"],
            dt_seconds,
        )
        features["wrist_distance_xy_v"] = _velocity(
            previous.derived_features["wrist_distance_xy"],
            derived_features["wrist_distance_xy"],
            dt_seconds,
        )

    _ensure_exact_feature_order(
        features,
        expected_names=CANONICAL_TEMPORAL_FEATURE_NAMES,
        context="temporal_features",
    )
    return features


def build_feature_vector(
    raw_features: dict[str, float],
    derived_features: dict[str, float],
    temporal_features: dict[str, float],
) -> tuple[float, ...]:
    """Build the final ordered feature vector used by the model.

    This function must be used by both offline dataset generation and live runtime
    inference to guarantee train/inference feature parity.
    """

    _ensure_exact_feature_order(
        raw_features,
        expected_names=CANONICAL_RAW_FEATURE_NAMES,
        context="raw_features",
    )
    _ensure_exact_feature_order(
        derived_features,
        expected_names=CANONICAL_DERIVED_FEATURE_NAMES,
        context="derived_features",
    )
    _ensure_exact_feature_order(
        temporal_features,
        expected_names=CANONICAL_TEMPORAL_FEATURE_NAMES,
        context="temporal_features",
    )
    combined = raw_features | derived_features | temporal_features
    _ensure_exact_feature_order(
        combined,
        expected_names=FEATURE_NAMES,
        context="combined feature mapping",
    )
    vector = tuple(float(combined[name]) for name in FEATURE_NAMES)
    assert_feature_vector_matches_schema(vector)
    return vector


def build_sequence_window(
    frames: Sequence[TrackerOutput | CanonicalFrameFeatures]
    | Iterable[TrackerOutput | CanonicalFrameFeatures],
    *,
    window_size: int | None = None,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> CanonicalSequenceWindow:
    """Build a fixed-order sequence matrix from tracker outputs or canonical frames.

    This function must be used by both offline dataset generation and live runtime
    inference to guarantee train/inference feature parity.
    """

    canonical_frames: list[CanonicalFrameFeatures] = []
    previous: CanonicalFrameFeatures | None = None
    for item in frames:
        if isinstance(item, CanonicalFrameFeatures):
            canonical_frame = item
        else:
            canonical_frame = extract_canonical_frame_features(
                item,
                previous=previous,
                fill_value=fill_value,
            )
        canonical_frames.append(canonical_frame)
        previous = canonical_frame

    if window_size is not None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero when provided.")
        canonical_frames = canonical_frames[-window_size:]

    rows = [frame.vector for frame in canonical_frames]
    if window_size is not None and len(rows) < window_size:
        pad_row = tuple(float(fill_value) for _ in FEATURE_NAMES)
        rows = ([pad_row] * (window_size - len(rows))) + rows

    return CanonicalSequenceWindow(
        frames=tuple(canonical_frames),
        matrix=tuple(tuple(float(value) for value in row) for row in rows),
        feature_names=FEATURE_NAMES,
        fill_value=float(fill_value),
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_count=FEATURE_COUNT,
    )


def assert_feature_schemas_match(
    expected: CanonicalFeatureSchema,
    actual: CanonicalFeatureSchema,
    *,
    context_expected: str = "expected schema",
    context_actual: str = "actual schema",
) -> None:
    """Raise when two feature schemas differ in version, count, or ordering."""

    differences = compare_feature_schemas(expected, actual)
    if differences:
        joined = "; ".join(differences)
        raise FeatureSchemaError(
            f"{context_expected} and {context_actual} differ: {joined}"
        )


def compare_feature_schemas(
    left: CanonicalFeatureSchema,
    right: CanonicalFeatureSchema,
) -> list[str]:
    """Return a human-readable list of schema differences."""

    differences: list[str] = []
    if left.version != right.version:
        differences.append(
            f"schema_version differs ({left.version!r} vs {right.version!r})"
        )
    if left.feature_count != right.feature_count:
        differences.append(
            f"feature_count differs ({left.feature_count} vs {right.feature_count})"
        )
    if left.feature_names != right.feature_names:
        differences.append("feature_names/order differs")
    return differences


def assert_feature_vector_matches_schema(
    vector: Sequence[float],
    *,
    schema: CanonicalFeatureSchema | None = None,
    context: str = "feature vector",
) -> None:
    """Raise when a feature vector width differs from the authoritative schema."""

    active_schema = schema or get_canonical_feature_schema()
    if len(vector) != active_schema.feature_count:
        raise FeatureSchemaError(
            f"{context} length does not match schema. "
            f"Expected {active_schema.feature_count}, got {len(vector)}."
        )


def compare_feature_vectors(
    left: Sequence[float],
    right: Sequence[float],
    *,
    schema: CanonicalFeatureSchema | None = None,
    abs_tolerance: float = 0.0,
) -> list[str]:
    """Return per-feature differences between two ordered feature vectors."""

    active_schema = schema or get_canonical_feature_schema()
    assert_feature_vector_matches_schema(left, schema=active_schema, context="left vector")
    assert_feature_vector_matches_schema(right, schema=active_schema, context="right vector")
    differences: list[str] = []
    for name, left_value, right_value in zip(
        active_schema.feature_names,
        left,
        right,
        strict=True,
    ):
        if not isclose(float(left_value), float(right_value), abs_tol=abs_tolerance):
            differences.append(f"{name}: {left_value} != {right_value}")
    return differences


def assert_feature_vectors_match(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    schema: CanonicalFeatureSchema | None = None,
    abs_tolerance: float = 0.0,
    context_expected: str = "expected vector",
    context_actual: str = "actual vector",
) -> None:
    """Raise when two vectors differ under the authoritative feature ordering."""

    differences = compare_feature_vectors(
        expected,
        actual,
        schema=schema,
        abs_tolerance=abs_tolerance,
    )
    if differences:
        joined = "; ".join(differences)
        raise FeatureSchemaError(
            f"{context_expected} and {context_actual} differ: {joined}"
        )


def _extract_raw_features(
    landmarks: dict[str, LandmarkPoint | None],
    *,
    fill_value: float,
) -> dict[str, float]:
    fill = float(fill_value)
    features: dict[str, float] = {}
    for name in CANONICAL_LANDMARKS:
        point = landmarks[name]
        if point is None:
            features[f"{name}_x"] = fill
            features[f"{name}_y"] = fill
            features[f"{name}_visibility"] = fill
            continue
        features[f"{name}_x"] = point.x
        features[f"{name}_y"] = point.y
        features[f"{name}_visibility"] = point.visibility
    _ensure_exact_feature_order(
        features,
        expected_names=CANONICAL_RAW_FEATURE_NAMES,
        context="raw_features",
    )
    return features


def _normalize_canonical_landmarks(frame: TrackerOutput) -> dict[str, LandmarkPoint | None]:
    return {name: _normalize_landmark(frame.get(name)) for name in CANONICAL_LANDMARKS}


def _normalize_landmark(point: LandmarkPoint | None) -> LandmarkPoint | None:
    if point is None:
        return None
    return LandmarkPoint(
        x=_clamp_unit_interval(point.x),
        y=_clamp_unit_interval(point.y),
        z=0.0,
        visibility=_clamp_unit_interval(point.visibility),
    )


def _ensure_exact_feature_order(
    mapping: Mapping[str, float],
    *,
    expected_names: tuple[str, ...],
    context: str,
) -> None:
    actual_names = tuple(mapping)
    missing = [name for name in expected_names if name not in mapping]
    extra = [name for name in mapping if name not in expected_names]
    if missing:
        joined = ", ".join(missing)
        raise FeatureSchemaError(f"{context} is missing required features: {joined}.")
    if extra:
        joined = ", ".join(extra)
        raise FeatureSchemaError(f"{context} contains unexpected features: {joined}.")
    if actual_names != expected_names:
        raise FeatureSchemaError(
            f"{context} ordering changed unexpectedly. "
            f"Expected {expected_names}, got {actual_names}."
        )


def _relative_positions_available(
    landmarks: dict[str, LandmarkPoint | None],
    derived_features: Mapping[str, float],
) -> dict[str, bool]:
    shoulders_present = (
        landmarks["left_shoulder"] is not None
        and landmarks["right_shoulder"] is not None
        and derived_features["shoulder_width"] > _SCALE_EPSILON
    )
    return {
        "left_wrist": shoulders_present and landmarks["left_wrist"] is not None,
        "right_wrist": shoulders_present and landmarks["right_wrist"] is not None,
    }


def _relative_positions_available_from_features(
    features: CanonicalFrameFeatures,
) -> dict[str, bool]:
    shoulders_present = (
        features.raw_features["left_shoulder_visibility"] > 0.0
        and features.raw_features["right_shoulder_visibility"] > 0.0
        and features.derived_features["shoulder_width"] > _SCALE_EPSILON
    )
    return {
        "left_wrist": shoulders_present and features.raw_features["left_wrist_visibility"] > 0.0,
        "right_wrist": shoulders_present
        and features.raw_features["right_wrist_visibility"] > 0.0,
    }


def _wrist_pair_available(landmarks: dict[str, LandmarkPoint | None]) -> bool:
    return landmarks["left_wrist"] is not None and landmarks["right_wrist"] is not None


def _wrist_pair_available_from_features(features: CanonicalFrameFeatures) -> bool:
    return (
        features.raw_features["left_wrist_visibility"] > 0.0
        and features.raw_features["right_wrist_visibility"] > 0.0
    )


def _velocity(previous_value: float, current_value: float, dt_seconds: float) -> float:
    return (current_value - previous_value) / dt_seconds


def _clamp_unit_interval(value: float) -> float:
    return min(1.0, max(0.0, float(value)))

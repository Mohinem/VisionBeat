"""Kick/snare classifier model and checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from visionbeat.cnn_model import require_torch, resolve_device
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

GESTURE_CLASSIFIER_MODEL_NAME: Final[str] = "visionbeat_gesture_classifier_cnn"
GESTURE_CLASSIFIER_CHECKPOINT_FORMAT_VERSION: Final[str] = (
    "visionbeat.gesture_classifier_checkpoint.v1"
)
DEFAULT_GESTURE_CLASS_LABELS: Final[tuple[str, ...]] = ("kick", "snare")
_EPSILON: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class VisionBeatGestureClassifierSpec:
    """Runtime-relevant metadata for the kick/snare classifier."""

    feature_count: int
    window_size: int
    hidden_channels: int
    dropout: float
    schema_version: str
    feature_names: tuple[str, ...]
    class_labels: tuple[str, ...] = DEFAULT_GESTURE_CLASS_LABELS
    source_target_name: str = "completion_within_next_k_frames"
    source_horizon_frames: int = 0
    model_name: str = GESTURE_CLASSIFIER_MODEL_NAME
    checkpoint_format_version: str = GESTURE_CLASSIFIER_CHECKPOINT_FORMAT_VERSION

    def __post_init__(self) -> None:
        """Validate feature schema and gesture-class metadata."""
        if self.feature_count <= 0:
            raise ValueError("feature_count must be greater than zero.")
        if self.window_size <= 0:
            raise ValueError("window_size must be greater than zero.")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be greater than zero.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if len(self.feature_names) != self.feature_count:
            raise ValueError("feature_names length must match feature_count.")
        if len(self.class_labels) < 2:
            raise ValueError("class_labels must contain at least two gesture classes.")
        normalized_labels = tuple(label.strip().lower() for label in self.class_labels)
        if any(label == "" for label in normalized_labels):
            raise ValueError("class_labels must not contain empty labels.")
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError("class_labels must be unique.")
        if self.schema_version == FEATURE_SCHEMA_VERSION and self.feature_names != FEATURE_NAMES:
            raise ValueError(
                "feature_names do not match the canonical schema for the active schema_version."
            )

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the expected `(window_size, feature_count)` model input shape."""
        return (self.window_size, self.feature_count)

    def to_checkpoint_metadata(self) -> dict[str, Any]:
        """Serialize the reusable checkpoint metadata."""
        return {
            "checkpoint_format_version": self.checkpoint_format_version,
            "model_name": self.model_name,
            "model_hyperparameters": {
                "feature_count": self.feature_count,
                "window_size": self.window_size,
                "hidden_channels": self.hidden_channels,
                "dropout": self.dropout,
                "class_count": len(self.class_labels),
            },
            "feature_schema": {
                "schema_version": self.schema_version,
                "feature_names": list(self.feature_names),
                "feature_count": self.feature_count,
                "window_size": self.window_size,
                "source_target_name": self.source_target_name,
                "source_horizon_frames": self.source_horizon_frames,
            },
            "gesture_classes": list(self.class_labels),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
    ) -> VisionBeatGestureClassifierSpec:
        """Load model metadata from a checkpoint."""
        metadata = checkpoint.get("model_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Checkpoint is missing gesture classifier model_metadata.")
        format_version = str(
            metadata.get(
                "checkpoint_format_version",
                GESTURE_CLASSIFIER_CHECKPOINT_FORMAT_VERSION,
            )
        )
        if format_version != GESTURE_CLASSIFIER_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint_format_version {format_version!r}.")
        model_name = str(metadata.get("model_name", GESTURE_CLASSIFIER_MODEL_NAME))
        if model_name != GESTURE_CLASSIFIER_MODEL_NAME:
            raise ValueError(f"Unsupported model_name {model_name!r}.")
        model_hyperparameters = metadata.get("model_hyperparameters")
        feature_schema = metadata.get("feature_schema")
        class_labels = metadata.get("gesture_classes")
        if not isinstance(model_hyperparameters, Mapping) or not isinstance(
            feature_schema,
            Mapping,
        ):
            raise ValueError(
                "Checkpoint model_metadata is missing model_hyperparameters or feature_schema."
            )
        if not isinstance(class_labels, list):
            raise ValueError("Checkpoint model_metadata is missing gesture_classes.")
        return cls(
            feature_count=int(model_hyperparameters["feature_count"]),
            window_size=int(model_hyperparameters["window_size"]),
            hidden_channels=int(model_hyperparameters["hidden_channels"]),
            dropout=float(model_hyperparameters.get("dropout", 0.0)),
            schema_version=str(feature_schema["schema_version"]),
            feature_names=tuple(str(name) for name in feature_schema["feature_names"]),
            class_labels=tuple(str(label) for label in class_labels),
            source_target_name=str(
                feature_schema.get("source_target_name", "completion_within_next_k_frames")
            ),
            source_horizon_frames=int(feature_schema.get("source_horizon_frames", 0)),
            model_name=model_name,
            checkpoint_format_version=format_version,
        )


def build_gesture_classifier(nn: Any, spec: VisionBeatGestureClassifierSpec):
    """Build the small temporal CNN used for kick/snare classification."""

    class GestureClassifierCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(
                    spec.feature_count,
                    spec.hidden_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    spec.hidden_channels,
                    spec.hidden_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    spec.hidden_channels,
                    spec.hidden_channels * 2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=spec.dropout),
                nn.Linear(spec.hidden_channels * 2, len(spec.class_labels)),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.features(x)
            x = self.classifier(x)
            return x

    return GestureClassifierCNN()


def build_gesture_classifier_checkpoint_payload(
    *,
    spec: VisionBeatGestureClassifierSpec,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a future-proof checkpoint payload for the gesture classifier."""
    payload = {
        "checkpoint_format_version": spec.checkpoint_format_version,
        "model_name": spec.model_name,
        "model_metadata": spec.to_checkpoint_metadata(),
        "input_shape": spec.input_shape,
        "feature_count": spec.feature_count,
        "window_size": spec.window_size,
        "hidden_channels": spec.hidden_channels,
        "dropout": spec.dropout,
        "schema_version": spec.schema_version,
        "feature_names": list(spec.feature_names),
        "class_labels": list(spec.class_labels),
        "source_target_name": spec.source_target_name,
        "source_horizon_frames": spec.source_horizon_frames,
        "model_state_dict": model_state_dict,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if extra:
        payload.update(extra)
    return payload


def load_gesture_classifier_checkpoint(path: Path, *, torch: Any, device: Any) -> dict[str, Any]:
    """Load a gesture classifier checkpoint with compatibility across PyTorch versions."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain model_state_dict.")
    return checkpoint


def load_gesture_classifier_from_checkpoint(
    *,
    checkpoint_path: Path,
    torch: Any,
    nn: Any,
    device: Any,
    runtime_feature_names: tuple[str, ...],
    runtime_schema_version: str,
    runtime_window_size: int,
) -> tuple[Any, VisionBeatGestureClassifierSpec, dict[str, Any]]:
    """Load a gesture classifier checkpoint and verify runtime compatibility."""
    checkpoint = load_gesture_classifier_checkpoint(
        checkpoint_path,
        torch=torch,
        device=device,
    )
    spec = VisionBeatGestureClassifierSpec.from_checkpoint(checkpoint)
    if spec.feature_names != runtime_feature_names:
        raise ValueError("Runtime feature_names do not match the gesture checkpoint schema.")
    if spec.schema_version != runtime_schema_version:
        raise ValueError("Runtime schema_version does not match the gesture checkpoint.")
    if spec.window_size != runtime_window_size:
        raise ValueError(
            "Runtime window size does not match the gesture checkpoint. "
            f"Expected {spec.window_size}, got {runtime_window_size}."
        )
    model = build_gesture_classifier(nn, spec)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, spec, checkpoint


def multiclass_classification_metrics(
    y_true: Any,
    probabilities: Any,
    *,
    class_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Return aggregate and per-class metrics for a multi-class classifier."""
    true_values = _as_rank1_int64(y_true, name="y_true")
    probability_values = probabilities
    if probability_values is None:
        raise ValueError("probabilities must not be None.")
    try:
        shape = tuple(int(dimension) for dimension in probability_values.shape)
    except AttributeError as exc:  # pragma: no cover - defensive only
        raise ValueError("probabilities must expose a shape attribute.") from exc
    if len(shape) != 2:
        raise ValueError(f"probabilities must be rank 2, got shape {shape}.")
    if shape[0] != true_values.shape[0]:
        raise ValueError(
            f"probabilities has sample count {shape[0]}, expected {true_values.shape[0]}."
        )
    if shape[1] != len(class_labels):
        raise ValueError(
            "probabilities class dimension does not match class_labels. "
            f"Expected {len(class_labels)}, got {shape[1]}."
        )

    if hasattr(probability_values, "detach"):
        probabilities_array = probability_values.detach().cpu().numpy()
    else:
        probabilities_array = probability_values
    predicted_indices = probabilities_array.argmax(axis=1)
    accuracy = float((predicted_indices == true_values).mean()) if true_values.size else 0.0

    confusion_by_class: dict[str, dict[str, int]] = {}
    per_class_metrics: dict[str, dict[str, float | int]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for class_index, class_label in enumerate(class_labels):
        true_positive = int(
            ((true_values == class_index) & (predicted_indices == class_index)).sum()
        )
        false_positive = int(
            ((true_values != class_index) & (predicted_indices == class_index)).sum()
        )
        false_negative = int(
            ((true_values == class_index) & (predicted_indices != class_index)).sum()
        )
        support = int((true_values == class_index).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, _EPSILON)
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        confusion_by_class[class_label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "support": support,
        }
        per_class_metrics[class_label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    class_count = max(len(class_labels), 1)
    return {
        "accuracy": accuracy,
        "sample_count": int(true_values.size),
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
        "predicted_class_counts": {
            class_label: int((predicted_indices == index).sum())
            for index, class_label in enumerate(class_labels)
        },
        "per_class": per_class_metrics,
        "confusion_by_class": confusion_by_class,
    }


def _as_rank1_int64(values: Any, *, name: str):
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else values
    if getattr(array, "ndim", None) != 1:
        raise ValueError(f"{name} must be rank 1.")
    return array.astype("int64", copy=False)


__all__ = [
    "DEFAULT_GESTURE_CLASS_LABELS",
    "GESTURE_CLASSIFIER_CHECKPOINT_FORMAT_VERSION",
    "GESTURE_CLASSIFIER_MODEL_NAME",
    "VisionBeatGestureClassifierSpec",
    "build_gesture_classifier",
    "build_gesture_classifier_checkpoint_payload",
    "load_gesture_classifier_checkpoint",
    "load_gesture_classifier_from_checkpoint",
    "multiclass_classification_metrics",
    "require_torch",
    "resolve_device",
]

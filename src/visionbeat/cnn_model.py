"""Reusable VisionBeat CNN model, metrics, and checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

CNN_MODEL_NAME: Final[str] = "visionbeat_completion_cnn"
CNN_CHECKPOINT_FORMAT_VERSION: Final[str] = "visionbeat.cnn_checkpoint.v1"
_EPSILON: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class VisionBeatCnnSpec:
    """Runtime-relevant model and schema metadata for the completion CNN."""

    feature_count: int
    window_size: int
    hidden_channels: int
    dropout: float
    schema_version: str
    feature_names: tuple[str, ...]
    target_name: str = "completion_frame_binary"
    horizon_frames: int = 0
    model_name: str = CNN_MODEL_NAME
    checkpoint_format_version: str = CNN_CHECKPOINT_FORMAT_VERSION

    def __post_init__(self) -> None:
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
        if self.horizon_frames < 0:
            raise ValueError("horizon_frames must be greater than or equal to zero.")
        if (
            self.target_name
            in {
                "completion_within_next_k_frames",
                "completion_within_last_k_frames",
                "arm_within_next_k_frames",
                "arm_within_last_k_frames",
            }
            and self.horizon_frames <= 0
        ):
            raise ValueError(
                "horizon_frames must be greater than zero for tolerant timing targets."
            )
        if (
            self.schema_version == FEATURE_SCHEMA_VERSION
            and self.feature_names != FEATURE_NAMES
        ):
            raise ValueError(
                "feature_names do not match the canonical schema for the active schema_version."
            )

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.window_size, self.feature_count)

    def to_checkpoint_metadata(self) -> dict[str, Any]:
        """Serialize the reusable metadata stored with checkpoints."""

        return {
            "checkpoint_format_version": self.checkpoint_format_version,
            "model_name": self.model_name,
            "model_hyperparameters": {
                "feature_count": self.feature_count,
                "window_size": self.window_size,
                "hidden_channels": self.hidden_channels,
                "dropout": self.dropout,
            },
            "feature_schema": {
                "schema_version": self.schema_version,
                "feature_names": list(self.feature_names),
                "feature_count": self.feature_count,
                "window_size": self.window_size,
                "target_name": self.target_name,
                "horizon_frames": self.horizon_frames,
            },
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> VisionBeatCnnSpec:
        """Load model metadata from a checkpoint, with legacy fallback support."""

        metadata = checkpoint.get("model_metadata")
        if isinstance(metadata, Mapping):
            format_version = str(
                metadata.get(
                    "checkpoint_format_version",
                    CNN_CHECKPOINT_FORMAT_VERSION,
                )
            )
            if format_version != CNN_CHECKPOINT_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported checkpoint_format_version {format_version!r}."
                )
            model_name = str(metadata.get("model_name", CNN_MODEL_NAME))
            if model_name != CNN_MODEL_NAME:
                raise ValueError(f"Unsupported model_name {model_name!r}.")
            model_hyperparameters = metadata.get("model_hyperparameters")
            feature_schema = metadata.get("feature_schema")
            if not isinstance(model_hyperparameters, Mapping) or not isinstance(
                feature_schema,
                Mapping,
            ):
                raise ValueError(
                    "Checkpoint model_metadata is missing model_hyperparameters or feature_schema."
                )
            return cls(
                feature_count=int(model_hyperparameters["feature_count"]),
                window_size=int(model_hyperparameters["window_size"]),
                hidden_channels=int(model_hyperparameters["hidden_channels"]),
                dropout=float(model_hyperparameters.get("dropout", 0.0)),
                schema_version=str(feature_schema["schema_version"]),
                feature_names=tuple(str(name) for name in feature_schema["feature_names"]),
                target_name=str(feature_schema.get("target_name", "completion_frame_binary")),
                horizon_frames=int(feature_schema.get("horizon_frames", 0)),
                model_name=model_name,
                checkpoint_format_version=format_version,
            )
        return _spec_from_legacy_checkpoint(checkpoint)


def require_torch():
    """Import torch lazily so pure-python tests can run without it."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CNN training or inference. Install it in the active environment, for example:\n"
            "  .venv/bin/pip install torch"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def resolve_device(device: str, torch: Any):
    """Resolve a requested device string into a torch.device."""

    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_completion_cnn(nn: Any, spec: VisionBeatCnnSpec):
    """Build the small baseline CNN using a reusable model spec."""

    class CompletionCNN(nn.Module):
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
                nn.Linear(spec.hidden_channels * 2, 1),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.features(x)
            x = self.classifier(x)
            return x.squeeze(-1)

    return CompletionCNN()


def build_checkpoint_payload(
    *,
    spec: VisionBeatCnnSpec,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a future-proof checkpoint payload with explicit schema metadata."""

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
        "target_name": spec.target_name,
        "horizon_frames": spec.horizon_frames,
        "model_state_dict": model_state_dict,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if extra:
        payload.update(extra)
    return payload


def load_checkpoint(path: Path, *, torch: Any, device: Any) -> dict[str, Any]:
    """Load a checkpoint with compatibility across PyTorch versions."""

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain model_state_dict.")
    return checkpoint


def load_completion_cnn_from_checkpoint(
    *,
    checkpoint_path: Path,
    torch: Any,
    nn: Any,
    device: Any,
    runtime_feature_names: tuple[str, ...],
    runtime_schema_version: str,
    runtime_window_size: int,
    runtime_target_name: str,
    runtime_horizon_frames: int = 0,
):
    """Load a checkpointed model and validate it against runtime schema metadata."""

    checkpoint = load_checkpoint(checkpoint_path, torch=torch, device=device)
    spec = VisionBeatCnnSpec.from_checkpoint(checkpoint)
    validate_runtime_compatibility(
        spec,
        feature_names=runtime_feature_names,
        schema_version=runtime_schema_version,
        window_size=runtime_window_size,
        target_name=runtime_target_name,
        horizon_frames=runtime_horizon_frames,
    )
    model = build_completion_cnn(nn, spec)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, spec, checkpoint


def validate_runtime_compatibility(
    spec: VisionBeatCnnSpec,
    *,
    feature_names: tuple[str, ...],
    schema_version: str,
    window_size: int,
    target_name: str,
    horizon_frames: int = 0,
) -> None:
    """Fail loudly when a checkpoint is used with the wrong runtime schema."""

    if spec.window_size != window_size:
        raise ValueError(
            f"Checkpoint window_size {spec.window_size} does not match runtime window_size {window_size}."
        )
    if spec.feature_count != len(feature_names):
        raise ValueError(
            f"Checkpoint feature_count {spec.feature_count} does not match runtime feature_count {len(feature_names)}."
        )
    if spec.schema_version != schema_version:
        raise ValueError(
            f"Checkpoint schema_version {spec.schema_version!r} does not match runtime schema_version {schema_version!r}."
        )
    if spec.target_name != target_name:
        raise ValueError(
            f"Checkpoint target_name {spec.target_name!r} does not match runtime target_name {target_name!r}."
        )
    if (
        spec.target_name
        in {
            "completion_within_next_k_frames",
            "completion_within_last_k_frames",
            "arm_within_next_k_frames",
            "arm_within_last_k_frames",
        }
        and spec.horizon_frames != horizon_frames
    ):
        raise ValueError(
            "Checkpoint horizon_frames "
            f"{spec.horizon_frames} does not match runtime horizon_frames {horizon_frames}."
        )
    if spec.feature_names != tuple(feature_names):
        raise ValueError("Checkpoint feature_names do not match the runtime feature_names.")


def infer_hidden_channels_from_state_dict(
    *,
    state_dict: Mapping[str, Any],
    feature_count: int,
) -> int:
    """Infer the baseline CNN width directly from the saved weights."""

    conv1_weight = state_dict.get("features.0.weight")
    linear_weight = state_dict.get("classifier.2.weight")
    if conv1_weight is None or linear_weight is None:
        raise ValueError(
            "Checkpoint state_dict is missing baseline CNN layers needed for inference."
        )
    if int(conv1_weight.shape[1]) != feature_count:
        raise ValueError(
            f"Checkpoint expects {int(conv1_weight.shape[1])} input features, got {feature_count}."
        )
    hidden_channels = int(conv1_weight.shape[0])
    if int(linear_weight.shape[1]) != hidden_channels * 2:
        raise ValueError("Checkpoint classifier width does not match the baseline CNN layout.")
    return hidden_channels


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, int | float | None]:
    """Compute binary classification metrics from probabilities and a threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")
    y_pred = (y_score >= threshold).astype(np.int64, copy=False)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, _EPSILON)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    predicted_positive_count = tp + fp
    predicted_positive_rate = predicted_positive_count / max(tp + tn + fp + fn, 1)
    roc_auc = binary_roc_auc(y_true, y_score)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "detected_positive_count": tp,
        "missed_positive_count": fn,
        "positive_detection_rate": recall,
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": predicted_positive_rate,
    }


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute ROC-AUC for binary labels without adding sklearn."""

    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_count = int(np.sum(positive_mask))
    negative_count = int(np.sum(negative_mask))
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[positive_mask].sum())
    auc = (
        positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)
    return float(auc)


def format_optional_metric(value: int | float | None) -> str:
    """Format optional scalar metrics such as ROC-AUC for logs."""

    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _spec_from_legacy_checkpoint(checkpoint: Mapping[str, Any]) -> VisionBeatCnnSpec:
    """Load older checkpoints created before model_metadata existed."""

    input_shape = checkpoint.get("input_shape")
    if input_shape is None or len(input_shape) != 2:
        raise ValueError("Legacy checkpoint is missing a valid input_shape.")
    window_size = int(input_shape[0])
    feature_count = int(input_shape[1])
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Legacy checkpoint is missing model_state_dict.")
    feature_names = checkpoint.get("feature_names", FEATURE_NAMES)
    return VisionBeatCnnSpec(
        feature_count=feature_count,
        window_size=window_size,
        hidden_channels=infer_hidden_channels_from_state_dict(
            state_dict=state_dict,
            feature_count=feature_count,
        ),
        dropout=float(checkpoint.get("dropout", 0.0)),
        schema_version=str(checkpoint.get("schema_version", FEATURE_SCHEMA_VERSION)),
        feature_names=tuple(str(name) for name in feature_names),
        target_name=str(checkpoint.get("target_name", "completion_frame_binary")),
        horizon_frames=int(checkpoint.get("horizon_frames", 0)),
    )

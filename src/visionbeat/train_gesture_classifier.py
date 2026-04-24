"""Train a kick/snare classifier on positive timing windows."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from visionbeat.gesture_classifier import (
    DEFAULT_GESTURE_CLASS_LABELS,
    VisionBeatGestureClassifierSpec,
    build_gesture_classifier,
    build_gesture_classifier_checkpoint_payload,
    multiclass_classification_metrics,
    require_torch,
    resolve_device,
)
from visionbeat.train_cnn import CombinedDataset, combine_archives, load_archive, split_dataset

_RUN_DIR_PREFIX: Final[str] = "visionbeat_gesture_classifier_run_"


@dataclass(frozen=True, slots=True)
class GestureClassificationDataset:
    """One filtered split used for kick/snare training or validation."""

    X: np.ndarray
    y: np.ndarray
    recording_ids: np.ndarray
    window_end_frame_indices: np.ndarray
    window_end_timestamps_seconds: np.ndarray
    gesture_labels: np.ndarray
    class_labels: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        """Return the number of samples in the filtered split."""
        return int(self.X.shape[0])


@dataclass(frozen=True, slots=True)
class GestureClassStats:
    """Simple class-count summary for the kick/snare classifier."""

    total_count: int
    counts_by_label: dict[str, int]

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-friendly class-count mapping."""
        return {"total_count": self.total_count, **self.counts_by_label}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for kick/snare classifier training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a second-stage kick/snare classifier on positive VisionBeat timing windows."
        )
    )
    parser.add_argument(
        "archives",
        nargs="+",
        help="Trusted NPZ archives produced by visionbeat.prepare_training_data.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gesture_classifier",
        help=(
            "Base output directory. If the final path component does not already look "
            "like a run directory, the script creates the next numbered "
            "visionbeat_gesture_classifier_run_### folder under it."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs. Default: 15.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size. Default: 256.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Default: 1e-3.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default: 1e-4.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate in the classifier head. Default: 0.2.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Base channel width for the CNN. Default: 64.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of grouped local gesture neighborhoods held out for validation "
            "inside the validation recording. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--validation-recording-id",
        default="",
        help="Recording id that should supply the grouped validation holdout.",
    )
    parser.add_argument(
        "--holdout-recording-id",
        action="append",
        default=[],
        help=(
            "Optional recording id to place entirely in validation. Repeatable. "
            "This mirrors the timing-model holdout behavior."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. Default: auto.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count. Default: 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility. Default: 7.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for kick/snare classifier training."""
    args = parse_args(argv)
    try:
        if args.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if args.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if args.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        if args.weight_decay < 0.0:
            raise ValueError("weight_decay must be greater than or equal to zero.")
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if args.hidden_channels <= 0:
            raise ValueError("hidden_channels must be greater than zero.")
        if args.num_workers < 0:
            raise ValueError("num_workers must be greater than or equal to zero.")

        archives = tuple(load_archive(Path(path)) for path in args.archives)
        dataset = combine_archives(archives)
        split = split_dataset(
            dataset,
            validation_fraction=args.validation_fraction,
            validation_recording_id=args.validation_recording_id,
            holdout_recording_ids=tuple(args.holdout_recording_id),
        )
        run_dir = _prepare_run_directory(Path(args.output_dir))
        result = train_gesture_classifier(
            dataset=dataset,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            output_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            config=_build_run_config(
                args=args,
                dataset=dataset,
                split_policy=split.policy,
                validation_recording_id=split.validation_recording_id,
                output_dir=run_dir,
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print("Training complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Best checkpoint: {result['best_checkpoint_path']}")
    print(f"Config: {result['config_path']}")
    print(f"Epoch metrics: {result['epoch_metrics_path']}")
    print(f"Evaluation report: {result['evaluation_report_path']}")
    print(f"Best validation loss: {result['best_val_loss']:.6f}")


def prepare_gesture_classification_dataset(
    dataset: CombinedDataset,
    *,
    indices: np.ndarray,
    class_labels: tuple[str, ...] = DEFAULT_GESTURE_CLASS_LABELS,
) -> GestureClassificationDataset:
    """Filter one split down to labeled positive windows for gesture classification."""
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    ordered_indices = np.asarray(indices, dtype=np.int64)
    if ordered_indices.ndim != 1:
        raise ValueError("indices must be rank 1.")

    selected_positions: list[int] = []
    encoded_labels: list[int] = []
    normalized_gesture_labels: list[str] = []
    for dataset_index in ordered_indices.tolist():
        binary_target = int(dataset.y[dataset_index])
        gesture_label = str(dataset.target_gesture_labels[dataset_index]).strip().lower()
        if binary_target == 0:
            continue
        if gesture_label == "":
            raise ValueError(
                "Positive timing window is missing target_gesture_labels. "
                f"Dataset index: {dataset_index}."
            )
        if gesture_label not in class_to_index:
            raise ValueError(
                f"Unsupported positive gesture label {gesture_label!r}. "
                f"Expected one of {class_labels}."
            )
        selected_positions.append(dataset_index)
        encoded_labels.append(class_to_index[gesture_label])
        normalized_gesture_labels.append(gesture_label)

    if not selected_positions:
        raise ValueError("Split contains no labeled positive windows for gesture classification.")

    selected_indices = np.asarray(selected_positions, dtype=np.int64)
    return GestureClassificationDataset(
        X=dataset.X[selected_indices].astype(np.float32, copy=False),
        y=np.asarray(encoded_labels, dtype=np.int64),
        recording_ids=dataset.recording_ids[selected_indices],
        window_end_frame_indices=dataset.window_end_frame_indices[selected_indices],
        window_end_timestamps_seconds=dataset.window_end_timestamps_seconds[selected_indices],
        gesture_labels=np.asarray(normalized_gesture_labels, dtype="<U16"),
        class_labels=class_labels,
    )


def summarize_gesture_classes(
    y: np.ndarray,
    *,
    class_labels: tuple[str, ...],
) -> GestureClassStats:
    """Summarize class counts for one classifier split."""
    values = np.asarray(y, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"Expected rank-1 class targets, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("Gesture classification split contains zero samples.")
    counts_by_label = {
        label: int(np.sum(values == index))
        for index, label in enumerate(class_labels)
    }
    if any(count <= 0 for count in counts_by_label.values()):
        raise ValueError(
            "Gesture classification split is missing one or more required classes: "
            f"{counts_by_label}."
        )
    return GestureClassStats(total_count=int(values.size), counts_by_label=counts_by_label)


def format_gesture_class_stats(name: str, stats: GestureClassStats) -> str:
    """Format one class-distribution line for stdout logging."""
    parts = [f"{label}={count}" for label, count in stats.counts_by_label.items()]
    return f"{name}: samples={stats.total_count} " + " ".join(parts)


def train_gesture_classifier(
    *,
    dataset: CombinedDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_channels: int,
    seed: int,
    device: str,
    num_workers: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train the kick/snare classifier and persist its artifacts."""
    class_labels = DEFAULT_GESTURE_CLASS_LABELS
    train_dataset = prepare_gesture_classification_dataset(
        dataset,
        indices=train_indices,
        class_labels=class_labels,
    )
    validation_dataset = prepare_gesture_classification_dataset(
        dataset,
        indices=validation_indices,
        class_labels=class_labels,
    )
    train_stats = summarize_gesture_classes(train_dataset.y, class_labels=class_labels)
    validation_stats = summarize_gesture_classes(validation_dataset.y, class_labels=class_labels)
    print("Gesture class distribution:")
    print(f"  {format_gesture_class_stats('train', train_stats)}")
    print(f"  {format_gesture_class_stats('validation', validation_stats)}")

    torch, nn, DataLoader, TensorDataset = require_torch()
    _set_seed(seed=seed, torch=torch)
    resolved_device = resolve_device(device, torch)
    class_weights = _build_class_weights(train_dataset.y, class_labels=class_labels)
    print(
        "Class weighting: CrossEntropyLoss with weights "
        + ", ".join(
            f"{label}={weight:.4f}"
            for label, weight in zip(class_labels, class_weights.tolist(), strict=True)
        )
    )

    train_tensor = torch.from_numpy(train_dataset.X)
    train_targets = torch.from_numpy(train_dataset.y).long()
    validation_tensor = torch.from_numpy(validation_dataset.X)
    validation_targets = torch.from_numpy(validation_dataset.y).long()
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_tensor, validation_targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model_spec = VisionBeatGestureClassifierSpec(
        feature_count=dataset.feature_count,
        window_size=dataset.window_size,
        hidden_channels=hidden_channels,
        dropout=dropout,
        schema_version=dataset.schema_version,
        feature_names=dataset.feature_names,
        class_labels=class_labels,
        source_target_name=dataset.target_name,
        source_horizon_frames=dataset.horizon_frames,
    )
    model = build_gesture_classifier(nn, model_spec)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=resolved_device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    checkpoints_dir = output_dir / "checkpoints"
    config_dir = output_dir / "config"
    reports_dir = output_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    epoch_history: list[dict[str, Any]] = []
    checkpoint_path = checkpoints_dir / "best_model.pt"
    for epoch_index in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_sample_count = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_size_actual = int(batch_X.shape[0])
            train_loss_total += float(loss.item()) * batch_size_actual
            train_sample_count += batch_size_actual

        train_loss = train_loss_total / max(train_sample_count, 1)
        validation_loss, validation_probabilities, validation_predictions = _evaluate_classifier(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=resolved_device,
            torch=torch,
        )
        validation_metrics = multiclass_classification_metrics(
            validation_targets.cpu().numpy(),
            validation_probabilities,
            class_labels=class_labels,
        )
        epoch_record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "val_metrics": validation_metrics,
        }
        epoch_history.append(epoch_record)
        print(
            f"Epoch {epoch_index + 1:02d}/{epochs:02d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"val_accuracy={float(validation_metrics['accuracy']):.4f} "
            f"val_macro_f1={float(validation_metrics['macro_f1']):.4f} "
            f"kick_f1={float(validation_metrics['per_class']['kick']['f1']):.4f} "
            f"snare_f1={float(validation_metrics['per_class']['snare']['f1']):.4f}"
        )
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch_index + 1
            best_metrics = validation_metrics
            checkpoint_payload = build_gesture_classifier_checkpoint_payload(
                spec=model_spec,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                extra={
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_metrics": best_metrics,
                },
            )
            torch.save(checkpoint_payload, checkpoint_path)

    if best_metrics is None:
        raise RuntimeError("Training did not produce any validation metrics.")

    epoch_metrics_path = reports_dir / "epoch_metrics.json"
    epoch_metrics_path.write_text(
        json.dumps(epoch_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation_report_path = reports_dir / "evaluation_report.json"
    evaluation_report = {
        "class_labels": list(class_labels),
        "train_class_distribution": train_stats.as_dict(),
        "validation_class_distribution": validation_stats.as_dict(),
        "class_weights": {
            label: float(weight)
            for label, weight in zip(class_labels, class_weights.tolist(), strict=True)
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_metrics,
        "epoch_metrics_path": str(epoch_metrics_path),
        "best_checkpoint_path": str(checkpoint_path),
    }
    evaluation_report_path.write_text(
        json.dumps(evaluation_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_dir": str(output_dir),
        "config_path": str(config_path),
        "best_checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "epoch_metrics_path": str(epoch_metrics_path),
        "evaluation_report_path": str(evaluation_report_path),
    }


def _prepare_run_directory(base_path: Path) -> Path:
    """Create a clear run directory without overwriting existing artifacts."""
    if base_path.name.startswith(_RUN_DIR_PREFIX):
        if base_path.exists() and any(base_path.iterdir()):
            raise ValueError(f"Run directory already exists and is not empty: {base_path}")
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    base_path.mkdir(parents=True, exist_ok=True)
    next_index = 1
    for child in base_path.iterdir():
        if not child.is_dir() or not child.name.startswith(_RUN_DIR_PREFIX):
            continue
        suffix = child.name[len(_RUN_DIR_PREFIX) :]
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    run_dir = base_path / f"{_RUN_DIR_PREFIX}{next_index:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _build_run_config(
    *,
    args: argparse.Namespace,
    dataset: CombinedDataset,
    split_policy: str,
    validation_recording_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a serializable record of the configuration used for one run."""
    return {
        "archives": [str(path) for path in args.archives],
        "output_dir": str(output_dir),
        "schema_version": dataset.schema_version,
        "feature_count": dataset.feature_count,
        "input_shape": list(dataset.input_shape),
        "source_target_name": dataset.target_name,
        "source_horizon_frames": dataset.horizon_frames,
        "class_labels": list(DEFAULT_GESTURE_CLASS_LABELS),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_channels": args.hidden_channels,
            "seed": args.seed,
            "device": args.device,
            "num_workers": args.num_workers,
        },
        "split": {
            "validation_fraction": args.validation_fraction,
            "validation_recording_id": validation_recording_id,
            "holdout_recording_ids": list(args.holdout_recording_id),
            "policy": split_policy,
        },
    }


def _build_class_weights(
    y: np.ndarray,
    *,
    class_labels: tuple[str, ...],
) -> np.ndarray:
    """Build simple inverse-frequency class weights for CrossEntropyLoss."""
    stats = summarize_gesture_classes(y, class_labels=class_labels)
    counts = np.asarray(
        [stats.counts_by_label[label] for label in class_labels],
        dtype=np.float32,
    )
    max_count = float(np.max(counts))
    return max_count / counts


def _evaluate_classifier(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    torch: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the gesture classifier on one split."""
    model.eval()
    loss_total = 0.0
    sample_count = 0
    probability_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            batch_size_actual = int(batch_X.shape[0])
            loss_total += float(loss.item()) * batch_size_actual
            sample_count += batch_size_actual
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probability_batches.append(probabilities)
            prediction_batches.append(np.argmax(probabilities, axis=1).astype(np.int64))

    if not probability_batches:
        raise RuntimeError("Validation loader produced zero batches.")
    return (
        loss_total / max(sample_count, 1),
        np.concatenate(probability_batches, axis=0),
        np.concatenate(prediction_batches, axis=0),
    )


def _set_seed(*, seed: int, torch: Any) -> None:
    """Set deterministic seeds for the gesture classifier run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "GestureClassificationDataset",
    "GestureClassStats",
    "format_gesture_class_stats",
    "prepare_gesture_classification_dataset",
    "summarize_gesture_classes",
    "train_gesture_classifier",
]


if __name__ == "__main__":
    main()

"""Live shadow-mode predictive inference using the locked timing front-end."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from visionbeat.cnn_model import (
    VisionBeatCnnSpec,
    build_completion_cnn,
    load_checkpoint,
    require_torch,
    resolve_device,
    validate_runtime_compatibility,
)
from visionbeat.config import ConfigError, PredictiveConfig
from visionbeat.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, CanonicalSequenceWindow
from visionbeat.gesture_classifier import (
    VisionBeatGestureClassifierSpec,
    build_gesture_classifier,
    load_gesture_classifier_checkpoint,
)
from visionbeat.models import FrameTimestamp, GestureEvent, GestureType


@dataclass(frozen=True, slots=True)
class ShadowPredictionEvent:
    """One accepted shadow-mode trigger and its kick/snare prediction."""

    frame_index: int
    timestamp: FrameTimestamp
    timing_probability: float
    threshold: float
    run_length: int
    gesture: GestureType
    gesture_confidence: float
    class_probabilities: dict[str, float]
    heuristic_triggered_on_peak_frame: bool
    heuristic_gesture_types_on_peak_frame: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the shadow trigger for logs and session bundles."""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.to_dict(),
            "timing_probability": self.timing_probability,
            "threshold": self.threshold,
            "run_length": self.run_length,
            "gesture": self.gesture.value,
            "gesture_confidence": self.gesture_confidence,
            "class_probabilities": dict(self.class_probabilities),
            "heuristic_triggered_on_peak_frame": self.heuristic_triggered_on_peak_frame,
            "heuristic_gesture_types_on_peak_frame": list(
                self.heuristic_gesture_types_on_peak_frame
            ),
        }


@dataclass(frozen=True, slots=True)
class PredictiveStatus:
    """Latest per-frame predictive telemetry exposed to the live HUD."""

    available_window_frames: int
    required_window_size: int
    threshold: float
    timing_probability: float | None = None
    predicted_gesture: GestureType | None = None
    predicted_gesture_confidence: float | None = None
    class_probabilities: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Render a compact status line for the live debug HUD."""
        if self.available_window_frames < self.required_window_size:
            return f"warmup {self.available_window_frames}/{self.required_window_size}"
        if self.timing_probability is None:
            return f"p=--/{self.threshold:.2f} top=--"
        if self.predicted_gesture is None or self.predicted_gesture_confidence is None:
            return f"p={self.timing_probability:.2f}/{self.threshold:.2f} top=--"
        return (
            f"p={self.timing_probability:.2f}/{self.threshold:.2f} "
            f"top={self.predicted_gesture.value} {self.predicted_gesture_confidence:.2f}"
        )


@dataclass(frozen=True, slots=True)
class _RunPeak:
    """The strongest above-threshold timing window inside one local run."""

    frame_index: int
    timestamp_seconds: float
    timing_probability: float
    run_length: int
    window_matrix: np.ndarray
    heuristic_gesture_types_on_peak_frame: tuple[str, ...]


class StreamingTriggerDecoder:
    """Streaming approximation of the offline trigger decoder for shadow-mode logging."""

    def __init__(
        self,
        *,
        threshold: float,
        cooldown_frames: int,
        max_gap_frames: int,
    ) -> None:
        """Initialize the streaming decoder with the locked offline decoder parameters."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0].")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be greater than or equal to zero.")
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames must be greater than or equal to zero.")
        self.threshold = float(threshold)
        self.cooldown_frames = int(cooldown_frames)
        self.max_gap_frames = int(max_gap_frames)
        self._active_run: _RunPeak | None = None
        self._last_positive_frame_index: int | None = None
        self._cooldown_candidate: _RunPeak | None = None

    def update(
        self,
        *,
        frame_index: int,
        timestamp_seconds: float,
        timing_probability: float,
        window_matrix: np.ndarray,
        heuristic_gesture_types: tuple[str, ...],
    ) -> tuple[_RunPeak, ...]:
        """Consume one timing probability and emit any accepted shadow triggers."""
        emitted: list[_RunPeak] = []
        emitted.extend(self._release_ready_candidates(frame_index=frame_index))

        if timing_probability >= self.threshold:
            candidate = _RunPeak(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                timing_probability=timing_probability,
                run_length=1,
                window_matrix=np.array(window_matrix, dtype=np.float32, copy=True),
                heuristic_gesture_types_on_peak_frame=heuristic_gesture_types,
            )
            if self._active_run is None:
                self._active_run = candidate
            else:
                if self._last_positive_frame_index is None:
                    gap = 0
                else:
                    gap = frame_index - self._last_positive_frame_index
                if gap <= self.max_gap_frames:
                    updated = replace(
                        self._active_run,
                        run_length=self._active_run.run_length + 1,
                    )
                    if timing_probability > updated.timing_probability:
                        updated = replace(
                            candidate,
                            run_length=updated.run_length,
                        )
                    self._active_run = updated
                else:
                    emitted.extend(self._enqueue_candidate(self._active_run))
                    self._active_run = candidate
            self._last_positive_frame_index = frame_index
            return tuple(emitted)

        if self._active_run is not None:
            emitted.extend(self._enqueue_candidate(self._active_run))
            self._active_run = None
            self._last_positive_frame_index = None
        return tuple(emitted)

    def flush(self) -> tuple[_RunPeak, ...]:
        """Finalize any buffered shadow trigger candidates."""
        emitted: list[_RunPeak] = []
        if self._active_run is not None:
            emitted.extend(self._enqueue_candidate(self._active_run))
            self._active_run = None
            self._last_positive_frame_index = None
        if self._cooldown_candidate is not None:
            emitted.append(self._cooldown_candidate)
            self._cooldown_candidate = None
        return tuple(emitted)

    def _release_ready_candidates(self, *, frame_index: int) -> list[_RunPeak]:
        emitted: list[_RunPeak] = []
        if self._cooldown_candidate is None:
            return emitted
        if frame_index - self._cooldown_candidate.frame_index > self.cooldown_frames:
            emitted.append(self._cooldown_candidate)
            self._cooldown_candidate = None
        return emitted

    def _enqueue_candidate(self, candidate: _RunPeak) -> list[_RunPeak]:
        emitted: list[_RunPeak] = []
        if self._cooldown_candidate is None:
            self._cooldown_candidate = candidate
            return emitted
        gap = candidate.frame_index - self._cooldown_candidate.frame_index
        if gap > self.cooldown_frames:
            emitted.append(self._cooldown_candidate)
            self._cooldown_candidate = candidate
            return emitted
        if candidate.timing_probability > self._cooldown_candidate.timing_probability:
            self._cooldown_candidate = candidate
        return emitted


@dataclass(frozen=True, slots=True)
class _PrimaryPeak:
    """Peak candidate tracked by the live primary decoder."""

    frame_index: int
    timestamp_seconds: float
    arm_timing_probability: float
    timing_probability: float
    run_length: int
    gesture: GestureType
    gesture_confidence: float
    class_probabilities: dict[str, float]


class PrimaryTriggerDecoder:
    """Causal live decoder tuned for primary-mode firing from predictive peaks."""

    def __init__(
        self,
        *,
        threshold: float,
        cooldown_frames: int,
        max_gap_frames: int,
        horizon_frames: int,
        arm_threshold: float | None = None,
        low_threshold: float | None = None,
        min_peak_drop: float | None = None,
        min_peak_gain: float | None = None,
        min_gesture_confidence: float = 0.6,
        peak_plateau_tolerance: float = 0.02,
    ) -> None:
        """Initialize a live decoder that fires on local peaks instead of run end."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0].")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be greater than or equal to zero.")
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames must be greater than or equal to zero.")
        if horizon_frames <= 0:
            raise ValueError("horizon_frames must be greater than zero.")
        if not 0.0 <= min_gesture_confidence <= 1.0:
            raise ValueError("min_gesture_confidence must be in [0.0, 1.0].")
        if peak_plateau_tolerance < 0.0:
            raise ValueError("peak_plateau_tolerance must be greater than or equal to zero.")
        self.threshold = float(threshold)
        self.cooldown_frames = int(cooldown_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.horizon_frames = int(horizon_frames)
        resolved_arm_threshold = (
            max(0.0, min(self.threshold, max(0.35, self.threshold - 0.08)))
            if arm_threshold is None
            else float(arm_threshold)
        )
        if not 0.0 <= resolved_arm_threshold <= self.threshold:
            raise ValueError("arm_threshold must be in [0.0, threshold].")
        self.arm_threshold = resolved_arm_threshold
        resolved_low_threshold = (
            max(0.0, min(self.arm_threshold - 0.08, self.arm_threshold * 0.85))
            if low_threshold is None
            else float(low_threshold)
        )
        if not 0.0 <= resolved_low_threshold < self.arm_threshold:
            raise ValueError("low_threshold must be in [0.0, arm_threshold).")
        self.low_threshold = resolved_low_threshold
        resolved_min_peak_drop = (
            max(0.03, min(0.08, self.threshold * 0.08))
            if min_peak_drop is None
            else float(min_peak_drop)
        )
        if resolved_min_peak_drop <= 0.0:
            raise ValueError("min_peak_drop must be greater than zero.")
        self.min_peak_drop = resolved_min_peak_drop
        resolved_min_peak_gain = (
            max(0.03, min(0.10, self.threshold * 0.08))
            if min_peak_gain is None
            else float(min_peak_gain)
        )
        if resolved_min_peak_gain <= 0.0:
            raise ValueError("min_peak_gain must be greater than zero.")
        self.min_peak_gain = resolved_min_peak_gain
        self.min_gesture_confidence = float(min_gesture_confidence)
        self.peak_plateau_tolerance = float(peak_plateau_tolerance)
        self._active_peak: _PrimaryPeak | None = None
        self._last_frame_index: int | None = None
        self._previous_probability: float | None = None
        self._frames_since_peak_update = 0
        self._refractory_until_frame_index: int | None = None
        self._awaiting_reset = False

    def update(
        self,
        *,
        frame_index: int,
        timestamp_seconds: float,
        timing_probability: float,
        predicted_gesture: GestureType,
        predicted_gesture_confidence: float,
        class_probabilities: dict[str, float],
    ) -> tuple[_PrimaryPeak, ...]:
        """Consume one live timing score and emit a trigger on a confirmed local peak."""
        previous_probability = self._previous_probability
        self._previous_probability = float(timing_probability)
        emitted: list[_PrimaryPeak] = []

        if (
            self._last_frame_index is not None
            and frame_index - self._last_frame_index > self.max_gap_frames + 1
        ):
            self._reset_active_peak()
        self._last_frame_index = frame_index

        if self._refractory_until_frame_index is not None:
            if frame_index <= self._refractory_until_frame_index:
                return ()
            self._refractory_until_frame_index = None

        if self._awaiting_reset:
            if timing_probability <= self.low_threshold:
                self._awaiting_reset = False
            return ()

        candidate = _PrimaryPeak(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            arm_timing_probability=float(timing_probability),
            timing_probability=float(timing_probability),
            run_length=1,
            gesture=predicted_gesture,
            gesture_confidence=float(predicted_gesture_confidence),
            class_probabilities=dict(class_probabilities),
        )
        if self._active_peak is None:
            if self._should_arm(
                timing_probability=float(timing_probability),
                previous_probability=previous_probability,
                gesture_confidence=float(predicted_gesture_confidence),
            ):
                self._active_peak = candidate
                self._frames_since_peak_update = 0
            return ()

        updated_peak = self._advance_active_peak(
            candidate=candidate,
            timing_probability=float(timing_probability),
        )
        peak_gain = updated_peak.timing_probability - updated_peak.arm_timing_probability
        if (
            self._frames_since_peak_update >= self.horizon_frames
            and timing_probability >= self.arm_threshold
        ):
            self._reset_active_peak()
            self._awaiting_reset = True
            return ()

        if self._should_fire(
            peak=updated_peak,
            timing_probability=float(timing_probability),
            previous_probability=previous_probability,
        ):
            emitted.append(updated_peak)
            self._reset_active_peak()
            self._refractory_until_frame_index = frame_index + self.cooldown_frames
            return tuple(emitted)

        if timing_probability <= self.low_threshold and self._frames_since_peak_update > 0:
            if self._peak_is_actionable(updated_peak) and peak_gain >= self.min_peak_gain * 0.5:
                emitted.append(updated_peak)
                self._reset_active_peak()
                self._refractory_until_frame_index = frame_index + self.cooldown_frames
                return tuple(emitted)
            self._reset_active_peak()
        return ()

    def flush(self) -> tuple[_PrimaryPeak, ...]:
        """Drop any buffered state without converting stale peaks into triggers."""
        self._reset_active_peak()
        self._refractory_until_frame_index = None
        self._awaiting_reset = False
        return ()

    def _should_arm(
        self,
        *,
        timing_probability: float,
        previous_probability: float | None,
        gesture_confidence: float,
    ) -> bool:
        if timing_probability < self.arm_threshold:
            return False
        if gesture_confidence < self.min_gesture_confidence:
            return False
        if previous_probability is None:
            return True
        return timing_probability >= previous_probability - self.peak_plateau_tolerance

    def _advance_active_peak(
        self,
        *,
        candidate: _PrimaryPeak,
        timing_probability: float,
    ) -> _PrimaryPeak:
        assert self._active_peak is not None
        updated_peak = replace(self._active_peak, run_length=self._active_peak.run_length + 1)
        if timing_probability > self._active_peak.timing_probability + self.peak_plateau_tolerance:
            updated_peak = replace(
                candidate,
                arm_timing_probability=self._active_peak.arm_timing_probability,
                run_length=updated_peak.run_length,
            )
            self._frames_since_peak_update = 0
        else:
            self._frames_since_peak_update += 1
        self._active_peak = updated_peak
        return updated_peak

    def _should_fire(
        self,
        *,
        peak: _PrimaryPeak,
        timing_probability: float,
        previous_probability: float | None,
    ) -> bool:
        if not self._peak_is_actionable(peak):
            return False
        peak_drop = peak.timing_probability - timing_probability
        if peak_drop < self.min_peak_drop:
            return False
        if previous_probability is not None and timing_probability > previous_probability:
            return False
        return True

    def _peak_is_actionable(self, peak: _PrimaryPeak) -> bool:
        if peak.gesture_confidence < self.min_gesture_confidence:
            return False
        required_peak_probability = max(
            self.arm_threshold + self.min_peak_gain,
            self.threshold - self.min_peak_drop,
        )
        return peak.timing_probability >= required_peak_probability

    def _reset_active_peak(self) -> None:
        self._active_peak = None
        self._frames_since_peak_update = 0


class CompletionTriggerDecoder:
    """Minimal learned decoder for completion-aligned timing checkpoints."""

    def __init__(
        self,
        *,
        threshold: float,
        cooldown_frames: int,
        low_threshold: float | None = None,
        min_gesture_confidence: float = 0.6,
    ) -> None:
        """Initialize a threshold-crossing decoder for learned completion models."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0].")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be greater than or equal to zero.")
        if not 0.0 <= min_gesture_confidence <= 1.0:
            raise ValueError("min_gesture_confidence must be in [0.0, 1.0].")
        self.threshold = float(threshold)
        self.cooldown_frames = int(cooldown_frames)
        resolved_low_threshold = (
            max(0.0, min(self.threshold - 0.1, self.threshold * 0.85))
            if low_threshold is None
            else float(low_threshold)
        )
        if not 0.0 <= resolved_low_threshold < self.threshold:
            raise ValueError("low_threshold must be in [0.0, threshold).")
        self.low_threshold = resolved_low_threshold
        self.min_gesture_confidence = float(min_gesture_confidence)
        self._refractory_until_frame_index: int | None = None
        self._awaiting_reset = False

    def update(
        self,
        *,
        frame_index: int,
        timestamp_seconds: float,
        timing_probability: float,
        predicted_gesture: GestureType,
        predicted_gesture_confidence: float,
        class_probabilities: dict[str, float],
    ) -> tuple[_PrimaryPeak, ...]:
        """Emit one trigger when a learned completion score crosses threshold."""
        if self._refractory_until_frame_index is not None:
            if frame_index <= self._refractory_until_frame_index:
                return ()
            self._refractory_until_frame_index = None
        if self._awaiting_reset:
            if timing_probability <= self.low_threshold:
                self._awaiting_reset = False
            return ()
        if timing_probability < self.threshold:
            return ()
        if predicted_gesture_confidence < self.min_gesture_confidence:
            return ()
        self._refractory_until_frame_index = frame_index + self.cooldown_frames
        self._awaiting_reset = True
        return (
            _PrimaryPeak(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                arm_timing_probability=float(timing_probability),
                timing_probability=float(timing_probability),
                run_length=1,
                gesture=predicted_gesture,
                gesture_confidence=float(predicted_gesture_confidence),
                class_probabilities=dict(class_probabilities),
            ),
        )

    def flush(self) -> tuple[_PrimaryPeak, ...]:
        """Drop any buffered completion-decoder state without emitting."""
        self._refractory_until_frame_index = None
        self._awaiting_reset = False
        return ()


class PredictiveShadowRunner:
    """Load the predictive timing front-end and gesture classifier for live inference."""

    def __init__(self, config: PredictiveConfig) -> None:
        """Load both checkpoints and initialize the runtime-mode decoder."""
        self.config = config
        torch, nn, _, _ = require_torch()
        self._torch = torch
        self._device = resolve_device(config.device, torch)

        timing_checkpoint = load_checkpoint(
            Path(config.timing_checkpoint_path or ""),
            torch=torch,
            device=self._device,
        )
        timing_spec = VisionBeatCnnSpec.from_checkpoint(timing_checkpoint)
        validate_runtime_compatibility(
            timing_spec,
            feature_names=FEATURE_NAMES,
            schema_version=FEATURE_SCHEMA_VERSION,
            window_size=timing_spec.window_size,
            target_name=timing_spec.target_name,
            horizon_frames=timing_spec.horizon_frames,
        )
        timing_model = build_completion_cnn(nn, timing_spec)
        timing_model.load_state_dict(timing_checkpoint["model_state_dict"])
        timing_model.to(self._device)
        timing_model.eval()

        gesture_checkpoint = load_gesture_classifier_checkpoint(
            Path(config.gesture_checkpoint_path or ""),
            torch=torch,
            device=self._device,
        )
        gesture_spec = VisionBeatGestureClassifierSpec.from_checkpoint(gesture_checkpoint)
        if gesture_spec.feature_names != FEATURE_NAMES:
            raise ValueError("Gesture classifier checkpoint feature_names do not match runtime.")
        if gesture_spec.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("Gesture classifier checkpoint schema_version does not match runtime.")
        if gesture_spec.window_size != timing_spec.window_size:
            raise ValueError(
                "Gesture classifier window_size does not match the timing checkpoint. "
                f"{gesture_spec.window_size} vs {timing_spec.window_size}."
            )
        gesture_model = build_gesture_classifier(nn, gesture_spec)
        gesture_model.load_state_dict(gesture_checkpoint["model_state_dict"])
        gesture_model.to(self._device)
        gesture_model.eval()

        self._timing_model = timing_model
        self._gesture_model = gesture_model
        self.timing_spec = timing_spec
        self.gesture_spec = gesture_spec
        self._latest_status = PredictiveStatus(
            available_window_frames=0,
            required_window_size=self.timing_spec.window_size,
            threshold=config.threshold,
        )
        self.decoder: (
            StreamingTriggerDecoder | PrimaryTriggerDecoder | CompletionTriggerDecoder | None
        )
        if config.predictive_logs_shadow:
            self.decoder = StreamingTriggerDecoder(
                threshold=config.threshold,
                cooldown_frames=config.trigger_cooldown_frames,
                max_gap_frames=config.trigger_max_gap_frames,
            )
        elif config.predictive_drives_audio:
            if self.timing_spec.target_name == "completion_frame_binary":
                self.decoder = CompletionTriggerDecoder(
                    threshold=config.threshold,
                    cooldown_frames=config.trigger_cooldown_frames,
                )
            elif self.timing_spec.target_name == "completion_within_last_k_frames":
                self.decoder = StreamingTriggerDecoder(
                    threshold=config.threshold,
                    cooldown_frames=config.trigger_cooldown_frames,
                    max_gap_frames=config.trigger_max_gap_frames,
                )
            else:
                self.decoder = PrimaryTriggerDecoder(
                    threshold=config.threshold,
                    cooldown_frames=config.trigger_cooldown_frames,
                    max_gap_frames=config.trigger_max_gap_frames,
                    horizon_frames=self.timing_spec.horizon_frames,
                )
        else:
            self.decoder = None

    @property
    def required_window_size(self) -> int:
        """Return the live history length required before shadow inference can start."""
        return self.timing_spec.window_size

    @property
    def prediction_horizon_frames(self) -> int:
        """Return the timing checkpoint horizon metadata used by live gating."""
        return self.timing_spec.horizon_frames

    @property
    def latest_status(self) -> PredictiveStatus:
        """Return the latest predictive telemetry for the current live window."""
        return self._latest_status

    def status_summary(self) -> str:
        """Return a compact summary of current predictive telemetry."""
        return self._latest_status.summary()

    def update(
        self,
        *,
        feature_window: CanonicalSequenceWindow,
        frame_index: int,
        timestamp: FrameTimestamp,
        heuristic_events: tuple[GestureEvent, ...],
    ) -> tuple[ShadowPredictionEvent, ...]:
        """Run one live inference step and emit any accepted shadow triggers."""
        if len(feature_window.frames) < self.required_window_size:
            self._latest_status = PredictiveStatus(
                available_window_frames=len(feature_window.frames),
                required_window_size=self.required_window_size,
                threshold=self.config.threshold,
            )
            return ()
        window_matrix = np.asarray(feature_window.matrix, dtype=np.float32)
        timing_probability = self._predict_timing_probability(window_matrix)
        class_probabilities = self._predict_gesture_probabilities(window_matrix)
        predicted_gesture, predicted_gesture_confidence = self._best_gesture_prediction(
            class_probabilities
        )
        probability_mapping = {
            label: float(class_probabilities[index])
            for index, label in enumerate(self.gesture_spec.class_labels)
        }
        self._latest_status = PredictiveStatus(
            available_window_frames=len(feature_window.frames),
            required_window_size=self.required_window_size,
            threshold=self.config.threshold,
            timing_probability=timing_probability,
            predicted_gesture=predicted_gesture,
            predicted_gesture_confidence=predicted_gesture_confidence,
            class_probabilities=probability_mapping,
        )
        if self.decoder is None:
            return ()
        if isinstance(self.decoder, StreamingTriggerDecoder):
            decoded_candidates = self.decoder.update(
                frame_index=frame_index,
                timestamp_seconds=timestamp.seconds,
                timing_probability=timing_probability,
                window_matrix=window_matrix,
                heuristic_gesture_types=tuple(event.gesture.value for event in heuristic_events),
            )
            return tuple(self._classify_candidate(candidate) for candidate in decoded_candidates)
        decoded_candidates = self.decoder.update(
            frame_index=frame_index,
            timestamp_seconds=timestamp.seconds,
            timing_probability=timing_probability,
            predicted_gesture=predicted_gesture,
            predicted_gesture_confidence=predicted_gesture_confidence,
            class_probabilities=probability_mapping,
        )
        return tuple(self._build_primary_event(candidate) for candidate in decoded_candidates)

    def flush(self) -> tuple[ShadowPredictionEvent, ...]:
        """Flush any buffered shadow triggers when the runtime stops."""
        if self.decoder is None:
            return ()
        if isinstance(self.decoder, StreamingTriggerDecoder):
            return tuple(self._classify_candidate(candidate) for candidate in self.decoder.flush())
        return tuple(self._build_primary_event(candidate) for candidate in self.decoder.flush())

    def _predict_timing_probability(self, window_matrix: np.ndarray) -> float:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(window_matrix).unsqueeze(0).to(self._device)
            logits = self._timing_model(tensor)
            probability = self._torch.sigmoid(logits).cpu().item()
        return float(probability)

    def _classify_candidate(self, candidate: _RunPeak) -> ShadowPredictionEvent:
        class_probabilities = self._predict_gesture_probabilities(candidate.window_matrix)
        predicted_gesture, predicted_confidence = self._best_gesture_prediction(class_probabilities)
        probability_mapping = {
            label: float(class_probabilities[index])
            for index, label in enumerate(self.gesture_spec.class_labels)
        }
        return ShadowPredictionEvent(
            frame_index=candidate.frame_index,
            timestamp=FrameTimestamp(seconds=candidate.timestamp_seconds),
            timing_probability=candidate.timing_probability,
            threshold=self.config.threshold,
            run_length=candidate.run_length,
            gesture=predicted_gesture,
            gesture_confidence=predicted_confidence,
            class_probabilities=probability_mapping,
            heuristic_triggered_on_peak_frame=bool(candidate.heuristic_gesture_types_on_peak_frame),
            heuristic_gesture_types_on_peak_frame=candidate.heuristic_gesture_types_on_peak_frame,
        )

    def _build_primary_event(self, candidate: _PrimaryPeak) -> ShadowPredictionEvent:
        return ShadowPredictionEvent(
            frame_index=candidate.frame_index,
            timestamp=FrameTimestamp(seconds=candidate.timestamp_seconds),
            timing_probability=candidate.timing_probability,
            threshold=self.config.threshold,
            run_length=candidate.run_length,
            gesture=candidate.gesture,
            gesture_confidence=candidate.gesture_confidence,
            class_probabilities=dict(candidate.class_probabilities),
            heuristic_triggered_on_peak_frame=False,
            heuristic_gesture_types_on_peak_frame=(),
        )

    def _predict_gesture_probabilities(self, window_matrix: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(window_matrix).unsqueeze(0).to(self._device)
            logits = self._gesture_model(tensor)
            probabilities = self._torch.softmax(logits, dim=1).cpu().numpy()[0]
        return np.asarray(probabilities, dtype=np.float32)

    def _best_gesture_prediction(
        self,
        class_probabilities: np.ndarray,
    ) -> tuple[GestureType, float]:
        predicted_index = int(np.argmax(class_probabilities))
        predicted_label = self.gesture_spec.class_labels[predicted_index]
        try:
            predicted_gesture = GestureType(predicted_label)
        except ValueError as exc:  # pragma: no cover - defensive only
            raise ValueError(
                "Unsupported gesture label in classifier checkpoint: "
                f"{predicted_label}"
            ) from exc
        return predicted_gesture, float(class_probabilities[predicted_index])


def build_predictive_shadow_runner(config: PredictiveConfig) -> PredictiveShadowRunner | None:
    """Build the optional predictive shadow runner from validated configuration."""
    if not config.enabled:
        return None
    try:
        return PredictiveShadowRunner(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"predictive: {exc}") from exc


__all__ = [
    "CompletionTriggerDecoder",
    "PrimaryTriggerDecoder",
    "PredictiveStatus",
    "PredictiveShadowRunner",
    "ShadowPredictionEvent",
    "StreamingTriggerDecoder",
    "build_predictive_shadow_runner",
]

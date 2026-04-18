"""Structured observability helpers for runtime logs and offline gesture analysis."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO

from visionbeat.models import GestureType

logger = logging.getLogger(__name__)


class StructuredLogFormatter(logging.Formatter):
    """Formatter that appends structured JSON payloads to standard log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and append any structured fields as compact JSON."""
        message = super().format(record)
        structured = getattr(record, "structured", None)
        if not structured:
            return message
        payload = json.dumps(structured, sort_keys=True, separators=(",", ":"))
        return f"{message} | {payload}"


@dataclass(frozen=True, slots=True)
class VelocityStats:
    """Motion statistics attached to a gesture-analysis decision."""

    elapsed: float
    delta_x: float
    delta_y: float
    delta_z: float
    net_velocity: float
    peak_x_velocity: float
    peak_y_velocity: float
    peak_z_velocity: float

    def to_dict(self) -> dict[str, float]:
        """Serialize the velocity statistics."""
        return {
            "elapsed": self.elapsed,
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "delta_z": self.delta_z,
            "net_velocity": self.net_velocity,
            "peak_x_velocity": self.peak_x_velocity,
            "peak_y_velocity": self.peak_y_velocity,
            "peak_z_velocity": self.peak_z_velocity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VelocityStats:
        """Deserialize a velocity-statistics payload."""
        return cls(
            elapsed=float(payload["elapsed"]),
            delta_x=float(payload["delta_x"]),
            delta_y=float(payload["delta_y"]),
            delta_z=float(payload["delta_z"]),
            net_velocity=float(payload["net_velocity"]),
            peak_x_velocity=float(payload["peak_x_velocity"]),
            peak_y_velocity=float(payload["peak_y_velocity"]),
            peak_z_velocity=float(payload["peak_z_velocity"]),
        )


@dataclass(frozen=True, slots=True)
class GestureObservationEvent:
    """Serializable gesture-observation event for logs and offline analysis."""

    timestamp: float
    event_kind: Literal[
        "candidate",
        "trigger",
        "cooldown_suppressed",
        "tracking_failure",
    ]
    gesture_type: GestureType | None
    accepted: bool
    reason: str
    velocity_stats: VelocityStats | None = None
    confidence: float | None = None
    hand: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observation event into a JSON-friendly structure."""
        return {
            "timestamp": self.timestamp,
            "event_kind": self.event_kind,
            "gesture_type": self.gesture_type.value if self.gesture_type is not None else None,
            "accepted": self.accepted,
            "reason": self.reason,
            "velocity_stats": (
                None if self.velocity_stats is None else self.velocity_stats.to_dict()
            ),
            "confidence": self.confidence,
            "hand": self.hand,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GestureObservationEvent:
        """Deserialize an observation event from a mapping payload."""
        gesture_value = payload.get("gesture_type")
        velocity_payload = payload.get("velocity_stats")
        return cls(
            timestamp=float(payload["timestamp"]),
            event_kind=payload["event_kind"],
            gesture_type=None if gesture_value is None else GestureType(gesture_value),
            accepted=bool(payload["accepted"]),
            reason=str(payload["reason"]),
            velocity_stats=(
                None if velocity_payload is None else VelocityStats.from_dict(velocity_payload)
            ),
            confidence=(
                None if payload.get("confidence") is None else float(payload["confidence"])
            ),
            hand=None if payload.get("hand") is None else str(payload["hand"]),
        )

    def to_csv_row(self) -> dict[str, str]:
        """Flatten the observation event into a CSV row."""
        velocity = self.velocity_stats.to_dict() if self.velocity_stats is not None else {}
        return {
            "timestamp": f"{self.timestamp:.6f}",
            "iso_timestamp": monotonic_to_iso8601(self.timestamp),
            "event_kind": self.event_kind,
            "gesture_type": "" if self.gesture_type is None else self.gesture_type.value,
            "accepted": str(self.accepted).lower(),
            "reason": self.reason,
            "confidence": "" if self.confidence is None else f"{self.confidence:.6f}",
            "hand": "" if self.hand is None else self.hand,
            "elapsed": _format_optional_float(velocity.get("elapsed")),
            "delta_x": _format_optional_float(velocity.get("delta_x")),
            "delta_y": _format_optional_float(velocity.get("delta_y")),
            "delta_z": _format_optional_float(velocity.get("delta_z")),
            "net_velocity": _format_optional_float(velocity.get("net_velocity")),
            "peak_x_velocity": _format_optional_float(velocity.get("peak_x_velocity")),
            "peak_y_velocity": _format_optional_float(velocity.get("peak_y_velocity")),
            "peak_z_velocity": _format_optional_float(velocity.get("peak_z_velocity")),
        }


CSV_FIELDNAMES = [
    "timestamp",
    "iso_timestamp",
    "event_kind",
    "gesture_type",
    "accepted",
    "reason",
    "confidence",
    "hand",
    "elapsed",
    "delta_x",
    "delta_y",
    "delta_z",
    "net_velocity",
    "peak_x_velocity",
    "peak_y_velocity",
    "peak_z_velocity",
]


class GestureEventLogger:
    """Optional CSV or JSONL sink for gesture-analysis observations."""

    def __init__(self, path: str | Path, *, fmt: Literal["jsonl", "csv"]) -> None:
        """Create an event log writer for the given path and format."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.format = fmt
        self._stream: TextIO = self.path.open("a", encoding="utf-8", newline="")
        self._csv_writer: csv.DictWriter[str] | None = None
        if self.format == "csv":
            self._csv_writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDNAMES)
            if self.path.stat().st_size == 0:
                self._csv_writer.writeheader()
                self._stream.flush()

    def write(self, event: GestureObservationEvent) -> None:
        """Persist one observation event."""
        if self.format == "jsonl":
            self._stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            self._stream.flush()
            return
        assert self._csv_writer is not None
        self._csv_writer.writerow(event.to_csv_row())
        self._stream.flush()

    def close(self) -> None:
        """Close the underlying stream."""
        self._stream.close()


@dataclass(slots=True)
class ObservabilityRecorder:
    """Central structured logger for runtime lifecycle and gesture-analysis events."""

    event_logger: GestureEventLogger | None = None

    def log_app_startup(self, *, config_summary: dict[str, Any]) -> None:
        """Record application startup with the effective configuration summary."""
        self._emit_lifecycle(
            logging.INFO,
            "app_startup",
            message="VisionBeat application startup",
            config=config_summary,
        )

    def log_app_shutdown(self) -> None:
        """Record application shutdown."""
        self._emit_lifecycle(
            logging.INFO,
            "app_shutdown",
            message="VisionBeat application shutdown",
        )

    def log_runtime_started(self, *, window_name: str) -> None:
        """Record the beginning of the runtime loop."""
        self._emit_lifecycle(
            logging.INFO,
            "runtime_started",
            message="VisionBeat runtime loop started",
            window_name=window_name,
        )

    def log_runtime_stopped(self, *, reason: str) -> None:
        """Record the runtime loop stopping."""
        self._emit_lifecycle(
            logging.INFO,
            "runtime_stopped",
            message="VisionBeat runtime loop stopped",
            reason=reason,
        )

    def log_camera_initialization(
        self,
        *,
        device_index: int,
        width: int,
        height: int,
        target_fps: int,
        mirror: bool,
        opened: bool,
    ) -> None:
        """Record camera initialization metadata."""
        self._emit_lifecycle(
            logging.INFO if opened else logging.ERROR,
            "camera_initialization",
            message="Camera initialization",
            device_index=device_index,
            width=width,
            height=height,
            target_fps=target_fps,
            mirror=mirror,
            opened=opened,
        )

    def log_tracking_failure(self, *, timestamp: float, status: str) -> None:
        """Record a tracking failure in structured logs and the optional event sink."""
        event = GestureObservationEvent(
            timestamp=timestamp,
            event_kind="tracking_failure",
            gesture_type=None,
            accepted=False,
            reason=status,
        )
        self._emit_event(logging.WARNING, "tracking_failure", event)

    def log_gesture_candidate(self, event: GestureObservationEvent) -> None:
        """Record a pending gesture candidate."""
        self._emit_event(logging.INFO, "gesture_candidate", event)

    def log_confirmed_trigger(self, event: GestureObservationEvent) -> None:
        """Record a confirmed trigger."""
        self._emit_event(logging.INFO, "gesture_trigger", event)

    def log_cooldown_suppression(self, event: GestureObservationEvent) -> None:
        """Record a trigger suppressed by cooldown."""
        self._emit_event(logging.INFO, "cooldown_suppression", event)

    def log_predictive_shadow_trigger(
        self,
        *,
        timestamp: float,
        frame_index: int,
        timing_probability: float,
        predicted_gesture: GestureType,
        predicted_gesture_confidence: float,
        heuristic_gesture_types: tuple[str, ...],
        class_probabilities: Mapping[str, float],
    ) -> None:
        """Record one accepted predictive shadow trigger in structured logs."""
        self._emit_lifecycle(
            logging.INFO,
            "predictive_shadow_trigger",
            message="Predictive shadow trigger",
            timestamp=timestamp,
            frame_index=frame_index,
            timing_probability=timing_probability,
            predicted_gesture=predicted_gesture.value,
            predicted_gesture_confidence=predicted_gesture_confidence,
            heuristic_triggered_on_peak_frame=bool(heuristic_gesture_types),
            heuristic_gesture_types_on_peak_frame=list(heuristic_gesture_types),
            class_probabilities=dict(class_probabilities),
        )

    def log_predictive_live_trigger(
        self,
        *,
        timestamp: float,
        frame_index: int,
        timing_probability: float,
        predicted_gesture: GestureType,
        predicted_gesture_confidence: float,
        hand: str,
        class_probabilities: Mapping[str, float],
    ) -> None:
        """Record one predictive trigger that drove the live instrument."""
        self._emit_lifecycle(
            logging.INFO,
            "predictive_live_trigger",
            message="Predictive live trigger",
            timestamp=timestamp,
            frame_index=frame_index,
            timing_probability=timing_probability,
            predicted_gesture=predicted_gesture.value,
            predicted_gesture_confidence=predicted_gesture_confidence,
            hand=hand,
            class_probabilities=dict(class_probabilities),
        )

    def close(self) -> None:
        """Flush and close any optional event sink."""
        if self.event_logger is not None:
            self.event_logger.close()

    def _emit_event(
        self,
        level: int,
        event_name: str,
        event: GestureObservationEvent,
    ) -> None:
        """Emit a structured log entry and optionally persist the event."""
        payload = event.to_dict()
        logger.log(
            level,
            "%s",
            event.reason,
            extra={"structured": {"event": event_name, **payload}},
        )
        if self.event_logger is not None:
            self.event_logger.write(event)

    def _emit_lifecycle(
        self,
        level: int,
        event_name: str,
        *,
        message: str,
        **fields: Any,
    ) -> None:
        """Emit a non-gesture lifecycle log entry."""
        logger.log(level, message, extra={"structured": {"event": event_name, **fields}})


def build_observability_recorder(logging_config: Any) -> ObservabilityRecorder:
    """Build an observability recorder from application logging configuration."""
    event_logger: GestureEventLogger | None = None
    if logging_config.event_log_path is not None:
        event_logger = GestureEventLogger(
            logging_config.event_log_path,
            fmt=logging_config.event_log_format,
        )
    return ObservabilityRecorder(event_logger=event_logger)


def configure_root_logging(level: str, *, log_format: str, structured: bool) -> None:
    """Configure process-wide logging with optional structured payload support."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter: logging.Formatter
    if structured:
        formatter = StructuredLogFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def monotonic_to_iso8601(timestamp: float) -> str:
    """Return a wall-clock timestamp representing when the event was recorded."""
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds")


def _format_optional_float(value: float | None) -> str:
    """Format an optional floating-point value for CSV output."""
    if value is None:
        return ""
    return f"{value:.6f}"

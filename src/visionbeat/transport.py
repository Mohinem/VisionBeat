"""Gesture event transport abstractions and UDP sender implementations."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable
from uuid import uuid4

from visionbeat.models import GestureEvent

SCHEMA_VERSION = "visionbeat.gesture.v1"


@dataclass(frozen=True, slots=True)
class GestureMessage:
    """Schema used for publishing confirmed gesture events to external systems."""

    schema: str
    event_id: str
    event_type: str
    gesture: str
    confidence: float
    intensity: float
    hand: str
    label: str
    timestamp_seconds: float
    source: str

    @classmethod
    def from_event(cls, event: GestureEvent, *, source: str = "visionbeat") -> GestureMessage:
        """Build a schema-compliant message from a confirmed gesture event."""
        return cls(
            schema=SCHEMA_VERSION,
            event_id=str(uuid4()),
            event_type="gesture.confirmed",
            gesture=event.gesture.value,
            confidence=event.confidence,
            intensity=event.confidence,
            hand=event.hand,
            label=event.label,
            timestamp_seconds=event.timestamp.seconds,
            source=source,
        )

    def to_dict(self) -> dict[str, str | float]:
        """Serialize message into a dictionary suitable for JSON encoding."""
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "gesture": self.gesture,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "hand": self.hand,
            "label": self.label,
            "timestamp_seconds": self.timestamp_seconds,
            "source": self.source,
        }

    def to_json_bytes(self) -> bytes:
        """Serialize message to compact UTF-8 JSON payload bytes."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")


@runtime_checkable
class GestureEventTransport(Protocol):
    """Transport abstraction for forwarding gesture events outside the process."""

    def emit(self, event: GestureEvent) -> None:
        """Send a confirmed gesture event to an external target."""

    def close(self) -> None:
        """Release transport resources."""


@dataclass(slots=True)
class NullGestureEventTransport:
    """No-op transport used when external gesture forwarding is disabled."""

    def emit(self, event: GestureEvent) -> None:
        """Ignore gesture events when no external transport is configured."""
        _ = event

    def close(self) -> None:
        """No-op close for API compatibility."""


@dataclass(slots=True)
class UdpGestureEventTransport:
    """UDP JSON transport that emits gesture events to a network endpoint."""

    host: str
    port: int
    source: str = "visionbeat"
    socket_factory: Callable[..., socket.socket] = socket.socket
    _socket: socket.socket = field(init=False)

    def __post_init__(self) -> None:
        """Initialize a datagram socket for one-way event delivery."""
        self._socket = self.socket_factory(socket.AF_INET, socket.SOCK_DGRAM)

    def emit(self, event: GestureEvent) -> None:
        """Serialize and send a confirmed gesture event over UDP."""
        message = GestureMessage.from_event(event, source=self.source)
        self._socket.sendto(message.to_json_bytes(), (self.host, self.port))

    def close(self) -> None:
        """Close the transport socket."""
        self._socket.close()

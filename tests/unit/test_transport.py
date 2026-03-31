from __future__ import annotations

import json

from visionbeat.models import FrameTimestamp, GestureEvent, GestureType
from visionbeat.transport import GestureMessage, NullGestureEventTransport, UdpGestureEventTransport


class FakeSocket:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.packets: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, payload: bytes, address: tuple[str, int]) -> None:
        self.packets.append((payload, address))

    def close(self) -> None:
        self.closed = True


def make_event() -> GestureEvent:
    return GestureEvent(
        gesture=GestureType.KICK,
        confidence=0.93,
        hand="right",
        timestamp=FrameTimestamp(seconds=12.5),
        label="Inward jab → kick",
    )


def test_gesture_message_serialization_round_trips_to_json() -> None:
    event = make_event()

    message = GestureMessage.from_event(event, source="visionbeat-tests")
    payload = json.loads(message.to_json_bytes().decode("utf-8"))

    assert payload["schema"] == "visionbeat.gesture.v1"
    assert payload["event_type"] == "gesture.confirmed"
    assert payload["gesture"] == "kick"
    assert payload["confidence"] == 0.93
    assert payload["intensity"] == 0.93
    assert payload["hand"] == "right"
    assert payload["timestamp_seconds"] == 12.5
    assert payload["source"] == "visionbeat-tests"
    assert payload["event_id"]


def test_udp_gesture_transport_sends_json_packet_to_configured_endpoint() -> None:
    socket = FakeSocket()
    transport = UdpGestureEventTransport(
        host="127.0.0.1",
        port=9100,
        source="visionbeat-tests",
        socket_factory=lambda *_args: socket,
    )

    transport.emit(make_event())
    transport.close()

    assert len(socket.packets) == 1
    payload, address = socket.packets[0]
    decoded = json.loads(payload.decode("utf-8"))
    assert address == ("127.0.0.1", 9100)
    assert decoded["gesture"] == "kick"
    assert decoded["source"] == "visionbeat-tests"
    assert socket.closed is True


def test_null_transport_accepts_events_without_side_effects() -> None:
    transport = NullGestureEventTransport()

    transport.emit(make_event())
    transport.close()

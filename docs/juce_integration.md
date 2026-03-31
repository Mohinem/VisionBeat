# VisionBeat → JUCE Integration Design

This document describes a future-ready integration path where VisionBeat's Python gesture detector drives a JUCE-based audio engine over the network.

## Goals

- Keep current Python sample playback working (pygame backend).
- Add an external event transport path that can be enabled later.
- Define a stable message schema that a JUCE app can parse.
- Make it easy to switch from local playback to JUCE-triggered playback without rewriting gesture logic.

## New transport architecture

VisionBeat now has a dedicated **gesture transport layer**:

- `GestureEventTransport` protocol: abstraction for emitting confirmed gesture events.
- `NullGestureEventTransport`: default no-op transport (keeps existing behavior unchanged).
- `UdpGestureEventTransport`: sends compact JSON packets over UDP.
- `GestureMessage`: schema object for serializing gesture events to network payloads.

The runtime flow is:

1. Gesture detector confirms a gesture.
2. VisionBeat triggers local audio (`audio.trigger(...)`) exactly as before.
3. VisionBeat also emits a transport message (`transport.emit(event)`).

This allows staged adoption:

- Today: `audio.backend=pygame`, `transport.backend=none`.
- Transition: keep pygame + also send UDP (`transport.backend=udp`).
- Future: disable local sample playback and let JUCE be the sound engine.

## Gesture event schema (`visionbeat.gesture.v1`)

All UDP packets contain one JSON object:

```json
{
  "schema": "visionbeat.gesture.v1",
  "event_id": "uuid-v4",
  "event_type": "gesture.confirmed",
  "gesture": "kick",
  "confidence": 0.93,
  "intensity": 0.93,
  "hand": "right",
  "label": "Inward jab → kick",
  "timestamp_seconds": 12.5,
  "source": "visionbeat"
}
```

### Field notes

- `schema`: versioned contract for compatibility (`visionbeat.gesture.v1`).
- `event_id`: unique identifier for deduplication/replay protection.
- `event_type`: currently `gesture.confirmed`.
- `gesture`: gesture name (`kick`, `snare`, ...).
- `confidence` and `intensity`: normalized `0.0..1.0`.
- `timestamp_seconds`: VisionBeat frame timestamp in seconds.
- `source`: sender identifier (useful for multi-camera/multi-sender setups).

## Configuration

Add a top-level `transport` section:

```toml
[transport]
backend = "udp"      # "none" (default) or "udp"
host = "127.0.0.1"
port = 9000
source = "visionbeat"
```

When omitted, VisionBeat defaults to `backend = "none"` and behaves exactly like the previous Python-only setup.

## JUCE receiver design

In a JUCE app, add a lightweight UDP listener:

1. Bind a UDP socket on the configured port (e.g., `9000`).
2. Parse UTF-8 JSON payloads.
3. Validate `schema == "visionbeat.gesture.v1"`.
4. Route `gesture` to sampler voices or MIDI trigger logic.
5. Map `intensity` to gain, velocity, or filter envelope amount.

### Suggested JUCE routing table

- `kick` -> kick sample / drum rack pad 36
- `snare` -> snare sample / drum rack pad 38
- unknown gesture -> ignore or send to fallback sample

### Latency recommendations

- Keep VisionBeat and JUCE on localhost when possible.
- Prefer UDP for low overhead and realtime behavior.
- Use `event_id` to ignore duplicates if your transport layer ever retries.

## Optional OSC bridge

If your JUCE project already uses OSC classes, you can:

- Continue receiving UDP JSON directly, **or**
- Add a small Python/JUCE bridge that converts JSON packets to OSC messages.

The current transport layer is intentionally backend-agnostic so an OSC sender can be added later without changing gesture detection logic.

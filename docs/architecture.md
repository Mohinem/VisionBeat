# VisionBeat Architecture

## Overview

VisionBeat is organized as a real-time pipeline with explicit boundaries between **I/O-heavy runtime components** and **pure gesture logic**. The key design goal is that gesture recognition can be tested from synthetic landmark sequences without requiring a webcam, MediaPipe runtime, or audio device.

## Runtime pipeline

1. **CameraStream** opens an OpenCV webcam source and yields mirrored BGR frames.
2. **PoseTracker** runs MediaPipe Pose on each frame and converts selected landmarks into a normalized `PoseFrame`.
3. **GestureDetector** consumes `PoseFrame` objects and emits `GestureEvent` values for kick and snare gestures.
4. **AudioEngine** is an abstraction that maps gestures to named samples; the current `PygameAudioEngine` backend triggers playback through `pygame.mixer`.
5. **OverlayRenderer** draws landmarks, labels, and debug state onto the preview frame.
6. **VisionBeatApp** orchestrates the loop, logging, cleanup, and keyboard exit handling.

## Module boundaries

- `visionbeat.camera`: webcam lifecycle and frame acquisition.
- `visionbeat.tracking`: MediaPipe integration and landmark normalization.
- `visionbeat.gestures`: pure trajectory analysis and cooldown handling.
- `visionbeat.audio`: sample loading and playback.
- `visionbeat.overlay`: OpenCV-based diagnostics and labels.
- `visionbeat.config`: strongly validated configuration models and YAML/TOML loading.
- `visionbeat.logging_config`: centralized logging setup.
- `visionbeat.models`: shared dataclasses and enums.

## Gesture detection strategy

The current detector uses a short rolling history of wrist positions for the configured active hand:

- **Forward punch → kick** is detected when normalized wrist `z` decreases sharply over the history window, vertical drift remains bounded, and the combined movement velocity exceeds a minimum threshold.
- **Downward strike → snare** is detected when normalized wrist `y` increases sharply, depth drift remains bounded, and velocity exceeds the threshold.
- A configurable cooldown prevents repeated triggers from a single motion arc.

This heuristic-based approach is intentionally simple, transparent, and easy to extend. Future iterations can swap in a classifier while preserving the `PoseFrame -> list[GestureEvent]` interface.

## Testability

The `GestureDetector` accepts plain dataclass models rather than MediaPipe protobufs. Unit tests construct `PoseFrame` sequences directly, making the most motion-sensitive logic deterministic and fast.

## Configuration and operations

- Runtime settings are loaded from `configs/default.yaml` via `load_config`.
- Logging is configured once at startup and uses Python's standard library logging.
- Placeholder samples are generated locally with `python scripts/generate_demo_samples.py` and are not committed as binary assets.

## Extension points

- Add multiple hand profiles or ambidextrous detection by extending `GestureConfig` and the hand loop.
- Introduce richer overlay diagnostics by exposing velocities and thresholds from the detector.
- Replace `pygame` with `sounddevice` if lower-latency output is needed on a target platform.
- Add calibration routines that learn per-user gesture baselines before live performance.

# VisionBeat Architecture

## Architectural intent

VisionBeat is designed as a **real-time interactive instrument pipeline** with clear boundaries between:

- hardware-facing I/O,
- landmark tracking,
- gesture reasoning,
- audio playback,
- visual feedback,
- configuration,
- and observability.

The central architectural decision is that **gesture recognition should remain testable without requiring a webcam, MediaPipe runtime, or actual audio output**. That decision keeps the motion logic reproducible for research and practical for continuous integration.

## End-to-end runtime pipeline

A single performance loop moves through these stages:

1. **CameraSource** opens an OpenCV capture device and returns timestamped frames.
2. **PoseTracker** converts each BGR frame to RGB, runs MediaPipe Pose, and emits a `TrackerOutput` containing the selected upper-body landmarks.
3. **GestureDetector** appends wrist samples to a short history for each hand, computes motion metrics, manages candidate state, and emits confirmed `GestureEvent` values.
4. **VisionBeatRuntime** forwards confirmed gestures to the audio layer and collects UI state for the overlay.
5. **OverlayRenderer** draws landmarks and a compact debug panel on the preview frame.
6. **PreviewWindow** displays the rendered output and checks for exit keys.
7. **ObservabilityRecorder** optionally records structured logs and event traces across the loop.

## Module responsibilities

### `src/visionbeat/camera.py`

Responsible for webcam lifecycle and frame acquisition.

Key responsibilities:

- open the configured camera device,
- apply requested width/height/FPS settings,
- mirror frames when configured,
- timestamp frames using a monotonic clock,
- and raise explicit runtime errors when capture fails.

### `src/visionbeat/tracking.py`

Responsible for MediaPipe integration and normalization into project-native models.

Key responsibilities:

- initialize MediaPipe Pose with validated tracking configuration,
- process webcam frames,
- keep only the subset of landmarks used by VisionBeat,
- filter landmarks by visibility/tracking confidence,
- and emit `TrackerOutput` with a descriptive status string.

### `src/visionbeat/gestures.py`

Responsible for **pure gesture analysis** over wrist trajectories.

Key responsibilities:

- maintain short motion histories for left and right wrists,
- compute displacement and peak-velocity metrics,
- detect candidate inward-jab kicks and downward-strike snares,
- confirm gestures within a bounded confirmation window,
- bias ambiguous inward-jab motion toward kick so snare does not steal it,
- enforce cooldown/debounce behavior,
- and emit typed `GestureEvent` values with confidence scores.

This is the most research-critical module because it operationalizes the mapping from movement to musical event.

### `src/visionbeat/audio.py`

Responsible for audio playback abstraction and backend-specific sample triggering.

Key responsibilities:

- define the `AudioEngine` interface,
- resolve sample assets from config,
- initialize the active backend,
- scale playback volume by trigger intensity,
- and expose a backend-agnostic trigger API.

### `src/visionbeat/overlay.py`

Responsible for visual instrumentation of the live performance loop.

Key responsibilities:

- draw simple upper-body landmark connections,
- label visible tracked points,
- show tracker status, FPS, candidates, confirmed gestures, and cooldown,
- and return a rendered frame for preview display.

### `src/visionbeat/app.py`

Responsible for composition and orchestration.

Key responsibilities:

- assemble the runtime dependencies,
- run the main frame-processing loop,
- compute instantaneous FPS,
- dispatch audio triggers,
- render overlay state,
- log key lifecycle events,
- and cleanly release external resources.

### `src/visionbeat/config.py`

Responsible for configuration loading and validation.

Key responsibilities:

- parse YAML or TOML,
- validate nested config structures,
- reject malformed or unknown fields,
- provide typed dataclass configuration objects,
- and fail fast with path-aware error messages.

### `src/visionbeat/models.py`

Responsible for shared domain models.

Key responsibilities:

- define enums and dataclasses shared across subsystems,
- validate model invariants,
- and support dictionary serialization for logging/tests.

### `src/visionbeat/observability.py` and `src/visionbeat/logging_config.py`

Responsible for structured logging and optional event recording.

Key responsibilities:

- configure root logging,
- write structured lifecycle and gesture-analysis events,
- and enable offline debugging of false positives, missed triggers, and cooldown suppression.

## Why the separation matters

### Testability

The detector accepts `TrackerOutput` and uses normalized landmark dataclasses rather than MediaPipe protobuf objects. That means tests can synthesize motion trajectories directly.

### Replaceability

The tracker and audio subsystems can change independently as long as they preserve their interfaces:

- MediaPipe Pose could later be replaced by a hand tracker or another pose estimator.
- `pygame` could be replaced by `sounddevice`, MIDI dispatch, or a JUCE bridge.

### Research transparency

Because the core gesture logic is heuristic and isolated, it is easier to explain:

- what variables matter,
- why a trigger fired,
- how thresholds interact,
- and which changes affect recall versus precision.

## Runtime state model

At any instant, the runtime carries several layers of state:

- **camera state**: whether the device is open and what frame index is being processed,
- **tracking state**: whether a person is detected and which landmarks survive confidence filtering,
- **gesture state**: motion history, pending candidate, last trigger time, and current cooldown,
- **audio state**: available sample assets and mixer readiness,
- **overlay state**: tracker status, candidate label/confidence, confirmed gesture, and FPS,
- **observability state**: optional event sinks and structured log output.

This explicit statefulness is especially important for percussion interaction because trigger quality depends on temporal context rather than a single frame.

## Gesture analysis sub-architecture

The gesture subsystem has its own internal mini-pipeline:

1. append the newest wrist sample,
2. discard samples outside the analysis window,
3. compute window-level motion metrics,
4. reject hands that are inactive, invisible, or in cooldown,
5. start a candidate when onset conditions or kick-preference heuristics are met,
6. expire the candidate if the motion weakens, becomes invalid, or confirmation takes too long,
7. confirm the gesture when full threshold conditions are met,
8. emit an event and start cooldown.

This candidate/confirmation structure is crucial for preventing immediate false triggers from noisy single-frame spikes.

## Data contracts between modules

### Tracker → Gesture detector

The tracker provides:

- timestamp,
- selected landmarks,
- person-detected flag,
- and tracking status.

The detector requires only the relevant wrist landmark plus timestamp, but the fuller payload supports overlays and future posture checks.

### Gesture detector → Audio engine

The detector emits a `GestureEvent`; the runtime converts it to an `AudioTrigger` carrying:

- gesture type,
- event timestamp,
- and confidence-based intensity.

### Runtime → Overlay

The overlay receives a `RenderState` containing:

- tracker output,
- frame index,
- estimated FPS,
- best active candidate,
- last confirmed gesture,
- and remaining cooldown.

## Failure handling philosophy

VisionBeat generally prefers **explicit degradation over hidden failure**:

- failed camera open/read raises runtime errors,
- missing landmarks produce a descriptive tracking status,
- missing audio assets log warnings rather than crashing,
- unsupported backends fail fast,
- and invalid config fields are rejected before the runtime starts.

That approach makes the system easier to operate in demos and easier to analyze during development.

## Current architectural limitations

- The system assumes a single performer and one camera viewpoint.
- Gesture recognition is currently single-hand active rather than fully polyphonic.
- The overlay is informative but intentionally lightweight.
- The audio path is sample-trigger based and not sample-accurate.
- Calibration is manual; there is no automated performer onboarding flow yet.

## Extension paths

### Clap detection

A clap feature would likely sit beside the current wrist-trajectory heuristics and incorporate:

- bilateral hand distance,
- time-to-collision style motion cues,
- and a short transient confirmation window.

### MIDI output

A MIDI backend can fit naturally behind the current audio abstraction by mapping gestures to note-on events instead of WAV playback.

### JUCE integration

A JUCE-based audio layer could preserve the Python vision stack while moving latency-critical sample playback and scheduling into a more specialized runtime.

### Expanded instrument vocabularies

The current architecture can support more gestures as long as the project preserves a stable event vocabulary and keeps the detector interpretable.

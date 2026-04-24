# VisionBeat

VisionBeat is a **camera-based gestural percussion instrument** built in Python. A webcam observes upper-body motion, MediaPipe estimates arm landmarks, a gesture detector classifies rhythmic strikes, and an audio engine triggers drum samples in real time. The current prototype focuses on two core performance gestures:

- **Downward strike → kick**
- **Wrist collision → snare**

The project is deliberately framed as both a **research prototype** and a **performable instrument**. It is useful for studying embodied rhythm interaction, but it is also structured so that a performer can stand in front of a single webcam and play drum-like patterns without wearing sensors or holding a controller.

## Research motivation

VisionBeat is motivated by a simple question: **how can rhythmic intent be inferred from full-body motion with minimal instrumentation?** Instead of attaching IMUs or custom gloves, VisionBeat treats the webcam as the sensing surface and the performer’s arm movement as the control vocabulary.

That framing matters for research in at least four ways:

1. **Embodied interaction**: percussion performance is inherently physical, so the system emphasizes gross, visible motion rather than abstract GUI control.
2. **Accessible sensing**: a commodity webcam lowers the barrier to building and reproducing gesture-controlled music systems.
3. **Transparent heuristics**: the gesture layer is rule-based and testable, making it easier to explain why a hit triggered or failed.
4. **Extensible architecture**: the current kick/snare prototype can evolve into a broader platform for gesture studies, multimodal rhythm interfaces, or hybrid audio engines.

## Instrument framing

VisionBeat should be presented as an **embodied rhythm instrument**, not just a pose-recognition demo. The player performs percussive intent through arm trajectories:

- a **downward hit** feels like a bass-drum attack,
- a **wrist-to-wrist collision** feels like a snare accent,
- the webcam becomes the sensing membrane,
- and the gesture detector becomes the instrument’s strike interpretation layer.

This framing helps clarify the design choices throughout the repository:

- the active gestures are intentionally percussive rather than symbolic,
- thresholds are exposed because performers have different movement styles,
- the overlay provides immediate instrumental feedback,
- and the audio path is separated so the system can later support lower-latency playback, MIDI, or plugin-based output.

## Research presentation narrative

If you need a short presentation story, use something like this:

> VisionBeat is a camera-based gestural percussion instrument for embodied rhythm performance. A single webcam tracks the performer’s upper body, interprets downward-strike and wrist-collision motions as drum hits, and plays percussion samples in real time. The project sits between HCI research and digital instrument design: it asks how rhythmic intention can be expressed through visible, full-body movement while keeping the sensing setup lightweight, transparent, and reproducible.

## What the system currently does

- captures live webcam frames with OpenCV,
- tracks shoulders, elbows, and wrists through MediaPipe Pose,
- converts wrist motion into normalized temporal trajectories,
- detects kick and snare gestures using configurable heuristic thresholds,
- plays local samples with a pygame-backed audio engine,
- renders landmark and debug overlays,
- validates YAML/TOML configuration,
- and supports unit/integration tests for the main subsystems.

## System overview

VisionBeat runs as a real-time pipeline:

1. **Camera capture** acquires mirrored webcam frames.
2. **Tracking** extracts upper-body landmarks.
3. **Gesture detection** evaluates recent wrist motion history.
4. **Audio dispatch** maps confirmed gestures to percussion samples.
5. **Overlay rendering** displays tracking state, candidates, confirmations, and cooldown.

For a deeper breakdown, see [`docs/architecture.md`](docs/architecture.md).

## Repository layout

```text
visionbeat/
├── assets/samples/           # Sample-asset policy and generated local WAVs
├── configs/                  # YAML/TOML runtime configuration
├── docs/                     # Project documentation
├── scripts/                  # Utility scripts such as demo sample generation
├── src/visionbeat/           # Application package
└── tests/                    # Unit and integration tests
```

## Requirements

### Software

- Python 3.11+
- pip
- platform support for OpenCV, MediaPipe, and pygame

### Hardware

- a webcam,
- speakers or headphones,
- enough light for upper-body tracking,
- and enough space for the performer’s wrists and forearms to remain visible.

## Quick start

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install VisionBeat and development tools

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If you want to use the MoveNet backend as well, install its runtime extra:

```bash
python -m pip install -e .[movenet]
```

If you are upgrading an older environment, rerun the install command so pip can
apply the backend's NumPy compatibility constraint.

### 3. Generate local placeholder drum samples

The repository intentionally does not commit the generated demo WAV files. Create them locally before the first run:

```bash
python scripts/generate_demo_samples.py
```

### 4. Run the application

```bash
visionbeat --config configs/default.yaml
```

You can also override the webcam index from the CLI:

```bash
visionbeat --config configs/default.yaml --camera-index 1
```

The pose backend is now selectable at runtime:

```bash
visionbeat --config configs/default.yaml --pose-backend mediapipe
```

For dataset capture, hide HUD text and trigger flashes while keeping the skeleton and point names:

```bash
visionbeat --config configs/default.yaml --skeleton-only-hud
```

Press `q` or `Esc` to exit.

## Live trigger modes

VisionBeat now supports four live trigger modes for the predictive CNN stack:

- `disabled` — heuristics only.
- `shadow` — heuristics drive audio while the CNN + decoder runs passively for logs and comparison.
- `primary` — the CNN + decoder drives live audio.
- `hybrid` — the CNN arms the gesture early, then a matching completion event releases audio.

You can select the mode from config or override it from the CLI.

### Heuristic live mode

```bash
visionbeat --config configs/default.yaml
```

### Predictive shadow mode

```bash
visionbeat \
  --config configs/default.yaml \
  --predictive-mode shadow \
  --timing-checkpoint path/to/timing.pt \
  --gesture-checkpoint path/to/gesture.pt
```

### Predictive primary mode

```bash
visionbeat \
  --config configs/default.yaml \
  --predictive-mode primary \
  --timing-checkpoint path/to/timing.pt \
  --gesture-checkpoint path/to/gesture.pt
```

`primary` is target-aware:

- `completion_within_next_k_frames` checkpoints use the live predictive peak decoder.
- `completion_frame_binary` checkpoints use a simple learned threshold-crossing completion decoder.
- `completion_within_last_k_frames` checkpoints use a local-peak run decoder aligned with the offline trigger analysis.

If you want `primary` to trigger at gesture completion instead of "a hit is coming soon," train a
`completion_frame_binary` or `completion_within_last_k_frames` timing checkpoint and use that at runtime.

### Predictive hybrid mode

```bash
visionbeat \
  --config configs/default.yaml \
  --predictive-mode hybrid \
  --timing-checkpoint path/to/timing.pt \
  --gesture-checkpoint path/to/gesture.pt
```

### Rhythm continuation mode

Repetition-based rhythm prediction can run on top of the existing live trigger
modes. It learns only from confirmed kick/snare sounds, whether those sounds came
from heuristics, primary CNN prediction, or hybrid completion. In direct mode, a
stable pulse can play the next expected kick/snare itself:

```yaml
predictive:
  rhythm_prediction_enabled: true
  rhythm_trigger_mode: direct
```

For a CNN-backed run, keep `mode: primary` or `mode: hybrid` and the normal
checkpoint paths. For a heuristic-only run, use `mode: disabled`. Direct rhythm
triggers do not feed themselves back into rhythm learning, so the system will not
keep playing indefinitely without new confirmed performer/model events.
Live trigger labels distinguish the source: rhythm-played beats appear as
`Kick (rhythm predictor)` / `Snare (rhythm predictor)`, while CNN predictive
beats appear as `Kick (CNN)` / `Snare (CNN)`. The debug HUD also includes a
`Rhythm:` line with the current expected beat time, interval, confidence,
jitter, repetition count, expiry countdown, and latest matched/missed outcome.

See [`docs/rhythm_prediction_evaluation.md`](docs/rhythm_prediction_evaluation.md)
for the ARJ rhythm-continuation test protocol.

Useful predictive overrides:

- `--predictive-threshold`
- `--predictive-trigger-cooldown-frames`
- `--predictive-trigger-max-gap-frames`
- `--predictive-device {auto,cpu,cuda}`

### Training A Learned-Completion Primary Stack

The cleanest path for `primary` is:

1. prepare training archives with `--target completion_within_last_k_frames` or `--target completion_frame_binary`,
2. train the timing CNN on those archives,
3. train the kick/snare classifier on the same archives,
4. run `visionbeat --predictive-mode primary` with those two checkpoints.

Prepare one archive per labeled recording:

```bash
./.venv/bin/python -m visionbeat.prepare_training_data \
  --video path/to/recording_1.mp4 \
  --labels path/to/recording_1_labels.csv \
  --config configs/default.yaml \
  --window-size 24 \
  --stride 1 \
  --target completion_within_last_k_frames \
  --horizon-frames 3 \
  --out data/recording_1_completion_w24.npz
```

Train the timing model:

```bash
./.venv/bin/python -m visionbeat.train_cnn \
  data/recording_1_completion_w24.npz \
  data/recording_2_completion_w24.npz \
  data/recording_3_completion_w24.npz \
  --holdout-recording-id "VisionBeat Dataset - Recording 3" \
  --output-dir outputs/completion_w24_r1r2_train_r3_val
```

Train the gesture classifier on the same completion-aligned archives:

```bash
./.venv/bin/python -m visionbeat.train_gesture_classifier \
  data/recording_1_completion_w24.npz \
  data/recording_2_completion_w24.npz \
  data/recording_3_completion_w24.npz \
  --holdout-recording-id "VisionBeat Dataset - Recording 3" \
  --output-dir outputs/gesture_classifier_completion_w24_r1r2_train_r3_val
```

Run the learned-completion `primary` stack live:

```bash
visionbeat \
  --config configs/default.yaml \
  --predictive-mode primary \
  --timing-checkpoint outputs/completion_w24_r1r2_train_r3_val/visionbeat_cnn_run_001/checkpoints/best_model.pt \
  --gesture-checkpoint outputs/gesture_classifier_completion_w24_r1r2_train_r3_val/visionbeat_gesture_classifier_run_001/checkpoints/best_model.pt
```

## Pose backends

VisionBeat now routes body tracking through a backend abstraction. The current options are:

- `mediapipe` — the default and fully supported backend.
- `movenet` — TensorFlow Lite-backed single-pose MoveNet Lightning.

Backends normalize their output into the shared `TrackerOutput` landmark schema, so gesture detection stays independent from MediaPipe or any future SDK-specific landmark object.

`movenet` requires the optional runtime extra:

```bash
python -m pip install -e .[movenet]
```

To add a future backend:

1. implement the `PoseProvider` interface in a new module under `src/visionbeat/`,
2. map backend-native keypoints into VisionBeat landmark names such as `left_shoulder` and `right_wrist`,
3. return normalized coordinates plus confidence through `TrackerOutput`,
4. register the backend in `create_pose_provider()` in [`src/visionbeat/pose_provider.py`](src/visionbeat/pose_provider.py).

## Local run workflow

A practical local development loop looks like this:

1. install dependencies,
2. generate demo samples,
3. stand centered in front of the webcam,
4. verify that shoulders, elbows, and wrists appear in the overlay,
5. test the **downward strike** gesture first,
6. then test the **wrist collision** gesture,
7. and tune thresholds if the detector is too sensitive or too conservative.

For step-by-step performer setup, see [`docs/demo_guide.md`](docs/demo_guide.md).

## Gesture vocabulary

### Downward strike → kick

A kick is triggered when the configured active wrist moves **downward on screen** quickly enough, with limited depth drift and sufficient vertical dominance in normalized `y`.

### Wrist collision → snare

A snare is triggered when the tracked wrists **move close together** quickly enough in image space, while also staying close in depth. The bilateral collision path takes priority over kick when both motions overlap on the same frame.

The detector uses a candidate/confirmation model plus cooldown to reduce accidental retriggers. The formal definitions are documented in [`docs/gesture_definitions.md`](docs/gesture_definitions.md).

## Calibration and threshold tuning

Thresholds are intentionally exposed in configuration because performer style, camera placement, lens angle, lighting, and distance from the camera all change the normalized motion profile.

Start with the defaults, then calibrate in this order:

1. **Tracking confidence first**
   - If landmarks flicker or disappear, adjust the camera setup before changing gesture thresholds.
2. **Primary gesture displacement**
   - Raise or lower `strike_down_delta_y` and `snare_collision_distance` based on how far the active hand must drop and how close the wrists must come together.
   - If kick candidates arm but stop short of confirming, lower `strike_confirmation_ratio`.
3. **Velocity threshold**
   - Increase `min_velocity` to reject slow drifting motion.
   - If snare candidates arm but stop short of confirming, lower `snare_confirmation_velocity_ratio`.
4. **Axis dominance**
   - Increase `axis_dominance_ratio` to reject diagonal or ambiguous kick motions.
5. **Drift tolerances**
   - Tighten `strike_max_depth_drift` or `snare_collision_max_depth_gap` if cross-axis motion causes false positives.
6. **Cooldown and confirmation**
   - Tune `trigger_seconds`, `analysis_window_seconds`, and `confirmation_window_seconds` for responsiveness versus stability.

Detailed calibration advice is in [`docs/configuration.md`](docs/configuration.md) and [`docs/tracking.md`](docs/tracking.md).

## Testing

Run the main local quality checks with:

```bash
ruff check .
mypy
pytest
```

For the full test strategy, markers, and hardware caveats, see [`docs/testing.md`](docs/testing.md).

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) — system architecture and module responsibilities
- [`docs/gesture_definitions.md`](docs/gesture_definitions.md) — formal gesture definitions and trigger logic
- [`docs/tracking.md`](docs/tracking.md) — tracking model, landmark selection, and calibration guidance
- [`docs/audio.md`](docs/audio.md) — audio subsystem design, latency constraints, and extension paths
- [`docs/configuration.md`](docs/configuration.md) — YAML/TOML configuration reference and threshold tuning
- [`docs/testing.md`](docs/testing.md) — test suite structure and recommended commands
- [`docs/demo_guide.md`](docs/demo_guide.md) — demo setup, presentation flow, and live operation tips
- [`docs/observability.md`](docs/observability.md) — structured logs and event recording

## Troubleshooting

### The webcam opens, but no gestures trigger

- Confirm the overlay shows wrists, elbows, and shoulders.
- Make sure the active hand in config matches the hand you are performing with.
- Move slightly slower first so the tracker can stay locked, then build speed.
- Reduce `strike_down_delta_y` or increase `snare_collision_distance` only after verifying stable tracking.

### Tracking flickers or frequently reports no person detected

- Improve lighting and avoid strong backlight.
- Step farther back so both shoulders and wrists remain in frame.
- Raise the camera so the arm motion stays visible during downward strikes.
- Consider increasing `tracker.min_detection_confidence` and `tracker.min_tracking_confidence` only if your setup is otherwise stable.

### Startup fails with "Unable to locate MediaPipe Pose API"

- VisionBeat depends on the classic `mediapipe.solutions.pose.Pose` API.
- Install a compatible package range: `python -m pip install "mediapipe>=0.10.14,<0.11"`.
- Confirm you are in a Python 3.11+ environment (`python --version`).
- If installation still fails on Linux, check whether your CPU architecture has an official MediaPipe wheel.

### Audio does not play

- Regenerate samples with `python scripts/generate_demo_samples.py`.
- Verify the configured sample paths exist.
- Check whether pygame mixer initialization failed in the logs.
- Try a different output device or a larger `audio.buffer_size`.

### Repeated hits are dropped

- The detector may be suppressing events during cooldown.
- Reduce `gestures.cooldowns.trigger_seconds` if you need faster repeated strikes.
- Check the debug panel or event log to confirm whether suppression is intentional.

### Too many false positives

- Increase `min_velocity`.
- Increase `axis_dominance_ratio`.
- Tighten cross-axis drift thresholds.
- Shorten the performer’s setup distance if the motion is too small in normalized coordinates.

## Known limitations

- The system is currently optimized for **one performer** in front of **one webcam**.
- Only two gestures are implemented by default.
- Detection is wrist-centric and does not use finger pose, hand shape, or explicit impact modeling.
- Fast motion, occlusion, motion blur, or poor lighting can degrade tracking quality.
- `pygame.mixer` is a practical baseline, not a fully professional low-latency audio engine.
- The current implementation provides sample triggering, not full timing quantization, groove modeling, or adaptive accompaniment.

## Future extensions

VisionBeat is intentionally modular so it can grow in several directions:

- **Clap detection** using bilateral hand convergence and transient motion energy.
- **Additional drum mappings** such as toms, hi-hats, or cymbal gestures.
- **MIDI output** for driving DAWs, drum racks, or external hardware.
- **JUCE integration** for a more robust low-latency audio engine or plugin workflow.
- **Performer calibration routines** to learn personal gesture baselines.
- **Alternative trackers** such as hand tracking or multimodal fusion.
- **Multi-hand / ambidextrous play** with simultaneous left/right mappings.

## Contributing

1. Create a Python 3.11+ virtual environment.
2. Install with `python -m pip install -e .[dev]`.
3. Generate demo samples locally.
4. Run linting, typing, and tests.
5. Update documentation when behavior or configuration changes.

## License

VisionBeat is released under the [MIT License](LICENSE).

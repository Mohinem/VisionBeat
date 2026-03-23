# VisionBeat Configuration

VisionBeat loads configuration at startup from either a **YAML** (`.yaml`/`.yml`) or **TOML** (`.toml`) file. The default configuration shipped with the project is [`configs/default.yaml`](../configs/default.yaml).

Use the CLI to point at a different file:

```bash
visionbeat --config configs/default.yaml
```

## Validation behavior

Configuration loading is strongly validated before the runtime starts.

VisionBeat will fail fast when it encounters:

- missing or malformed sections
- wrong value types, such as a string where an integer is required
- out-of-range numeric values
- empty sample paths
- unknown fields caused by typos or stale config keys
- unsupported file extensions

Error messages include the exact config path that failed, for example:

- `camera.width: expected an integer`
- `gestures.thresholds.candidate_ratio: must be less than or equal to 1.0`
- `debug.overlays: unknown field(s): show_landmakrs`

## Default configuration

```yaml
camera:
  device_index: 0
  width: 1280
  height: 720
  fps: 30
  mirror: true
  window_name: VisionBeat

tracker:
  model_complexity: 1
  min_detection_confidence: 0.55
  min_tracking_confidence: 0.55
  enable_segmentation: false

gestures:
  thresholds:
    punch_forward_delta_z: 0.2
    punch_max_vertical_drift: 0.1
    strike_down_delta_y: 0.26
    strike_max_depth_drift: 0.1
    min_velocity: 0.75
    candidate_ratio: 0.7
    axis_dominance_ratio: 1.7
  cooldowns:
    trigger_seconds: 0.2
    analysis_window_seconds: 0.18
    confirmation_window_seconds: 0.12
  history_size: 6
  active_hand: right

audio:
  backend: pygame
  sample_rate: 44100
  buffer_size: 256
  output_channels: 2
  simultaneous_voices: 16
  output_device_name: null
  sample_mapping:
    kick: assets/samples/kick.wav
    snare: assets/samples/snare.wav
  volume: 0.9

debug:
  overlays:
    draw_landmarks: true
    draw_velocity_vectors: true
    show_debug_panel: true

logging:
  level: INFO
  format: '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
```

## Field reference

### `camera`

- `device_index` (`int`, default `0`): OpenCV camera device index.
- `width` (`int`, default `1280`): Requested capture width in pixels.
- `height` (`int`, default `720`): Requested capture height in pixels.
- `fps` (`int`, default `30`): Requested camera frame rate.
- `mirror` (`bool`, default `true`): Whether the preview should be mirrored horizontally.
- `window_name` (`str`, default `VisionBeat`): Name shown on the preview window.

### `tracker`

- `model_complexity` (`0 | 1 | 2`, default `1`): MediaPipe pose complexity level.
- `min_detection_confidence` (`float`, default `0.55`): Minimum person-detection confidence.
- `min_tracking_confidence` (`float`, default `0.55`): Minimum pose-tracking confidence.
- `enable_segmentation` (`bool`, default `false`): Whether to enable MediaPipe segmentation output.

### `gestures`

#### `gestures.thresholds`

- `punch_forward_delta_z` (`float`, default `0.2`): Minimum forward wrist travel for punch detection.
- `punch_max_vertical_drift` (`float`, default `0.1`): Maximum vertical drift tolerated during a punch.
- `strike_down_delta_y` (`float`, default `0.26`): Minimum downward wrist travel for downward-strike detection.
- `strike_max_depth_drift` (`float`, default `0.1`): Maximum depth drift tolerated during a downward strike.
- `min_velocity` (`float`, default `0.75`): Minimum motion velocity required for either gesture.
- `candidate_ratio` (`float`, default `0.7`): Fraction of the threshold needed to keep a pending candidate alive.
- `axis_dominance_ratio` (`float`, default `1.7`): How strongly the primary gesture axis must dominate other movement axes.

#### `gestures.cooldowns`

- `trigger_seconds` (`float`, default `0.2`): Cooldown after a confirmed gesture before the same hand can trigger again.
- `analysis_window_seconds` (`float`, default `0.18`): Rolling time window used to evaluate motion samples.
- `confirmation_window_seconds` (`float`, default `0.12`): Maximum time allowed to confirm a started candidate.

#### `gestures` root fields

- `history_size` (`int`, default `6`): Maximum wrist samples stored per hand.
- `active_hand` (`left | right`, default `right`): Hand used for gesture detection.

### `audio`

- `backend` (`str`, default `pygame`): Audio backend. Currently only `pygame` is supported.
- `sample_rate` (`int`, default `44100`): Mixer output sample rate.
- `buffer_size` (`int`, default `256`): Mixer buffer size. Lower values may reduce latency but increase glitch risk.
- `output_channels` (`int`, default `2`): Number of audio output channels.
- `simultaneous_voices` (`int`, default `16`): Number of mixer channels available for overlapping sounds.
- `output_device_name` (`str | null`, default `null`): Optional device name passed to `pygame.mixer`.
- `sample_mapping` (`map[str, str]`): Gesture/sample-name to audio-file mapping.
  - `kick` (`str`): Sample used for the punch-triggered kick.
  - `snare` (`str`): Sample used for the downward-strike-triggered snare.
  - Additional keys are allowed for future gesture or pad mappings.
- `volume` (`float`, default `0.9`): Master sample playback volume in the range `0.0..1.0`.

### `debug`

#### `debug.overlays`

- `draw_landmarks` (`bool`, default `true`): Draw tracked pose landmarks.
- `draw_velocity_vectors` (`bool`, default `true`): Reserved toggle for velocity-vector diagnostics.
- `show_debug_panel` (`bool`, default `true`): Draw the text debug panel with state, candidates, and cooldowns.

### `logging`

- `level` (`str`, default `INFO`): Python logging level used at startup.
- `format` (`str`, default `%(asctime)s | %(levelname)s | %(name)s | %(message)s`): Logging format string.

## Tuned gesture examples

These values are good starting points for the two built-in gestures:

### Forward punch tuning

```yaml
gestures:
  thresholds:
    punch_forward_delta_z: 0.2
    punch_max_vertical_drift: 0.1
    min_velocity: 0.75
    candidate_ratio: 0.7
    axis_dominance_ratio: 1.7
```

This tuning favors a clear forward extension while rejecting diagonal arm swings.

### Downward strike tuning

```yaml
gestures:
  thresholds:
    strike_down_delta_y: 0.26
    strike_max_depth_drift: 0.1
    min_velocity: 0.75
    candidate_ratio: 0.7
    axis_dominance_ratio: 1.7
```

This tuning favors a confident downward hit while filtering out shallow drops and forward jabs.

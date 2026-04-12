# Configuration

## Overview

VisionBeat loads runtime settings from either:

- a YAML file (`.yaml` or `.yml`), or
- a TOML file (`.toml`).

The default shipped config is [`configs/default.yaml`](../configs/default.yaml), and there is also a TOML example in [`configs/default.toml`](../configs/default.toml).

Launch with:

```bash
visionbeat --config configs/default.yaml
```

Or override the camera index from the CLI:

```bash
visionbeat --config configs/default.yaml --camera-index 1
```

For dataset capture, keep only the tracked skeleton HUD:

```bash
visionbeat --config configs/default.yaml --skeleton-only-hud
```

## Validation behavior

Configuration is strongly validated before the runtime starts. VisionBeat fails fast for:

- missing required sections,
- wrong value types,
- out-of-range numeric values,
- empty strings where non-empty paths/names are expected,
- unknown fields caused by typos,
- and unsupported file extensions.

This is important for reproducible research demos: invalid configuration should fail clearly before live operation.

## Default configuration reference

```yaml
camera:
  device_index: 0
  width: 1280
  height: 720
  fps: 30
  mirror: true
  window_name: VisionBeat

tracker:
  model_complexity: 0
  max_input_width: 640
  min_detection_confidence: 0.55
  min_tracking_confidence: 0.55
  enable_segmentation: false

gestures:
  thresholds:
    punch_forward_delta_z: 0.006
    punch_max_vertical_drift: 0.75
    strike_down_delta_y: 0.12
    strike_confirmation_ratio: 0.65
    strike_max_depth_drift: 0.18
    snare_collision_distance: 0.26
    snare_confirmation_velocity_ratio: 0.8
    snare_collision_max_depth_gap: 0.24
    min_velocity: 0.37
    candidate_ratio: 0.6
    axis_dominance_ratio: 1.2
  cooldowns:
    trigger_seconds: 0.2
    analysis_window_seconds: 0.18
    confirmation_window_seconds: 0.12
  history_size: 6
  active_hand: right
  velocity_smoothing_alpha: 0.8
  rearm_threshold_ratio: 0.45

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

transport:
  backend: none
  host: 127.0.0.1
  port: 9000
  source: visionbeat

debug:
  overlays:
    draw_landmarks: true
    draw_velocity_vectors: true
    show_landmark_labels: true
    show_debug_panel: true
    show_trigger_flash: true

logging:
  level: INFO
  format: '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
  structured: true
  event_log_path: null
  event_log_format: jsonl
  session_recording_path: null
  session_recording_mode: tracker_outputs
```

## Section-by-section reference

### `camera`

Controls webcam acquisition and preview-window labeling.

- `device_index` (`int`, default `0`): OpenCV capture device index.
- `width` (`int`, default `1280`): requested capture width.
- `height` (`int`, default `720`): requested capture height.
- `fps` (`int`, default `30`): requested camera FPS.
- `mirror` (`bool`, default `true`): mirrors the preview and tracked image horizontally.
- `window_name` (`str`, default `VisionBeat`): preview-window title.

### `tracker`

Controls MediaPipe pose-tracking behavior.

- `model_complexity` (`0 | 1 | 2`, default `0`): pose model complexity.
- `max_input_width` (`int`, default `640`): maximum frame width used for pose inference; wider frames are downscaled before tracking to reduce latency.
- `min_detection_confidence` (`float`, default `0.55`): minimum confidence for person detection.
- `min_tracking_confidence` (`float`, default `0.55`): minimum confidence required for retained landmarks.
- `enable_segmentation` (`bool`, default `false`): whether to request segmentation from MediaPipe.

### `gestures`

Controls wrist-history sizing, active-hand selection, timing windows, and gesture thresholds.

#### `gestures.thresholds`

- `punch_forward_delta_z` (`float`, default `0.006`): legacy compatibility field retained for older configs and sensitivity presets.
- `punch_max_vertical_drift` (`float`, default `0.75`): legacy compatibility field retained for older configs and sensitivity presets.
- `strike_down_delta_y` (`float`, default `0.12`): minimum downward travel for kick detection.
- `strike_confirmation_ratio` (`float`, default `0.65`): multiplier applied to `strike_down_delta_y` during final kick confirmation.
- `strike_max_depth_drift` (`float`, default `0.18`): maximum allowed depth drift during a downward-strike kick.
- `snare_collision_distance` (`float`, default `0.26`): maximum image-plane wrist distance allowed for snare confirmation.
- `snare_confirmation_velocity_ratio` (`float`, default `0.8`): multiplier applied to `min_velocity` during final snare closing-speed confirmation.
- `snare_collision_max_depth_gap` (`float`, default `0.24`): maximum wrist depth gap allowed for snare confirmation.
- `min_velocity` (`float`, default `0.37`): shared baseline motion speed used by both gestures.
- `candidate_ratio` (`float`, default `0.6`): ratio used to derive lower onset thresholds from the base displacement thresholds.
- `axis_dominance_ratio` (`float`, default `1.2`): baseline dominance factor for the downward-strike kick axis.

`punch_forward_delta_z` and `punch_max_vertical_drift` are no longer part of the active gesture logic, but they remain in the schema for backward compatibility with older configs.

#### `gestures.cooldowns`

- `trigger_seconds` (`float`, default `0.2`): debounce period after a confirmed hit.
- `analysis_window_seconds` (`float`, default `0.18`): time span of wrist history considered for metrics.
- `confirmation_window_seconds` (`float`, default `0.12`): maximum time between candidate onset and confirmation.

#### `gestures` root fields

- `history_size` (`int`, default `6`): maximum wrist samples retained per hand.
- `active_hand` (`left | right`, default `right`): the hand eligible to trigger events.
- `velocity_smoothing_alpha` (`float`, default `0.8`): exponential smoothing factor applied before peak-velocity measurement.
- `rearm_threshold_ratio` (`float`, default `0.45`): fraction of the base gesture travel required before the same hand can retrigger after recovery.

### `audio`

Controls sample playback.

- `backend` (`str`, default `pygame`): backend name. Currently only `pygame` is supported.
- `sample_rate` (`int`, default `44100`): mixer sample rate.
- `buffer_size` (`int`, default `256`): mixer buffer size.
- `output_channels` (`int`, default `2`): channel count for mixer initialization.
- `simultaneous_voices` (`int`, default `16`): number of overlapping mixer channels.
- `output_device_name` (`str | null`, default `null`): optional device selection hint.
- `sample_mapping` (`map[str, str]`): mapping from sound names to file paths.
- `volume` (`float`, default `0.9`): master playback level in `0.0..1.0`.

### `transport`

Controls optional external gesture-event forwarding.

- `backend` (`none | udp`, default `none`): transport backend selection.
- `host` (`str`, default `127.0.0.1`): UDP destination host when `backend = udp`.
- `port` (`int`, default `9000`): UDP destination port.
- `source` (`str`, default `visionbeat`): sender identifier embedded in emitted transport messages.

### `debug.overlays`

Controls what the preview overlay renders.

- `draw_landmarks` (`bool`, default `true`): draw the tracked upper-body skeleton.
- `draw_velocity_vectors` (`bool`, default `true`): reserved overlay toggle for velocity diagnostics.
- `show_landmark_labels` (`bool`, default `true`): show landmark-name text next to joints.
- `show_debug_panel` (`bool`, default `true`): show text status/candidate/cooldown information.
- `show_trigger_flash` (`bool`, default `true`): show the red trigger confirmation flash.

### `logging`

Controls log output and optional event tracing.

- `level` (`str`, default `INFO`): root logging level.
- `format` (`str`): standard log format string.
- `structured` (`bool`, default `true`): append structured JSON payloads to log lines.
- `event_log_path` (`str | null`, default `null`): optional file path for event tracing.
- `event_log_format` (`jsonl | csv`, default `jsonl`): file format for event traces.
- `session_recording_path` (`str | null`, default `null`): optional directory where VisionBeat creates timestamped session bundles for replay and analysis.
- `session_recording_mode` (`tracker_outputs | raw_frames | both`, default `tracker_outputs`): whether a session bundle stores normalized tracker outputs only, raw camera frames only, or both.

## Calibration guide

Thresholds are not one-size-fits-all. Calibration should be treated as part of setting up the instrument.

### 1. Start from a stable camera setup

Do not tune gestures while tracking is unstable. First ensure the performer remains well framed and the wrist stays visible through the full motion arc.

### 2. Tune the kick gesture

If downward strikes do not trigger:

- lower `strike_down_delta_y` slightly,
- or lower `min_velocity` if the performer uses softer attacks.

If kicks false-trigger during unrelated diagonal movement:

- increase `axis_dominance_ratio`,
- or tighten `strike_max_depth_drift`.

### 3. Tune the snare gesture

If wrist collisions do not trigger:

- increase `snare_collision_distance`,
- or slightly extend `analysis_window_seconds` if the motion unfolds more gradually.

If near-crossings or loose arm passes trigger snares:

- increase `min_velocity`,
- tighten `snare_collision_max_depth_gap`,
- or require a smaller `snare_collision_distance`.

### 4. Tune timing behavior

- reduce `trigger_seconds` for faster repeated hits,
- increase it to prevent double-triggering,
- extend `confirmation_window_seconds` if gestures begin correctly but fail to confirm,
- shorten it if stale partial movements confirm too late.

## Example tuning scenarios

### Performer stands farther from the camera

Likely symptom: gestures feel too small in normalized space.

Try:

- lowering `strike_down_delta_y`,
- increasing `snare_collision_distance`,
- and keeping `min_velocity` high enough to avoid false positives.

### Performer uses large theatrical movements

Likely symptom: the system triggers too easily on expressive transitions.

Try:

- raising `axis_dominance_ratio`,
- tightening drift thresholds,
- and increasing `trigger_seconds` if one motion arc produces two hits.

### Lighting is inconsistent

Likely symptom: candidates disappear because landmarks flicker.

Try improving the environment first, then revisit tracker confidence thresholds if needed.

## Troubleshooting config errors

VisionBeat reports errors with qualified config paths. Typical examples include:

- `camera.width: expected an integer.`
- `gestures.thresholds.candidate_ratio: must be less than or equal to 1.0.`
- `logging: unknown field(s): strucutred.`

When a config fails, compare it against the default config before debugging runtime behavior.

## Recommended workflow for experiments

If you are doing research runs or repeated demos:

1. keep a copy of the baseline config,
2. make one threshold change at a time,
3. save event logs when evaluating changes,
4. document the performer/camera setup alongside the config,
5. and treat the config as part of the experimental method.

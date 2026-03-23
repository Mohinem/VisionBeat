# Gesture Definitions

VisionBeat gesture detection operates only on normalized wrist landmark samples and timestamps. It does not depend on webcam frame objects or MediaPipe-specific classes.

## Inputs

For each tracked hand, the detector keeps a temporal window of wrist samples:

- `s_i = (t_i, x_i, y_i, z_i)`
- `t_i` is the frame timestamp in seconds.
- `x_i`, `y_i`, and `z_i` are normalized wrist coordinates.
- The detector retains only samples with `t_i >= t_now - analysis_window_seconds`.

Given the oldest sample `s_0` and newest sample `s_n` in the active window:

- `Δx = x_n - x_0`
- `Δy = y_n - y_0`
- `Δz = z_n - z_0`
- `Δt = t_n - t_0`
- `net_velocity = (|Δx| + |Δy| + |Δz|) / max(Δt, ε)`

For each adjacent pair of samples, the detector also computes per-frame velocities:

- `v_x(i) = (x_i - x_{i-1}) / (t_i - t_{i-1})`
- `v_y(i) = (y_i - y_{i-1}) / (t_i - t_{i-1})`
- `v_z(i) = (z_i - z_{i-1}) / (t_i - t_{i-1})`

And then derives:

- `forward_velocity = max(0, -max_abs(v_z))`
- `downward_velocity = max(0, max_abs(v_y))`
- `punch_axis_ratio = |Δz| / max(|Δx| + |Δy|, ε)`
- `strike_axis_ratio = |Δy| / max(|Δx| + |Δz|, ε)`

`ε` is a very small positive constant that prevents division by zero.

## Candidate vs Confirmed Trigger

Gesture detection is intentionally two-stage and stateful.

### Candidate stage

A gesture becomes a **candidate** when motion in the temporal window reaches an onset threshold:

```text
candidate_threshold = configured_threshold * candidate_ratio
```

The detector stores a pending gesture candidate with a `started_at` timestamp. Candidates expire if either:

1. the elapsed candidate age exceeds `confirmation_window_seconds`, or
2. the motion window no longer satisfies the candidate constraints.

### Confirmed trigger stage

A gesture becomes a **confirmed trigger** only when the pending candidate reaches the full configured threshold before the confirmation window expires. On confirmation, the detector emits a `GestureEvent` and starts cooldown for that hand.

## Forward Punch → Kick

A forward punch is motion dominated by decreasing normalized `z` (toward the camera) with limited vertical drift.

### Candidate rule

```text
if (
    Δz <= -(punch_forward_delta_z * candidate_ratio)
    and |Δy| <= punch_max_vertical_drift
    and forward_velocity >= min_velocity * candidate_ratio
    and net_velocity >= min_velocity * candidate_ratio
    and punch_axis_ratio >= axis_dominance_ratio
):
    pending = KICK
```

### Confirmation rule

```text
if pending == KICK and (
    Δz <= -punch_forward_delta_z
    and |Δy| <= punch_max_vertical_drift
    and forward_velocity >= min_velocity
    and net_velocity >= min_velocity
    and punch_axis_ratio >= axis_dominance_ratio
):
    emit GestureEvent(KICK)
```

## Downward Strike → Snare

A downward strike is motion dominated by increasing normalized `y` (downward on screen) with limited forward/back depth drift.

### Candidate rule

```text
if (
    Δy >= strike_down_delta_y * candidate_ratio
    and |Δz| <= strike_max_depth_drift
    and downward_velocity >= min_velocity * candidate_ratio
    and net_velocity >= min_velocity * candidate_ratio
    and strike_axis_ratio >= axis_dominance_ratio
):
    pending = SNARE
```

### Confirmation rule

```text
if pending == SNARE and (
    Δy >= strike_down_delta_y
    and |Δz| <= strike_max_depth_drift
    and downward_velocity >= min_velocity
    and net_velocity >= min_velocity
    and strike_axis_ratio >= axis_dominance_ratio
):
    emit GestureEvent(SNARE)
```

## Cooldown / Debounce

After any confirmed trigger, the detector ignores new candidates on that hand for `cooldown_seconds`:

```text
if t_now - last_trigger_time < cooldown_seconds:
    suppress candidate creation and suppress triggers
```

This debounce logic prevents one continuous punch or strike from producing repeated rapid-fire hits.

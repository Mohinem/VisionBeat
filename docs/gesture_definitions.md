# Gesture Definitions

## Purpose of this document

This document specifies VisionBeat’s current gesture vocabulary in a formal, implementation-aligned way. The goal is to describe gestures as **temporal movement patterns over normalized wrist trajectories**, not as vague labels like “punch” or “strike.”

The current detector operates on the configured **active hand** and recognizes two gesture classes:

- **Forward punch → kick**
- **Downward strike → snare**

## Coordinate interpretation

VisionBeat uses normalized MediaPipe pose coordinates:

- `x`: horizontal image position,
- `y`: vertical image position, where larger values move downward on screen,
- `z`: normalized depth, where more negative values move toward the camera.

The signs matter:

- a **forward** motion tends to produce **negative `Δz`**, and
- a **downward** motion tends to produce **positive `Δy`**.

## Temporal input model

For each hand, the detector stores a bounded history of wrist samples:

- `s_i = (t_i, x_i, y_i, z_i)`
- `t_i` is the sample timestamp in seconds,
- `(x_i, y_i, z_i)` are normalized wrist coordinates.

Only samples inside the configured analysis window are retained.

## Motion metrics

Given the oldest and newest samples in the active history window:

- `Δx = x_n - x_0`
- `Δy = y_n - y_0`
- `Δz = z_n - z_0`
- `Δt = t_n - t_0`

The detector computes a simple L1-style net velocity:

- `net_velocity = (|Δx| + |Δy| + |Δz|) / max(Δt, ε)`

For each adjacent sample pair, the detector also computes axis-wise velocities:

- `v_x(i) = (x_i - x_{i-1}) / (t_i - t_{i-1})`
- `v_y(i) = (y_i - y_{i-1}) / (t_i - t_{i-1})`
- `v_z(i) = (z_i - z_{i-1}) / (t_i - t_{i-1})`

From those, it derives peak directional magnitudes:

- `forward_velocity = max(0, -peak_z_velocity)`
- `downward_velocity = max(0, peak_y_velocity)`

and axis-dominance ratios:

- `punch_axis_ratio = |Δz| / max(|Δx| + |Δy|, ε)`
- `strike_axis_ratio = |Δy| / max(|Δx| + |Δz|, ε)`

`ε` is a small positive constant used to avoid division by zero and to keep threshold comparisons numerically stable.

## State machine overview

Gesture detection is **stateful**. Each hand history may be in one of these practical states:

1. **Idle** — no valid gesture candidate is active.
2. **Candidate active** — onset motion resembles a supported gesture but has not yet reached full confirmation.
3. **Confirmed trigger** — the gesture reached its confirmation threshold and emitted a `GestureEvent`.
4. **Cooldown** — new candidates are temporarily suppressed after a confirmed hit.

This design improves musical behavior because it separates **gesture onset** from **gesture confirmation**.

## Candidate threshold scaling

The detector uses a configurable `candidate_ratio` to create a lower onset threshold:

```text
candidate_threshold = full_threshold × candidate_ratio
```

This means the system can notice a promising gesture early, then demand stronger evidence before emitting audio.

## Forward punch → kick

### Informal description

A forward punch is a wrist trajectory dominated by **movement toward the camera** with limited vertical drift and sufficient motion energy.

### Candidate condition

A kick candidate is started when all of the following are true:

```text
Δz <= -(punch_forward_delta_z × candidate_ratio)
|Δy| <= punch_max_vertical_drift
forward_velocity >= min_velocity × candidate_ratio
net_velocity >= min_velocity × candidate_ratio
punch_axis_ratio >= axis_dominance_ratio
```

### Confirmation condition

The candidate becomes a confirmed kick when all full-threshold conditions are true before the confirmation window expires:

```text
Δz <= -punch_forward_delta_z
|Δy| <= punch_max_vertical_drift
forward_velocity >= min_velocity
net_velocity >= min_velocity
punch_axis_ratio >= axis_dominance_ratio
```

### Musical interpretation

This gesture is intended to feel like a **bass-drum hit**: direct, frontal, and impact-oriented.

## Downward strike → snare

### Informal description

A downward strike is a wrist trajectory dominated by **movement downward in image space** with limited forward/back depth drift and sufficient motion energy.

### Candidate condition

A snare candidate is started when all of the following are true:

```text
Δy >= strike_down_delta_y × candidate_ratio
|Δz| <= strike_max_depth_drift
downward_velocity >= min_velocity × candidate_ratio
net_velocity >= min_velocity × candidate_ratio
strike_axis_ratio >= axis_dominance_ratio
```

### Confirmation condition

The candidate becomes a confirmed snare when all full-threshold conditions are true before the confirmation window expires:

```text
Δy >= strike_down_delta_y
|Δz| <= strike_max_depth_drift
downward_velocity >= min_velocity
net_velocity >= min_velocity
strike_axis_ratio >= axis_dominance_ratio
```

### Musical interpretation

This gesture is intended to feel like a **snare hit**: vertical, accented, and visually legible.

## Cooldown and debounce

After a confirmed trigger, the detector ignores new candidates on the active hand for `cooldown_seconds`:

```text
if now - last_trigger_time < cooldown_seconds:
    suppress candidates and triggers
```

This prevents a single sweeping movement from causing multiple repeated drum hits.

## Candidate expiration

A pending candidate is cleared when either:

1. the gesture no longer satisfies the candidate constraints, or
2. the elapsed time since candidate start exceeds `confirmation_window_seconds`.

This protects against stale partial motions that should not linger and suddenly confirm later.

## Active-hand constraint

Although the detector stores histories for both left and right wrists, it only evaluates the configured `active_hand` for triggering. The other hand is still tracked, but it does not produce kick/snare events under the current configuration model.

## Confidence estimation

The detector assigns a confidence score to candidates and confirmed events. The score is derived from how strongly the motion exceeds the primary displacement and velocity thresholds, then clamped into `[0.0, 1.0]`.

Interpret confidence as a **relative trigger strength**, not as a probabilistic classifier confidence.

## Formal summary table

| Gesture | Primary axis | Required signed displacement | Restricted drift | Directional velocity | Axis dominance |
|---|---|---:|---:|---:|---:|
| Kick | `z` toward camera | `Δz < 0` and magnitude above threshold | `|Δy|` must stay small | forward velocity must exceed threshold | `|Δz| / (|Δx| + |Δy|)` |
| Snare | `y` downward on screen | `Δy > 0` and magnitude above threshold | `|Δz|` must stay small | downward velocity must exceed threshold | `|Δy| / (|Δx| + |Δz|)` |

## Calibration implications

The definitions above explain why false positives and false negatives happen:

- If diagonal motion is misclassified, `axis_dominance_ratio` is probably too low.
- If slow drifts trigger sounds, `min_velocity` is too low.
- If energetic hits never confirm, the displacement thresholds may be too high.
- If a gesture starts but expires before confirming, the confirmation window may be too short or the movement may not sustain momentum.

## Future gesture extensions

The formalism can be extended without discarding the current architecture.

### Clap detection

A clap would likely depend less on single-wrist displacement and more on:

- bilateral wrist distance,
- inward relative velocity,
- and a small hand-separation threshold near impact.

### Cymbal or tom gestures

Additional percussive gestures could be defined through distinct dominant axes, bilateral asymmetry, or torso-relative strike zones.

### Classifier-based recognition

If VisionBeat later adopts learned gesture recognition, these current heuristics still provide a useful baseline and an interpretable benchmark.

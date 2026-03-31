# Gesture Definitions

## Purpose of this document

This document specifies VisionBeat's current gesture vocabulary in an implementation-aligned way. The goal is to describe gestures as **temporal movement patterns over normalized wrist trajectories**, not as vague labels like "punch" or "strike."

The current detector operates on the configured **active hand** and recognizes two gesture classes:

- **Inward side-jab → kick**
- **Downward strike → snare**

## Coordinate interpretation

VisionBeat uses normalized MediaPipe pose coordinates:

- `x`: horizontal image position,
- `y`: vertical image position, where larger values move downward on screen,
- `z`: normalized depth, where more negative values move toward the camera.

For kick detection, the important detail is that the detector prefers **shoulder-relative horizontal motion** when shoulder tracking is available:

- if the shoulder is visible, the detector stores `x = wrist_x - shoulder_x`, so inward kick motion means the wrist moves **toward the shoulder / centerline** and `|x|` shrinks,
- if the shoulder is not visible, the detector falls back to raw wrist `x` and interprets inward motion by hand:
  - right hand: inward means `Δx < 0`
  - left hand: inward means `Δx > 0`

Depth `z` still matters for net velocity and for snare drift rejection, but it is **not** the primary kick axis in the current implementation.

## Temporal input model

For each hand, the detector stores a bounded history of wrist samples:

- `s_i = (t_i, x_i, y_i, z_i)`
- `t_i` is the sample timestamp in seconds
- `(x_i, y_i, z_i)` are either shoulder-relative or raw wrist coordinates depending on shoulder visibility

Only samples inside the configured analysis window are retained, and the history is reset if the detector has to switch between shoulder-relative and raw-coordinate mode mid-stream.

## Motion metrics

Given the oldest and newest samples in the active history window:

- `Δx = x_n - x_0`
- `Δabs_x = |x_n| - |x_0|`
- `Δy = y_n - y_0`
- `Δz = z_n - z_0`
- `Δt = t_n - t_0`

The detector computes a simple L1-style net velocity:

- `net_velocity = (|Δx| + |Δy| + |Δz|) / max(Δt, ε)`

For each adjacent sample pair, the detector also computes axis-wise velocities:

- `v_x(i) = (x_i - x_{i-1}) / (t_i - t_{i-1})`
- `v_abs_x(i) = (|x_i| - |x_{i-1}|) / (t_i - t_{i-1})`
- `v_y(i) = (y_i - y_{i-1}) / (t_i - t_{i-1})`
- `v_z(i) = (z_i - z_{i-1}) / (t_i - t_{i-1})`

From those, it derives peak directional magnitudes:

- `inward_velocity = max(0, -peak_abs_x_velocity)` in shoulder-relative mode
- `downward_velocity = max(0, peak_y_velocity)`

When the detector is in raw-coordinate fallback mode, inward kick velocity is interpreted by hand:

- right hand: `max(0, -peak_x_velocity)`
- left hand: `max(0, peak_x_velocity)`

It also derives gesture-shape ratios:

- `lateral_axis_ratio = |Δabs_x| / max(|Δy| + |Δz|, ε)` for kick analysis
- `strike_axis_ratio = |Δy| / max(|Δx| + |Δz|, ε)` for snare analysis

`ε` is a small positive constant used to avoid division by zero and to keep comparisons numerically stable.

## State machine overview

Gesture detection is stateful. Each hand history may be in one of these practical states:

1. **Idle** - no valid gesture candidate is active.
2. **Candidate active** - onset motion resembles a supported gesture but has not yet reached full confirmation.
3. **Confirmed trigger** - the gesture reached its confirmation threshold and emitted a `GestureEvent`.
4. **Recovery / cooldown** - new candidates are temporarily suppressed until the post-trigger reset completes and the cooldown elapses.

This design improves musical behavior because it separates **gesture onset** from **gesture confirmation** and allows the detector to bias ambiguous motion toward the more likely class.

## Derived thresholds

The current kick path is intentionally permissive. The detector derives these helper thresholds from config:

```text
kick_distance_threshold = punch_forward_delta_z × candidate_ratio
kick_velocity_floor = min_velocity × 0.08
kick_axis_threshold = max(0.4, axis_dominance_ratio × candidate_ratio × 0.5)

kick_bias_distance = kick_distance_threshold × 0.75
kick_bias_velocity = kick_velocity_floor × 0.75
kick_bias_axis_threshold = max(0.3, kick_axis_threshold × 0.75)
```

Important consequence: **kick confirmation currently uses the same relaxed distance threshold as kick candidate detection**. This is deliberate so short inward jabs can still produce audio.

Snare still uses the more traditional candidate-vs-confirm split:

```text
snare_candidate_distance = strike_down_delta_y × candidate_ratio
snare_confirm_distance = strike_down_delta_y
```

## Inward side-jab → kick

### Informal description

A kick is a quick **inward lateral jab** of the active wrist toward the player's centerline. The motion can include some downward drift, but it should still read as primarily sideways-inward rather than purely downward.

### Kick-preference heuristic

Before the detector considers snare, it checks whether the motion is already "kick-like enough" to deserve kick priority. That heuristic is:

```text
inward_displacement >= kick_bias_distance
|Δy| <= punch_max_vertical_drift
inward_velocity >= kick_bias_velocity
net_velocity >= kick_bias_velocity
lateral_axis_ratio >= kick_bias_axis_threshold
```

If those conditions hold, the detector:

- may start a kick candidate early,
- may keep an existing kick candidate alive,
- and blocks snare candidate/confirmation on the same motion window.

### Candidate condition

A kick candidate is considered fully valid when all of the following are true:

```text
inward_displacement >= kick_distance_threshold
|Δy| <= punch_max_vertical_drift
inward_velocity >= kick_velocity_floor
net_velocity >= kick_velocity_floor
lateral_axis_ratio >= kick_axis_threshold
```

`inward_displacement` means:

- `max(0, -Δabs_x)` in shoulder-relative mode,
- `max(0, -Δx)` for the right hand in raw fallback mode,
- `max(0, Δx)` for the left hand in raw fallback mode.

### Confirmation condition

The candidate becomes a confirmed kick when the same full kick condition above is true before the confirmation window expires.

### Musical interpretation

This gesture is intended to feel like a **bass-drum hit**: short, direct, and easy to repeat without a large forward lunge.

## Downward strike → snare

### Informal description

A snare is a downward wrist trajectory dominated by **movement downward in image space** with limited depth drift and sufficient motion energy.

### Candidate condition

A snare candidate is started when all of the following are true:

```text
not kick_preference
Δy >= strike_down_delta_y × candidate_ratio
|Δz| <= strike_max_depth_drift
downward_velocity >= min_velocity × candidate_ratio
net_velocity >= min_velocity × candidate_ratio
strike_axis_ratio >= axis_dominance_ratio
```

The explicit `not kick_preference` term is important: it makes snare significantly less likely to steal diagonal inward-jab motion that should really remain on the kick path.

### Confirmation condition

The candidate becomes a confirmed snare when all full-threshold conditions are true before the confirmation window expires:

```text
not kick_preference
Δy >= strike_down_delta_y
|Δz| <= strike_max_depth_drift
downward_velocity >= min_velocity
net_velocity >= min_velocity
strike_axis_ratio >= axis_dominance_ratio
```

### Musical interpretation

This gesture is intended to feel like a **snare hit**: vertical, accented, and visually legible.

## Cooldown and recovery

After a confirmed trigger, the detector will not immediately allow another hit from the same motion arc. It requires:

1. the configured cooldown to elapse, and
2. the hand to move back in the opposite direction far enough to re-arm.

For kick, this means the hand must recover **outward** after the inward jab. For snare, the hand must recover **upward** after the downward strike.

## Candidate expiration

A pending candidate is cleared when either:

1. the gesture no longer satisfies its candidate constraints,
2. the elapsed time since candidate start exceeds `confirmation_window_seconds`, or
3. the detector has to reset the wrist history because the underlying coordinate mode changed.

This protects against stale partial motions that should not linger and suddenly confirm later.

## Active-hand constraint

Although the detector stores histories for both left and right wrists, it only evaluates the configured `active_hand` for triggering. The other hand is still tracked, but it does not produce kick/snare events under the current configuration model.

## Confidence estimation

The detector assigns a confidence score to candidates and confirmed events. The score is derived from how strongly the motion exceeds the primary displacement and velocity thresholds, then clamped into `[0.0, 1.0]`.

Interpret confidence as a **relative trigger strength**, not as a probabilistic classifier confidence.

## Formal summary table

| Gesture | Primary axis | Required signed displacement | Restricted drift | Directional velocity | Axis dominance | Bias logic |
|---|---|---:|---:|---:|---:|---|
| Kick | inward lateral `x` toward centerline | inward displacement above `punch_forward_delta_z × candidate_ratio` | `|Δy|` must stay below `punch_max_vertical_drift` | inward velocity above `min_velocity × 0.08` | `|Δabs_x| / (|Δy| + |Δz|)` | preferred over snare when sufficiently kick-like |
| Snare | downward `y` on screen | `Δy > 0` and magnitude above threshold | `|Δz|` must stay below `strike_max_depth_drift` | downward velocity above threshold | `|Δy| / (|Δx| + |Δz|)` | blocked when kick-preference heuristic is active |

## Calibration implications

The definitions above explain why false positives and false negatives happen:

- If inward jabs never confirm, `punch_forward_delta_z` or `min_velocity` is probably too high.
- If inward jabs frequently become snares, the kick path is still too strict for the performer; reduce `punch_forward_delta_z` or increase `punch_max_vertical_drift` before loosening snare.
- If accidental sideways movement produces too many kicks, raise `axis_dominance_ratio` or `punch_forward_delta_z`.
- If slow drifts trigger sounds, `min_velocity` is too low.
- If downward hits never confirm, `strike_down_delta_y` may be too high or the performer may be introducing too much depth drift.
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

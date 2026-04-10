# Gesture Definitions

## Purpose

This document describes VisionBeat's current gesture vocabulary in terms that match the implementation in [`src/visionbeat/gestures.py`](../src/visionbeat/gestures.py).

The live instrument currently recognizes two gesture classes:

- **Downward strike → kick**
- **Wrist collision → snare**

## Coordinate interpretation

VisionBeat uses normalized pose coordinates:

- `x`: horizontal image position
- `y`: vertical image position, increasing downward on screen
- `z`: normalized depth

For the active-hand kick path, the detector prefers **shoulder-relative wrist coordinates** when the relevant shoulder is visible. This reduces false positives from torso sway by measuring wrist travel relative to the shoulder rather than raw screen position.

For the bilateral snare path, the detector uses the raw left/right wrist landmarks directly and tracks the **gap between the wrists** in image space and depth.

## State model

The detector maintains two independent trigger paths:

1. an **active-hand motion path** for kick,
2. a **bilateral wrist-distance path** for snare.

Each path follows the same high-level state machine:

1. **Idle**
2. **Candidate active**
3. **Confirmed trigger**
4. **Recovery / cooldown**

Candidates must confirm before `confirmation_window_seconds` expires. After a trigger, the gesture must both clear cooldown and satisfy a recovery movement before it can fire again.

## Kick: downward strike

### Informal description

A kick is now a clear **downward strike** performed by the configured `active_hand`.

### Motion metrics

For the active hand, the detector stores wrist samples over a rolling time window and derives:

- `Δx`, `Δy`, `Δz`
- `net_velocity = (|Δx| + |Δy| + |Δz|) / Δt`
- `downward_velocity = max(0, peak_y_velocity)`
- `strike_axis_ratio = |Δy| / max(|Δx| + |Δz|, ε)`

### Candidate condition

A kick candidate starts when:

```text
Δy >= strike_down_delta_y × candidate_ratio
|Δz| <= strike_max_depth_drift
downward_velocity >= min_velocity × candidate_ratio
net_velocity >= min_velocity × candidate_ratio
strike_axis_ratio >= axis_dominance_ratio
```

### Confirmation condition

The kick confirms when the same shape holds at full threshold:

```text
Δy >= strike_down_delta_y × strike_confirmation_ratio
|Δz| <= strike_max_depth_drift
downward_velocity >= min_velocity
net_velocity >= min_velocity
strike_axis_ratio >= axis_dominance_ratio
```

### Recovery behavior

After a kick trigger, the active hand must move **upward** enough to re-arm. This prevents a single long downward sweep from producing repeated kick hits.

## Snare: wrist collision

### Informal description

A snare is a bilateral **wrist collision**: the performer brings the right wrist into the left wrist area until the two tracked wrists converge tightly in image space.

### Bilateral separation metrics

The snare path tracks:

- horizontal wrist gap
- vertical wrist gap
- depth gap
- image-plane wrist distance

From those samples it derives:

- `current_distance_xy`
- `current_depth_gap`
- `closing_velocity = max(0, -peak_distance_velocity)`
- `net_velocity` over the wrist-gap deltas

### Candidate condition

A snare candidate starts when:

```text
current_distance_xy <= snare_collision_distance / candidate_ratio
current_depth_gap <= snare_collision_max_depth_gap
closing_velocity >= min_velocity × candidate_ratio
net_velocity >= min_velocity × candidate_ratio
```

### Confirmation condition

The snare confirms when:

```text
current_distance_xy <= snare_collision_distance
current_depth_gap <= snare_collision_max_depth_gap
closing_velocity >= min_velocity × snare_confirmation_velocity_ratio
net_velocity >= min_velocity
```

### Recovery behavior

After a snare trigger, the wrists must separate again before another snare can fire. This prevents one sustained hand-touch from retriggering repeatedly.

## Priority between gestures

When a valid snare collision candidate is active on the current frame, the detector suppresses the simultaneous kick path. This gives the bilateral collision gesture priority when the active hand is also moving downward during the approach.

## Active-hand behavior

- `active_hand` still selects which hand can produce the **kick**.
- **Snare** remains bilateral and uses both wrists.
- Snare events currently report the configured `active_hand` in the event payload so the public event model stays compatible with the existing `left|right` hand enum contract.

## Confidence

Candidate and trigger confidence values are normalized strength scores derived from threshold exceedance. They should be interpreted as relative trigger strength, not classifier probability.

## Practical calibration notes

- If kicks arm but do not confirm, reduce `strike_confirmation_ratio`, reduce `strike_down_delta_y`, or reduce `min_velocity`.
- If kicks false-trigger on diagonal arm motion, increase `axis_dominance_ratio` or tighten `strike_max_depth_drift`.
- If snares arm but do not confirm, lower `snare_confirmation_velocity_ratio`, increase the physical closeness of the wrists, or raise `snare_collision_distance`.
- If accidental near-crossings trigger snares, lower `snare_collision_distance`, lower `snare_collision_max_depth_gap`, or raise `min_velocity`.

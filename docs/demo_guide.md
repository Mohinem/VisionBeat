# Demo Guide

## Purpose

This guide explains how to present and operate VisionBeat as a live demo or research prototype. It assumes you want to show VisionBeat as a **camera-based gestural percussion instrument for embodied rhythm performance** rather than as a generic pose-estimation application.

## Demo goal

A successful demo should communicate three things clearly:

1. **The sensing setup is lightweight** — one webcam, no wearables.
2. **The gesture vocabulary is musically legible** — punch-like and strike-like movements create distinct percussion events.
3. **The project is extensible** — it is a structured instrument prototype, not a one-off effect.

## Recommended demo narrative

A concise live explanation could be:

> VisionBeat uses a webcam to track upper-body motion and interpret arm gestures as percussion hits. A forward punch triggers a kick, and a downward strike triggers a snare. The goal is to explore embodied rhythm performance with minimal sensing hardware while keeping the gesture logic transparent and configurable.

## Pre-demo checklist

### Software

- activate the project virtual environment,
- install dependencies,
- generate local samples,
- verify the config path you want to use,
- and keep a fallback config ready if you have tuned thresholds experimentally.

### Hardware

- webcam connected and selected correctly,
- speakers/headphones connected,
- stable lighting,
- enough free floor space for visible arm motion,
- and a clear background if possible.

### Runtime

- launch the app before the presentation starts,
- verify the overlay shows tracked landmarks,
- perform a few warm-up gestures,
- and confirm both kick and snare trigger cleanly.

## Basic launch sequence

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python scripts/generate_demo_samples.py
visionbeat --config configs/default.yaml
```

If you need a different camera:

```bash
visionbeat --config configs/default.yaml --camera-index 1
```

## Room setup advice

For a robust demo:

- place the camera roughly front-facing,
- keep the performer centered,
- avoid strong windows or lights behind the performer,
- and ensure the full upper body remains visible during the biggest gesture.

In practice, the demo works best when the audience can see both:

- the performer’s body motion,
- and the VisionBeat preview window with overlay feedback.

## Suggested demo flow

### 1. Introduce the concept

Briefly explain that VisionBeat is a gestural percussion instrument controlled by visible upper-body movement.

### 2. Show the sensing view

Point out the overlay:

- shoulder/elbow/wrist landmarks,
- tracker status,
- active candidate,
- confirmed gesture,
- and cooldown feedback.

### 3. Demonstrate single gestures in isolation

Perform several clean examples of:

- forward punch → kick,
- downward strike → snare.

Let the audience associate each movement class with the sound.

### 4. Demonstrate rhythm building

Alternate kick/snare gestures to create a simple groove. Keep the pattern sparse at first so the mapping remains obvious.

### 5. Discuss tuning and extensibility

Explain that thresholds are configurable and that future versions can add clap detection, MIDI output, or a JUCE audio backend.

## Performance tips

- Use **clear, accented gestures** instead of subtle motions.
- Return to a neutral pose between hits when possible.
- Keep the active wrist visible.
- Avoid crossing the striking arm across the torso too aggressively.
- If the system is missing hits, slightly exaggerate the primary gesture axis before changing thresholds live.

## Calibration tips before the audience arrives

### Kick calibration

Test whether a clean forward extension triggers reliably. If not:

- lower `punch_forward_delta_z`,
- or step a bit closer to the camera.

### Snare calibration

Test whether a clean downward accent triggers reliably. If not:

- lower `strike_down_delta_y`,
- or raise the camera slightly so the downward motion remains visible.

### Repeat-hit calibration

If a quick succession of hits is blocked:

- reduce `gestures.cooldowns.trigger_seconds`.

If one gesture causes multiple triggers:

- increase `trigger_seconds`,
- or tighten the gesture thresholds.

## Troubleshooting during a live demo

### Landmarks disappear

- pause,
- re-center in frame,
- improve body visibility,
- and reduce motion speed briefly until tracking recovers.

### Gestures are visible but don’t sound

- verify the configured active hand,
- try a slightly more axis-clean motion,
- and confirm audio is actually still working.

### Audio works intermittently

- check system output routing,
- reduce competing applications,
- and consider increasing `audio.buffer_size` if the environment is unstable.

## Known live-demo limitations

- The prototype is sensitive to lighting and framing.
- Very fast gestures may challenge tracking in low light.
- The current instrument has only two built-in gesture classes.
- `pygame` playback is good for demos, but not the final word on latency-sensitive performance.

## Future demo extensions

A compelling next-stage demo could include:

- **clap detection** as an additional accent gesture,
- **MIDI output** to trigger an external drum rack,
- **JUCE-backed playback** for improved latency and reliability,
- or a larger gesture vocabulary for full drum-kit performance.

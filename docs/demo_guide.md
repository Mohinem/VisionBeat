# VisionBeat Demo Guide (Research Presentation)

This guide is optimized for a **2-minute live demo** during a research presentation.

## Camera positioning

- Place the webcam **at chest-to-shoulder height**.
- Keep the camera **facing the performer head-on** (not at a steep side angle).
- Tilt slightly downward only if needed so both wrists stay visible.
- Use even front lighting; avoid bright windows behind the performer.

## Performer distance from camera

- Stand about **1.5 to 2.5 meters (5 to 8 feet)** from the camera.
- Keep the upper body fully visible: shoulders, elbows, and wrists should remain in frame.
- If wrists leave the frame during downward strikes, either:
  - step back a little, or
  - raise the camera slightly.

## Performing gestures for reliable detection

### Downward strike (kick)

- Start from a comfortable raised position.
- Move the wrist **clearly downward** in one accented motion.
- Avoid excessive forward/backward travel during the strike.
- Re-center before the next gesture to reduce ambiguity.

### Wrist collision (snare)

- Keep both wrists visible before starting the gesture.
- Move the right wrist toward the left wrist in one clear approach.
- Let the wrists come **close together in image space** rather than only passing each other at different depths.
- Separate again after the hit so the detector can re-arm cleanly.

## Common failure modes and quick fixes

- **No webcam feed / startup error**
  - Check camera permissions.
  - Ensure no other app is using the webcam.
  - Try another device index: `visionbeat --config configs/default.yaml --camera-index 1`.

- **Landmarks flicker or disappear**
  - Improve lighting.
  - Step back so upper body stays fully in frame.
  - Slow down briefly until tracking stabilizes.

- **Gestures visible, but few triggers**
  - Use larger, cleaner motion on the primary axis.
  - For kick, exaggerate the downward strike.
  - For snare, bring the wrists genuinely close together rather than only crossing paths.
  - Try `--sensitivity expressive` for demos.
  - Confirm the configured active hand matches your performing hand.

- **Too many accidental triggers**
  - Use more axis-clean gestures.
  - Try `--sensitivity conservative`.

- **No audio output / missing sample warnings**
  - Regenerate samples: `python scripts/generate_demo_samples.py`.
  - Verify configured sample paths exist.

## Demo HUD and keyboard controls

During the demo, the overlay shows:

- title and runtime status,
- currently detected motion candidate,
- last confirmed trigger,
- armed cooldown state,
- quick controls.

Keyboard shortcuts while running:

- `o` → toggle overlays on/off,
- `d` → toggle debug panel,
- `q` or `Esc` → quit.

For dataset capture, start with `--skeleton-only-hud` to hide debug text, landmark labels, and trigger flashes while keeping the skeleton overlay.

## Recommended 2-minute demo script

### 0:00 - 0:20 (Context)

“VisionBeat is a camera-based gestural percussion instrument. A single webcam tracks arm motion and maps gestures to drum hits in real time.”

### 0:20 - 0:45 (Show sensing + HUD)

- Point to landmarks and status HUD.
- Mention that the system displays candidates, confirmed triggers, and cooldown state.

### 0:45 - 1:15 (Gesture mapping)

- Perform 3–4 clear downward strikes (kick).
- Perform 3–4 clear wrist-collision snares.
- Briefly state the mapping aloud as you perform.

### 1:15 - 1:45 (Mini groove)

- Alternate kick/snare to create a simple rhythmic pattern.
- Keep gestures clear and separated so the mapping remains understandable.

### 1:45 - 2:00 (Research takeaway)

“This prototype explores embodied rhythm interaction with minimal hardware, transparent heuristics, and configurable sensitivity for different performers and room setups.”

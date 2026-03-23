# Tracking Notes

## Landmarks used

VisionBeat tracks only six MediaPipe pose landmarks: left/right shoulders, left/right elbows, and left/right wrists. This is the minimum upper-body set needed by the current gesture detector because the detector only reasons about arm chain motion and wrist trajectories.

## Why these landmarks are sufficient

For forward punch detection, the detector primarily looks for a wrist moving quickly toward the camera (negative `z` delta) while keeping vertical drift relatively small. Shoulder and elbow landmarks provide enough context for overlays and future posture validation without requiring the full 33-point pose skeleton.

For downward strike detection, the detector watches for a wrist dropping quickly in normalized image `y` while limiting depth drift. Again, the shoulder-elbow-wrist chain is enough to understand whether the arm is moving through a plausible striking arc for percussion-style input.

## Known limitations

- The tracker does not estimate hand pose or finger articulation, so open/closed hand state is ignored.
- The detector assumes one performer and is optimized for upper-body visibility in front of a single webcam.
- Fast motion, heavy occlusion, poor lighting, or wrists leaving the frame can cause low-confidence landmarks or temporary tracking loss.
- Because gesture detection is wrist-driven, body turns and diagonal hits may require threshold tuning for different camera placements.

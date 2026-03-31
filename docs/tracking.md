# Tracking

## Role of tracking in VisionBeat

Tracking is the sensing layer that turns webcam imagery into a motion representation suitable for musical interaction. In VisionBeat, that means converting raw frames into a small, stable set of upper-body landmarks that can drive gesture recognition.

The tracking system is intentionally conservative: it uses **MediaPipe Pose** and keeps only the landmarks needed by the current gesture vocabulary.

## Landmarks currently used

VisionBeat extracts six pose landmarks:

- left shoulder,
- right shoulder,
- left elbow,
- right elbow,
- left wrist,
- right wrist.

This subset is enough for the current instrument because kick/snare detection is driven mainly by the wrist trajectory, while shoulder landmarks also support the shoulder-relative inward-jab kick model and elbow landmarks support overlay rendering and future arm-chain validation.

## Why pose tracking instead of hand tracking

The current system emphasizes **gross rhythmic movement**, not fine-grained finger articulation. Pose tracking is therefore a good fit because it:

- works with a single RGB webcam,
- captures the arm chain needed for visible striking gestures,
- is robust enough for early prototyping,
- and fits the instrument framing of whole-arm percussive motion.

A future hand-tracking mode may improve detail, but pose tracking currently matches the project’s research emphasis on embodied, upper-body rhythm gestures.

## Tracking pipeline

1. OpenCV captures a BGR webcam frame.
2. The frame is converted to RGB for MediaPipe.
3. MediaPipe Pose processes the frame.
4. VisionBeat selects only the landmark indices it cares about.
5. Landmarks below the configured visibility threshold are discarded.
6. The tracker emits a `TrackerOutput` with timestamps, remaining landmarks, and a human-readable status.

## Tracking status values

The tracker reports a small set of useful states:

- `tracking` — required landmarks are available above confidence threshold.
- `no_person_detected` — MediaPipe did not detect a pose in the frame.
- `landmarks_below_confidence_threshold` — a person was found, but the retained landmarks did not survive filtering.

These states are useful for troubleshooting because not every failed gesture is a gesture-recognition problem; many are tracking problems first.

## Environmental setup guidelines

For the most stable tracking:

- keep the upper body fully visible,
- avoid backlighting or very dim light,
- use a background with moderate visual contrast,
- place the camera far enough away to include shoulders and both wrists,
- and avoid moving so close that the wrists leave frame during strikes.

A good default setup is a camera placed roughly chest-to-head height, facing the performer frontally with enough distance to preserve full arm motion.

## Instrument-oriented framing for tracking

In VisionBeat, tracking is not just about anatomical estimation; it is about making **drumming intent visible enough to be interpreted**. That leads to a practical rule:

> The best tracking setup is the one that makes the performer’s rhythmic accents legible as clean wrist trajectories.

That is why the project prefers visually bold percussive motions over subtle micro-gestures.

## Calibration workflow for tracking and gestures

Before changing gesture thresholds, first stabilize tracking.

### Step 1: confirm landmark visibility

Run VisionBeat and verify that the overlay consistently shows:

- both shoulders,
- the active elbow,
- and the active wrist.

If landmarks flicker, solve camera/lighting/framing first.

### Step 2: choose the active hand

Set `gestures.active_hand` to the hand you will use in performance. This avoids confusion when the non-dominant hand remains visible but is not meant to trigger sounds.

### Step 3: rehearse each gesture slowly

Test the inward side-jab kick and downward strike at moderate speed. The goal is to confirm the tracker keeps up before you optimize for energy or latency.

### Step 4: inspect gesture failures by type

- If the motion is visible but no candidate appears, the thresholds may be too high.
- If a candidate appears but never confirms, the full threshold or confirmation timing may be too strict.
- If gestures trigger on unrelated motion, the detector needs tighter velocity, drift, or axis-dominance settings.

## Threshold calibration heuristics

### When to adjust tracker confidence

Adjust `tracker.min_detection_confidence` and `tracker.min_tracking_confidence` when:

- the performer is consistently visible but the tracker flickers,
- detection seems unstable across lighting changes,
- or landmarks disappear even during obvious arm motion.

Be careful: raising confidence thresholds too far can make the system overly brittle.

### When to adjust gesture thresholds instead

Adjust gesture thresholds when:

- tracking is stable,
- landmarks remain visible,
- but the musical trigger behavior is wrong.

That distinction matters: tracker confidence affects whether motion is seen at all, while gesture thresholds affect how seen motion is interpreted.

## Common tracking failure modes

### Occlusion

If the wrist crosses the torso or exits the frame, the tracker may drop visibility and the gesture candidate will disappear.

### Motion blur

Very fast gestures can temporarily reduce reliability, especially in low light.

### Camera angle mismatch

A very low or high camera angle can distort the apparent motion axis and make an inward side-jab look diagonal or a downward strike look shallow.

### Distance scaling

If the performer stands too far away, the normalized wrist displacement may become too small to pass gesture thresholds.

## Known limitations

- No finger pose or hand openness estimation.
- No multi-person disambiguation.
- No explicit torso-orientation compensation.
- No learned robustness to unusual camera viewpoints.
- No automatic recovery strategy beyond the tracker’s native behavior.

## Future tracking extensions

### Clap detection support

Clap detection would likely require better bilateral hand localization and perhaps more direct hand landmarks.

### Hand-tracking mode

A dedicated hand tracker could support finer gestures, better clap timing, or hand-shape-conditioned triggers.

### Multi-view setups

A second camera could reduce occlusion and improve the reliability of depth-sensitive gestures.

### Sensor fusion

A future system could combine webcam tracking with IMUs, audio onset cues, or inertial wearables while preserving the current architecture.

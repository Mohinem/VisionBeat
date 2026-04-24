# Early-Arm Labels v2

This schema is the source-of-truth event table for the early-arm restart.

## Required columns

- `recording_id`: recording or session identifier.
- `event_id`: stable unique identifier for one labeled gesture event.
- `gesture` or `gesture_label`: gesture class such as `kick` or `snare`.
- `arm_start_frame` or `arm_start_seconds`: first frame/time where early arming is acceptable.
- `completion_frame` or `completion_seconds`: physical completion / hit frame or time.

Use either frame-based boundaries or time-based boundaries for an event row. Do not mix them within a single row.

## Optional columns

- `recovery_end_frame` or `recovery_end_seconds`
- any metadata columns such as `annotator`, `split`, `confidence`, `notes`

## Semantics

- The positive early-arm interval is `[arm_start, completion]`, inclusive.
- `arm_start` should be the first committed pre-hit frame, not the first visible twitch.
- `completion` should be the physical hit frame used for timing evaluation.
- `recovery_end` is preserved as metadata but is not expanded into a separate per-frame recovery span in the current extractor.

## Extracted per-frame columns

When `visionbeat.extract_dataset_features` ingests a v2 event table, the output feature CSV keeps the existing completion columns and adds:

- `event_id`
- `is_arm_frame`
- `arm_start_frame` / `arm_start_seconds`
- `completion_frame` / `completion_seconds`
- `recovery_end_frame` / `recovery_end_seconds` when provided

Per-frame behavior:

- frames inside `[arm_start, completion]` get `gesture_label`, `event_id`, and `is_arm_frame=True`
- the completion frame also gets `is_completion_frame=True`
- frames outside the arm interval remain blank / `False`

## Example

```csv
recording_id,event_id,gesture_label,arm_start_frame,completion_frame,recovery_end_frame,split
session_01,evt-001,kick,118,121,126,train
session_01,evt-002,snare,164,166,170,train
```

## Dataset prep

Use the dedicated early-arm dataset prep entrypoint to generate train/validation `.npz` archives from v2 labels:

```bash
./.venv/bin/python -m visionbeat.prepare_early_arm_dataset \
  --recording session_01=/path/to/session_01.mp4 \
  --recording session_02=/path/to/session_02.mp4 \
  --labels session_01=/path/to/session_01_labels_v2.csv \
  --labels session_02=/path/to/session_02_labels_v2.csv \
  --out-dir outputs/early_arm_dataset \
  --target arm_frame_binary
```

Supported early-arm targets:

- `arm_frame_binary`
- `arm_within_next_k_frames`
- `arm_within_last_k_frames`

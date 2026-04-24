# CNN Baselines

This document records reference offline CNN runs whose generated artifacts live
under ignored `outputs/` directories and therefore are not committed to Git.

## Locked Timing Front-End: `future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001`

- Commit intent: freeze the first practical predictive baseline after the
  offline CNN training/inference pipeline landed.
- Run directory:
  `outputs/future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001`
- Checkpoint:
  `outputs/future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001/checkpoints/best_model.pt`

### Dataset split

- Train recordings: Recording 1 and Recording 2
- Validation holdout: Recording 3
- Prepared archives:
  - `/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_1_future8_w24.npz`
  - `/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_2_future8_w24.npz`
  - `/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_future8_w24.npz`

### Training configuration

- Target: `completion_within_next_k_frames`
- Horizon frames: `8`
- Window size: `24`
- Stride: `1`
- Holdout split: full Recording 3 validation holdout
- Hard negative margin frames: `8`
- Max train negative:positive ratio: `3`
- Hidden channels: `64`
- Dropout: `0.2`
- Learning rate: `1e-3`
- Batch size: `256`
- Seed: `7`

### Best validation metrics

- Best epoch: `5`
- Validation loss: `0.5381`
- Accuracy: `0.7674`
- Precision: `0.2892`
- Recall: `0.7425`
- F1: `0.4163`
- ROC AUC: `0.8472`

### Locked decoder configuration

- Threshold: `0.60`
- Cooldown frames: `6`
- Max gap frames: `1`
- Decoder sweep summary:
  `outputs/decoder_sweeps/recording_3_baseline_sweep/recording_3_future8_w24_decoder_sweep_summary.csv`

### Trigger-level evaluation on Recording 3

Initial manual decoder check:

```bash
./.venv/bin/python -m visionbeat.predict_cnn \
  outputs/future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001/checkpoints/best_model.pt \
  "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_future8_w24.npz" \
  --threshold 0.8 \
  --trigger-cooldown-frames 24 \
  --trigger-max-gap-frames 2
```

Observed decoded-trigger metrics:

- Decoded trigger precision: `0.6102`
- Decoded trigger recall: `0.5373`
- Decoded trigger F1: `0.5714`
- Detected events: `36`
- False triggers: `23`
- Missed events: `31`

### Decoder sweep on Recording 3

Sweep command:

```bash
./.venv/bin/python -m visionbeat.sweep_decoder \
  outputs/future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001/checkpoints/best_model.pt \
  "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_future8_w24.npz" \
  --thresholds 0.55 0.60 0.65 0.70 \
  --cooldowns 6 8 10 \
  --max-gaps 1 2 \
  --output-dir outputs/decoder_sweeps/recording_3_baseline_sweep
```

Sweep outputs:

- Sweep root:
  `outputs/decoder_sweeps/recording_3_baseline_sweep`
- Summary CSV:
  `outputs/decoder_sweeps/recording_3_baseline_sweep/recording_3_future8_w24_decoder_sweep_summary.csv`

Best decoded-trigger configuration from the sweep:

- Threshold: `0.60`
- Cooldown frames: `6`
- Max gap frames: `1`
- Decoded trigger precision: `0.5227`
- Decoded trigger recall: `0.6866`
- Decoded trigger F1: `0.5935`
- Detected events: `46`
- False triggers: `42`
- Missed events: `21`

`max_gap=2` tied with `max_gap=1` on this dataset, so `max_gap=1` remains the
simpler reference setting.

### Timing relative to labeled completion

Timing analysis command:

```bash
./.venv/bin/python -m visionbeat.analyze_decoder_timing \
  --sweep-summary outputs/decoder_sweeps/recording_3_baseline_sweep/recording_3_future8_w24_decoder_sweep_summary.csv \
  --dataset "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_future8_w24.npz" \
  --labels "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_frames.csv" \
  --output-dir outputs/decoder_sweeps/recording_3_baseline_sweep/timing_top3
```

Timing outputs:

- Timing summary:
  `outputs/decoder_sweeps/recording_3_baseline_sweep/timing_top3/recording_3_future8_w24_timing_summary_top3.csv`
- Per-event CSV for locked decoder:
  `outputs/decoder_sweeps/recording_3_baseline_sweep/timing_top3/th_0p60_cd_6_gap_1_matched_timing.csv`

Observed timing for the locked decoder:

- Mean trigger delta: `-4.41` frames, `-147.10 ms`
- Median trigger delta: `-4.0` frames, `-133.33 ms`
- Triggered before completion: `100%`
- Triggered at completion: `0%`
- Triggered after completion: `0%`
- Marked too early under the explicit analysis threshold (`delta_frames < -4`): `47.83%`

Interpretation:

- The timing model is genuinely predictive rather than late.
- The main cost of the current front-end is early firing plus false triggers, not lag.

### Rejected retrain: `future4_w24`

Comparison run:

- Training root:
  `outputs/retraining_future4_w24/train_r1r2_val_r3/visionbeat_cnn_run_001`
- Sweep summary:
  `outputs/retraining_future4_w24/decoder_sweeps/recording_3_future4_w24/recording_3_future4_w24_decoder_sweep_summary.csv`
- Timing summary:
  `outputs/retraining_future4_w24/decoder_sweeps/recording_3_future4_w24/timing_top3/recording_3_future4_w24_timing_summary_top3.csv`

Observed tradeoff:

- `future4` improved timing lead from about `-147 ms` to about `-80 ms`
- But event-level decoded-trigger F1 dropped from `0.5935` to `0.4306`
- Best `future4` decoded-trigger config still underperformed the locked `future8` baseline

Decision:

- Keep `future8_w24` as the active timing front-end
- Keep decoder `threshold=0.60`, `cooldown=6`, `max_gap=1`

This timing configuration is the current front-end baseline to beat for future
model or decoding changes.

## Kick/Snare Second Stage on Top of the Locked Front-End

### Stage-2 training run

Command:

```bash
./.venv/bin/python -m visionbeat.train_gesture_classifier \
  "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_1_future8_w24.npz" \
  "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_2_future8_w24.npz" \
  "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_future8_w24.npz" \
  --holdout-recording-id "VisionBeat Dataset - Recording 3" \
  --output-dir outputs/gesture_classifier_future8_r1r2_train_r3_val
```

Outputs:

- Run directory:
  `outputs/gesture_classifier_future8_r1r2_train_r3_val/visionbeat_gesture_classifier_run_001`
- Checkpoint:
  `outputs/gesture_classifier_future8_r1r2_train_r3_val/visionbeat_gesture_classifier_run_001/checkpoints/best_model.pt`
- Evaluation report:
  `outputs/gesture_classifier_future8_r1r2_train_r3_val/visionbeat_gesture_classifier_run_001/reports/evaluation_report.json`

Best validation metrics on Recording 3 positive windows:

- Best epoch: `9`
- Validation loss: `0.1074`
- Accuracy: `0.9701`
- Macro F1: `0.9700`
- Kick F1: `0.9680`
- Snare F1: `0.9720`

### Combined timing-front-end plus gesture typing run

Command:

```bash
./.venv/bin/python -m visionbeat.classify_decoded_triggers \
  outputs/future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001/checkpoints/best_model.pt \
  "/home/rasdaman/Desktop/VisionBeat Test/Dataset/recording_3_future8_w24.npz" \
  outputs/gesture_classifier_future8_r1r2_train_r3_val/visionbeat_gesture_classifier_run_001/checkpoints/best_model.pt \
  --threshold 0.60 \
  --trigger-cooldown-frames 6 \
  --trigger-max-gap-frames 1 \
  --output-dir outputs/gesture_pipeline_future8_r3
```

Outputs:

- Combined report:
  `outputs/gesture_pipeline_future8_r3/recording_3_future8_w24_gesture_pipeline_report.json`
- Classified decoded triggers:
  `outputs/gesture_pipeline_future8_r3/recording_3_future8_w24_decoded_trigger_classes.csv`
- Matched trigger gesture rows:
  `outputs/gesture_pipeline_future8_r3/recording_3_future8_w24_matched_trigger_gestures.csv`

Observed end-to-end results on Recording 3:

- Timing front-end remained unchanged:
  - decoded triggers: `88`
  - matched events: `46 / 67`
  - false triggers: `42`
- On the `46` matched triggers, kick/snare typing accuracy was `0.9565`
- Correctly typed end-to-end events: `44 / 67`
- Correctly typed event recall: `0.6567`

Interpretation:

- The second-stage classifier is already strong once the timing front-end lands on the right event.
- The dominant bottleneck remains the timing front-end, especially false triggers and missed events.
- The current stack for offline experiments is:
  - timing front-end: `future8_w24` checkpoint
  - decoder: `threshold=0.60`, `cooldown=6`, `max_gap=1`
  - second stage: `visionbeat_gesture_classifier_run_001`

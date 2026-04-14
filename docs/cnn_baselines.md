# CNN Baselines

This document records reference offline CNN runs whose generated artifacts live
under ignored `outputs/` directories and therefore are not committed to Git.

## Reference Baseline: `future8_w24_r1r2_train_r3_val/visionbeat_cnn_run_001`

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

### Trigger-level evaluation on Recording 3

Command family:

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

This is the current baseline to beat for future model or decoding changes.

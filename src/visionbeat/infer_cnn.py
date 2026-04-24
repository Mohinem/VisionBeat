"""Backward-compatible alias for the offline VisionBeat CNN predictor."""

from __future__ import annotations

from visionbeat.cnn_model import (
    infer_hidden_channels_from_state_dict as _infer_hidden_channels_from_state_dict,
)
from visionbeat.predict_cnn import (
    InferenceDataset,
    analyze_thresholds,
    build_threshold_grid,
    save_decoded_triggers_csv,
    evaluate_predictions,
    load_inference_dataset,
    main,
    run_inference,
    save_inference_report,
    save_predictions_csv,
    save_threshold_analysis_csv,
    summarize_threshold_analysis,
)

__all__ = [
    "InferenceDataset",
    "_infer_hidden_channels_from_state_dict",
    "analyze_thresholds",
    "build_threshold_grid",
    "save_decoded_triggers_csv",
    "evaluate_predictions",
    "load_inference_dataset",
    "main",
    "run_inference",
    "save_inference_report",
    "save_predictions_csv",
    "save_threshold_analysis_csv",
    "summarize_threshold_analysis",
]


if __name__ == "__main__":
    main()

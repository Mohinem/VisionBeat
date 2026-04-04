"""Compatibility exports for the default MediaPipe tracking backend."""

from __future__ import annotations

from visionbeat.mediapipe_provider import MediaPipePoseProvider, _TasksPoseAdapter

PoseTracker = MediaPipePoseProvider

__all__ = ["MediaPipePoseProvider", "PoseTracker", "_TasksPoseAdapter"]

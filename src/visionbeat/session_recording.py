"""Research-oriented session recording helpers for reproducible replay."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np

from visionbeat.camera import CameraFrame
from visionbeat.models import GestureEvent, TrackerOutput

SessionRecordingMode = Literal["tracker_outputs", "raw_frames", "both"]


@dataclass(slots=True)
class SessionRecorder:
    """Persist session bundles containing config, frames, tracker output, and triggers."""

    root_path: Path
    mode: SessionRecordingMode
    config_payload: Mapping[str, Any]
    session_dir: Path = field(init=False)
    _manifest_path: Path = field(init=False)
    _frames_dir: Path | None = field(init=False, default=None)
    _camera_stream: TextIO | None = field(init=False, default=None)
    _tracker_stream: TextIO | None = field(init=False, default=None)
    _trigger_stream: TextIO = field(init=False)
    _started_at: str = field(init=False)
    _stopped_at: str | None = field(init=False, default=None)
    _camera_frame_count: int = field(init=False, default=0)
    _tracker_output_count: int = field(init=False, default=0)
    _trigger_count: int = field(init=False, default=0)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Create a unique session bundle directory and initialize its artifacts."""
        self.root_path = Path(self.root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.session_dir = self._create_session_dir()
        self._manifest_path = self.session_dir / "manifest.json"
        self._started_at = datetime.now(tz=UTC).isoformat(timespec="milliseconds")
        if self.captures_raw_frames:
            self._frames_dir = self.session_dir / "frames"
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            self._camera_stream = (self.session_dir / "camera_frames.jsonl").open(
                "a",
                encoding="utf-8",
            )
        if self.captures_tracker_outputs:
            self._tracker_stream = (self.session_dir / "tracker_outputs.jsonl").open(
                "a",
                encoding="utf-8",
            )
        self._trigger_stream = (self.session_dir / "triggers.jsonl").open("a", encoding="utf-8")
        self._write_manifest()

    @property
    def captures_raw_frames(self) -> bool:
        """Return whether the recorder stores lossless raw camera frames."""
        return self.mode in {"raw_frames", "both"}

    @property
    def captures_tracker_outputs(self) -> bool:
        """Return whether the recorder stores normalized tracker outputs."""
        return self.mode in {"tracker_outputs", "both"}

    def record_camera_frame(self, camera_frame: CameraFrame) -> None:
        """Persist one raw camera frame and its capture metadata when enabled."""
        if not self.captures_raw_frames:
            return
        assert self._frames_dir is not None
        assert self._camera_stream is not None
        frame = camera_frame.image
        if not isinstance(frame, np.ndarray):
            raise TypeError("SessionRecorder raw frame recording expects numpy.ndarray frames.")
        frame_path = self._frames_dir / f"{camera_frame.frame_index:06d}.npy"
        np.save(frame_path, frame, allow_pickle=False)
        self._write_jsonl(
            self._camera_stream,
            {
                "frame_index": camera_frame.frame_index,
                "captured_at": camera_frame.captured_at,
                "mirrored_for_display": camera_frame.mirrored_for_display,
                "frame_path": frame_path.relative_to(self.session_dir).as_posix(),
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
            },
        )
        self._camera_frame_count += 1

    def record_tracker_output(
        self,
        camera_frame: CameraFrame,
        tracker_output: TrackerOutput,
    ) -> None:
        """Persist one normalized tracker output when enabled."""
        if not self.captures_tracker_outputs:
            return
        assert self._tracker_stream is not None
        self._write_jsonl(
            self._tracker_stream,
            {
                "frame_index": camera_frame.frame_index,
                "captured_at": camera_frame.captured_at,
                "tracker_output": tracker_output.to_dict(),
            },
        )
        self._tracker_output_count += 1

    def record_trigger(self, event: GestureEvent) -> None:
        """Persist one confirmed gesture event."""
        self._write_jsonl(self._trigger_stream, event.to_dict())
        self._trigger_count += 1

    def close(self) -> None:
        """Finalize the manifest and close any open session streams."""
        if self._closed:
            return
        self._stopped_at = datetime.now(tz=UTC).isoformat(timespec="milliseconds")
        self._write_manifest()
        if self._camera_stream is not None:
            self._camera_stream.close()
        if self._tracker_stream is not None:
            self._tracker_stream.close()
        self._trigger_stream.close()
        self._closed = True

    def _create_session_dir(self) -> Path:
        """Create and return a unique session directory within the configured root."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = self.root_path / f"session-{timestamp}"
        suffix = 1
        while candidate.exists():
            candidate = self.root_path / f"session-{timestamp}-{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _write_manifest(self) -> None:
        """Write the current manifest snapshot to disk."""
        artifacts: dict[str, str] = {"triggers": "triggers.jsonl"}
        if self.captures_raw_frames:
            artifacts["camera_frames"] = "camera_frames.jsonl"
            artifacts["frames_directory"] = "frames"
        if self.captures_tracker_outputs:
            artifacts["tracker_outputs"] = "tracker_outputs.jsonl"
        payload = {
            "schema": "visionbeat.session.v1",
            "session_id": self.session_dir.name,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "recording_mode": self.mode,
            "session_directory": self.session_dir.as_posix(),
            "artifacts": artifacts,
            "counts": {
                "camera_frames": self._camera_frame_count,
                "tracker_outputs": self._tracker_output_count,
                "triggers": self._trigger_count,
            },
            "config": self.config_payload,
        }
        self._manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_jsonl(self, stream: TextIO, payload: Mapping[str, Any]) -> None:
        """Append one JSONL payload to a stream and flush it immediately."""
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        stream.flush()


def build_session_recorder(
    logging_config: Any,
    *,
    config_payload: Mapping[str, Any],
) -> SessionRecorder | None:
    """Build a session recorder from logging configuration, if enabled."""
    path = getattr(logging_config, "session_recording_path", None)
    if path is None:
        return None
    return SessionRecorder(
        Path(path),
        mode=getattr(logging_config, "session_recording_mode", "tracker_outputs"),
        config_payload=config_payload,
    )

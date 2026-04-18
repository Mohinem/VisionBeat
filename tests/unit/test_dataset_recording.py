from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from visionbeat.camera import CameraFrame
from visionbeat.config import AppConfig
from visionbeat.config import CameraConfig
from visionbeat.dataset_recording import _build_ffmpeg_record_command, record_dataset_video


class FakeCamera:
    def __init__(self, frames: list[CameraFrame]) -> None:
        self.frames = list(frames)
        self.last_frame: CameraFrame | None = None
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1

    def read_frame(self) -> CameraFrame:
        if self.frames:
            self.last_frame = self.frames.pop(0)
            return self.last_frame
        if self.last_frame is None:
            raise RuntimeError("No camera frames configured")
        self.last_frame = replace(
            self.last_frame,
            captured_at=self.last_frame.captured_at + 0.15,
            frame_index=self.last_frame.frame_index + 1,
        )
        return self.last_frame

    def close(self) -> None:
        self.close_calls += 1


class FakePreview:
    def __init__(self, should_close_sequence: list[bool]) -> None:
        self.should_close_sequence = list(should_close_sequence)
        self.show_calls: list[tuple[str, object]] = []
        self.close_calls = 0

    def show(self, window_name: str, frame: object) -> None:
        self.show_calls.append((window_name, frame))

    def poll_key(self) -> int | None:
        return None

    def should_close(self, key_code: int | None = None) -> bool:
        if self.should_close_sequence:
            return self.should_close_sequence.pop(0)
        return True

    def close(self) -> None:
        self.close_calls += 1


class FakeVideoWriter:
    def __init__(self) -> None:
        self.frames: list[object] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def write(self, frame: object) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    def __init__(self) -> None:
        self.writer = FakeVideoWriter()
        self.writer_fps: float | None = None

    def VideoWriter_fourcc(self, *_: str) -> int:  # noqa: N802
        return 1234

    def VideoWriter(
        self,
        path: str,
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> FakeVideoWriter:  # noqa: N802
        assert path.endswith(".mp4")
        assert fourcc == 1234
        assert fps > 0.0
        assert frame_size == (4, 3)
        self.writer_fps = fps
        return self.writer


def test_record_dataset_video_respects_start_delay_and_writes_metadata(tmp_path: Path) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    frames = [
        CameraFrame(image=image, display_image="preview-0", captured_at=0.00, frame_index=0),
        CameraFrame(image=image, display_image="preview-1", captured_at=0.10, frame_index=1),
        CameraFrame(image=image, display_image="preview-2", captured_at=0.25, frame_index=2),
        CameraFrame(image=image, display_image="preview-3", captured_at=0.40, frame_index=3),
        CameraFrame(image=image, display_image="preview-4", captured_at=0.55, frame_index=4),
    ]
    camera = FakeCamera(frames)
    preview = FakePreview([False, False, False, False, True])
    cv2_module = FakeCV2()

    result = record_dataset_video(
        AppConfig(),
        output_path=tmp_path / "dataset.mp4",
        start_delay_seconds=0.20,
        camera_source=camera,
        preview_window=preview,
        cv2_module=cv2_module,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert camera.open_calls == 1
    assert camera.close_calls == 1
    assert preview.close_calls == 1
    assert len(preview.show_calls) >= 1
    assert preview.show_calls[-1][1] == "preview-4"
    assert len(cv2_module.writer.frames) >= 1
    assert cv2_module.writer.released is True
    assert result.frames_recorded >= 1
    assert result.output_fps == pytest.approx(cv2_module.writer_fps)
    assert result.frame_width == 4
    assert result.frame_height == 3
    assert metadata["frames_recorded"] >= 1
    assert metadata["output_fps"] == pytest.approx(cv2_module.writer_fps)
    assert metadata["start_delay_seconds"] == 0.2
    assert metadata["measured_capture_fps"] is None or metadata["measured_capture_fps"] > 0.0


def test_build_ffmpeg_record_command_uses_v4l2_mjpeg_copy_for_mkv(tmp_path: Path) -> None:
    config = AppConfig(camera=CameraConfig(device_index=0, width=1280, height=720, fps=30, backend="v4l2", fourcc="MJPG"))

    command = _build_ffmpeg_record_command(
        ffmpeg_path="/usr/bin/ffmpeg",
        config=config,
        output_path=tmp_path / "dataset.mkv",
        duration_seconds=15.0,
        include_preview=False,
    )

    assert command[:8] == [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "v4l2",
        "-input_format",
    ]
    assert "mjpeg" in command
    assert "/dev/video0" in command
    assert "1280x720" in command
    assert str(tmp_path / "dataset.mkv") in command
    assert "-map" not in command
    assert "-f" in command

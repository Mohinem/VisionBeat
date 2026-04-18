"""Raw webcam dataset-recording helpers for annotation-friendly capture."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from visionbeat.app import OpenCVPreviewWindow, PreviewWindow
from visionbeat.camera import CameraFrame, CameraSource
from visionbeat.config import AppConfig, CameraConfig, ConfigError, load_config
from visionbeat.logging_config import configure_logging

logger = logging.getLogger(__name__)
_DEFAULT_FOURCC = "mp4v"
_PREVIEW_POLL_INTERVAL_SECONDS = 0.005
_TIMESTAMP_HISTORY_SIZE = 240
_WRITER_QUEUE_MAXSIZE = 120
_FFMPEG_LOGLEVEL = "warning"


class VideoWriterProtocol(Protocol):
    """Subset of OpenCV's video-writer API used by dataset recording."""

    def write(self, frame: Any) -> None:
        """Append one frame to the output stream."""

    def release(self) -> None:
        """Finalize the output stream."""


@dataclass
class _FrameWriterWorker:
    """Background worker that serializes recorded frames to disk."""

    writer: VideoWriterProtocol
    max_queue_size: int = _WRITER_QUEUE_MAXSIZE
    _stop_token: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._stop_token = object()
        self._queue: queue.Queue[CameraFrame | object] = queue.Queue(maxsize=self.max_queue_size)
        self._thread = threading.Thread(target=self._run, name="visionbeat-frame-writer", daemon=True)
        self.frames_written = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: CameraFrame) -> None:
        self._queue.put(frame)

    def close(self) -> None:
        self._queue.put(self._stop_token)
        self._thread.join()
        self.writer.release()
        if self.error is not None:
            raise RuntimeError("Dataset recording writer failed.") from self.error

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._stop_token:
                    return
                assert isinstance(item, CameraFrame)
                self.writer.write(item.image)
                self.frames_written += 1
        except BaseException as exc:  # pragma: no cover - defensive thread wrapper
            self.error = exc


@dataclass
class _CaptureWorker:
    """Background worker that reads camera frames independently of preview speed."""

    camera: CameraSource
    start_delay_seconds: float
    duration_seconds: float | None

    def __post_init__(self) -> None:
        self._timestamps: deque[float] = deque(maxlen=_TIMESTAMP_HISTORY_SIZE)
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="visionbeat-camera-capture", daemon=True)
        self._latest_frame: CameraFrame | None = None
        self._writer: _FrameWriterWorker | None = None
        self._error: BaseException | None = None
        self._recording_start_time: float | None = None
        self._recording_end_time: float | None = None
        self._first_recorded_time: float | None = None
        self._last_recorded_time: float | None = None
        self._frames_recorded = 0
        self._stopped_by_duration = False
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def attach_writer(self, writer: _FrameWriterWorker) -> None:
        with self._lock:
            self._writer = writer

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Dataset recording capture failed.") from self._error

    def latest_frame(self) -> CameraFrame | None:
        with self._lock:
            return self._latest_frame

    def wait_for_frame(self, *, timeout: float) -> CameraFrame | None:
        deadline = time.monotonic() + timeout
        with self._new_frame:
            if self._latest_frame is not None:
                return self._latest_frame
            while self._latest_frame is None and self._running and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._new_frame.wait(timeout=remaining)
            return self._latest_frame

    def recent_timestamps(self) -> tuple[float, ...]:
        with self._lock:
            return tuple(self._timestamps)

    def stats(self) -> tuple[int, float | None, float | None]:
        with self._lock:
            return self._frames_recorded, self._first_recorded_time, self._last_recorded_time

    def stopped_by_duration(self) -> bool:
        with self._lock:
            return self._stopped_by_duration

    def recording_window(self) -> tuple[float | None, float | None]:
        with self._lock:
            return self._recording_start_time, self._recording_end_time

    def is_running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                camera_frame = self.camera.read_frame()
                writer: _FrameWriterWorker | None
                should_stop = False
                with self._new_frame:
                    self._latest_frame = camera_frame
                    self._timestamps.append(camera_frame.captured_at)
                    if self._recording_start_time is None:
                        self._recording_start_time = camera_frame.captured_at + self.start_delay_seconds
                        if self.duration_seconds is not None:
                            self._recording_end_time = (
                                self._recording_start_time + self.duration_seconds
                            )
                    writer = self._writer
                    if (
                        writer is not None
                        and self._recording_start_time is not None
                        and camera_frame.captured_at >= self._recording_start_time
                    ):
                        writer.submit(camera_frame)
                        self._frames_recorded += 1
                        if self._first_recorded_time is None:
                            self._first_recorded_time = camera_frame.captured_at
                        self._last_recorded_time = camera_frame.captured_at
                        if (
                            self._recording_end_time is not None
                            and camera_frame.captured_at >= self._recording_end_time
                        ):
                            self._stopped_by_duration = True
                            should_stop = True
                    self._new_frame.notify_all()
                if should_stop:
                    self._stop_event.set()
                    return
        except BaseException as exc:  # pragma: no cover - defensive thread wrapper
            self._error = exc
            self._stop_event.set()
            with self._new_frame:
                self._new_frame.notify_all()
        finally:
            self._running = False


@dataclass(frozen=True, slots=True)
class DatasetRecordingResult:
    """Summary of one completed dataset-recording run."""

    output_path: Path
    metadata_path: Path
    frames_recorded: int
    target_fps: int
    output_fps: float
    frame_width: int
    frame_height: int
    start_delay_seconds: float
    recorded_duration_seconds: float
    measured_capture_fps: float | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for raw dataset recording."""

    parser = argparse.ArgumentParser(
        description="Record raw webcam video for dataset annotation."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a VisionBeat YAML or TOML configuration file.",
    )
    parser.add_argument(
        "--output-video",
        required=True,
        help="Destination path for the recorded raw video.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override the configured camera device index.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "v4l2", "dshow", "msmf", "avfoundation", "gstreamer", "ffmpeg"),
        default=None,
        help="Override the configured OpenCV camera backend used for capture negotiation.",
    )
    parser.add_argument(
        "--camera-fourcc",
        default=None,
        help="Override the configured camera pixel format as a four-character code such as MJPG.",
    )
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=0.0,
        help="Delay before recording starts, while the clean preview remains visible.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional fixed recording duration. Omit to stop manually with q/esc.",
    )
    return parser.parse_args(argv)


def record_dataset_video(
    config: AppConfig,
    *,
    output_path: str | Path,
    start_delay_seconds: float = 0.0,
    duration_seconds: float | None = None,
    camera_source: CameraSource | None = None,
    preview_window: PreviewWindow | None = None,
    cv2_module: Any | None = None,
) -> DatasetRecordingResult:
    """Record raw webcam video with capture decoupled from preview refresh."""

    if _should_use_ffmpeg_dataset_recording(
        config,
        duration_seconds=duration_seconds,
        camera_source=camera_source,
        preview_window=preview_window,
    ):
        return _record_dataset_video_with_ffmpeg(
            config,
            output_path=output_path,
            start_delay_seconds=start_delay_seconds,
            duration_seconds=duration_seconds,
            cv2_module=cv2_module,
        )

    if start_delay_seconds < 0.0:
        raise ValueError("start_delay_seconds must be greater than or equal to zero.")
    if duration_seconds is not None and duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be greater than zero when provided.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_name(f"{destination.name}.metadata.json")

    camera = camera_source or CameraSource(config.camera, _cv2=cv2_module)
    preview = preview_window or OpenCVPreviewWindow(cv2_module=cv2_module)
    writer_worker: _FrameWriterWorker | None = None

    preview_window_name = f"{config.camera.window_name} Dataset Recording"
    last_announced_delay_second: int | None = None
    frame_width = int(config.camera.width)
    frame_height = int(config.camera.height)
    output_fps: float = float(config.camera.fps)
    capture_worker: _CaptureWorker | None = None
    last_previewed_frame_index: int | None = None

    logger.info(
        "Starting dataset recording output=%s target_fps=%s start_delay_seconds=%.2f "
        "duration_seconds=%s",
        destination,
        config.camera.fps,
        start_delay_seconds,
        "manual" if duration_seconds is None else f"{duration_seconds:.2f}",
    )
    camera.open()
    try:
        capture_worker = _CaptureWorker(
            camera,
            start_delay_seconds=start_delay_seconds,
            duration_seconds=duration_seconds,
        )
        capture_worker.start()
        if start_delay_seconds > 0.0:
            logger.info(
                "Dataset recording armed; raw video capture starts in %.2f seconds",
                start_delay_seconds,
            )
        while True:
            camera_frame = capture_worker.wait_for_frame(timeout=_PREVIEW_POLL_INTERVAL_SECONDS)
            if camera_frame is None:
                if not capture_worker.is_running():
                    break
                continue
            display_frame = (
                camera_frame.display_image
                if camera_frame.display_image is not None
                else camera_frame.image
            )
            frame_width = int(getattr(camera_frame.image, "shape", [frame_height, frame_width])[1])
            frame_height = int(getattr(camera_frame.image, "shape", [frame_height, frame_width])[0])
            if camera_frame.frame_index != last_previewed_frame_index:
                preview.show(preview_window_name, display_frame)
                last_previewed_frame_index = camera_frame.frame_index
            key_code = preview.poll_key()
            if preview.should_close(key_code):
                logger.info("Stopping dataset recording on user request")
                capture_worker.request_stop()
                break

            recording_start_time, _ = capture_worker.recording_window()
            if recording_start_time is not None and camera_frame.captured_at < recording_start_time:
                remaining_seconds = max(0.0, recording_start_time - camera_frame.captured_at)
                announced_second = int(remaining_seconds) + (0 if remaining_seconds.is_integer() else 1)
                if announced_second != last_announced_delay_second:
                    logger.info("Dataset recording starts in %ss", announced_second)
                    last_announced_delay_second = announced_second
            if writer_worker is None:
                recent_capture_timestamps = capture_worker.recent_timestamps()
                if len(recent_capture_timestamps) >= 2 or (
                    recording_start_time is not None
                    and camera_frame.captured_at >= recording_start_time
                ):
                    output_fps = _resolve_output_fps(
                        target_fps=config.camera.fps,
                        recent_capture_timestamps=recent_capture_timestamps,
                    )
                    writer = _open_video_writer(
                        destination,
                        fps=output_fps,
                        frame_size=(frame_width, frame_height),
                        cv2_module=cv2_module,
                    )
                    writer_worker = _FrameWriterWorker(writer)
                    writer_worker.start()
                    capture_worker.attach_writer(writer_worker)
                    logger.info(
                        "Dataset recording writer ready output=%s resolution=%sx%s target_fps=%s output_fps=%.3f",
                        destination,
                        frame_width,
                        frame_height,
                        config.camera.fps,
                        output_fps,
                    )
                    if abs(output_fps - float(config.camera.fps)) >= 1.0:
                        logger.warning(
                            "Dataset recording writer FPS differs from target FPS: target=%s output=%.3f. "
                            "This usually means the webcam or system cannot sustain the requested capture rate.",
                            config.camera.fps,
                            output_fps,
                        )
            if recording_start_time is not None and camera_frame.captured_at < recording_start_time:
                continue

            if capture_worker.stopped_by_duration():
                logger.info(
                    "Stopping dataset recording after reaching duration_seconds=%.2f",
                    duration_seconds,
                )
                break
    finally:
        shutdown_error: BaseException | None = None
        try:
            if capture_worker is not None:
                capture_worker.request_stop()
                capture_worker.join()
        except BaseException as exc:  # pragma: no cover - defensive cleanup wrapper
            shutdown_error = exc
        try:
            if writer_worker is not None:
                writer_worker.close()
        except BaseException as exc:  # pragma: no cover - defensive cleanup wrapper
            shutdown_error = shutdown_error or exc
        try:
            preview.close()
        finally:
            camera.close()
        if shutdown_error is not None:
            raise shutdown_error

    recorded_duration_seconds = 0.0
    measured_capture_fps: float | None = None
    frames_recorded = 0
    first_recorded_time: float | None = None
    last_recorded_time: float | None = None
    if capture_worker is not None:
        frames_recorded, first_recorded_time, last_recorded_time = capture_worker.stats()
    if first_recorded_time is not None and last_recorded_time is not None:
        recorded_duration_seconds = max(0.0, last_recorded_time - first_recorded_time)
        if recorded_duration_seconds > 0.0 and frames_recorded > 1:
            measured_capture_fps = (frames_recorded - 1) / recorded_duration_seconds

    metadata = {
        "schema": "visionbeat.dataset_recording.v1",
        "output_path": destination.as_posix(),
        "target_fps": config.camera.fps,
        "output_fps": output_fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "start_delay_seconds": start_delay_seconds,
        "duration_seconds": duration_seconds,
        "frames_recorded": frames_recorded,
        "recorded_duration_seconds": recorded_duration_seconds,
        "measured_capture_fps": measured_capture_fps,
        "camera": {
            "device_index": config.camera.device_index,
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps,
            "mirror_preview": config.camera.mirror,
        },
    }
    capture_mode = getattr(camera, "capture_mode", None)
    if callable(capture_mode):
        resolved_capture_mode = capture_mode()
        if resolved_capture_mode is not None:
            metadata["negotiated_camera_mode"] = {
                "backend": resolved_capture_mode.backend,
                "width": resolved_capture_mode.width,
                "height": resolved_capture_mode.height,
                "fps": resolved_capture_mode.fps,
                "fourcc": resolved_capture_mode.fourcc,
            }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return DatasetRecordingResult(
        output_path=destination,
        metadata_path=metadata_path,
        frames_recorded=frames_recorded,
        target_fps=config.camera.fps,
        output_fps=output_fps,
        frame_width=frame_width,
        frame_height=frame_height,
        start_delay_seconds=start_delay_seconds,
        recorded_duration_seconds=recorded_duration_seconds,
        measured_capture_fps=measured_capture_fps,
    )


def _record_dataset_video_with_ffmpeg(
    config: AppConfig,
    *,
    output_path: str | Path,
    start_delay_seconds: float,
    duration_seconds: float | None,
    cv2_module: Any | None,
) -> DatasetRecordingResult:
    if duration_seconds is None:
        raise ValueError("FFmpeg-backed dataset recording requires duration_seconds.")
    if start_delay_seconds < 0.0:
        raise ValueError("start_delay_seconds must be greater than or equal to zero.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be greater than zero when provided.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_name(f"{destination.name}.metadata.json")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for ffmpeg-backed dataset recording.")

    preview_frame = _run_preview_countdown(
        config,
        start_delay_seconds=start_delay_seconds,
        cv2_module=cv2_module,
    )

    command = _build_ffmpeg_record_command(
        ffmpeg_path=ffmpeg_path,
        config=config,
        output_path=destination,
        duration_seconds=duration_seconds,
        include_preview=False,
    )
    logger.info("Starting ffmpeg-backed dataset recording output=%s", destination)
    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _run_recording_status_window(
            config,
            duration_seconds=duration_seconds,
            process=process,
            preview_frame=preview_frame,
            cv2_module=cv2_module,
        )
        _, stderr_text = process.communicate()
        if process.returncode != 0:
            stderr = stderr_text.strip() if stderr_text else f"exit status {process.returncode}"
            raise RuntimeError(f"ffmpeg dataset recording failed: {stderr}")
        completed_stderr = stderr_text
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(f"ffmpeg dataset recording failed: {exc}") from exc
    finished_at = time.monotonic()

    probe = _probe_recorded_video(destination, cv2_module=cv2_module)
    metadata = {
        "schema": "visionbeat.dataset_recording.v1",
        "output_path": destination.as_posix(),
        "target_fps": config.camera.fps,
        "output_fps": probe["fps"],
        "frame_width": probe["width"],
        "frame_height": probe["height"],
        "start_delay_seconds": start_delay_seconds,
        "duration_seconds": duration_seconds,
        "frames_recorded": probe["frame_count"],
        "recorded_duration_seconds": probe["duration_seconds"],
        "measured_capture_fps": probe["fps"],
        "recording_backend": "ffmpeg",
        "camera": {
            "device_index": config.camera.device_index,
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps,
            "mirror_preview": config.camera.mirror,
        },
        "ffmpeg": {
            "command": command,
            "elapsed_seconds": max(0.0, finished_at - started_at),
            "stderr_tail": _tail_lines(completed_stderr, limit=20),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return DatasetRecordingResult(
        output_path=destination,
        metadata_path=metadata_path,
        frames_recorded=int(probe["frame_count"]),
        target_fps=config.camera.fps,
        output_fps=float(probe["fps"]),
        frame_width=int(probe["width"]),
        frame_height=int(probe["height"]),
        start_delay_seconds=start_delay_seconds,
        recorded_duration_seconds=float(probe["duration_seconds"]),
        measured_capture_fps=float(probe["fps"]),
    )


def _should_use_ffmpeg_dataset_recording(
    config: AppConfig,
    *,
    duration_seconds: float | None,
    camera_source: CameraSource | None,
    preview_window: PreviewWindow | None,
) -> bool:
    if duration_seconds is None:
        return False
    if camera_source is not None or preview_window is not None:
        return False
    if config.camera.backend != "v4l2":
        return False
    return shutil.which("ffmpeg") is not None


def _run_preview_countdown(
    config: AppConfig,
    *,
    start_delay_seconds: float,
    cv2_module: Any | None,
) -> Any | None:
    if start_delay_seconds <= 0.0:
        return None
    camera = CameraSource(config.camera, _cv2=cv2_module)
    preview = OpenCVPreviewWindow(cv2_module=cv2_module)
    preview_window_name = f"{config.camera.window_name} Dataset Recording"
    deadline = time.monotonic() + start_delay_seconds
    last_announced_second: int | None = None
    last_display_frame: Any | None = None
    camera.open()
    try:
        while True:
            camera_frame = camera.read_frame()
            display_frame = (
                camera_frame.display_image
                if camera_frame.display_image is not None
                else camera_frame.image
            )
            last_display_frame = display_frame
            preview.show(preview_window_name, display_frame)
            key_code = preview.poll_key()
            if preview.should_close(key_code):
                raise RuntimeError("Dataset recording cancelled before ffmpeg capture started.")
            remaining_seconds = max(0.0, deadline - time.monotonic())
            announced_second = int(remaining_seconds) + (0 if remaining_seconds.is_integer() else 1)
            if announced_second != last_announced_second:
                logger.info("Dataset recording starts in %ss", announced_second)
                last_announced_second = announced_second
            if remaining_seconds <= 0.0:
                break
    finally:
        preview.close()
        camera.close()
    return last_display_frame


def _build_ffmpeg_record_command(
    *,
    ffmpeg_path: str,
    config: AppConfig,
    output_path: Path,
    duration_seconds: float,
    include_preview: bool,
) -> list[str]:
    device_source = _camera_device_source(config.camera)
    input_format = _ffmpeg_input_format(config.camera.fourcc)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        _FFMPEG_LOGLEVEL,
        "-y",
        "-f",
        "v4l2",
    ]
    if input_format is not None:
        command.extend(["-input_format", input_format])
    command.extend(
        [
            "-framerate",
            str(config.camera.fps),
            "-video_size",
            f"{config.camera.width}x{config.camera.height}",
            "-i",
            device_source,
            "-t",
            f"{duration_seconds:.3f}",
        ]
    )
    suffix = output_path.suffix.lower()
    if suffix in {".avi", ".mkv"} and input_format == "mjpeg":
        command.extend(["-c:v", "copy"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
    command.append(output_path.as_posix())
    return command


def _camera_device_source(camera_config: CameraConfig) -> str:
    if camera_config.backend == "v4l2":
        return f"/dev/video{camera_config.device_index}"
    return str(camera_config.device_index)


def _ffmpeg_input_format(fourcc: str | None) -> str | None:
    if fourcc is None:
        return None
    normalized = fourcc.upper()
    if normalized == "MJPG":
        return "mjpeg"
    if normalized == "YUYV":
        return "yuyv422"
    return None


def _run_recording_status_window(
    config: AppConfig,
    *,
    duration_seconds: float,
    process: subprocess.Popen[str],
    preview_frame: Any | None,
    cv2_module: Any | None,
) -> None:
    preview = OpenCVPreviewWindow(cv2_module=cv2_module)
    window_name = f"{config.camera.window_name} Dataset Recording"
    started_at = time.monotonic()
    try:
        while True:
            if process.poll() is not None:
                break
            frame = _build_status_frame(
                config,
                preview_frame=preview_frame,
                elapsed_seconds=max(0.0, time.monotonic() - started_at),
                duration_seconds=duration_seconds,
                cv2_module=cv2_module,
            )
            preview.show(window_name, frame)
            key_code = preview.poll_key()
            if preview.should_close(key_code):
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise RuntimeError("Dataset recording cancelled during ffmpeg capture.")
        frame = _build_status_frame(
            config,
            preview_frame=preview_frame,
            elapsed_seconds=duration_seconds,
            duration_seconds=duration_seconds,
            cv2_module=cv2_module,
            completed=True,
        )
        preview.show(window_name, frame)
        preview.poll_key()
    finally:
        preview.close()


def _build_status_frame(
    config: AppConfig,
    *,
    preview_frame: Any | None,
    elapsed_seconds: float,
    duration_seconds: float,
    cv2_module: Any | None,
    completed: bool = False,
) -> Any:
    if cv2_module is None:
        import cv2

        cv2_module = cv2
    if preview_frame is not None and hasattr(preview_frame, "copy"):
        frame = preview_frame.copy()
    else:
        frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
    overlay = frame.copy()
    alpha = 0.35
    cv2_module.rectangle(overlay, (32, 32), (min(frame.shape[1] - 32, 640), 180), (0, 0, 0), -1)
    cv2_module.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    status_text = "Recording..." if not completed else "Recording Complete"
    remaining_seconds = max(0.0, duration_seconds - elapsed_seconds)
    timer_text = (
        f"Elapsed {elapsed_seconds:05.1f}s / {duration_seconds:05.1f}s"
        if not completed
        else f"Saved {duration_seconds:05.1f}s clip"
    )
    remaining_text = f"Remaining {remaining_seconds:05.1f}s" if not completed else "Window closes now"
    color = (30, 30, 220) if not completed else (40, 170, 40)
    cv2_module.circle(frame, (72, 78), 14, color, -1)
    cv2_module.putText(frame, status_text, (104, 88), cv2_module.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2_module.putText(frame, timer_text, (48, 130), cv2_module.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2_module.putText(frame, remaining_text, (48, 164), cv2_module.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
    return frame


def _probe_recorded_video(output_path: Path, *, cv2_module: Any | None) -> dict[str, float | int]:
    if cv2_module is None:
        import cv2

        cv2_module = cv2
    capture = cv2_module.VideoCapture(output_path.as_posix())
    if not bool(capture.isOpened()):
        capture.release()
        raise RuntimeError(f"Unable to open recorded dataset video for probing: {output_path}")
    frame_count = int(round(float(capture.get(getattr(cv2_module, "CAP_PROP_FRAME_COUNT", 7)))))
    fps = float(capture.get(getattr(cv2_module, "CAP_PROP_FPS", 5)))
    width = int(round(float(capture.get(getattr(cv2_module, "CAP_PROP_FRAME_WIDTH", 3)))))
    height = int(round(float(capture.get(getattr(cv2_module, "CAP_PROP_FRAME_HEIGHT", 4)))))
    capture.release()
    duration_seconds = frame_count / fps if fps > 0.0 and frame_count > 0 else 0.0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def _tail_lines(text: str | None, *, limit: int) -> list[str]:
    if not text:
        return []
    return [line for line in text.splitlines()[-limit:] if line.strip()]


def _open_video_writer(
    output_path: Path,
    *,
    fps: int,
    frame_size: tuple[int, int],
    cv2_module: Any | None,
) -> VideoWriterProtocol:
    if cv2_module is None:
        import cv2

        cv2_module = cv2
    fourcc = cv2_module.VideoWriter_fourcc(*_DEFAULT_FOURCC)
    writer = cv2_module.VideoWriter(
        output_path.as_posix(),
        fourcc,
        float(fps),
        frame_size,
    )
    is_opened = getattr(writer, "isOpened", None)
    if callable(is_opened) and not bool(is_opened()):
        writer.release()
        raise RuntimeError(f"Unable to open video writer for dataset recording: {output_path}")
    return writer


def _resolve_output_fps(
    *,
    target_fps: int,
    recent_capture_timestamps: tuple[float, ...],
) -> float:
    estimated_fps = _estimate_capture_fps(recent_capture_timestamps)
    if estimated_fps is not None:
        return estimated_fps
    return float(target_fps)


def _estimate_capture_fps(capture_timestamps: tuple[float, ...]) -> float | None:
    if len(capture_timestamps) < 2:
        return None
    elapsed_seconds = capture_timestamps[-1] - capture_timestamps[0]
    if elapsed_seconds <= 0.0:
        return None
    return (len(capture_timestamps) - 1) / elapsed_seconds


def _build_recording_config(
    config_path: str | Path,
    *,
    camera_index: int | None = None,
    camera_backend: str | None = None,
    camera_fourcc: str | None = None,
) -> AppConfig:
    config = load_config(Path(config_path))
    if camera_index is None and camera_backend is None and camera_fourcc is None:
        return config
    return replace(
        config,
        camera=CameraConfig(
            device_index=config.camera.device_index if camera_index is None else camera_index,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            backend=config.camera.backend if camera_backend is None else camera_backend.lower(),
            fourcc=(
                config.camera.fourcc
                if camera_fourcc is None
                else camera_fourcc.strip().upper()
            ),
            mirror=config.camera.mirror,
            window_name=config.camera.window_name,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for dataset recording."""

    args = parse_args(argv)
    try:
        config = _build_recording_config(
            args.config,
            camera_index=args.camera_index,
            camera_backend=args.camera_backend,
            camera_fourcc=args.camera_fourcc,
        )
        configure_logging(
            config.logging.level,
            log_format=config.logging.format,
            structured=config.logging.structured,
        )
        result = record_dataset_video(
            config,
            output_path=args.output_video,
            start_delay_seconds=args.start_delay_seconds,
            duration_seconds=args.duration_seconds,
        )
    except (ConfigError, RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Recorded {result.frames_recorded} frames to {result.output_path} "
        f"({result.frame_width}x{result.frame_height} @ output {result.output_fps:.3f} fps, "
        f"target {result.target_fps} fps)"
    )
    print(f"Metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()

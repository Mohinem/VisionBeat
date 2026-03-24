"""Top-level application orchestration for VisionBeat."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from visionbeat.audio import AudioEngine, create_audio_engine
from visionbeat.camera import CameraFrame, CameraSource
from visionbeat.config import AppConfig
from visionbeat.gestures import GestureDetector
from visionbeat.models import (
    AudioTrigger,
    DetectionCandidate,
    FrameTimestamp,
    GestureEvent,
    RenderState,
)
from visionbeat.observability import ObservabilityRecorder, build_observability_recorder
from visionbeat.overlay import OverlayRenderer
from visionbeat.tracking import PoseTracker

logger = logging.getLogger(__name__)


class PreviewWindow(Protocol):
    """Interface for preview windows that show rendered frames and poll for exit keys."""

    def show(self, window_name: str, frame: Any) -> None:
        """Display a rendered frame in the named preview window."""

    def poll_key(self) -> int | None:
        """Return the most recent keyboard key code, if any."""

    def should_close(self, key_code: int | None = None) -> bool:
        """Return whether the user requested the loop to stop."""

    def close(self) -> None:
        """Release any preview-window resources."""


@dataclass(slots=True)
class OpenCVPreviewWindow:
    """OpenCV-backed preview window for VisionBeat's rendered output."""

    cv2_module: Any | None = None
    exit_keys: tuple[int, ...] = (27, ord("q"))
    _cv2: Any = field(init=False)

    def __post_init__(self) -> None:
        """Store or lazily import the OpenCV module used for display."""
        if self.cv2_module is not None:
            self._cv2 = self.cv2_module
            return

        import cv2

        self._cv2 = cv2

    def show(self, window_name: str, frame: Any) -> None:
        """Show the provided frame in the configured OpenCV window."""
        self._cv2.imshow(window_name, frame)

    def poll_key(self) -> int | None:
        """Poll for keyboard input and return a normalized key code."""
        pressed_key = self._cv2.waitKey(1)
        if pressed_key < 0:
            return None
        return pressed_key & 0xFF

    def should_close(self, key_code: int | None = None) -> bool:
        """Return whether an exit key was pressed."""
        if key_code is None:
            return False
        return key_code in self.exit_keys

    def close(self) -> None:
        """Destroy any OpenCV preview windows."""
        self._cv2.destroyAllWindows()


@dataclass(slots=True)
class VisionBeatRuntime:
    """Composable real-time loop that orchestrates capture, tracking, gestures, audio, and UI."""

    config: AppConfig
    camera: CameraSource
    tracker: PoseTracker
    detector: GestureDetector
    audio: AudioEngine
    overlay: OverlayRenderer
    preview: PreviewWindow
    recorder: ObservabilityRecorder | None = None
    _last_confirmed_gesture: GestureEvent | None = field(default=None, init=False)
    _last_frame_time: float | None = field(default=None, init=False)
    _overlays_enabled: bool = field(default=True, init=False)
    _debug_enabled: bool = field(default=True, init=False)

    def run(self) -> None:
        """Run the application until the preview window or frame source requests shutdown."""
        logger.info("Starting VisionBeat runtime loop")
        self._overlays_enabled = self.config.overlay.draw_landmarks or self.config.overlay.show_debug_panel
        self._debug_enabled = self.config.overlay.show_debug_panel
        self.overlay.set_overlay_enabled(self._overlays_enabled)
        self.overlay.set_debug_enabled(self._debug_enabled)
        if self.recorder is not None:
            self.recorder.log_runtime_started(window_name=self.config.camera.window_name)
        self.camera.open()
        logger.info("Camera opened; entering processing loop")
        try:
            while self.process_next_frame():
                continue
        finally:
            self.close()

    def process_next_frame(self) -> bool:
        """Process one frame and return whether the runtime should continue."""
        camera_frame = self.camera.read_frame()
        timestamp = FrameTimestamp(seconds=camera_frame.captured_at)
        pose = self.tracker.process(camera_frame.image, timestamp)
        events = list(self.detector.update(pose))
        current_candidate = self._select_candidate()

        logger.debug(
            "Frame index=%s tracking_status=%s detected=%s candidates=%s events=%s",
            camera_frame.frame_index,
            pose.status,
            pose.person_detected,
            len(self.detector.candidates),
            len(events),
        )

        if pose.status != "tracking" and self.recorder is not None:
            self.recorder.log_tracking_failure(timestamp=timestamp.seconds, status=pose.status)

        for event in events:
            self._handle_confirmed_gesture(event)

        render_state = RenderState(
            pose=pose,
            frame_index=camera_frame.frame_index,
            fps=self._compute_fps(camera_frame),
            current_candidate=current_candidate,
            confirmed_gesture=self._last_confirmed_gesture,
            cooldown_remaining_seconds=self.detector.cooldown_remaining(timestamp),
        )
        rendered_frame = self.overlay.render(camera_frame.image, render_state)
        self.preview.show(self.config.camera.window_name, rendered_frame)

        key_code = self.preview.poll_key()
        self._handle_key_command(key_code)
        if self.preview.should_close(key_code):
            logger.info("Stopping VisionBeat runtime loop on user request")
            if self.recorder is not None:
                self.recorder.log_runtime_stopped(reason="user_request")
            return False
        return True

    def _handle_key_command(self, key_code: int | None) -> None:
        """Handle interactive keyboard controls for overlay visibility."""
        if key_code is None:
            return
        if key_code == ord("o"):
            self._overlays_enabled = not self._overlays_enabled
            self.overlay.set_overlay_enabled(self._overlays_enabled)
            logger.info(
                "Overlay visibility toggled to %s via keyboard shortcut",
                "on" if self._overlays_enabled else "off",
            )
            return
        if key_code == ord("d"):
            self._debug_enabled = not self._debug_enabled
            self.overlay.set_debug_enabled(self._debug_enabled)
            logger.info(
                "Debug panel toggled to %s via keyboard shortcut",
                "on" if self._debug_enabled else "off",
            )

    def _select_candidate(self) -> DetectionCandidate | None:
        """Return the highest-confidence active gesture candidate, if any."""
        if not self.detector.candidates:
            return None
        return max(self.detector.candidates, key=lambda candidate: candidate.confidence)

    def _compute_fps(self, camera_frame: CameraFrame) -> float | None:
        """Compute an instantaneous FPS estimate from captured frame timestamps."""
        if self._last_frame_time is None:
            self._last_frame_time = camera_frame.captured_at
            return None

        elapsed = camera_frame.captured_at - self._last_frame_time
        self._last_frame_time = camera_frame.captured_at
        if elapsed <= 0.0:
            return None
        return 1.0 / elapsed

    def _handle_confirmed_gesture(self, event: GestureEvent) -> None:
        """Log, persist, and play audio for a confirmed gesture event."""
        logger.info(
            "Confirmed gesture=%s hand=%s confidence=%.2f timestamp=%.3f",
            event.gesture,
            event.hand,
            event.confidence,
            event.timestamp.seconds,
        )
        self._last_confirmed_gesture = event
        self.audio.trigger(
            AudioTrigger(
                gesture=event.gesture,
                timestamp=event.timestamp,
                intensity=event.confidence,
            )
        )

    def close(self) -> None:
        """Release external resources owned by the runtime."""
        logger.info("Shutting down VisionBeat runtime resources")
        if self.recorder is not None:
            self.recorder.log_app_shutdown()
        self.camera.close()
        self.tracker.close()
        self.audio.close()
        self.preview.close()
        if self.recorder is not None:
            self.recorder.close()


@dataclass(slots=True)
class VisionBeatApp:
    """Default dependency container for the VisionBeat runtime."""

    config: AppConfig
    camera: CameraSource = field(init=False)
    tracker: PoseTracker = field(init=False)
    detector: GestureDetector = field(init=False)
    recorder: ObservabilityRecorder = field(init=False)
    audio: AudioEngine = field(init=False)
    overlay: OverlayRenderer = field(init=False)
    preview: PreviewWindow = field(init=False)
    runtime: VisionBeatRuntime = field(init=False)

    def __post_init__(self) -> None:
        """Initialize runtime dependencies."""
        self.recorder = build_observability_recorder(self.config.logging)
        self.camera = CameraSource(self.config.camera, recorder=self.recorder)
        self.tracker = PoseTracker(self.config.tracker)
        self.detector = GestureDetector(self.config.gestures, observer=self.recorder)
        self.audio = create_audio_engine(self.config.audio)
        self.overlay = OverlayRenderer(self.config.overlay)
        self.preview = OpenCVPreviewWindow()
        self.recorder.log_app_startup(
            config_summary={
                "camera_device_index": self.config.camera.device_index,
                "camera_resolution": f"{self.config.camera.width}x{self.config.camera.height}",
                "camera_fps": self.config.camera.fps,
                "active_hand": self.config.gestures.active_hand,
                "event_log_format": self.config.logging.event_log_format,
                "event_log_path": self.config.logging.event_log_path,
            }
        )
        self.runtime = VisionBeatRuntime(
            config=self.config,
            camera=self.camera,
            tracker=self.tracker,
            detector=self.detector,
            audio=self.audio,
            overlay=self.overlay,
            preview=self.preview,
            recorder=self.recorder,
        )

    def run(self) -> None:
        """Run the real-time webcam processing loop until the user exits."""
        self.runtime.run()

    def close(self) -> None:
        """Release external resources used by the app."""
        self.runtime.close()

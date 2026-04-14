"""Top-level application orchestration for VisionBeat."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from visionbeat.audio import AudioEngine, create_audio_engine
from visionbeat.camera import CameraFrame, CameraSource
from visionbeat.config import AppConfig
from visionbeat.features import (
    CanonicalFeatureSchema,
    CanonicalFeatureExtractor,
    CanonicalFrameFeatures,
    CanonicalSequenceWindow,
    assert_feature_schemas_match,
    build_sequence_window,
    get_canonical_feature_schema,
)
from visionbeat.gestures import GestureDetector
from visionbeat.models import (
    AudioTrigger,
    DetectionCandidate,
    FrameTimestamp,
    GestureEvent,
    RenderState,
    TrackerOutput,
)
from visionbeat.observability import ObservabilityRecorder, build_observability_recorder
from visionbeat.overlay import OverlayRenderer
from visionbeat.pose_provider import PoseProvider, create_pose_provider
from visionbeat.session_recording import SessionRecorder, build_session_recorder
from visionbeat.transport import (
    GestureEventTransport,
    NullGestureEventTransport,
    UdpGestureEventTransport,
)

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
        pressed_key = int(self._cv2.waitKey(1))
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
    tracker: PoseProvider
    detector: GestureDetector
    audio: AudioEngine
    overlay: OverlayRenderer
    preview: PreviewWindow
    transport: GestureEventTransport = field(default_factory=NullGestureEventTransport)
    recorder: ObservabilityRecorder | None = None
    session_recorder: SessionRecorder | None = None
    overlay_toggle_key: int = ord("o")
    debug_toggle_key: int = ord("d")
    live_feature_history_size: int = 32
    feature_extractor: CanonicalFeatureExtractor = field(default_factory=CanonicalFeatureExtractor)
    live_feature_schema: CanonicalFeatureSchema = field(default_factory=get_canonical_feature_schema)
    _last_confirmed_gesture: GestureEvent | None = field(default=None, init=False)
    _last_frame_time: float | None = field(default=None, init=False)
    _overlays_enabled: bool = field(default=True, init=False)
    _debug_enabled: bool = field(default=True, init=False)
    _feature_history: deque[CanonicalFrameFeatures] = field(init=False)
    _latest_frame_features: CanonicalFrameFeatures | None = field(default=None, init=False)
    _latest_feature_vector: tuple[float, ...] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize live canonical-feature state for future model inference."""
        if self.live_feature_history_size <= 0:
            raise ValueError("live_feature_history_size must be greater than zero.")
        assert_feature_schemas_match(
            self.live_feature_schema,
            self.feature_extractor.schema,
            context_expected="live runtime schema",
            context_actual="feature extractor schema",
        )
        self._feature_history = deque(maxlen=self.live_feature_history_size)

    def run(self) -> None:
        """Run the application until the preview window or frame source requests shutdown."""
        logger.info("Starting VisionBeat runtime loop")
        self._overlays_enabled = (
            self.config.overlay.draw_landmarks
            or self.config.overlay.show_debug_panel
            or self.config.overlay.show_trigger_flash
        )
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
        if self.session_recorder is not None:
            self.session_recorder.record_camera_frame(camera_frame)
        timestamp = FrameTimestamp(seconds=camera_frame.captured_at)
        pose = self.tracker.process(camera_frame.image, timestamp)
        # The canonical live feature vector becomes available here. A future CNN
        # inference path should consume `frame_features.vector` instead of
        # reimplementing any pose-to-feature formulas locally.
        frame_features = self._extract_live_features(pose)
        if self.session_recorder is not None:
            self.session_recorder.record_tracker_output(camera_frame, pose)
        events = list(self.detector.update(pose))
        current_candidate = self._select_candidate()
        display_pose = self._pose_for_display(pose, mirrored=camera_frame.mirrored_for_display)
        display_frame = (
            camera_frame.display_image
            if camera_frame.display_image is not None
            else camera_frame.image
        )

        logger.debug(
            "Frame index=%s tracking_status=%s detected=%s feature_dims=%s candidates=%s events=%s",
            camera_frame.frame_index,
            pose.status,
            pose.person_detected,
            len(frame_features.vector),
            len(self.detector.candidates),
            len(events),
        )

        if pose.status != "tracking" and self.recorder is not None:
            self.recorder.log_tracking_failure(timestamp=timestamp.seconds, status=pose.status)

        for event in events:
            self._handle_confirmed_gesture(event)

        render_state = RenderState(
            pose=display_pose,
            frame_index=camera_frame.frame_index,
            fps=self._compute_fps(camera_frame),
            current_candidate=current_candidate,
            confirmed_gesture=self._last_confirmed_gesture,
            cooldown_remaining_seconds=self.detector.cooldown_remaining(timestamp),
            detector_status=self._detector_status(timestamp),
            audio_status=self._audio_status(),
        )
        rendered_frame = self.overlay.render(display_frame, render_state)
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
        if key_code == self.overlay_toggle_key:
            self._overlays_enabled = not self._overlays_enabled
            self.overlay.set_overlay_enabled(self._overlays_enabled)
            logger.info(
                "Overlay visibility toggled to %s via keyboard shortcut",
                "on" if self._overlays_enabled else "off",
            )
            return
        if key_code == self.debug_toggle_key:
            if not self.config.overlay.show_debug_panel:
                logger.info("Debug panel toggle ignored because debug panel is disabled")
                return
            self._debug_enabled = not self._debug_enabled
            self.overlay.set_debug_enabled(self._debug_enabled)
            logger.info(
                "Debug panel toggled to %s via keyboard shortcut",
                "on" if self._debug_enabled else "off",
            )

    @property
    def latest_frame_features(self) -> CanonicalFrameFeatures | None:
        """Return the most recent canonical per-frame features from the live path."""
        return self._latest_frame_features

    @property
    def latest_feature_vector(self) -> tuple[float, ...] | None:
        """Return the latest ordered CNN-ready feature vector from the live path."""
        return self._latest_feature_vector

    def build_live_feature_window(
        self,
        *,
        window_size: int | None = None,
    ) -> CanonicalSequenceWindow:
        """Return a sliding canonical feature window for future live model inference."""
        return build_sequence_window(
            tuple(self._feature_history),
            window_size=window_size,
        )

    def _select_candidate(self) -> DetectionCandidate | None:
        """Return the highest-confidence active gesture candidate, if any."""
        if not self.detector.candidates:
            return None
        return max(self.detector.candidates, key=lambda candidate: candidate.confidence)

    def _extract_live_features(self, pose: TrackerOutput) -> CanonicalFrameFeatures:
        """Extract and retain live canonical features without affecting gesture logic."""
        frame_features = self.feature_extractor.update(pose)
        self._latest_frame_features = frame_features
        self._latest_feature_vector = frame_features.vector
        # Keep a bounded sliding history of canonical frames so future live CNN
        # prediction can request `build_live_feature_window(window_size=...)`
        # without recomputing any per-frame formulas.
        self._feature_history.append(frame_features)
        return frame_features

    def _detector_status(self, timestamp: FrameTimestamp) -> str | None:
        """Return a short detector-phase summary when the implementation exposes one."""
        summary = getattr(self.detector, "status_summary", None)
        if not callable(summary):
            return None
        result = summary(timestamp)
        if not isinstance(result, str):
            return None
        normalized = result.strip()
        return normalized or None

    def _pose_for_display(self, pose: Any, *, mirrored: bool) -> Any:
        """Return pose data aligned to the preview frame orientation."""
        if not mirrored:
            return pose
        mirror = getattr(pose, "mirrored_horizontally", None)
        if not callable(mirror):
            return pose
        return mirror()

    def _audio_status(self) -> str | None:
        """Return a short audio readiness summary when the implementation exposes one."""
        summary = getattr(self.audio, "status_summary", None)
        if not callable(summary):
            return None
        result = summary()
        if not isinstance(result, str):
            return None
        normalized = result.strip()
        return normalized or None

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
        if self.session_recorder is not None:
            self.session_recorder.record_trigger(event)
        self.audio.trigger(
            AudioTrigger(
                gesture=event.gesture,
                timestamp=event.timestamp,
                intensity=event.confidence,
            )
        )
        self.transport.emit(event)

    def close(self) -> None:
        """Release external resources owned by the runtime."""
        logger.info("Shutting down VisionBeat runtime resources")
        if self.recorder is not None:
            self.recorder.log_app_shutdown()
        self.camera.close()
        self.tracker.close()
        self.audio.close()
        self.transport.close()
        self.preview.close()
        if self.session_recorder is not None:
            self.session_recorder.close()
        if self.recorder is not None:
            self.recorder.close()


@dataclass(slots=True)
class VisionBeatApp:
    """Default dependency container for the VisionBeat runtime."""

    config: AppConfig
    overlay_toggle_key: str = "o"
    debug_toggle_key: str = "d"
    camera: CameraSource = field(init=False)
    tracker: PoseProvider = field(init=False)
    detector: GestureDetector = field(init=False)
    recorder: ObservabilityRecorder = field(init=False)
    session_recorder: SessionRecorder | None = field(init=False, default=None)
    audio: AudioEngine = field(init=False)
    transport: GestureEventTransport = field(init=False)
    overlay: OverlayRenderer = field(init=False)
    preview: PreviewWindow = field(init=False)
    runtime: VisionBeatRuntime = field(init=False)

    def __post_init__(self) -> None:
        """Initialize runtime dependencies."""
        if len(self.overlay_toggle_key) != 1:
            raise ValueError("overlay_toggle_key must be a single character.")
        if len(self.debug_toggle_key) != 1:
            raise ValueError("debug_toggle_key must be a single character.")
        self.recorder = build_observability_recorder(self.config.logging)
        self.session_recorder = build_session_recorder(
            self.config.logging,
            config_payload=self.config.to_dict(),
        )
        self.camera = CameraSource(self.config.camera, recorder=self.recorder)
        self.tracker = create_pose_provider(self.config.tracker)
        self.detector = GestureDetector(self.config.gestures, observer=self.recorder)
        self.audio = create_audio_engine(self.config.audio)
        if self.config.transport.backend == "udp":
            self.transport = UdpGestureEventTransport(
                host=self.config.transport.host,
                port=self.config.transport.port,
                source=self.config.transport.source,
            )
        else:
            self.transport = NullGestureEventTransport()
        self.overlay = OverlayRenderer(self.config.overlay)
        self.preview = OpenCVPreviewWindow()
        self.recorder.log_app_startup(
            config_summary={
                "camera_device_index": self.config.camera.device_index,
                "camera_resolution": f"{self.config.camera.width}x{self.config.camera.height}",
                "camera_fps": self.config.camera.fps,
                "active_hand": self.config.gestures.active_hand,
                "pose_backend": self.config.tracker.backend,
                "audio_status": self._audio_status(),
                "event_log_format": self.config.logging.event_log_format,
                "event_log_path": self.config.logging.event_log_path,
                "session_recording_mode": (
                    None
                    if self.session_recorder is None
                    else self.config.logging.session_recording_mode
                ),
                "session_recording_path": (
                    None
                    if self.session_recorder is None
                    else self.session_recorder.session_dir.as_posix()
                ),
            }
        )
        self.runtime = VisionBeatRuntime(
            config=self.config,
            camera=self.camera,
            tracker=self.tracker,
            detector=self.detector,
            audio=self.audio,
            transport=self.transport,
            overlay=self.overlay,
            preview=self.preview,
            recorder=self.recorder,
            session_recorder=self.session_recorder,
            overlay_toggle_key=ord(self.overlay_toggle_key.lower()),
            debug_toggle_key=ord(self.debug_toggle_key.lower()),
        )

    def run(self) -> None:
        """Run the real-time webcam processing loop until the user exits."""
        self.runtime.run()

    def _audio_status(self) -> str | None:
        """Return a short audio readiness summary when the implementation exposes one."""
        summary = getattr(self.audio, "status_summary", None)
        if not callable(summary):
            return None
        result = summary()
        if not isinstance(result, str):
            return None
        normalized = result.strip()
        return normalized or None

    def close(self) -> None:
        """Release external resources used by the app."""
        self.runtime.close()

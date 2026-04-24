"""Top-level application orchestration for VisionBeat."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from visionbeat.audio import AudioEngine, create_audio_engine
from visionbeat.camera import CameraFrame, CameraSource
from visionbeat.config import AppConfig
from visionbeat.features import (
    CanonicalFeatureExtractor,
    CanonicalFeatureSchema,
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
    GestureType,
    RenderState,
    TrackerOutput,
)
from visionbeat.observability import ObservabilityRecorder, build_observability_recorder
from visionbeat.overlay import OverlayRenderer
from visionbeat.pose_provider import PoseProvider, create_pose_provider
from visionbeat.predictive_shadow import (
    PredictiveShadowRunner,
    ShadowPredictionEvent,
    build_predictive_shadow_runner,
)
from visionbeat.session_recording import SessionRecorder, build_session_recorder
from visionbeat.transport import (
    GestureEventTransport,
    NullGestureEventTransport,
    UdpGestureEventTransport,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PredictiveCompletionArm:
    """One predictive arm that can be released by a matching completion event."""

    gesture: GestureType
    timing_probability: float
    gesture_confidence: float
    class_probabilities: dict[str, float]
    armed_frame_index: int
    expires_after_frame_index: int

    def frames_remaining(self, *, frame_index: int) -> int:
        """Return the remaining completion-gate lifetime in frames."""
        return max(0, self.expires_after_frame_index - frame_index + 1)


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


@dataclass(frozen=True, slots=True)
class _ProcessedFrameSnapshot:
    """Fully processed runtime state for one analyzed camera frame."""

    camera_frame: CameraFrame
    display_frame: Any
    render_state: RenderState
    processed_at: float


@dataclass(slots=True)
class _AsyncCaptureState:
    """Shared latest-frame state written by the capture worker."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)
    latest_frame: CameraFrame | None = None
    latest_capture_fps: float | None = None
    error: Exception | None = None
    stopped: bool = False

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)


@dataclass(slots=True)
class _AsyncProcessingState:
    """Shared latest-analysis state written by the inference worker."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_snapshot: _ProcessedFrameSnapshot | None = None
    latest_inference_fps: float | None = None
    error: Exception | None = None
    stopped: bool = False


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
    feature_extractor: CanonicalFeatureExtractor = field(
        default_factory=CanonicalFeatureExtractor
    )
    live_feature_schema: CanonicalFeatureSchema = field(
        default_factory=get_canonical_feature_schema
    )
    predictive_shadow_runner: PredictiveShadowRunner | None = None
    _last_confirmed_gesture: GestureEvent | None = field(default=None, init=False)
    _last_frame_time: float | None = field(default=None, init=False)
    _overlays_enabled: bool = field(default=True, init=False)
    _debug_enabled: bool = field(default=True, init=False)
    _feature_history: deque[CanonicalFrameFeatures] = field(init=False)
    _latest_frame_features: CanonicalFrameFeatures | None = field(default=None, init=False)
    _latest_feature_vector: tuple[float, ...] | None = field(default=None, init=False)
    _predictive_completion_arm: _PredictiveCompletionArm | None = field(
        default=None,
        init=False,
    )

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
        if (
            self.predictive_shadow_runner is not None
            and self.live_feature_history_size < self.predictive_shadow_runner.required_window_size
        ):
            raise ValueError(
                "live_feature_history_size must be at least the predictive shadow window size."
            )
        self._feature_history = deque(maxlen=self.live_feature_history_size)

    def run(self) -> None:
        """Run the application until the preview window or frame source requests shutdown."""
        logger.info(
            "Starting VisionBeat runtime loop async_pipeline=%s target_render_fps=%s",
            self.config.runtime.async_pipeline,
            self.config.runtime.target_render_fps,
        )
        self._initialize_runtime()
        try:
            if self.config.runtime.async_pipeline:
                self._run_async_loop()
            else:
                while self.process_next_frame():
                    continue
        finally:
            self.close()

    def process_next_frame(self) -> bool:
        """Process one frame and return whether the runtime should continue."""
        camera_frame = self.camera.read_frame()
        capture_fps = self._compute_fps(camera_frame)
        snapshot = self._process_camera_frame(
            camera_frame,
            capture_fps=capture_fps,
            inference_fps=capture_fps,
        )
        return self._render_snapshot(
            snapshot,
            display_camera_frame=camera_frame,
            capture_fps=capture_fps,
            inference_fps=capture_fps,
            render_fps=capture_fps,
        )

    def _initialize_runtime(self) -> None:
        """Prepare overlay state, logging, and the camera before entering the loop."""
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

    def _run_async_loop(self) -> None:
        """Run capture, inference, and preview refresh on separate schedules."""
        capture_state = _AsyncCaptureState()
        processing_state = _AsyncProcessingState()
        stop_event = threading.Event()
        capture_thread = threading.Thread(
            target=self._capture_worker,
            name="visionbeat-capture",
            args=(capture_state, stop_event),
            daemon=True,
        )
        processing_thread = threading.Thread(
            target=self._processing_worker,
            name="visionbeat-processing",
            args=(capture_state, processing_state, stop_event),
            daemon=True,
        )
        capture_thread.start()
        processing_thread.start()
        target_interval_seconds = 1.0 / float(self.config.runtime.target_render_fps)
        last_render_at: float | None = None
        try:
            while not stop_event.is_set():
                self._raise_async_worker_errors(capture_state, processing_state)
                render_started_at = time.monotonic()
                render_fps = self._compute_loop_rate(last_render_at, render_started_at)
                last_render_at = render_started_at
                should_continue = self._render_async_frame(
                    capture_state,
                    processing_state,
                    render_fps=render_fps,
                )
                if not should_continue:
                    stop_event.set()
                    break
                elapsed = time.monotonic() - render_started_at
                sleep_for = target_interval_seconds - elapsed
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
        finally:
            stop_event.set()
            with capture_state.condition:
                capture_state.condition.notify_all()
            capture_thread.join(timeout=1.0)
            processing_thread.join(timeout=1.0)
            self._raise_async_worker_errors(capture_state, processing_state)

    def _capture_worker(
        self,
        capture_state: _AsyncCaptureState,
        stop_event: threading.Event,
    ) -> None:
        """Continuously grab the newest camera frame without waiting on inference."""
        last_capture_at: float | None = None
        try:
            while not stop_event.is_set():
                camera_frame = self.camera.read_frame()
                capture_fps = self._compute_loop_rate(last_capture_at, camera_frame.captured_at)
                last_capture_at = camera_frame.captured_at
                with capture_state.condition:
                    capture_state.latest_frame = camera_frame
                    capture_state.latest_capture_fps = capture_fps
                    capture_state.condition.notify_all()
        except Exception as exc:
            with capture_state.condition:
                capture_state.error = exc
                capture_state.stopped = True
                capture_state.condition.notify_all()
            return
        with capture_state.condition:
            capture_state.stopped = True
            capture_state.condition.notify_all()

    def _processing_worker(
        self,
        capture_state: _AsyncCaptureState,
        processing_state: _AsyncProcessingState,
        stop_event: threading.Event,
    ) -> None:
        """Analyze only the freshest captured frame and drop stale backlog."""
        last_processed_frame_index = -1
        last_processed_at: float | None = None
        try:
            while not stop_event.is_set():
                with capture_state.condition:
                    capture_state.condition.wait_for(
                        lambda: stop_event.is_set()
                        or capture_state.error is not None
                        or (
                            capture_state.latest_frame is not None
                            and capture_state.latest_frame.frame_index != last_processed_frame_index
                        )
                        or capture_state.stopped,
                        timeout=self.config.runtime.idle_sleep_seconds,
                    )
                    if stop_event.is_set():
                        break
                    if capture_state.error is not None:
                        raise capture_state.error
                    camera_frame = capture_state.latest_frame
                    capture_fps = capture_state.latest_capture_fps
                    stopped = capture_state.stopped
                if camera_frame is None:
                    if stopped:
                        break
                    continue
                if camera_frame.frame_index == last_processed_frame_index:
                    continue
                processed_at = time.monotonic()
                inference_fps = self._compute_loop_rate(last_processed_at, processed_at)
                snapshot = self._process_camera_frame(
                    camera_frame,
                    capture_fps=capture_fps,
                    inference_fps=inference_fps,
                )
                last_processed_at = snapshot.processed_at
                last_processed_frame_index = camera_frame.frame_index
                with processing_state.lock:
                    processing_state.latest_snapshot = snapshot
                    processing_state.latest_inference_fps = inference_fps
        except Exception as exc:
            with processing_state.lock:
                processing_state.error = exc
                processing_state.stopped = True
            return
        with processing_state.lock:
            processing_state.stopped = True

    def _process_camera_frame(
        self,
        camera_frame: CameraFrame,
        *,
        capture_fps: float | None,
        inference_fps: float | None,
    ) -> _ProcessedFrameSnapshot:
        """Run tracking, gesture logic, and predictive inference for one camera frame."""
        if self.session_recorder is not None:
            self.session_recorder.record_camera_frame(camera_frame)
        timestamp = FrameTimestamp(seconds=camera_frame.captured_at)
        pose = self.tracker.process(camera_frame.image, timestamp)
        frame_features = self._extract_live_features(pose)
        if self.session_recorder is not None:
            self.session_recorder.record_tracker_output(camera_frame, pose)
        heuristic_events = tuple(self.detector.update(pose))
        current_candidate = self._select_candidate()
        display_pose = self._pose_for_display(pose, mirrored=camera_frame.mirrored_for_display)
        display_frame = self._display_frame(camera_frame)

        logger.debug(
            "Frame index=%s tracking_status=%s detected=%s feature_dims=%s candidates=%s events=%s",
            camera_frame.frame_index,
            pose.status,
            pose.person_detected,
            len(frame_features.vector),
            len(self.detector.candidates),
            len(heuristic_events),
        )

        if pose.status != "tracking" and self.recorder is not None:
            self.recorder.log_tracking_failure(timestamp=timestamp.seconds, status=pose.status)

        predictive_events = self._update_predictive_shadow(
            camera_frame,
            pose,
            heuristic_events,
        )
        self._refresh_predictive_completion_arm(frame_index=camera_frame.frame_index)

        if self.config.predictive.heuristic_drives_audio:
            for event in heuristic_events:
                self._handle_confirmed_gesture(event, source="heuristic")
        elif self.config.predictive.predictive_uses_completion_gate:
            for event in heuristic_events:
                self._handle_predictive_completion_gate(
                    event,
                    frame_index=camera_frame.frame_index,
                )

        for predictive_event in predictive_events:
            self._handle_predictive_event(
                predictive_event,
                playback_timestamp=pose.timestamp,
                playback_frame_index=camera_frame.frame_index,
            )

        render_state = RenderState(
            pose=display_pose,
            frame_index=camera_frame.frame_index,
            fps=inference_fps or capture_fps,
            capture_fps=capture_fps,
            inference_fps=inference_fps,
            current_candidate=current_candidate,
            confirmed_gesture=self._last_confirmed_gesture,
            cooldown_remaining_seconds=self.detector.cooldown_remaining(timestamp),
            detector_status=self._detector_status(timestamp),
            predictive_status=self._predictive_status(camera_frame.frame_index),
            audio_status=self._audio_status(),
            pipeline_latency_ms=max(0.0, (time.monotonic() - camera_frame.captured_at) * 1000.0),
        )
        return _ProcessedFrameSnapshot(
            camera_frame=camera_frame,
            display_frame=display_frame,
            render_state=render_state,
            processed_at=time.monotonic(),
        )

    def _render_async_frame(
        self,
        capture_state: _AsyncCaptureState,
        processing_state: _AsyncProcessingState,
        *,
        render_fps: float | None,
    ) -> bool:
        """Render the freshest camera frame with the latest completed analysis."""
        with capture_state.lock:
            latest_capture = capture_state.latest_frame
            capture_fps = capture_state.latest_capture_fps
        with processing_state.lock:
            latest_snapshot = processing_state.latest_snapshot
            inference_fps = processing_state.latest_inference_fps
        if latest_capture is None and latest_snapshot is None:
            time.sleep(self.config.runtime.idle_sleep_seconds)
            return True
        if latest_snapshot is None:
            return self._render_warmup_frame(
                latest_capture,
                capture_fps=capture_fps,
                render_fps=render_fps,
            )
        return self._render_snapshot(
            latest_snapshot,
            display_camera_frame=latest_capture,
            capture_fps=capture_fps,
            inference_fps=inference_fps,
            render_fps=render_fps,
        )

    def _render_warmup_frame(
        self,
        camera_frame: CameraFrame | None,
        *,
        capture_fps: float | None,
        render_fps: float | None,
    ) -> bool:
        """Render a live preview before the first tracker result becomes available."""
        if camera_frame is None:
            return True
        render_state = RenderState(
            pose=TrackerOutput(
                timestamp=FrameTimestamp(seconds=camera_frame.captured_at),
                status="warming_up",
            ),
            frame_index=camera_frame.frame_index,
            fps=capture_fps or render_fps,
            capture_fps=capture_fps,
            render_fps=render_fps,
            detector_status="warming up",
            predictive_status=(
                "warming up" if self.config.predictive.enabled else None
            ),
            audio_status=self._audio_status(),
        )
        rendered_frame = self.overlay.render(self._display_frame(camera_frame), render_state)
        return self._show_rendered_frame(rendered_frame)

    def _render_snapshot(
        self,
        snapshot: _ProcessedFrameSnapshot,
        *,
        display_camera_frame: CameraFrame | None,
        capture_fps: float | None,
        inference_fps: float | None,
        render_fps: float | None,
    ) -> bool:
        """Render one processed snapshot and return whether the runtime should continue."""
        display_frame = (
            self._display_frame(display_camera_frame)
            if display_camera_frame is not None
            else snapshot.display_frame
        )
        display_frame_index = (
            display_camera_frame.frame_index
            if display_camera_frame is not None
            else snapshot.camera_frame.frame_index
        )
        render_state = replace(
            snapshot.render_state,
            frame_index=display_frame_index,
            fps=inference_fps or capture_fps or render_fps or snapshot.render_state.fps,
            capture_fps=capture_fps or snapshot.render_state.capture_fps,
            inference_fps=inference_fps or snapshot.render_state.inference_fps,
            render_fps=render_fps,
            pipeline_latency_ms=max(
                0.0,
                (time.monotonic() - snapshot.camera_frame.captured_at) * 1000.0,
            ),
        )
        rendered_frame = self.overlay.render(display_frame, render_state)
        return self._show_rendered_frame(rendered_frame)

    def _show_rendered_frame(self, rendered_frame: Any) -> bool:
        """Display one rendered frame and process keyboard shortcuts."""
        self.preview.show(self.config.camera.window_name, rendered_frame)
        key_code = self.preview.poll_key()
        self._handle_key_command(key_code)
        if self.preview.should_close(key_code):
            logger.info("Stopping VisionBeat runtime loop on user request")
            if self.recorder is not None:
                self.recorder.log_runtime_stopped(reason="user_request")
            return False
        return True

    def _raise_async_worker_errors(
        self,
        capture_state: _AsyncCaptureState,
        processing_state: _AsyncProcessingState,
    ) -> None:
        """Re-raise asynchronous worker failures on the main thread."""
        with capture_state.lock:
            capture_error = capture_state.error
        if capture_error is not None:
            raise RuntimeError("Camera capture worker failed.") from capture_error
        with processing_state.lock:
            processing_error = processing_state.error
        if processing_error is not None:
            raise RuntimeError("Tracking worker failed.") from processing_error

    def _display_frame(self, camera_frame: CameraFrame) -> Any:
        """Return the camera frame oriented for preview display."""
        return (
            camera_frame.display_image
            if camera_frame.display_image is not None
            else camera_frame.image
        )

    @staticmethod
    def _compute_loop_rate(previous_time: float | None, current_time: float) -> float | None:
        """Return the instantaneous rate for a repeating task measured in seconds."""
        if previous_time is None:
            return None
        elapsed = current_time - previous_time
        if elapsed <= 0.0:
            return None
        return 1.0 / elapsed

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

    def _update_predictive_shadow(
        self,
        camera_frame: CameraFrame,
        pose: TrackerOutput,
        heuristic_events: tuple[GestureEvent, ...],
    ) -> tuple[ShadowPredictionEvent, ...]:
        """Run the optional predictive path and return accepted predictive events."""
        if self.predictive_shadow_runner is None:
            return ()
        window = self.build_live_feature_window(
            window_size=self.predictive_shadow_runner.required_window_size
        )
        return self.predictive_shadow_runner.update(
            feature_window=window,
            frame_index=camera_frame.frame_index,
            timestamp=pose.timestamp,
            heuristic_events=heuristic_events,
        )

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

    def _predictive_status(self, frame_index: int) -> str | None:
        """Return a short predictive-model summary when the runtime exposes one."""
        if self.predictive_shadow_runner is None:
            return None
        summary = getattr(self.predictive_shadow_runner, "status_summary", None)
        if not callable(summary):
            return None
        result = summary()
        if not isinstance(result, str):
            return None
        normalized = result.strip()
        if not normalized:
            return None
        if not self.config.predictive.predictive_uses_completion_gate:
            return normalized
        arm = self._active_predictive_completion_arm(frame_index=frame_index)
        if arm is None:
            return f"{normalized} arm=--"
        return (
            f"{normalized} arm={arm.gesture.value} {arm.gesture_confidence:.2f} "
            f"ttl={arm.frames_remaining(frame_index=frame_index)}"
        )

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

    def _handle_confirmed_gesture(self, event: GestureEvent, *, source: str) -> None:
        """Log, persist, and play audio for one live trigger event."""
        logger.info(
            "Confirmed %s gesture=%s hand=%s confidence=%.2f timestamp=%.3f",
            source,
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

    def _handle_predictive_event(
        self,
        event: ShadowPredictionEvent,
        *,
        playback_timestamp: FrameTimestamp,
        playback_frame_index: int,
    ) -> None:
        """Dispatch one predictive event according to the configured runtime mode."""
        if self.config.predictive.predictive_logs_shadow:
            self._handle_predictive_shadow_trigger(event)
        if self.config.predictive.predictive_drives_audio:
            self._handle_predictive_live_trigger(
                event,
                playback_timestamp=playback_timestamp,
                playback_frame_index=playback_frame_index,
            )

    def _predictive_completion_horizon_frames(self) -> int:
        """Return the predictive arm lifetime window in frames."""
        if self.predictive_shadow_runner is None:
            return 1
        horizon = getattr(self.predictive_shadow_runner, "prediction_horizon_frames", None)
        if isinstance(horizon, int) and horizon > 0:
            return horizon
        return 1

    def _active_predictive_completion_arm(
        self,
        *,
        frame_index: int,
    ) -> _PredictiveCompletionArm | None:
        """Return the current predictive arm after expiring stale state."""
        arm = self._predictive_completion_arm
        if arm is None:
            return None
        if frame_index <= arm.expires_after_frame_index:
            return arm
        logger.debug(
            "Predictive completion arm expired gesture=%s armed_frame=%s current_frame=%s",
            arm.gesture,
            arm.armed_frame_index,
            frame_index,
        )
        self._predictive_completion_arm = None
        return None

    def _refresh_predictive_completion_arm(self, *, frame_index: int) -> None:
        """Update the predictive completion gate from the latest live model status."""
        current_arm = self._active_predictive_completion_arm(frame_index=frame_index)
        if (
            not self.config.predictive.predictive_uses_completion_gate
            or self.predictive_shadow_runner is None
        ):
            return
        status = self.predictive_shadow_runner.latest_status
        if (
            status.available_window_frames < status.required_window_size
            or status.timing_probability is None
            or status.predicted_gesture is None
            or status.predicted_gesture_confidence is None
            or status.timing_probability < status.threshold
        ):
            return
        refreshed_arm = _PredictiveCompletionArm(
            gesture=status.predicted_gesture,
            timing_probability=status.timing_probability,
            gesture_confidence=status.predicted_gesture_confidence,
            class_probabilities=dict(status.class_probabilities),
            armed_frame_index=frame_index,
            expires_after_frame_index=frame_index + self._predictive_completion_horizon_frames(),
        )
        if current_arm is not None and current_arm.gesture != refreshed_arm.gesture:
            logger.debug(
                "Predictive completion arm retained gesture=%s frame=%s despite "
                "conflicting status gesture=%s",
                current_arm.gesture,
                frame_index,
                refreshed_arm.gesture,
            )
            return
        if (
            current_arm is not None
            and current_arm.gesture == refreshed_arm.gesture
            and current_arm.timing_probability > refreshed_arm.timing_probability
        ):
            refreshed_arm = _PredictiveCompletionArm(
                gesture=current_arm.gesture,
                timing_probability=current_arm.timing_probability,
                gesture_confidence=current_arm.gesture_confidence,
                class_probabilities=dict(current_arm.class_probabilities),
                armed_frame_index=current_arm.armed_frame_index,
                expires_after_frame_index=refreshed_arm.expires_after_frame_index,
            )
        elif current_arm is None:
            logger.debug(
                "Predictive completion arm set gesture=%s timing_probability=%.2f "
                "gesture_confidence=%.2f frame=%s expires_after=%s",
                refreshed_arm.gesture,
                refreshed_arm.timing_probability,
                refreshed_arm.gesture_confidence,
                frame_index,
                refreshed_arm.expires_after_frame_index,
            )
        self._predictive_completion_arm = refreshed_arm

    def _handle_predictive_completion_gate(
        self,
        event: GestureEvent,
        *,
        frame_index: int,
    ) -> None:
        """Release an armed predictive gesture on a matching completion event."""
        arm = self._active_predictive_completion_arm(frame_index=frame_index)
        if arm is None:
            logger.debug(
                "Predictive completion gate ignored heuristic %s at frame=%s: no arm",
                event.gesture,
                frame_index,
            )
            return
        if event.gesture is not arm.gesture:
            logger.info(
                "Predictive completion gate ignored mismatched heuristic predicted=%s "
                "heuristic=%s frame=%s",
                arm.gesture,
                event.gesture,
                frame_index,
            )
            return
        live_event = GestureEvent(
            gesture=arm.gesture,
            confidence=event.confidence,
            hand=event.hand,
            timestamp=event.timestamp,
            label=f"Predictive-arm {arm.gesture.value} completion",
        )
        logger.info(
            "Predictive completion trigger gesture=%s completion_confidence=%.2f "
            "timing_probability=%.2f class_confidence=%.2f frame=%s",
            arm.gesture,
            event.confidence,
            arm.timing_probability,
            arm.gesture_confidence,
            frame_index,
        )
        if self.recorder is not None:
            self.recorder.log_predictive_live_trigger(
                timestamp=event.timestamp.seconds,
                frame_index=frame_index,
                timing_probability=arm.timing_probability,
                predicted_gesture=arm.gesture,
                predicted_gesture_confidence=arm.gesture_confidence,
                hand=live_event.hand,
                class_probabilities=arm.class_probabilities,
            )
        self._predictive_completion_arm = None
        self._handle_confirmed_gesture(live_event, source="predictive_completion")

    def _handle_predictive_live_trigger(
        self,
        event: ShadowPredictionEvent,
        *,
        playback_timestamp: FrameTimestamp,
        playback_frame_index: int,
    ) -> None:
        """Convert one predictive event into a live gesture trigger."""
        live_event = GestureEvent(
            gesture=event.gesture,
            confidence=event.timing_probability,
            hand=self.config.gestures.active_hand,
            timestamp=playback_timestamp,
            label=f"Predictive {event.gesture.value} trigger",
        )
        logger.info(
            "Predictive live trigger gesture=%s timing_probability=%.2f "
            "class_confidence=%.2f peak_frame=%s emit_frame=%s",
            event.gesture,
            event.timing_probability,
            event.gesture_confidence,
            event.frame_index,
            playback_frame_index,
        )
        if self.recorder is not None:
            self.recorder.log_predictive_live_trigger(
                timestamp=playback_timestamp.seconds,
                frame_index=playback_frame_index,
                timing_probability=event.timing_probability,
                predicted_gesture=event.gesture,
                predicted_gesture_confidence=event.gesture_confidence,
                hand=live_event.hand,
                class_probabilities=event.class_probabilities,
            )
        self._handle_confirmed_gesture(live_event, source="predictive")

    def _handle_predictive_shadow_trigger(self, event: ShadowPredictionEvent) -> None:
        """Log and persist one shadow-mode predictive trigger without touching audio."""
        logger.info(
            "Predictive shadow trigger gesture=%s confidence=%.2f timing_probability=%.2f frame=%s",
            event.gesture,
            event.gesture_confidence,
            event.timing_probability,
            event.frame_index,
        )
        if self.recorder is not None:
            self.recorder.log_predictive_shadow_trigger(
                timestamp=event.timestamp.seconds,
                frame_index=event.frame_index,
                timing_probability=event.timing_probability,
                predicted_gesture=event.gesture,
                predicted_gesture_confidence=event.gesture_confidence,
                heuristic_gesture_types=event.heuristic_gesture_types_on_peak_frame,
                class_probabilities=event.class_probabilities,
            )
        if self.session_recorder is not None:
            self.session_recorder.record_predictive_shadow_trigger(event.to_dict())

    def close(self) -> None:
        """Release external resources owned by the runtime."""
        logger.info("Shutting down VisionBeat runtime resources")
        if self.predictive_shadow_runner is not None:
            for event in self.predictive_shadow_runner.flush():
                self._handle_predictive_event(
                    event,
                    playback_timestamp=event.timestamp,
                    playback_frame_index=event.frame_index,
                )
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
        predictive_shadow_runner = build_predictive_shadow_runner(self.config.predictive)
        self.recorder.log_app_startup(
            config_summary={
                "camera_device_index": self.config.camera.device_index,
                "camera_resolution": f"{self.config.camera.width}x{self.config.camera.height}",
                "camera_fps": self.config.camera.fps,
                "async_pipeline": self.config.runtime.async_pipeline,
                "target_render_fps": self.config.runtime.target_render_fps,
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
                "predictive_enabled": self.config.predictive.enabled,
                "predictive_mode": self.config.predictive.mode,
                "predictive_window_size": (
                    None
                    if predictive_shadow_runner is None
                    else predictive_shadow_runner.required_window_size
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
            predictive_shadow_runner=predictive_shadow_runner,
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

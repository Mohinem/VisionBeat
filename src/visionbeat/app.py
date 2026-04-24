"""Top-level application orchestration for VisionBeat."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Protocol

from visionbeat.audio import AudioEngine, create_audio_engine
from visionbeat.camera import CameraFrame, CameraSource
from visionbeat.config import AppConfig, PredictiveConfig
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
from visionbeat.rhythm_prediction import (
    RhythmPrediction,
    RhythmPredictionConfig,
    RhythmPredictionOutcome,
    RhythmPredictionTracker,
)
from visionbeat.session_recording import SessionRecorder, build_session_recorder
from visionbeat.transport import (
    GestureEventTransport,
    NullGestureEventTransport,
    UdpGestureEventTransport,
)

logger = logging.getLogger(__name__)


def _predictor_trigger_label(gesture: GestureType, predictor: str) -> str:
    """Return the short live label used for predictor-driven audio triggers."""
    normalized = predictor.strip().lower()
    if normalized == "cnn":
        return f"{gesture.value.title()} (CNN)"
    if normalized == "rhythm":
        return f"{gesture.value.title()} (rhythm predictor)"
    return f"{gesture.value.title()} ({predictor.strip()})"


def _format_delta_ms(seconds: float) -> str:
    """Format a signed time delta for compact live status text."""
    return f"{seconds * 1000.0:+.0f}ms"


@dataclass(frozen=True, slots=True)
class _PredictiveCompletionArm:
    """One predictive arm that can be released by a matching completion event."""

    gesture: GestureType
    timing_probability: float
    gesture_confidence: float
    class_probabilities: dict[str, float]
    armed_frame_index: int
    expires_after_frame_index: int
    source: str = "cnn"
    armed_timestamp_seconds: float | None = None
    expected_timestamp_seconds: float | None = None
    expires_after_timestamp_seconds: float | None = None

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


def build_rhythm_prediction_tracker(
    config: PredictiveConfig,
) -> RhythmPredictionTracker | None:
    """Build the optional passive rhythm tracker from predictive configuration."""
    if not config.rhythm_prediction_enabled:
        return None
    default_history_size = RhythmPredictionConfig().history_size
    return RhythmPredictionTracker(
        RhythmPredictionConfig(
            min_hits=config.rhythm_min_hits,
            history_size=max(default_history_size, config.rhythm_min_hits),
            min_interval_seconds=config.rhythm_min_interval_seconds,
            max_interval_seconds=config.rhythm_max_interval_seconds,
            jitter_tolerance=config.rhythm_jitter_tolerance,
            expiry_ratio=config.rhythm_expiry_ratio,
            max_horizon_seconds=config.rhythm_max_horizon_seconds,
            match_tolerance_seconds=config.rhythm_match_tolerance_seconds,
        )
    )


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
    rhythm_tracker: RhythmPredictionTracker | None = None
    _rhythm_direct_triggered_prediction_ids: set[str] = field(
        default_factory=set,
        init=False,
    )
    _last_rhythm_outcome: RhythmPredictionOutcome | None = field(default=None, init=False)
    _last_rhythm_outcome_timestamp_seconds: float | None = field(default=None, init=False)
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
        if self.rhythm_tracker is None:
            self.rhythm_tracker = build_rhythm_prediction_tracker(self.config.predictive)

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
                    expected_next_frame_index = last_processed_frame_index
                    def fresh_frame_available(
                        expected_frame_index: int = expected_next_frame_index,
                    ) -> bool:
                        return (
                            stop_event.is_set()
                            or capture_state.error is not None
                            or (
                                capture_state.latest_frame is not None
                                and capture_state.latest_frame.frame_index
                                != expected_frame_index
                            )
                            or capture_state.stopped
                        )

                    capture_state.condition.wait_for(
                        fresh_frame_available,
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
        self._refresh_predictive_completion_arm(
            frame_index=camera_frame.frame_index,
            timestamp_seconds=pose.timestamp.seconds,
        )

        if self.config.predictive.heuristic_drives_audio:
            for event in heuristic_events:
                self._handle_confirmed_gesture(
                    event,
                    source="heuristic",
                    frame_index=camera_frame.frame_index,
                )
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

        self._advance_rhythm_predictions(
            timestamp=pose.timestamp,
            frame_index=camera_frame.frame_index,
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
            predictive_status=self._predictive_status(
                camera_frame.frame_index,
                timestamp_seconds=pose.timestamp.seconds,
            ),
            rhythm_status=self._rhythm_status(timestamp_seconds=pose.timestamp.seconds),
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
            rhythm_status=(
                "warming up" if self.config.predictive.rhythm_prediction_enabled else None
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

    def _predictive_status(
        self,
        frame_index: int,
        *,
        timestamp_seconds: float | None = None,
    ) -> str | None:
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
        arm = self._active_predictive_completion_arm(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )
        if arm is None:
            return f"{normalized} arm=--"
        arm_label = (
            arm.gesture.value
            if arm.source == "cnn"
            else f"{arm.source}:{arm.gesture.value}"
        )
        return (
            f"{normalized} arm={arm_label} {arm.gesture_confidence:.2f} "
            f"ttl={arm.frames_remaining(frame_index=frame_index)}"
        )

    def _rhythm_status(self, *, timestamp_seconds: float) -> str | None:
        """Return a compact live summary of rhythm-continuation state."""
        if not self.config.predictive.rhythm_prediction_enabled:
            return None
        if self.rhythm_tracker is None:
            return "enabled but tracker unavailable"
        active_predictions = self.rhythm_tracker.active_predictions(
            timestamp_seconds=timestamp_seconds,
        )
        if active_predictions:
            return " | ".join(
                self._rhythm_prediction_status(prediction, timestamp_seconds=timestamp_seconds)
                for prediction in active_predictions
            )
        recent_outcome = self._recent_rhythm_outcome_status(
            timestamp_seconds=timestamp_seconds,
        )
        if recent_outcome is not None:
            return recent_outcome
        return self._rhythm_learning_status()

    def _rhythm_prediction_status(
        self,
        prediction: RhythmPrediction,
        *,
        timestamp_seconds: float,
    ) -> str:
        """Describe one active rhythm expectation for the HUD."""
        seconds_until_expected = prediction.seconds_until_expected(
            timestamp_seconds=timestamp_seconds,
        )
        seconds_until_expiry = prediction.expires_after_seconds - timestamp_seconds
        phase = "next" if seconds_until_expected > 0.0 else "due"
        if prediction.confidence < self.config.predictive.rhythm_confidence_threshold:
            phase = f"{phase}/low-conf"
        return (
            f"mode={self.config.predictive.rhythm_trigger_mode} "
            f"{prediction.gesture.value} {phase} @{prediction.expected_timestamp_seconds:.3f}s "
            f"({_format_delta_ms(seconds_until_expected)}) "
            f"exp={_format_delta_ms(seconds_until_expiry)} "
            f"int={prediction.interval_seconds:.3f}s "
            f"conf={prediction.confidence:.2f}/"
            f"{self.config.predictive.rhythm_confidence_threshold:.2f} "
            f"jit={prediction.jitter_ratio:.2f} reps={prediction.repetition_count}"
        )

    def _recent_rhythm_outcome_status(self, *, timestamp_seconds: float) -> str | None:
        """Describe the most recent rhythm outcome briefly after it happens."""
        if (
            self._last_rhythm_outcome is None
            or self._last_rhythm_outcome_timestamp_seconds is None
        ):
            return None
        outcome = self._last_rhythm_outcome
        age_seconds = timestamp_seconds - self._last_rhythm_outcome_timestamp_seconds
        if age_seconds < 0.0 or age_seconds > max(2.0, outcome.interval_seconds * 2.0):
            return None
        seconds_until_expiry = outcome.expires_after_seconds - timestamp_seconds
        expired_suffix = "/expired" if seconds_until_expiry <= 0.0 else ""
        error_text = "--" if outcome.error_ms is None else f"{outcome.error_ms:+.0f}ms"
        actual_text = (
            "--"
            if outcome.actual_time_seconds is None
            else f"{outcome.actual_time_seconds:.3f}s"
        )
        return (
            f"mode={self.config.predictive.rhythm_trigger_mode} "
            f"last={outcome.outcome}{expired_suffix} {outcome.gesture.value} "
            f"expected={outcome.predicted_time_seconds:.3f}s actual={actual_text} "
            f"err={error_text} exp={_format_delta_ms(seconds_until_expiry)} "
            f"int={outcome.interval_seconds:.3f}s conf={outcome.confidence:.2f} "
            f"jit={outcome.jitter_ratio:.2f} reps={outcome.repetition_count}"
        )

    def _rhythm_learning_status(self) -> str:
        """Describe why no rhythm prediction is currently active."""
        assert self.rhythm_tracker is not None
        histories = [
            (gesture, self.rhythm_tracker.history_for(gesture))
            for gesture in GestureType
            if self.rhythm_tracker.history_for(gesture)
        ]
        if not histories:
            return (
                f"mode={self.config.predictive.rhythm_trigger_mode} "
                f"learning hits=0/{self.config.predictive.rhythm_min_hits}"
            )
        gesture, history = max(histories, key=lambda item: item[1][-1].timestamp_seconds)
        hit_count = len(history)
        last_timestamp = history[-1].timestamp_seconds
        if hit_count < self.config.predictive.rhythm_min_hits:
            return (
                f"mode={self.config.predictive.rhythm_trigger_mode} "
                f"learning {gesture.value} hits={hit_count}/"
                f"{self.config.predictive.rhythm_min_hits} last={last_timestamp:.3f}s"
            )
        reason, interval_seconds, jitter_ratio = self._rhythm_inactive_reason(history)
        detail = ""
        if interval_seconds is not None and jitter_ratio is not None:
            detail = f" int={interval_seconds:.3f}s jit={jitter_ratio:.2f}"
        return (
            f"mode={self.config.predictive.rhythm_trigger_mode} inactive {gesture.value} "
            f"hits={hit_count} last={last_timestamp:.3f}s reason={reason}{detail}"
        )

    def _rhythm_inactive_reason(
        self,
        history: tuple[Any, ...],
    ) -> tuple[str, float | None, float | None]:
        """Return a compact reason why a retained history is not predictive."""
        intervals = tuple(
            current.timestamp_seconds - previous.timestamp_seconds
            for previous, current in zip(history, history[1:], strict=False)
        )
        if not intervals:
            return ("not-enough-intervals", None, None)
        if any(
            interval < self.config.predictive.rhythm_min_interval_seconds
            or interval > self.config.predictive.rhythm_max_interval_seconds
            for interval in intervals
        ):
            return ("interval-out-of-range", float(median(intervals)), None)
        interval_seconds = float(median(intervals))
        jitter_ratio = float(
            median(abs(interval - interval_seconds) for interval in intervals)
        ) / max(interval_seconds, 1.0e-9)
        if jitter_ratio > self.config.predictive.rhythm_jitter_tolerance:
            return ("unstable-jitter", interval_seconds, jitter_ratio)
        if interval_seconds > self.config.predictive.rhythm_max_horizon_seconds:
            return ("beyond-horizon", interval_seconds, jitter_ratio)
        return ("waiting", interval_seconds, jitter_ratio)

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

    def _handle_confirmed_gesture(
        self,
        event: GestureEvent,
        *,
        source: str,
        frame_index: int | None = None,
        observe_rhythm: bool = True,
    ) -> None:
        """Log, persist, and play audio for one live trigger event."""
        rhythm_duplicate = (
            self._rhythm_direct_duplicate_for_event(event) if observe_rhythm else None
        )
        if rhythm_duplicate is not None:
            logger.info(
                "Suppressed duplicate %s gesture=%s after rhythm direct trigger "
                "prediction_id=%s timestamp=%.3f",
                source,
                event.gesture,
                rhythm_duplicate.prediction_id,
                event.timestamp.seconds,
            )
            self._observe_rhythm_event(
                event,
                source=f"{source}_after_rhythm_direct",
                frame_index=frame_index,
            )
            return
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
        if observe_rhythm:
            self._observe_rhythm_event(event, source=source, frame_index=frame_index)

    def _rhythm_prediction_enabled(self) -> bool:
        """Return whether rhythm prediction should run in the live runtime."""
        return self.config.predictive.rhythm_prediction_enabled and self.rhythm_tracker is not None

    def _rhythm_direct_duplicate_for_event(
        self,
        event: GestureEvent,
    ) -> RhythmPrediction | None:
        """Return a direct rhythm prediction already played for this real event."""
        if not self.config.predictive.rhythm_triggers_audio or self.rhythm_tracker is None:
            return None
        for prediction in self.rhythm_tracker.active_predictions(
            timestamp_seconds=event.timestamp.seconds,
        ):
            if prediction.prediction_id not in self._rhythm_direct_triggered_prediction_ids:
                continue
            if prediction.gesture is not event.gesture:
                continue
            if (
                abs(event.timestamp.seconds - prediction.expected_timestamp_seconds)
                <= self.config.predictive.rhythm_match_tolerance_seconds
            ):
                return prediction
        return None

    def _advance_rhythm_predictions(
        self,
        *,
        timestamp: FrameTimestamp,
        frame_index: int,
    ) -> None:
        """Advance rhythm state for the current live timestamp and log expirations."""
        if not self._rhythm_prediction_enabled():
            return
        assert self.rhythm_tracker is not None
        self._trigger_due_rhythm_predictions(
            timestamp=timestamp,
            frame_index=frame_index,
        )
        update = self.rhythm_tracker.advance_with_outcomes(timestamp_seconds=timestamp.seconds)
        self._active_predictive_completion_arm(
            frame_index=frame_index,
            timestamp_seconds=timestamp.seconds,
        )
        for outcome in update.outcomes:
            self._log_rhythm_prediction_outcome(
                outcome,
                current_timestamp=timestamp.seconds,
                frame_index=frame_index,
                source="advance",
            )

    def _observe_rhythm_event(
        self,
        event: GestureEvent,
        *,
        source: str,
        frame_index: int | None,
    ) -> None:
        """Feed one accepted live gesture into the passive rhythm predictor."""
        if not self._rhythm_prediction_enabled():
            return
        assert self.rhythm_tracker is not None
        try:
            update = self.rhythm_tracker.observe_event_with_outcomes(
                event,
                source=source,
                frame_index=frame_index,
            )
        except ValueError as exc:
            logger.warning("Rhythm observation ignored: %s", exc)
            return
        for outcome in update.outcomes:
            self._log_rhythm_prediction_outcome(
                outcome,
                current_timestamp=event.timestamp.seconds,
                frame_index=frame_index,
                source=source,
            )
        if update.prediction is not None:
            self._refresh_rhythm_completion_arm(
                update.prediction,
                current_timestamp_seconds=event.timestamp.seconds,
                frame_index=frame_index,
            )

    def _trigger_due_rhythm_predictions(
        self,
        *,
        timestamp: FrameTimestamp,
        frame_index: int,
    ) -> None:
        """Trigger direct rhythm-predicted audio for due expectations."""
        if not self.config.predictive.rhythm_triggers_audio:
            return
        assert self.rhythm_tracker is not None
        for prediction in self.rhythm_tracker.active_predictions(
            timestamp_seconds=timestamp.seconds,
        ):
            if prediction.prediction_id in self._rhythm_direct_triggered_prediction_ids:
                continue
            if prediction.confidence < self.config.predictive.rhythm_confidence_threshold:
                continue
            if timestamp.seconds < prediction.expected_timestamp_seconds:
                continue
            if (
                timestamp.seconds - prediction.expected_timestamp_seconds
                > self.config.predictive.rhythm_match_tolerance_seconds
            ):
                continue
            self._handle_rhythm_direct_trigger(
                prediction,
                timestamp=timestamp,
                frame_index=frame_index,
            )

    def _handle_rhythm_direct_trigger(
        self,
        prediction: RhythmPrediction,
        *,
        timestamp: FrameTimestamp,
        frame_index: int,
    ) -> None:
        """Play one rhythm-predicted beat without feeding it back into rhythm learning."""
        self._rhythm_direct_triggered_prediction_ids.add(prediction.prediction_id)
        scheduling_error_ms = (
            timestamp.seconds - prediction.expected_timestamp_seconds
        ) * 1000.0
        logger.info(
            "Rhythm direct trigger gesture=%s confidence=%.2f expected=%.3f "
            "actual=%.3f scheduling_error_ms=%.1f frame=%s",
            prediction.gesture,
            prediction.confidence,
            prediction.expected_timestamp_seconds,
            timestamp.seconds,
            scheduling_error_ms,
            frame_index,
        )
        if self.recorder is not None:
            self.recorder.log_rhythm_live_trigger(
                timestamp=timestamp.seconds,
                frame_index=frame_index,
                prediction_id=prediction.prediction_id,
                predicted_time_seconds=prediction.expected_timestamp_seconds,
                scheduling_error_ms=scheduling_error_ms,
                gesture=prediction.gesture,
                confidence=prediction.confidence,
                interval_seconds=prediction.interval_seconds,
                repetition_count=prediction.repetition_count,
                jitter=prediction.jitter_ratio,
                source="rhythm_direct",
            )
        live_event = GestureEvent(
            gesture=prediction.gesture,
            confidence=prediction.confidence,
            hand=self.config.gestures.active_hand,
            timestamp=timestamp,
            label=_predictor_trigger_label(prediction.gesture, "rhythm"),
        )
        self._handle_confirmed_gesture(
            live_event,
            source="rhythm_direct",
            frame_index=frame_index,
            observe_rhythm=False,
        )

    def _refresh_rhythm_completion_arm(
        self,
        prediction: RhythmPrediction,
        *,
        current_timestamp_seconds: float,
        frame_index: int | None,
    ) -> None:
        """Use a confident rhythm expectation to arm hybrid completion firing."""
        if (
            frame_index is None
            or not self.config.predictive.predictive_uses_completion_gate
            or not self.config.predictive.rhythm_arms_completion_gate
            or prediction.confidence < self.config.predictive.rhythm_confidence_threshold
        ):
            return
        current_arm = self._active_predictive_completion_arm(
            frame_index=frame_index,
            timestamp_seconds=current_timestamp_seconds,
        )
        if current_arm is not None and current_arm.source == "cnn":
            logger.debug(
                "Rhythm completion arm ignored because CNN arm is active gesture=%s frame=%s",
                current_arm.gesture,
                frame_index,
            )
            return
        if current_arm is not None and current_arm.gesture is not prediction.gesture:
            logger.debug(
                "Rhythm completion arm retained gesture=%s frame=%s despite "
                "conflicting rhythm gesture=%s",
                current_arm.gesture,
                frame_index,
                prediction.gesture,
            )
            return
        expires_after_frame_index = frame_index + max(
            1,
            math.ceil(
                max(0.0, prediction.expires_after_seconds - current_timestamp_seconds)
                * float(self.config.camera.fps)
            ),
        )
        refreshed_arm = _PredictiveCompletionArm(
            gesture=prediction.gesture,
            timing_probability=prediction.confidence,
            gesture_confidence=prediction.confidence,
            class_probabilities={prediction.gesture.value: prediction.confidence},
            armed_frame_index=(
                current_arm.armed_frame_index
                if current_arm is not None
                else frame_index
            ),
            expires_after_frame_index=expires_after_frame_index,
            source="rhythm",
            armed_timestamp_seconds=(
                current_arm.armed_timestamp_seconds
                if current_arm is not None
                else current_timestamp_seconds
            ),
            expected_timestamp_seconds=prediction.expected_timestamp_seconds,
            expires_after_timestamp_seconds=prediction.expires_after_seconds,
        )
        logger.info(
            "Rhythm completion arm set gesture=%s confidence=%.2f frame=%s "
            "expected=%.3f expires_after=%.3f",
            refreshed_arm.gesture,
            refreshed_arm.gesture_confidence,
            frame_index,
            prediction.expected_timestamp_seconds,
            prediction.expires_after_seconds,
        )
        self._predictive_completion_arm = refreshed_arm

    def _log_rhythm_prediction_outcome(
        self,
        outcome: RhythmPredictionOutcome,
        *,
        current_timestamp: float,
        frame_index: int | None,
        source: str,
    ) -> None:
        """Write one structured rhythm-prediction evaluation event."""
        self._last_rhythm_outcome = outcome
        self._last_rhythm_outcome_timestamp_seconds = current_timestamp
        seconds_until_prediction = outcome.predicted_time_seconds - current_timestamp
        seconds_until_expiry = outcome.expires_after_seconds - current_timestamp
        status_description = self._rhythm_outcome_description(
            outcome,
            current_timestamp=current_timestamp,
            seconds_until_prediction=seconds_until_prediction,
            seconds_until_expiry=seconds_until_expiry,
        )
        logger.info(
            "Rhythm prediction gesture=%s outcome=%s confidence=%.2f expected=%.3f "
            "interval=%.3f expiry=%.3f source=%s description=%s",
            outcome.gesture,
            outcome.outcome,
            outcome.confidence,
            outcome.predicted_time_seconds,
            outcome.interval_seconds,
            outcome.expires_after_seconds,
            source,
            status_description,
        )
        if self.recorder is None:
            return
        self.recorder.log_rhythm_prediction(
            timestamp=current_timestamp,
            frame_index=frame_index,
            prediction_id=outcome.prediction_id,
            outcome=outcome.outcome,
            gesture=outcome.gesture,
            predicted_time_seconds=outcome.predicted_time_seconds,
            actual_time_seconds=outcome.actual_time_seconds,
            actual_gesture=outcome.actual_gesture,
            error_ms=outcome.error_ms,
            last_event_timestamp=outcome.last_observed_timestamp_seconds,
            interval_seconds=outcome.interval_seconds,
            expires_after_seconds=outcome.expires_after_seconds,
            seconds_until_prediction=seconds_until_prediction,
            seconds_until_expiry=seconds_until_expiry,
            match_tolerance_seconds=self.config.predictive.rhythm_match_tolerance_seconds,
            confidence=outcome.confidence,
            repetition_count=outcome.repetition_count,
            jitter=outcome.jitter_ratio,
            active=outcome.outcome == "pending",
            shadow_only=self.config.predictive.rhythm_trigger_mode == "shadow",
            source=source,
            trigger_mode=self.config.predictive.rhythm_trigger_mode,
            status_description=status_description,
        )

    def _rhythm_outcome_description(
        self,
        outcome: RhythmPredictionOutcome,
        *,
        current_timestamp: float,
        seconds_until_prediction: float,
        seconds_until_expiry: float,
    ) -> str:
        """Build a human-readable description for rhythm experiment logs."""
        gesture = outcome.gesture.value
        if outcome.outcome == "pending":
            return (
                f"predicted next {gesture} at {outcome.predicted_time_seconds:.3f}s "
                f"({_format_delta_ms(seconds_until_prediction)} from now); "
                f"interval={outcome.interval_seconds:.3f}s, "
                f"confidence={outcome.confidence:.2f}, jitter={outcome.jitter_ratio:.2f}, "
                f"repetitions={outcome.repetition_count}, "
                f"expires at {outcome.expires_after_seconds:.3f}s "
                f"({_format_delta_ms(seconds_until_expiry)})"
            )
        if outcome.outcome == "matched":
            actual_time = (
                current_timestamp
                if outcome.actual_time_seconds is None
                else outcome.actual_time_seconds
            )
            error_text = "--" if outcome.error_ms is None else f"{outcome.error_ms:+.1f} ms"
            return (
                f"matched predicted {gesture}: expected "
                f"{outcome.predicted_time_seconds:.3f}s, actual {actual_time:.3f}s, "
                f"error={error_text}"
            )
        if outcome.outcome == "missed":
            expiry_text = (
                "prediction state has expired"
                if seconds_until_expiry <= 0.0
                else f"state expires in {_format_delta_ms(seconds_until_expiry)}"
            )
            actual_gesture = (
                outcome.actual_gesture.value
                if outcome.actual_gesture is not None
                else "unknown"
            )
            actual_text = (
                "no matching hit arrived"
                if outcome.actual_time_seconds is None
                else (
                    f"actual {actual_gesture} arrived at "
                    f"{outcome.actual_time_seconds:.3f}s"
                )
            )
            return (
                f"missed predicted {gesture} at {outcome.predicted_time_seconds:.3f}s; "
                f"{actual_text}; {expiry_text}"
            )
        return (
            f"expired predicted {gesture} at {outcome.predicted_time_seconds:.3f}s; "
            f"expiry was {outcome.expires_after_seconds:.3f}s"
        )

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
        timestamp_seconds: float | None = None,
    ) -> _PredictiveCompletionArm | None:
        """Return the current predictive arm after expiring stale state."""
        arm = self._predictive_completion_arm
        if arm is None:
            return None
        frame_active = frame_index <= arm.expires_after_frame_index
        time_active = (
            True
            if timestamp_seconds is None or arm.expires_after_timestamp_seconds is None
            else timestamp_seconds <= arm.expires_after_timestamp_seconds
        )
        if frame_active and time_active:
            return arm
        logger.debug(
            "Predictive completion arm expired gesture=%s source=%s armed_frame=%s "
            "current_frame=%s",
            arm.gesture,
            arm.source,
            arm.armed_frame_index,
            frame_index,
        )
        self._predictive_completion_arm = None
        return None

    def _refresh_predictive_completion_arm(
        self,
        *,
        frame_index: int,
        timestamp_seconds: float | None = None,
    ) -> None:
        """Update the predictive completion gate from the latest live model status."""
        current_arm = self._active_predictive_completion_arm(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )
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
            source="cnn",
            armed_timestamp_seconds=timestamp_seconds,
        )
        if (
            current_arm is not None
            and current_arm.source == "cnn"
            and current_arm.gesture != refreshed_arm.gesture
        ):
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
            and current_arm.source == "cnn"
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
                source=current_arm.source,
                armed_timestamp_seconds=current_arm.armed_timestamp_seconds,
                expected_timestamp_seconds=current_arm.expected_timestamp_seconds,
                expires_after_timestamp_seconds=current_arm.expires_after_timestamp_seconds,
            )
        elif current_arm is None:
            logger.debug(
                "Predictive completion arm set gesture=%s source=%s timing_probability=%.2f "
                "gesture_confidence=%.2f frame=%s expires_after=%s",
                refreshed_arm.gesture,
                refreshed_arm.source,
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
        arm = self._active_predictive_completion_arm(
            frame_index=frame_index,
            timestamp_seconds=event.timestamp.seconds,
        )
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
                "heuristic=%s source=%s frame=%s",
                arm.gesture,
                event.gesture,
                arm.source,
                frame_index,
            )
            return
        predictor_label = "rhythm" if arm.source == "rhythm" else "CNN"
        live_event = GestureEvent(
            gesture=arm.gesture,
            confidence=event.confidence,
            hand=event.hand,
            timestamp=event.timestamp,
            label=_predictor_trigger_label(arm.gesture, predictor_label),
        )
        logger.info(
            "Predictive completion trigger gesture=%s completion_confidence=%.2f "
            "timing_probability=%.2f class_confidence=%.2f source=%s label=%s frame=%s",
            arm.gesture,
            event.confidence,
            arm.timing_probability,
            arm.gesture_confidence,
            arm.source,
            live_event.label,
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
                source=arm.source,
            )
        self._predictive_completion_arm = None
        self._handle_confirmed_gesture(
            live_event,
            source=(
                "rhythm_completion"
                if arm.source == "rhythm"
                else "predictive_completion"
            ),
            frame_index=frame_index,
        )

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
            label=_predictor_trigger_label(event.gesture, "CNN"),
        )
        logger.info(
            "Predictive live trigger gesture=%s timing_probability=%.2f "
            "class_confidence=%.2f label=%s peak_frame=%s emit_frame=%s",
            event.gesture,
            event.timing_probability,
            event.gesture_confidence,
            live_event.label,
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
        self._handle_confirmed_gesture(
            live_event,
            source="predictive",
            frame_index=playback_frame_index,
        )

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
                "rhythm_prediction_enabled": (
                    self.config.predictive.rhythm_prediction_enabled
                ),
                "rhythm_trigger_mode": self.config.predictive.rhythm_trigger_mode,
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

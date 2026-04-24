"""Frame overlay rendering utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast

from visionbeat.config import OverlayConfig
from visionbeat.models import RenderState

Frame = Any


class Cv2Protocol(Protocol):
    """Subset of OpenCV drawing APIs used by overlay rendering."""

    FONT_HERSHEY_SIMPLEX: int

    def line(
        self,
        img: Frame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> Any:
        """Draw a line on a frame."""
        raise NotImplementedError

    def circle(
        self,
        img: Frame,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        thickness: int,
    ) -> Any:
        """Draw a circle on a frame."""
        raise NotImplementedError

    def putText(
        self,
        img: Frame,
        text: str,
        org: tuple[int, int],
        fontFace: int,
        fontScale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> Any:
        """Draw text on a frame."""
        raise NotImplementedError

    def rectangle(
        self,
        img: Frame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> Any:
        """Draw a rectangle on a frame."""
        raise NotImplementedError


_LANDMARK_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)
_TRIGGER_FLASH_DURATION_SECONDS = 0.2
_TRIGGER_FLASH_WIDTH_RATIO = 0.4
_TRIGGER_FLASH_HEIGHT_RATIO = 0.4
_TRIGGER_FLASH_MIN_WIDTH = 96
_TRIGGER_FLASH_MIN_HEIGHT = 72


class OverlayRenderer:
    """Draw tracking and gesture feedback onto webcam frames."""

    def __init__(self, config: OverlayConfig, *, cv2_module: Cv2Protocol | None = None) -> None:
        """Store overlay configuration and an optional OpenCV-compatible module."""
        self.config = config
        self._cv2 = cv2_module
        self._overlay_enabled = (
            config.draw_landmarks or config.show_debug_panel or config.show_trigger_flash
        )
        self._debug_enabled = config.show_debug_panel

    def set_overlay_enabled(self, enabled: bool) -> None:
        """Enable or disable all overlay rendering."""
        self._overlay_enabled = bool(enabled)

    def set_debug_enabled(self, enabled: bool) -> None:
        """Enable or disable debug-panel rendering."""
        self._debug_enabled = bool(enabled) and self.config.show_debug_panel

    def render(self, frame: Frame, state: RenderState) -> Frame:
        """Render landmarks and runtime state onto a frame copy."""
        output = frame.copy()

        if self._overlay_enabled and self.config.draw_landmarks:
            draw_pose_landmarks(
                output,
                state.pose,
                cv2_module=self._cv2,
                show_labels=self.config.show_landmark_labels,
            )

        if self._overlay_enabled and self.config.show_trigger_flash:
            draw_trigger_flash(output, state, cv2_module=self._cv2)

        if self._overlay_enabled and self.config.show_debug_panel and self._debug_enabled:
            draw_labels(output, _build_debug_labels(state), cv2_module=self._cv2)

        return output


def _build_debug_labels(state: RenderState) -> list[str]:
    """Build human-readable overlay lines from render state."""
    performance_labels: list[str] = []
    if state.capture_fps is not None:
        performance_labels.append(f"Capture {state.capture_fps:.1f}")
    if state.inference_fps is not None:
        performance_labels.append(f"Infer {state.inference_fps:.1f}")
    if state.render_fps is not None:
        performance_labels.append(f"Render {state.render_fps:.1f}")
    if not performance_labels and state.fps is not None:
        performance_labels.append(f"Loop {state.fps:.1f}")

    labels = [
        "VisionBeat • Research Demo HUD",
        f"Status: {state.pose.status}",
        f"Frame: {state.frame_index}",
        f"Rates: {' | '.join(performance_labels)}" if performance_labels else "Rates: --",
        (
            f"Pipeline latency: {state.pipeline_latency_ms:.1f} ms"
            if state.pipeline_latency_ms is not None
            else "Pipeline latency: --"
        ),
        (
            f"Detected motion: {state.current_candidate.label} "
            f"[{state.current_candidate.confidence:.2f}]"
        )
        if state.current_candidate is not None
        else "Detected motion: none",
        (
            f"Trigger: {state.confirmed_gesture.label} "
            f"[{state.confirmed_gesture.confidence:.2f}]"
        )
        if state.confirmed_gesture is not None
        else "Trigger: none",
        (
            f"ARMED COOLDOWN: {state.cooldown_remaining_seconds:.2f}s remaining"
            if state.cooldown_remaining_seconds > 0.0
            else "ARMED COOLDOWN: READY"
        ),
        (
            f"Detector: {state.detector_status}"
            if state.detector_status is not None
            else "Detector: --"
        ),
        (
            f"Predictive: {state.predictive_status}"
            if state.predictive_status is not None
            else "Predictive: --"
        ),
        (
            f"Rhythm: {state.rhythm_status}"
            if state.rhythm_status is not None
            else "Rhythm: --"
        ),
        (
            f"Audio: {state.audio_status}"
            if state.audio_status is not None
            else "Audio: --"
        ),
        "Shortcuts: overlay toggle, debug toggle, q/esc quit",
    ]
    return labels


def _get_cv2_module(cv2_module: Cv2Protocol | None) -> Cv2Protocol:
    """Return the configured cv2 module or import the real OpenCV module."""
    if cv2_module is not None:
        return cv2_module

    import cv2

    return cast(Cv2Protocol, cv2)


def draw_pose_landmarks(
    frame: Frame,
    pose: Any,
    *,
    cv2_module: Cv2Protocol | None = None,
    show_labels: bool = True,
) -> Frame:
    """Draw tracked upper-body landmarks and simple bone connections."""
    cv2 = _get_cv2_module(cv2_module)

    height, width = frame.shape[:2]
    for start_name, end_name in _LANDMARK_CONNECTIONS:
        start = pose.get(start_name)
        end = pose.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(
            frame,
            (int(start.x * width), int(start.y * height)),
            (int(end.x * width), int(end.y * height)),
            (80, 180, 255),
            2,
        )

    for name, landmark in pose.landmarks.items():
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 6, (50, 220, 120), -1)
        if not show_labels:
            continue
        cv2.putText(
            frame,
            name.replace("_", " "),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    return frame


def draw_labels(
    frame: Frame,
    labels: Iterable[str],
    *,
    cv2_module: Cv2Protocol | None = None,
    origin: tuple[int, int] = (12, 12),
) -> Frame:
    """Draw a compact label panel for tracker status and recent events."""
    cv2 = _get_cv2_module(cv2_module)

    label_list = [label for label in labels if label.strip()]
    if not label_list:
        return frame

    x0, y0 = origin
    line_height = 28
    panel_height = 20 + line_height * len(label_list)
    panel_width = 440
    cv2.rectangle(frame, (x0, y0), (x0 + panel_width, y0 + panel_height), (20, 20, 20), -1)
    for index, label in enumerate(label_list, start=1):
        cv2.putText(
            frame,
            label,
            (x0 + 12, y0 + 12 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (100, 220, 255) if index > 1 else (255, 255, 255),
            2,
        )
    return frame


def _trigger_flash_text(state: RenderState) -> str:
    """Return the visible sound label for the centered trigger flash."""
    assert state.confirmed_gesture is not None
    event = state.confirmed_gesture
    label = event.label.strip()
    normalized_label = label.lower()
    if (
        "rhythm predictor" in normalized_label
        or normalized_label.startswith("rhythm predicted")
        or normalized_label.startswith("rhythm-arm")
    ):
        return f"{event.gesture.value.title()} (rhythm predictor)"
    if (
        "cnn predictor" in normalized_label
        or normalized_label.endswith("(cnn)")
        or normalized_label.startswith("predictive ")
        or normalized_label.startswith("predictive-arm")
    ):
        return f"{event.gesture.value.title()} (CNN)"
    return event.gesture.value.upper()


def _trigger_flash_font_scale(text: str, flash_width: int) -> float:
    """Keep longer predictor labels inside the trigger flash."""
    if "(" not in text:
        return 1.4
    return max(0.45, min(1.0, flash_width / max(18.0 * len(text), 1.0)))


def draw_trigger_flash(
    frame: Frame,
    state: RenderState,
    *,
    cv2_module: Cv2Protocol | None = None,
) -> Frame:
    """Flash a centered trigger marker for recently confirmed gestures."""
    if state.confirmed_gesture is None:
        return frame

    flash_age = state.pose.timestamp.seconds - state.confirmed_gesture.timestamp.seconds
    if flash_age < 0.0 or flash_age > _TRIGGER_FLASH_DURATION_SECONDS:
        return frame

    cv2 = _get_cv2_module(cv2_module)
    height, width = frame.shape[:2]
    flash_width = min(width, max(_TRIGGER_FLASH_MIN_WIDTH, int(width * _TRIGGER_FLASH_WIDTH_RATIO)))
    flash_height = min(
        height,
        max(_TRIGGER_FLASH_MIN_HEIGHT, int(height * _TRIGGER_FLASH_HEIGHT_RATIO)),
    )
    left = max(0, (width - flash_width) // 2)
    top = max(0, (height - flash_height) // 2)
    right = min(width, left + flash_width)
    bottom = min(height, top + flash_height)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), -1)
    sound_name = _trigger_flash_text(state)
    font_scale = _trigger_flash_font_scale(sound_name, flash_width)
    thickness = 4 if font_scale >= 1.0 else 2
    text_origin = (
        left + max(10, flash_width // 10),
        top + max(28, int(flash_height * 0.62)),
    )
    cv2.putText(
        frame,
        sound_name,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    return frame

"""Top-level application orchestration for VisionBeat."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from visionbeat.audio import AudioEngine
from visionbeat.camera import CameraSource
from visionbeat.config import AppConfig
from visionbeat.gestures import GestureDetector
from visionbeat.models import AudioTrigger, FrameTimestamp
from visionbeat.overlay import OverlayRenderer
from visionbeat.tracking import PoseTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisionBeatApp:
    """Compose camera, tracking, gesture, audio, and overlay subsystems."""

    config: AppConfig
    camera: CameraSource = field(init=False)
    tracker: PoseTracker = field(init=False)
    detector: GestureDetector = field(init=False)
    audio: AudioEngine = field(init=False)
    overlay: OverlayRenderer = field(init=False)

    def __post_init__(self) -> None:
        """Initialize runtime dependencies."""
        self.camera = CameraSource(self.config.camera)
        self.tracker = PoseTracker(self.config.tracker)
        self.detector = GestureDetector(self.config.gestures)
        self.audio = AudioEngine(self.config.audio)
        self.overlay = OverlayRenderer(self.config.overlay)

    def run(self) -> None:
        """Run the real-time webcam processing loop until the user exits."""
        import cv2

        logger.info("Starting VisionBeat frame loop")
        self.camera.open()
        try:
            while True:
                camera_frame = self.camera.read_frame()
                timestamp = FrameTimestamp(seconds=camera_frame.captured_at)
                pose = self.tracker.process(camera_frame.image, timestamp)
                logger.debug(
                    "Frame loop index=%s tracking_status=%s detected=%s",
                    camera_frame.frame_index,
                    pose.status,
                    pose.person_detected,
                )
                events = self.detector.update(pose)
                for event in events:
                    logger.info("Detected %s with confidence %.2f", event.gesture, event.confidence)
                    self.audio.trigger(
                        AudioTrigger(
                            gesture=event.gesture,
                            timestamp=event.timestamp,
                            intensity=event.confidence,
                        )
                    )
                output = self.overlay.render(camera_frame.image, pose, events)
                cv2.imshow(self.config.camera.window_name, output)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    logger.info("Stopping VisionBeat frame loop on user request")
                    break
        finally:
            self.close()

    def close(self) -> None:
        """Release external resources used by the app."""
        import cv2

        logger.info("Closing VisionBeat resources")
        self.camera.close()
        self.tracker.close()
        self.audio.close()
        cv2.destroyAllWindows()

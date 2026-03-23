"""Top-level application orchestration for VisionBeat."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from visionbeat.audio import AudioEngine
from visionbeat.camera import CameraStream
from visionbeat.config import AppConfig
from visionbeat.gestures import GestureDetector
from visionbeat.overlay import OverlayRenderer
from visionbeat.tracking import PoseTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisionBeatApp:
    """Compose camera, tracking, gesture, audio, and overlay subsystems."""

    config: AppConfig
    camera: CameraStream = field(init=False)
    tracker: PoseTracker = field(init=False)
    detector: GestureDetector = field(init=False)
    audio: AudioEngine = field(init=False)
    overlay: OverlayRenderer = field(init=False)

    def __post_init__(self) -> None:
        """Initialize runtime dependencies."""
        self.camera = CameraStream(self.config.camera)
        self.tracker = PoseTracker(self.config.tracker)
        self.detector = GestureDetector(self.config.gestures)
        self.audio = AudioEngine(self.config.audio)
        self.overlay = OverlayRenderer(self.config.overlay)

    def run(self) -> None:
        """Run the real-time webcam processing loop until the user exits."""
        import cv2

        self.camera.open()
        try:
            while True:
                frame = self.camera.read()
                timestamp = time.monotonic()
                pose = self.tracker.process(frame, timestamp)
                events = self.detector.update(pose)
                for event in events:
                    logger.info("Detected %s with confidence %.2f", event.gesture, event.confidence)
                    self.audio.trigger(event.gesture)
                output = self.overlay.render(frame, pose, events)
                cv2.imshow(self.config.camera.window_name, output)
                if cv2.waitKey(1) & 0xFF in {27, ord('q')}:
                    break
        finally:
            self.close()

    def close(self) -> None:
        """Release external resources used by the app."""
        import cv2

        self.camera.close()
        self.tracker.close()
        self.audio.close()
        cv2.destroyAllWindows()

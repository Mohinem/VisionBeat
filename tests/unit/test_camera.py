import pytest

from visionbeat.camera import CameraSource
from visionbeat.config import CameraConfig


class FakeCapture:
    def __init__(self, opened: bool = True, frames: list[object] | None = None) -> None:
        self._opened = opened
        self.frames = list(frames or [])
        self.properties: list[tuple[int, int]] = []
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: int) -> None:
        self.properties.append((prop, value))

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 6

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.flip_calls: list[tuple[object, int]] = []

    def VideoCapture(self, device_index: int) -> FakeCapture:  # noqa: N802
        assert device_index == 0
        return self.capture

    def flip(self, frame: object, flip_code: int) -> object:
        self.flip_calls.append((frame, flip_code))
        return ("flipped", frame, flip_code)


def test_camera_source_open_and_read_frame() -> None:
    capture = FakeCapture(frames=["frame-a"])
    camera = CameraSource(CameraConfig(), _cv2=FakeCV2(capture))

    camera.open()
    frame = camera.read_frame()

    assert frame.image == "frame-a"
    assert frame.display_image == ("flipped", "frame-a", 1)
    assert frame.mirrored_for_display is True
    assert frame.frame_index == 0
    assert capture.properties == [(3, 1280), (4, 720), (5, 30), (6, 1)]


def test_camera_source_raises_when_device_cannot_open() -> None:
    capture = FakeCapture(opened=False)
    camera = CameraSource(CameraConfig(), _cv2=FakeCV2(capture))

    with pytest.raises(RuntimeError, match="Unable to open webcam"):
        camera.open()

    assert capture.released is True


def test_camera_source_raises_when_read_fails() -> None:
    capture = FakeCapture(frames=[])
    camera = CameraSource(CameraConfig(), _cv2=FakeCV2(capture))
    camera.open()

    with pytest.raises(RuntimeError, match="Failed to read frame"):
        camera.read_frame()


def test_camera_source_close_releases_capture() -> None:
    capture = FakeCapture(frames=["frame-a"])
    camera = CameraSource(CameraConfig(), _cv2=FakeCV2(capture))
    camera.open()

    camera.close()

    assert capture.released is True

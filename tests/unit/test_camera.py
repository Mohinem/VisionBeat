import pytest

from visionbeat.camera import CameraSource
from visionbeat.config import CameraConfig


class FakeCapture:
    def __init__(self, opened: bool = True, frames: list[object] | None = None) -> None:
        self._opened = opened
        self.frames = list(frames or [])
        self.properties: list[tuple[int, int]] = []
        self.get_properties: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: int) -> None:
        self.properties.append((prop, value))
        self.get_properties[prop] = float(value)

    def get(self, prop: int) -> float:
        return self.get_properties.get(prop, 0.0)

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
    CAP_PROP_FOURCC = 7
    CAP_PROP_BACKEND = 8
    CAP_V4L2 = 200

    def __init__(self, capture: FakeCapture | None = None, *, capture_by_source: dict[object, FakeCapture] | None = None) -> None:
        self.capture = capture
        self.capture_by_source = dict(capture_by_source or {})
        self.flip_calls: list[tuple[object, int]] = []
        self.capture_calls: list[tuple[object, int | None]] = []

    def VideoCapture(self, source: object, api_preference: int | None = None) -> FakeCapture:  # noqa: N802
        self.capture_calls.append((source, api_preference))
        capture = self.capture_by_source.get(source, self.capture)
        assert capture is not None
        if api_preference is not None:
            capture.get_properties[self.CAP_PROP_BACKEND] = float(api_preference)
        return capture

    def flip(self, frame: object, flip_code: int) -> object:
        self.flip_calls.append((frame, flip_code))
        return ("flipped", frame, flip_code)

    def VideoWriter_fourcc(self, *chars: str) -> int:  # noqa: N802
        assert len(chars) == 4
        return sum(ord(char) << (8 * index) for index, char in enumerate(chars))


def test_camera_source_open_and_read_frame() -> None:
    capture = FakeCapture(frames=["frame-a"])
    camera = CameraSource(CameraConfig(backend="v4l2", fourcc="MJPG"), _cv2=FakeCV2(capture))

    camera.open()
    frame = camera.read_frame()

    assert frame.image == "frame-a"
    assert frame.display_image == ("flipped", "frame-a", 1)
    assert frame.mirrored_for_display is True
    assert frame.frame_index == 0
    assert capture.properties == [(7, 1196444237), (3, 1280), (4, 720), (5, 30), (6, 1)]
    assert camera.capture_mode() is not None
    assert camera.capture_mode().backend == "v4l2"
    assert camera.capture_mode().fourcc == "MJPG"


def test_camera_source_uses_default_backend_when_configured_auto() -> None:
    capture = FakeCapture(frames=["frame-a"])
    cv2_module = FakeCV2(capture)
    camera = CameraSource(CameraConfig(), _cv2=cv2_module)

    camera.open()

    assert cv2_module.capture_calls == [(0, None)]


def test_camera_source_falls_back_to_v4l2_device_path_when_backend_index_open_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_capture = FakeCapture(opened=False)
    succeeded_capture = FakeCapture(frames=["frame-a"])
    cv2_module = FakeCV2(
        capture_by_source={
            0: failed_capture,
            "/dev/video0": succeeded_capture,
        }
    )
    camera = CameraSource(CameraConfig(backend="v4l2", fourcc="MJPG"), _cv2=cv2_module)

    monkeypatch.setattr("visionbeat.camera.Path.exists", lambda self: self.as_posix() == "/dev/video0")

    camera.open()

    assert failed_capture.released is True
    assert cv2_module.capture_calls == [(0, 200), ("/dev/video0", 200)]
    assert camera.capture_mode() is not None
    assert camera.capture_mode().backend == "v4l2"


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

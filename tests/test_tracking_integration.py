import pytest

from visionbeat.camera import CameraSource
from visionbeat.config import CameraConfig


@pytest.mark.webcam
def test_default_webcam_can_capture_frame() -> None:
    pytest.importorskip("cv2", exc_type=ImportError)

    camera = CameraSource(CameraConfig())
    try:
        camera.open()
        frame = camera.read_frame()
    except RuntimeError as exc:
        pytest.skip(f"Webcam integration test skipped: {exc}")
    else:
        assert frame.image is not None
        assert frame.frame_index == 0
    finally:
        camera.close()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionbeat.models import FrameTimestamp, LandmarkPoint, TrackerOutput
from visionbeat.render_pose_video import render_pose_video


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, fps: float) -> None:
        self.frames = list(frames)
        self.fps = fps
        self.released = False
        self._last_frame_index = -1

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        self._last_frame_index += 1
        return True, self.frames.pop(0)

    def get(self, prop: int) -> float:  # noqa: N802
        if prop == FakeCV2.CAP_PROP_FPS:
            return self.fps
        if prop == FakeCV2.CAP_PROP_POS_MSEC:
            if self._last_frame_index < 0:
                return 0.0
            return ((self._last_frame_index + 1) / self.fps) * 1000.0
        return 0.0

    def release(self) -> None:
        self.released = True


class FakePreview:
    def __init__(self) -> None:
        self.show_calls: list[tuple[str, object]] = []
        self.close_calls = 0

    def show(self, window_name: str, frame: object) -> None:
        self.show_calls.append((window_name, frame))

    def poll_key(self) -> int | None:
        return None

    def should_close(self, key_code: int | None = None) -> bool:
        return False

    def close(self) -> None:
        self.close_calls += 1


class FakeVideoWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 6
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.writer = FakeVideoWriter()

    def VideoCapture(self, _: str) -> FakeCapture:  # noqa: N802
        return self.capture

    def VideoWriter_fourcc(self, *_: str) -> int:  # noqa: N802
        return 4321

    def VideoWriter(
        self,
        path: str,
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> FakeVideoWriter:  # noqa: N802
        assert path.endswith(".mp4")
        assert fourcc == 4321
        assert fps == 20.0
        assert frame_size == (4, 3)
        return self.writer

    def line(self, *args, **kwargs) -> None:
        return None

    def circle(self, *args, **kwargs) -> None:
        return None

    def putText(self, *args, **kwargs) -> None:  # noqa: N802
        return None

    def rectangle(self, *args, **kwargs) -> None:
        return None

    def destroyAllWindows(self) -> None:
        return None


class FakePoseProvider:
    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self.closed = False

    def process(self, frame: object, timestamp: FrameTimestamp | float) -> TrackerOutput:
        seconds = timestamp.seconds if isinstance(timestamp, FrameTimestamp) else float(timestamp)
        self.timestamps.append(seconds)
        return TrackerOutput(
            timestamp=FrameTimestamp(seconds=seconds),
            landmarks={
                "left_shoulder": LandmarkPoint(x=0.25, y=0.5, z=0.0, visibility=0.9),
                "right_shoulder": LandmarkPoint(x=0.75, y=0.5, z=0.0, visibility=0.9),
                "left_wrist": LandmarkPoint(x=0.3, y=0.7, z=0.0, visibility=0.95),
            },
            person_detected=True,
            status="tracking",
        )

    def close(self) -> None:
        self.closed = True


def test_render_pose_video_writes_overlay_video_and_preview(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    capture = FakeCapture(
        [np.zeros((3, 4, 3), dtype=np.uint8) for _ in range(2)],
        fps=20.0,
    )
    cv2_module = FakeCV2(capture)
    preview = FakePreview()
    provider = FakePoseProvider()

    output_path = render_pose_video(
        video_path,
        output_path=tmp_path / "clip.pose.mp4",
        pose_provider_factory=lambda _: provider,
        cv2_module=cv2_module,
        preview_window=preview,
        show_preview=True,
    )

    metadata = json.loads((tmp_path / "clip.pose.mp4.metadata.json").read_text(encoding="utf-8"))

    assert output_path == tmp_path / "clip.pose.mp4"
    assert len(cv2_module.writer.frames) == 2
    assert cv2_module.writer.released is True
    assert preview.close_calls == 1
    assert len(preview.show_calls) == 2
    assert provider.closed is True
    assert provider.timestamps == [0.05, 0.1]
    assert metadata["frames_processed"] == 2
    assert metadata["output_fps"] == 20.0

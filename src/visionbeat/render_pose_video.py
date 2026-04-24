"""Render a saved video with pose landmarks for visual inspection."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from visionbeat.app import OpenCVPreviewWindow, PreviewWindow
from visionbeat.config import ConfigError, OverlayConfig, TrackerConfig, load_config
from visionbeat.features import get_canonical_feature_schema
from visionbeat.logging_config import configure_logging
from visionbeat.models import FrameTimestamp, RenderState
from visionbeat.overlay import OverlayRenderer
from visionbeat.pose_provider import PoseBackendError, create_pose_provider


class _NullPreviewWindow:
    """No-op preview window used for headless rendering."""

    def show(self, window_name: str, frame: object) -> None:
        return None

    def poll_key(self) -> int | None:
        return None

    def should_close(self, key_code: int | None = None) -> bool:
        return False

    def close(self) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for pose-video rendering."""

    parser = argparse.ArgumentParser(
        description="Render a saved video with VisionBeat pose landmarks."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Input raw video path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output overlay-video path. Defaults to '<video-stem>.pose.mp4'.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a VisionBeat YAML or TOML configuration file.",
    )
    parser.add_argument(
        "--pose-backend",
        default=None,
        choices=("mediapipe", "movenet"),
        help="Override the configured pose backend.",
    )
    parser.add_argument(
        "--show-landmark-labels",
        action="store_true",
        help="Render landmark names next to each point.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window while rendering the overlay video.",
    )
    return parser.parse_args(argv)


def render_pose_video(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    tracker_config: TrackerConfig | None = None,
    pose_provider_factory: Any = create_pose_provider,
    cv2_module: Any | None = None,
    preview_window: PreviewWindow | None = None,
    show_preview: bool = True,
    show_landmark_labels: bool = False,
) -> Path:
    """Render a saved video with pose landmarks and optionally preview it live."""

    source_video = Path(video_path)
    if not source_video.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_video}")

    destination = (
        Path(output_path)
        if output_path is not None
        else source_video.with_name(f"{source_video.stem}.pose.mp4")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_name(f"{destination.name}.metadata.json")

    if cv2_module is None:
        import cv2

        cv2_module = cv2

    capture = cv2_module.VideoCapture(source_video.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {source_video}")

    tracker = pose_provider_factory(tracker_config or TrackerConfig())
    if show_preview:
        preview = preview_window or OpenCVPreviewWindow(cv2_module=cv2_module)
    else:
        preview = _NullPreviewWindow()
    overlay = OverlayRenderer(
        OverlayConfig(
            draw_landmarks=True,
            draw_velocity_vectors=False,
            show_landmark_labels=show_landmark_labels,
            show_debug_panel=False,
            show_trigger_flash=False,
        ),
        cv2_module=cv2_module,
    )
    overlay.set_overlay_enabled(True)
    overlay.set_debug_enabled(False)

    writer = None
    frames_processed = 0
    output_fps = _resolve_video_fps(capture=capture, cv2_module=cv2_module)
    preview_window_name = f"{source_video.stem} Pose Preview"

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp_seconds = _resolve_frame_timestamp_seconds(
                capture,
                frame_index=frames_processed,
                cv2_module=cv2_module,
            )
            pose = tracker.process(frame, FrameTimestamp(seconds=timestamp_seconds))
            rendered_frame = overlay.render(
                frame,
                RenderState(
                    pose=pose,
                    frame_index=frames_processed,
                ),
            )
            if writer is None:
                writer = _open_video_writer(
                    destination,
                    fps=output_fps,
                    frame_size=(int(frame.shape[1]), int(frame.shape[0])),
                    cv2_module=cv2_module,
                )
            writer.write(rendered_frame)
            frames_processed += 1

            if show_preview:
                preview.show(preview_window_name, rendered_frame)
                key_code = preview.poll_key()
                if preview.should_close(key_code):
                    break
    finally:
        capture.release()
        tracker.close()
        preview.close()
        if writer is not None:
            writer.release()

    metadata = {
        "schema": "visionbeat.pose_render.v1",
        "input_video": source_video.as_posix(),
        "output_video": destination.as_posix(),
        "frames_processed": frames_processed,
        "output_fps": output_fps,
        "pose_backend": (tracker_config or TrackerConfig()).backend,
        "show_landmark_labels": show_landmark_labels,
        "feature_schema": get_canonical_feature_schema().to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _resolve_video_fps(*, capture: Any, cv2_module: Any) -> float:
    fps_property = getattr(cv2_module, "CAP_PROP_FPS", None)
    fps = float(capture.get(fps_property)) if fps_property is not None else 0.0
    if fps > 0.0:
        return fps
    return 30.0


def _resolve_frame_timestamp_seconds(
    capture: Any,
    *,
    frame_index: int,
    cv2_module: Any,
) -> float:
    pos_msec_property = getattr(cv2_module, "CAP_PROP_POS_MSEC", None)
    timestamp_msec = (
        float(capture.get(pos_msec_property))
        if pos_msec_property is not None
        else 0.0
    )
    if timestamp_msec > 0.0:
        return timestamp_msec / 1000.0
    fps = _resolve_video_fps(capture=capture, cv2_module=cv2_module)
    return frame_index / fps


def _open_video_writer(
    output_path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
    cv2_module: Any,
) -> Any:
    fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")
    writer = cv2_module.VideoWriter(
        output_path.as_posix(),
        fourcc,
        float(fps),
        frame_size,
    )
    is_opened = getattr(writer, "isOpened", None)
    if callable(is_opened) and not bool(is_opened()):
        writer.release()
        raise RuntimeError(f"Unable to open rendered-video writer: {output_path}")
    return writer


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for pose-video rendering."""

    args = parse_args(argv)
    try:
        app_config = load_config(Path(args.config))
        tracker_config = app_config.tracker
        if args.pose_backend is not None:
            tracker_config = replace(tracker_config, backend=args.pose_backend.lower())
        configure_logging(
            app_config.logging.level,
            log_format=app_config.logging.format,
            structured=app_config.logging.structured,
        )
        output_path = render_pose_video(
            args.video,
            output_path=args.out,
            tracker_config=tracker_config,
            show_preview=not args.no_preview,
            show_landmark_labels=args.show_landmark_labels,
        )
    except (ConfigError, PoseBackendError, RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Rendered pose video to {output_path}")


if __name__ == "__main__":
    main()

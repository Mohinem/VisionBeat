# VisionBeat

VisionBeat is a production-oriented Python project for a **webcam-based gestural percussion instrument**. It uses OpenCV for live camera input, MediaPipe for upper-body landmark tracking, and `pygame.mixer` for low-latency sample triggering. The initial gesture set maps:

- **Forward punch → kick**
- **Downward strike → snare**

The codebase is intentionally modular so that the gesture recognition layer can be tested in isolation from webcam hardware or MediaPipe runtime dependencies.

## Features

- Real-time webcam frame capture and display with OpenCV
- MediaPipe pose-based wrist and upper-body tracking
- Gesture detection isolated in a pure Python module
- Low-latency kick/snare sample playback with pygame
- Structured configuration using dataclasses and TOML files
- Logging with Python's built-in `logging`
- Unit tests with pytest
- CI for linting, type checking, and tests
- Architecture and contributor documentation

## Project layout

```text
visionbeat/
├── assets/samples/           # Text docs + generated local drum samples
├── configs/                  # TOML configuration files
├── docs/                     # Architecture and project documentation
├── scripts/                  # Utility scripts
├── src/visionbeat/           # Application package
├── tests/                    # Pytest suite
└── .github/workflows/        # CI definitions
```

## Requirements

- Python 3.11+
- A webcam
- Platform support for OpenCV, MediaPipe, and pygame

## Quick start

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### 3. Generate local placeholder drum samples

Generate the local WAV files before the first run (they are intentionally not committed):

```bash
python scripts/generate_demo_samples.py
```

### 4. Run VisionBeat

```bash
visionbeat --config configs/default.toml
```

Press `q` or `Esc` to exit.

## Configuration

VisionBeat loads settings from a TOML file. The included sample config lives at `configs/default.toml` and controls:

- camera device, resolution, and mirroring
- MediaPipe confidence thresholds
- gesture thresholds and cooldown windows
- sample file paths and audio buffer size
- overlay visibility and debug options

Example:

```toml
[gestures]
punch_forward_delta_z = 0.18
strike_down_delta_y = 0.22
cooldown_seconds = 0.18
active_hand = "right"
```

## Gesture architecture

The most important design decision is the separation between tracking and gesture recognition:

- `PoseTracker` converts MediaPipe output into a `PoseFrame` dataclass.
- `GestureDetector` accepts `PoseFrame` objects and emits typed `GestureEvent` results.
- Tests build `PoseFrame` values directly, so the core gesture logic remains deterministic and unit-testable.

This means you can improve or replace the tracker without rewriting gesture tests.

## Developer workflow

### Run quality checks locally

```bash
ruff check .
mypy
pytest
```

### Contributor setup instructions

1. Fork or clone the repository.
2. Create and activate a Python 3.11+ virtual environment.
3. Install editable dependencies with `python -m pip install -e .[dev]`.
4. Generate local placeholder samples with `python scripts/generate_demo_samples.py`.
5. Run `ruff check .`, `mypy`, and `pytest` before opening a pull request.

## Runtime notes

- MediaPipe Pose is used as the initial tracking backend because it can estimate shoulder, elbow, and wrist landmarks from a single webcam feed.
- Gesture thresholds are intentionally exposed in config so they can be tuned per lighting condition, camera angle, and performer style.
- `pygame.mixer` provides a practical baseline for sample playback. If your deployment needs lower or more deterministic audio latency, the architecture is ready for a `sounddevice` backend.

## Documentation

- Architecture overview: [`docs/architecture.md`](docs/architecture.md)
- Default configuration: [`configs/default.toml`](configs/default.toml)
- Sample generation utility: [`scripts/generate_demo_samples.py`](scripts/generate_demo_samples.py)
- Sample asset policy: [`assets/samples/README.md`](assets/samples/README.md)

## Roadmap ideas

- Add calibration routines for performer-specific gesture baselines
- Support left- and right-hand drum mappings simultaneously
- Add on-screen latency and confidence diagnostics
- Introduce hand-tracking mode as an alternative to pose tracking
- Add MIDI output for DAW integration

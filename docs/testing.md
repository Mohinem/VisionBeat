# Testing

## Philosophy

VisionBeat’s tests are designed around a key architectural promise: **the musically important logic should be testable without requiring a physical webcam or active audio hardware**.

That means most tests validate:

- data-model invariants,
- configuration validation,
- gesture heuristics,
- orchestration behavior,
- overlay rendering logic,
- and audio/tracking integration through mocks or scaffolding.

This strategy supports both software quality and research reproducibility.

## Test suite structure

The repository contains two main categories of tests:

### Unit tests

Located under `tests/unit/`.

They cover:

- `test_models.py` — domain model validation and serialization,
- `test_config.py` — configuration parsing and validation errors,
- `test_gestures.py` — synthetic motion sequences, thresholds, cooldown, and candidate logic,
- `test_audio.py` — asset loading, backend behavior, and trigger dispatch,
- `test_tracking.py` — structured tracking output with mocked MediaPipe paths,
- `test_camera.py` — camera lifecycle and failure paths,
- `test_overlay.py` — overlay drawing behavior,
- `test_app.py` — runtime orchestration and integration across components,
- `test_cli.py` — command-line parsing,
- `test_observability.py` — event logging and recorder behavior.

### Integration tests

Located under `tests/integration/`.

They cover:

- audio engine creation and config preservation,
- tracking scaffolding across MediaPipe/OpenCV pathways,
- and one optional webcam-marked test for environments with actual camera hardware.

## Recommended commands

### Run the full suite

```bash
pytest
```

### Run only unit tests

```bash
pytest tests/unit
```

### Run only integration tests

```bash
pytest tests/integration
```

### Run a focused gesture test pass

```bash
pytest tests/unit/test_gestures.py
```

### Run linting and type checking

```bash
ruff check .
mypy
```

## Test environment setup

Use a Python 3.11+ virtual environment and install development dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If you want local audio assets available during manual runs, also generate demo samples:

```bash
python scripts/generate_demo_samples.py
```

## Why gesture tests matter most

The detector is the musical core of VisionBeat. Its tests use synthetic timestamped trajectories to validate that:

- stationary motion does not trigger,
- the expected kick/snare trajectories do trigger,
- threshold boundaries remain stable,
- cooldown suppresses duplicate hits,
- low visibility clears pending candidates,
- inactive hands are ignored,
- and metric computation behaves correctly.

These tests are especially valuable because they validate the instrument’s semantic behavior without relying on a noisy real-world webcam loop.

## Hardware-dependent testing

Most of the suite is hardware independent, but one integration test is marked with `@pytest.mark.webcam` and expects a real default webcam.

If your environment does not have a webcam, skip hardware-only tests or run the regular suite without requiring that marker explicitly.

## Interpreting failures

### Config tests fail

A config test failure usually indicates that:

- a validation rule changed,
- a default field moved,
- or an error message format regressed.

### Gesture tests fail

A gesture failure often means one of the following:

- threshold semantics changed,
- candidate/confirmation timing changed,
- velocity computations changed,
- or cooldown handling regressed.

This category deserves special attention because it changes the instrument’s playability.

### Tracking or audio tests fail

These often indicate:

- interface drift,
- initialization behavior changes,
- or assumptions that need to stay stable for local development environments.

## Suggested validation workflow before merging doc-adjacent code changes

Even if you are only changing documentation, it is useful to confirm the repository still passes the normal project checks:

1. `ruff check .`
2. `mypy`
3. `pytest`

If you change gesture logic, prefer also running:

- `pytest tests/unit/test_gestures.py`
- `pytest tests/unit/test_app.py`
- `pytest tests/unit/test_observability.py`

## Testing limitations

- The automated tests do not measure true motion-to-sound latency.
- Real tracking robustness still depends on lighting, camera angle, and performer behavior.
- Mocked audio tests cannot guarantee device compatibility on every host.
- A passing suite does not replace live rehearsal before a demo.

## Future testing extensions

As VisionBeat grows, useful additions would include:

- recorded trajectory fixtures from real performances,
- benchmark-style latency profiling,
- regression datasets for clap detection,
- backend-specific tests for MIDI output,
- and integration tests around a future JUCE bridge.

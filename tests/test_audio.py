from pathlib import Path

import pytest

from visionbeat.audio import AudioEngine
from visionbeat.config import AudioConfig


def test_audio_engine_raises_helpful_error_for_missing_samples(tmp_path: Path) -> None:
    engine = AudioEngine.__new__(AudioEngine)
    engine.config = AudioConfig(
        kick_sample=(tmp_path / "missing-kick.wav").as_posix(),
        snare_sample=(tmp_path / "missing-snare.wav").as_posix(),
    )

    with pytest.raises(FileNotFoundError, match="generate_demo_samples.py"):
        engine._validate_sample_paths()

from __future__ import annotations

from pathlib import Path

import pytest

from visionbeat.audio import AudioEngine, PygameAudioEngine, SampleAsset, create_audio_engine
from visionbeat.config import AudioConfig
from visionbeat.models import AudioTrigger, FrameTimestamp, GestureType


class FakeSound:
    def __init__(self, path: str) -> None:
        self.path = path
        self.volumes: list[float] = []
        self.play_calls = 0

    def set_volume(self, value: float) -> None:
        self.volumes.append(value)

    def play(self) -> None:
        self.play_calls += 1


class FakeMixer:
    def __init__(self, *, fail_init: bool = False) -> None:
        self.fail_init = fail_init
        self.init_calls: list[dict[str, object]] = []
        self.sound_paths: list[str] = []
        self.sounds: dict[str, FakeSound] = {}
        self.num_channels: int | None = None
        self.quit_calls = 0

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("mixer unavailable")

    def set_num_channels(self, value: int) -> None:
        self.num_channels = value

    def Sound(self, path: str) -> FakeSound:
        self.sound_paths.append(path)
        sound = FakeSound(path)
        self.sounds[path] = sound
        return sound

    def quit(self) -> None:
        self.quit_calls += 1


class FakePygame:
    def __init__(self, *, fail_init: bool = False) -> None:
        self.mixer = FakeMixer(fail_init=fail_init)


def create_sample(path: Path) -> None:
    path.write_bytes(b"RIFFdemoWAVE")


def build_engine(
    tmp_path: Path,
    *,
    volume: float = 0.8,
    fail_init: bool = False,
) -> tuple[PygameAudioEngine, FakePygame, Path, Path]:
    kick = tmp_path / "kick.wav"
    snare = tmp_path / "snare.wav"
    create_sample(kick)
    create_sample(snare)
    fake_pygame = FakePygame(fail_init=fail_init)
    engine = PygameAudioEngine(
        config=AudioConfig(
            sample_mapping={"kick": kick.as_posix(), "snare": snare.as_posix()},
            volume=volume,
        ),
        pygame_module=fake_pygame,
    )
    return engine, fake_pygame, kick, snare


def test_audio_engine_loads_available_assets_and_tracks_missing_samples(tmp_path: Path) -> None:
    kick = tmp_path / "kick.wav"
    create_sample(kick)
    fake_pygame = FakePygame()
    config = AudioConfig(
        sample_mapping={
            "kick": kick.as_posix(),
            "snare": (tmp_path / "missing-snare.wav").as_posix(),
        },
        output_device_name="Studio Output",
    )

    engine = PygameAudioEngine(config=config, pygame_module=fake_pygame)

    assert engine.available_sounds() == ("kick",)
    assert engine.missing_sounds() == ("snare",)
    assert fake_pygame.mixer.init_calls == [
        {
            "frequency": config.sample_rate,
            "buffer": config.buffer_size,
            "channels": config.output_channels,
            "devicename": config.output_device_name,
        }
    ]
    assert fake_pygame.mixer.num_channels == config.simultaneous_voices
    assert fake_pygame.mixer.sound_paths == [kick.as_posix()]


@pytest.mark.parametrize(
    ("trigger", "intensity", "expected_sound"),
    [
        ("kick", 0.5, "kick"),
        (GestureType.SNARE, 0.25, "snare"),
        (
            AudioTrigger(
                gesture=GestureType.KICK,
                timestamp=FrameTimestamp(seconds=1.0),
                intensity=1.0,
            ),
            None,
            "kick",
        ),
    ],
)
def test_audio_engine_dispatches_named_enum_and_structured_triggers(
    trigger: AudioTrigger | GestureType | str,
    intensity: float | None,
    expected_sound: str,
    tmp_path: Path,
) -> None:
    engine, fake_pygame, kick, snare = build_engine(tmp_path)
    path_by_name = {"kick": kick.as_posix(), "snare": snare.as_posix()}

    assert engine.trigger(trigger, intensity=intensity) is True

    sound = fake_pygame.mixer.sounds[path_by_name[expected_sound]]
    assert sound.play_calls == 1


@pytest.mark.parametrize(
    ("intensity", "expected_volume"),
    [(-1.0, 0.0), (0.5, 0.4), (2.0, 0.8)],
)
def test_audio_engine_clamps_intensity_before_dispatch(
    intensity: float,
    expected_volume: float,
    tmp_path: Path,
) -> None:
    engine, fake_pygame, kick, _ = build_engine(tmp_path)

    assert engine.trigger_sound("  KiCk  ", intensity=intensity) is True

    kick_sound = fake_pygame.mixer.sounds[kick.as_posix()]
    assert kick_sound.play_calls == 1
    assert kick_sound.volumes[-1] == pytest.approx(expected_volume)


def test_audio_engine_handles_repeated_triggers_without_crashing(tmp_path: Path) -> None:
    engine, fake_pygame, kick, _ = build_engine(tmp_path, volume=0.9)

    for _ in range(10):
        assert engine.trigger_sound("kick", intensity=0.7) is True

    kick_sound = fake_pygame.mixer.sounds[kick.as_posix()]
    assert kick_sound.play_calls == 10
    assert kick_sound.volumes[-1] == pytest.approx(0.63)


def test_audio_engine_returns_false_for_missing_named_sound(tmp_path: Path) -> None:
    fake_pygame = FakePygame()
    engine = PygameAudioEngine(
        config=AudioConfig(
            sample_mapping={
                "kick": (tmp_path / "missing-kick.wav").as_posix(),
                "snare": (tmp_path / "missing-snare.wav").as_posix(),
            },
        ),
        pygame_module=fake_pygame,
    )

    assert engine.available_sounds() == ()
    assert engine.trigger_sound("kick") is False

    engine.close()
    assert fake_pygame.mixer.quit_calls == 1


def test_audio_engine_disables_loading_when_mixer_initialization_fails(tmp_path: Path) -> None:
    engine, fake_pygame, _, _ = build_engine(tmp_path, fail_init=True)

    assert engine.available_sounds() == ()
    assert set(engine.missing_sounds()) == {"kick", "snare"}
    assert engine.status_summary().startswith("audio unavailable")
    assert fake_pygame.mixer.quit_calls == 0


def test_audio_engine_status_summary_reports_partial_sample_availability(tmp_path: Path) -> None:
    kick = tmp_path / "kick.wav"
    create_sample(kick)
    engine = PygameAudioEngine(
        config=AudioConfig(
            sample_mapping={
                "kick": kick.as_posix(),
                "snare": (tmp_path / "missing-snare.wav").as_posix(),
            },
        ),
        pygame_module=FakePygame(),
    )

    assert engine.is_ready() is True
    assert engine.status_summary() == "audio partial (snare missing)"


@pytest.mark.parametrize(
    ("sample_mapping", "expected_missing"),
    [
        ({"kick": "kick.wav"}, False),
        ({"kick": "kick.wav", "snare": "missing.wav"}, True),
    ],
)
def test_sample_asset_resolution_tracks_file_existence(
    sample_mapping: dict[str, str],
    expected_missing: bool,
    tmp_path: Path,
) -> None:
    kick = tmp_path / "kick.wav"
    create_sample(kick)
    normalized_mapping = {
        name: (kick.as_posix() if path == "kick.wav" else (tmp_path / path).as_posix())
        for name, path in sample_mapping.items()
    }
    engine = PygameAudioEngine(
        config=AudioConfig(sample_mapping=normalized_mapping),
        pygame_module=FakePygame(),
    )

    assets = engine._resolve_assets()

    assert all(isinstance(asset, SampleAsset) for asset in assets.values())
    assert any(asset.exists is False for asset in assets.values()) is expected_missing


def test_create_audio_engine_rejects_unsupported_backends() -> None:
    config = AudioConfig()
    object.__setattr__(config, "backend", "unknown")

    with pytest.raises(ValueError, match="Unsupported audio backend"):
        create_audio_engine(config)


class DummyEngine(AudioEngine):
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def trigger_sound(self, name: str, intensity: float = 1.0) -> bool:
        self.calls.append((name, intensity))
        return True

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("trigger", "intensity", "expected_name", "expected_intensity"),
    [
        (GestureType.KICK, None, "kick", 1.0),
        (GestureType.KICK, 0.2, "kick", 0.2),
        (
            AudioTrigger(
                gesture=GestureType.SNARE,
                timestamp=FrameTimestamp(seconds=2.0),
                intensity=0.4,
            ),
            None,
            "snare",
            0.4,
        ),
        ("kick", 0.7, "kick", 0.7),
    ],
)
def test_audio_engine_base_trigger_dispatch_rules(
    trigger: AudioTrigger | GestureType | str,
    intensity: float | None,
    expected_name: str,
    expected_intensity: float,
) -> None:
    engine = DummyEngine()

    assert engine.trigger(trigger, intensity=intensity) is True
    assert engine.calls == [(expected_name, expected_intensity)]

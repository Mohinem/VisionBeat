from __future__ import annotations

from pathlib import Path

from visionbeat.audio import PygameAudioEngine
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
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.sound_paths: list[str] = []
        self.sounds: dict[str, FakeSound] = {}
        self.num_channels: int | None = None
        self.quit_calls = 0

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

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
    def __init__(self) -> None:
        self.mixer = FakeMixer()


def create_sample(path: Path) -> None:
    path.write_bytes(b"RIFFdemoWAVE")


def test_audio_engine_loads_available_assets_and_tracks_missing_samples(tmp_path: Path) -> None:
    kick = tmp_path / "kick.wav"
    create_sample(kick)
    fake_pygame = FakePygame()
    config = AudioConfig(
        kick_sample=kick.as_posix(),
        snare_sample=(tmp_path / "missing-snare.wav").as_posix(),
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


def test_audio_engine_dispatches_named_and_structured_triggers(tmp_path: Path) -> None:
    kick = tmp_path / "kick.wav"
    snare = tmp_path / "snare.wav"
    create_sample(kick)
    create_sample(snare)
    fake_pygame = FakePygame()
    config = AudioConfig(
        kick_sample=kick.as_posix(),
        snare_sample=snare.as_posix(),
        volume=0.8,
    )
    engine = PygameAudioEngine(config=config, pygame_module=fake_pygame)

    assert engine.trigger_sound("kick", intensity=0.5) is True
    assert engine.trigger(GestureType.SNARE, intensity=0.25) is True
    assert (
        engine.trigger(
            AudioTrigger(
                gesture=GestureType.KICK,
                timestamp=FrameTimestamp(seconds=1.0),
                intensity=1.0,
            )
        )
        is True
    )

    kick_sound = fake_pygame.mixer.sounds[kick.as_posix()]
    snare_sound = fake_pygame.mixer.sounds[snare.as_posix()]
    assert kick_sound.play_calls == 2
    assert snare_sound.play_calls == 1
    assert kick_sound.volumes == [0.8, 0.4, 0.8]
    assert snare_sound.volumes == [0.8, 0.2]


def test_audio_engine_handles_repeated_triggers_without_crashing(tmp_path: Path) -> None:
    kick = tmp_path / "kick.wav"
    snare = tmp_path / "snare.wav"
    create_sample(kick)
    create_sample(snare)
    fake_pygame = FakePygame()
    engine = PygameAudioEngine(
        config=AudioConfig(kick_sample=kick.as_posix(), snare_sample=snare.as_posix()),
        pygame_module=fake_pygame,
    )

    for _ in range(10):
        assert engine.trigger_sound("kick", intensity=0.7) is True

    kick_sound = fake_pygame.mixer.sounds[kick.as_posix()]
    assert kick_sound.play_calls == 10
    assert kick_sound.volumes[-1] == 0.63


def test_audio_engine_returns_false_for_missing_named_sound(tmp_path: Path) -> None:
    fake_pygame = FakePygame()
    engine = PygameAudioEngine(
        config=AudioConfig(
            kick_sample=(tmp_path / "missing-kick.wav").as_posix(),
            snare_sample=(tmp_path / "missing-snare.wav").as_posix(),
        ),
        pygame_module=fake_pygame,
    )

    assert engine.available_sounds() == ()
    assert engine.trigger_sound("kick") is False

    engine.close()
    assert fake_pygame.mixer.quit_calls == 1

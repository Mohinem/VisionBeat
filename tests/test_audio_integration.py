from __future__ import annotations

from pathlib import Path

from visionbeat.audio import PygameAudioEngine, create_audio_engine
from visionbeat.config import AudioConfig, load_config


class IntegrationSound:
    def __init__(self, path: str) -> None:
        self.path = path
        self.play_count = 0
        self.last_volume = 0.0

    def set_volume(self, value: float) -> None:
        self.last_volume = value

    def play(self) -> None:
        self.play_count += 1


class IntegrationMixer:
    def __init__(self) -> None:
        self.loaded: dict[str, IntegrationSound] = {}
        self.initialized = False
        self.quit_called = False

    def init(self, **_: object) -> None:
        self.initialized = True

    def set_num_channels(self, _: int) -> None:
        return None

    def Sound(self, path: str) -> IntegrationSound:
        sound = IntegrationSound(path)
        self.loaded[path] = sound
        return sound

    def quit(self) -> None:
        self.quit_called = True


class IntegrationPygame:
    def __init__(self) -> None:
        self.mixer = IntegrationMixer()


def create_sample(path: Path) -> None:
    path.write_bytes(b"RIFFdemoWAVE")


def test_create_audio_engine_uses_pygame_backend(monkeypatch, tmp_path: Path) -> None:
    kick = tmp_path / "kick.wav"
    snare = tmp_path / "snare.wav"
    create_sample(kick)
    create_sample(snare)
    fake_pygame = IntegrationPygame()

    monkeypatch.setattr(PygameAudioEngine, "_import_pygame", lambda self: fake_pygame)

    engine = create_audio_engine(
        AudioConfig(kick_sample=kick.as_posix(), snare_sample=snare.as_posix())
    )

    assert isinstance(engine, PygameAudioEngine)
    assert engine.trigger_sound("snare", intensity=0.5) is True
    assert fake_pygame.mixer.loaded[snare.as_posix()].play_count == 1

    engine.close()
    assert fake_pygame.mixer.quit_called is True


def test_load_config_preserves_audio_output_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "audio-config.toml"
    config_path.write_text(
        """
log_level = "INFO"

[audio]
backend = "pygame"
sample_rate = 48000
buffer_size = 512
output_channels = 2
simultaneous_voices = 24
output_device_name = "USB Interface"
kick_sample = "assets/samples/kick.wav"
snare_sample = "assets/samples/snare.wav"
volume = 0.75
""".strip()
    )

    config = load_config(config_path)

    assert config.audio == AudioConfig(
        backend="pygame",
        sample_rate=48_000,
        buffer_size=512,
        output_channels=2,
        simultaneous_voices=24,
        output_device_name="USB Interface",
        kick_sample="assets/samples/kick.wav",
        snare_sample="assets/samples/snare.wav",
        volume=0.75,
    )

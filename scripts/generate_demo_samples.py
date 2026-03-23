"""Generate simple placeholder drum samples for local development."""

from __future__ import annotations

import math
import wave
from pathlib import Path

SAMPLE_RATE = 44_100
OUTPUT_DIR = Path("assets/samples")


def write_wave(path: Path, samples: list[float]) -> None:
    """Write a mono 16-bit WAV file."""
    with wave.open(path.as_posix(), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        pcm = bytearray()
        for sample in samples:
            value = max(-1.0, min(1.0, sample))
            pcm.extend(int(value * 32767).to_bytes(2, "little", signed=True))
        wav_file.writeframes(bytes(pcm))


def synthesize_kick(duration: float = 0.25) -> list[float]:
    """Create a decaying low-frequency kick drum sample."""
    sample_count = int(SAMPLE_RATE * duration)
    return [
        math.sin(2 * math.pi * (90 - 55 * i / sample_count) * (i / SAMPLE_RATE))
        * math.exp(-7 * i / sample_count)
        for i in range(sample_count)
    ]


def synthesize_snare(duration: float = 0.18) -> list[float]:
    """Create a decaying noise-based snare sample."""
    sample_count = int(SAMPLE_RATE * duration)
    return [
        math.sin(2 * math.pi * 180 * (i / SAMPLE_RATE)) * 0.35 * math.exp(-18 * i / sample_count)
        + math.sin(2 * math.pi * 330 * (i / SAMPLE_RATE)) * 0.2 * math.exp(-10 * i / sample_count)
        for i in range(sample_count)
    ]


def main() -> None:
    """Write placeholder kick and snare samples into the assets folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_wave(OUTPUT_DIR / "kick.wav", synthesize_kick())
    write_wave(OUTPUT_DIR / "snare.wav", synthesize_snare())


if __name__ == "__main__":
    main()

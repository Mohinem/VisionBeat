# Audio System

## Architecture

VisionBeat's audio subsystem is split into two layers:

1. **`AudioEngine` abstraction** in `src/visionbeat/audio.py` defines the trigger-oriented interface used by the app.
2. **`PygameAudioEngine`** is the current concrete backend. It initializes `pygame.mixer`, loads named samples from configuration, and exposes both `trigger_sound(name, intensity=...)` and `trigger(...)` helpers.

The app only depends on the abstraction through `create_audio_engine(config)`, so a future backend can be introduced without changing the frame loop.

### Sample mapping

The default mapping is intentionally small and explicit:

- `kick` → `assets/samples/kick.wav`
- `snare` → `assets/samples/snare.wav`

These paths come from `AudioConfig`, alongside output controls such as sample rate, mixer buffer size, output channels, and simultaneous voice count.

### Trigger flow

At runtime:

1. Gesture recognition emits an `AudioTrigger`.
2. The app forwards the trigger to the active audio engine.
3. The engine resolves the named sample.
4. The backend scales volume by the trigger intensity/velocity value.
5. `pygame.mixer.Sound.play()` starts playback.

If a sample is missing, the engine logs a warning and simply returns `False` from the trigger call instead of crashing the application.

## Limitations

- The current implementation uses `pygame.mixer`, which is practical but not a true pro-audio engine.
- Device selection is configuration-driven, but behavior still depends on how SDL/pygame exposes devices on the host platform.
- Volume is currently the only intensity mapping; there is no pitch-shift, envelope shaping, or round-robin sample variation.
- Missing assets are handled gracefully, but the corresponding trigger becomes a no-op until the file is restored.
- Tests use mocked pygame objects for determinism, so they do not measure real output latency.

## Future path toward JUCE or a lower-latency engine

There are two likely upgrade paths:

### 1. Add a lower-latency Python backend first

A `sounddevice` backend could be added behind the existing `AudioEngine` abstraction. That would allow:

- tighter control over stream setup,
- lower-latency callback-based playback,
- more explicit output device handling,
- and a migration path that keeps the application architecture stable.

### 2. Move performance-critical audio into JUCE

For the most robust real-time behavior, VisionBeat can eventually split audio into a dedicated JUCE-based engine or plugin layer. The existing abstraction already helps prepare for that by isolating the app from backend-specific calls.

A JUCE migration would likely involve:

- replacing sample loading/playback internals while keeping the same trigger vocabulary (`kick`, `snare`, etc.),
- moving timing-sensitive scheduling into the JUCE engine,
- exposing configuration through a bridge layer or IPC boundary,
- and preserving Python for camera, tracking, gesture recognition, and rapid iteration.

That approach would let VisionBeat keep the current computer-vision pipeline while upgrading only the latency-sensitive audio path.

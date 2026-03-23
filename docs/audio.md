# Audio

## Audio system role

The audio subsystem turns confirmed gesture events into audible percussion. In VisionBeat’s current prototype, this means triggering local WAV samples with a lightweight Python backend. The implementation is intentionally modest but architecturally separated so that lower-latency or more expressive backends can be added later.

## Current architecture

VisionBeat’s audio layer is split into two parts:

1. **`AudioEngine`** — an abstraction that defines how the rest of the app requests playback.
2. **`PygameAudioEngine`** — the concrete backend that initializes `pygame.mixer`, loads samples, and plays them on demand.

The runtime does not talk directly to pygame. Instead, it calls `create_audio_engine(config)` and works against the `AudioEngine` interface.

## Trigger path

The current end-to-end trigger path is:

1. a confirmed `GestureEvent` is emitted,
2. the runtime converts it to an `AudioTrigger`,
3. the trigger carries gesture type, timestamp, and intensity,
4. the audio engine resolves the corresponding sample name,
5. intensity is clamped into `0.0..1.0`,
6. the engine scales playback volume,
7. the backend plays the sample.

## Sample mapping

By default, the instrument uses two sample names:

- `kick`
- `snare`

These names map to configured sample paths in `audio.sample_mapping`, typically:

- `assets/samples/kick.wav`
- `assets/samples/snare.wav`

Because the mapping is config-driven, the audio layer already supports swapping sample files without changing gesture logic.

## Mixer configuration

The following audio settings are configurable:

- `backend`
- `sample_rate`
- `buffer_size`
- `output_channels`
- `simultaneous_voices`
- `output_device_name`
- `sample_mapping`
- `volume`

These parameters matter for demo stability and latency experiments.

### Practical guidance

- Lower `buffer_size` may reduce latency but can increase glitch risk.
- Higher `simultaneous_voices` helps with overlapping hits.
- `volume` acts as a master level before per-trigger intensity scaling.
- `output_device_name` depends on what SDL/pygame exposes on the host system.

## Asset handling

Sample files are resolved at startup. For each configured sample, VisionBeat records whether the file exists and attempts to load it only if:

- the mixer initialized successfully, and
- the sample path exists.

If a file is missing, the system logs a warning and leaves that sound unavailable instead of crashing the app.

## Intensity model

The runtime currently maps gesture confidence to audio intensity. That means more decisive gestures can play slightly louder than weaker ones.

Important caveat: this is a simple intensity model, not a physically realistic velocity-to-drum synthesis system. It is useful for expressive variation but intentionally lightweight.

## Why `pygame.mixer` is used right now

`pygame.mixer` is a pragmatic prototype backend because it:

- is easy to install,
- supports sample playback with low implementation overhead,
- fits well with Python-first experimentation,
- and is adequate for demonstrating the instrument concept.

This choice optimizes for iteration speed rather than maximum audio performance.

## Latency considerations

VisionBeat’s perceived latency comes from several layers:

1. camera capture timing,
2. tracker inference time,
3. gesture confirmation timing,
4. Python runtime overhead,
5. audio backend buffering,
6. host operating system audio scheduling.

So even though the audio backend matters, end-to-end responsiveness is a system property rather than a mixer-only property.

## Troubleshooting audio issues

### No sound plays

- verify sample files exist,
- regenerate them if needed,
- inspect logs for mixer initialization failure,
- confirm the selected output device is valid,
- and try increasing the buffer size.

### Sound plays but feels delayed

- reduce `audio.buffer_size` gradually,
- shorten gesture confirmation/cooldown settings only if the detector feels sluggish,
- and check runtime FPS because low frame rates reduce responsiveness before audio playback even starts.

### Repeated hits cut each other off

- increase `audio.simultaneous_voices`,
- verify the backend actually initialized with the requested channel count,
- and consider whether your sample tails are overlapping more than expected.

## Known limitations

- Only a pygame backend is implemented today.
- Playback is sample-trigger based, not sample-accurate.
- The engine does not do time stretching, pitch shifting, envelopes, or round-robin articulation.
- Device support depends on pygame/SDL behavior on the host machine.
- Tests validate logic and backend interaction, but they do not measure real acoustic latency.

## Future extensions

### Clap detection and sample layering

If clap detection is added, the audio layer could map it to:

- a clap sample,
- a layered accent stack,
- or a composite transient effect.

### MIDI output

A MIDI backend could interpret gesture events as note-on/note-off or velocity-bearing percussion messages, enabling VisionBeat to control:

- DAWs,
- drum samplers,
- external synths,
- or live performance rigs.

### JUCE integration

JUCE is the most natural route for a more professional audio engine. A likely future architecture would:

- keep Python for camera, tracking, gesture prototyping, and control logic,
- move latency-critical sample scheduling into JUCE,
- expose a transport protocol or bridge between the systems,
- and preserve the existing gesture vocabulary so the rest of the app remains conceptually stable.

### Richer expressive audio mapping

Beyond simple volume scaling, future versions could map gesture features to:

- velocity layers,
- filter brightness,
- sample selection,
- stereo position,
- or timing humanization.

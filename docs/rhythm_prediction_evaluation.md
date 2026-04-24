# Rhythm Prediction Evaluation

This guide tests ARJ's requested behavior: when a performer establishes a repeated
pulse, VisionBeat should assume the pulse may continue and predict the next expected
beat. The evaluation should show both the useful case, direct rhythm-triggered
continuation, and the musical side effect, false expectations when the performer
violates the pulse.

Use these current baselines when interpreting results:

- shadow-mode latency: **210 ms**
- hybrid-mode latency: **240 ms**
- camera: **30 fps**, about 33.3 ms per captured frame
- live inference: **8-9 fps**, about 111-125 ms per inference update

The rhythm tracker is timestamp-based, so the analysis should use seconds from
the logs rather than frame counts.

## Runtime Setup

Create a temporary evaluation config from `configs/default.yaml` and change only
the relevant fields:

```yaml
logging:
  level: INFO
  structured: true

predictive:
  mode: hybrid
  rhythm_prediction_enabled: true
  rhythm_trigger_mode: direct
  rhythm_min_hits: 3
  rhythm_jitter_tolerance: 0.18
  rhythm_confidence_threshold: 0.70
  rhythm_match_tolerance_seconds: 0.12
```

`direct` mode is the closest mode to ARJ's request: confirmed kick/snare sounds
train the rhythm tracker, and a stable pulse may play the next predicted kick/snare
directly. Rhythm-generated sounds do not train the tracker again, which prevents
an accidental self-playing loop.
During live testing, the HUD and trigger flash label the source of played sounds:
`Kick (rhythm predictor)` / `Snare (rhythm predictor)` means rhythm continuation
played the sound, while `Kick (CNN)` / `Snare (CNN)` means the CNN predictive
path played it.

The debug HUD also has a `Rhythm:` line. Use it as the live explanation of the
tracker:

- `learning hits=0/3` or `learning kick hits=2/3`: not enough repeated accepted
  sounds yet.
- `kick next @2.500s (+500ms)`: the tracker has predicted the next kick/snare
  time from the repeated pulse.
- `due`: the expected beat time has arrived and direct mode may play if
  confidence is above threshold and the frame is still inside the match window.
- `exp=+875ms` / `exp=-25ms`: how long until the rhythm state expires, or how
  long ago it expired.
- `last=matched`, `last=missed`, or `last=missed/expired`: the latest expected
  beat was matched by a real event, missed within the tolerance window, or missed
  and the rhythm state has now expired.
- `int`, `conf`, `jit`, and `reps`: estimated interval, confidence versus
  threshold, timing jitter, and repetition count.

For the live `configs/rhythm-hybrid.yaml` experiment, the rhythm settings are
slightly more permissive than the defaults: `rhythm_jitter_tolerance: 0.25`,
`rhythm_confidence_threshold: 0.60`, and
`rhythm_match_tolerance_seconds: 0.18`. This is intentional because 8-9 FPS live
inference can place the first post-beat frame roughly 110-125 ms after the
expected beat. If you change these values, report them with the results.

Keep the normal timing and gesture checkpoints configured when you use `mode:
hybrid` or `mode: primary`. To evaluate heuristics without the CNN path, use
`mode: disabled` with the same rhythm fields above.

Record each pattern as a separate run so the log summary maps cleanly to one test:

```bash
mkdir -p logs/rhythm-eval
PYTHONPATH=src python -m visionbeat --config configs/rhythm_eval.yaml 2> logs/rhythm-eval/steady-medium.log
```

VisionBeat writes rhythm-prediction events to the structured application log. The
`logging.event_log_path` gesture-analysis file is useful for detector debugging,
but rhythm-prediction summaries should be taken from the structured runtime log.

During each take, also write a short listening note:

- Did the rhythm-predicted sound happen close to the expected beat?
- Did the direct rhythm trigger feel like a continuation of the established pulse?
- Did a skipped or changed beat create a visible false expectation?
- Did false expectations feel musically acceptable or distracting?

## Test Patterns

Use a metronome if possible. Count aloud or use a visible click so the pattern is
repeatable.

| Pattern | Tempo | Suggested take | Expected behavior |
| --- | ---: | --- | --- |
| steady slow pulse | 1.0 s interval | 8-12 same-gesture hits | prediction activates after 3 hits; direct trigger should land near the 4th beat |
| steady medium pulse | 0.5 s interval | 12-20 same-gesture hits | prediction activates after 3 hits; direct trigger should land near the 4th beat |
| steady fast pulse | 0.33 s interval | 16-24 same-gesture hits | prediction activates after 3 hits; check whether frame cadence makes the direct trigger late |
| skip test | hit-hit-hit-rest-hit | skip the 4th expected beat | the skipped beat should become a missed rhythm expectation |
| delayed beat test | hit-hit-hit-delayed hit | delay the 4th hit by 150-300 ms | near delays may match; larger delays should miss and log error |
| tempo change test | 1.0 s then 0.5 s, or 0.5 s then 0.33 s | at least 4 hits at each tempo | old pulse may miss, then tracker should adapt after new repetitions |
| gesture variation test | kick-kick-kick-snare | switch gesture at expected time | kick expectation should be false; snare history should stay separate |

For each pattern, repeat at least 3 takes. The first take usually tests whether
the procedure is clear; use later takes for reporting.

## Metrics

Collect these metrics for each take:

- `time until rhythm prediction activates`: how long from the estimated first hit
  until the first pending rhythm expectation. With `rhythm_min_hits = 3`, this
  should be about 2 intervals.
- `prediction timing error in ms`: signed and absolute error for matched predictions.
  Negative means the actual hit was earlier than expected; positive means later.
- `matched prediction count`: expected beats that were followed by a real matching
  gesture inside the tolerance window.
- `missed prediction count`: expected beats that were not matched.
- `false expectation count`: missed plus expired expectations. In variation tests,
  wrong-gesture outcomes are the key false-expectation subtype.
- `direct rhythm trigger scheduling error`: how late or early the direct rhythm
  trigger was relative to the predicted timestamp.
- `effective latency compared with baseline hybrid`: prediction-side estimate using
  the 240 ms hybrid baseline.
- `whether direct rhythm triggering improved perceived timing`: listening judgment,
  supported by scheduling error, match/error metrics, and false expectations.

The analysis helper estimates prediction-side latency as:

```text
prediction_lead_ms = (predicted_time_seconds - prediction_log_timestamp) * 1000
estimated_effective_latency_ms = max(0, 240 - prediction_lead_ms)
estimated_latency_reduction_ms = 240 - estimated_effective_latency_ms
```

This estimates whether the rhythm expectation was available early enough to beat
the normal 240 ms hybrid baseline. In `direct` mode, the actual sound is emitted
when the live loop first reaches the predicted timestamp, so the important live
metric is `rhythm_live_trigger.scheduling_error_ms`.

## Analysis Helper

Run the helper on one or more structured log files:

```bash
PYTHONPATH=src python -m visionbeat.analyze_rhythm_predictions \
  logs/rhythm-eval/steady-medium.log \
  --label steady-medium-0.5s \
  --baseline-hybrid-latency-ms 240 \
  --baseline-shadow-latency-ms 210 \
  --output-json logs/rhythm-eval/steady-medium-summary.json
```

For mixed kick/snare runs, add `--by-gesture`:

```bash
PYTHONPATH=src python -m visionbeat.analyze_rhythm_predictions \
  logs/rhythm-eval/gesture-variation.log \
  --label gesture-variation \
  --by-gesture
```

The helper reports:

- activation hit count and activation delay
- pending, matched, missed, and expired counts
- false-expectation rate
- wrong-gesture false expectations
- median signed and absolute prediction error
- median prediction lead
- estimated prediction-side latency reduction versus the 240 ms hybrid baseline
- a coarse `yes`, `no`, or `review` verdict for perceived timing

Treat the verdict as a triage signal, not a final musical judgment. ARJ's request
explicitly allows false expectations when the performer violates the pulse, so the
important question is whether stable-pulse takes improve timing while violation
takes remain observable and explainable.

## Reporting Template

Use this table after each evaluation session:

| Pattern | Interval | Takes | Activation | Median abs error | Matched | False expectations | Prediction lead | Direct trigger timing | Listening note |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| steady slow | 1.0 s | 3 | after 3 hits |  |  |  |  |  |  |
| steady medium | 0.5 s | 3 | after 3 hits |  |  |  |  |  |  |
| steady fast | 0.33 s | 3 | after 3 hits |  |  |  |  |  |  |
| skip | mixed | 3 | after 3 hits |  |  |  |  |  |  |
| delayed beat | mixed | 3 | after 3 hits |  |  |  |  |  |  |
| tempo change | mixed | 3 | old pulse misses, new pulse adapts |  |  |  |  |  |  |
| gesture variation | mixed | 3 | per gesture |  |  |  |  |  |  |

## Interpreting Results

Good evidence that ARJ's rhythm-continuation idea is working:

- steady patterns activate after the expected number of hits
- median prediction lead is greater than 240 ms, especially for 0.5 s and 0.33 s pulses
- matched predictions outnumber false expectations in steady-pattern takes
- median absolute timing error is below the match tolerance, normally 120 ms
- `rhythm_live_trigger` logs show small scheduling errors
- rhythm-generated sounds do not create a self-sustaining loop without performer or model-confirmed beats

Expected evidence of pulse violation:

- skip tests produce missed expectations
- delayed beats produce signed timing errors or misses
- tempo changes temporarily miss, then re-establish a new interval
- gesture changes produce wrong-gesture false expectations without corrupting the
  separate gesture history

If stable takes show many misses, first check tracking quality and gesture detector
completion consistency before changing rhythm thresholds. If violation takes show
no misses, the match tolerance or expiry window is probably too permissive.

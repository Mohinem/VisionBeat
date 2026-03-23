# Observability

VisionBeat now emits two complementary telemetry streams:

1. **Structured application logs** for startup, shutdown, camera initialization, tracking failures, gesture candidates, confirmed triggers, and cooldown suppression.
2. **Optional event logs** in **JSONL** or **CSV** for offline analysis of false positives, missed gestures, and end-to-end latency.

## Configuration

Add these fields under `logging` in your YAML or TOML config:

```yaml
logging:
  level: INFO
  format: '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
  structured: true
  event_log_path: logs/visionbeat-events.jsonl
  event_log_format: jsonl
```

- `structured`: appends machine-readable JSON payloads to standard log lines.
- `event_log_path`: when set, writes gesture-analysis events to disk.
- `event_log_format`: choose `jsonl` or `csv`.

## Structured log coverage

VisionBeat emits structured log payloads for:

- application startup and shutdown
- runtime loop start and stop
- camera initialization success/failure
- tracker failures such as `no_person_detected`
- gesture candidates
- confirmed triggers
- cooldown suppression

A typical trigger log line looks like this:

```text
2026-03-23 12:00:00,000 | INFO | visionbeat.observability | Forward punch → kick | {"accepted":true,"confidence":0.94,"event":"gesture_trigger",...}
```

## Event schema

Each event-log row includes the following debugging fields:

- `timestamp`
- `gesture_type`
- `accepted`
- `reason`
- `velocity_stats`
- `confidence` when available

VisionBeat also records `event_kind` and `hand` to help separate candidate, confirmed, cooldown, and tracking-failure records.

### JSONL example

```json
{"timestamp": 12.5, "event_kind": "trigger", "gesture_type": "kick", "accepted": true, "reason": "Forward punch → kick", "velocity_stats": {"elapsed": 0.08, "delta_x": 0.01, "delta_y": 0.02, "delta_z": -0.27, "net_velocity": 3.75, "peak_x_velocity": 0.4, "peak_y_velocity": 0.6, "peak_z_velocity": -3.9}, "confidence": 0.94, "hand": "right"}
```

### CSV columns

The CSV writer flattens `velocity_stats` into separate columns:

- `elapsed`
- `delta_x`
- `delta_y`
- `delta_z`
- `net_velocity`
- `peak_x_velocity`
- `peak_y_velocity`
- `peak_z_velocity`

## Debugging false positives

Use the event log to answer these questions:

1. **Was the gesture only a candidate, or did it fully trigger?**
   - Filter `event_kind == candidate` to see motions that almost triggered.
   - Filter `accepted == true` to isolate confirmed triggers.
2. **Which threshold was most likely exceeded?**
   - High negative `delta_z` and `peak_z_velocity` suggest a kick-like forward punch.
   - High positive `delta_y` and `peak_y_velocity` suggest a snare-like downward strike.
3. **Was a repeated hit intentionally blocked?**
   - Look for `event_kind == cooldown_suppressed` with reason `cooldown_active`.
4. **Was bad tracking the real culprit?**
   - `tracking_failure` rows tell you whether the body temporarily disappeared or landmarks dropped below confidence thresholds.

A practical workflow:

- reproduce the false positive
- save a JSONL event log
- find the unexpected `trigger` row
- inspect the immediately preceding `candidate` and `tracking_failure` rows
- compare its velocity fields against your configured gesture thresholds

If false positives cluster around borderline values, raise thresholds such as `min_velocity`, `punch_forward_delta_z`, or `axis_dominance_ratio`.

## Debugging latency

To diagnose sluggish triggering:

1. Compare the spacing between `candidate` and `trigger` events.
   - Large gaps can indicate an overly long confirmation window or insufficient motion energy.
2. Inspect `cooldown_suppressed` events.
   - If deliberate rapid hits are being dropped, reduce `gestures.cooldowns.trigger_seconds`.
3. Watch for repeated `tracking_failure` rows.
   - Inconsistent tracking can delay candidates from ever reaching confirmation.
4. Correlate with runtime FPS in the live overlay/log output.
   - Lower frame rates reduce the temporal resolution of velocity estimates.

For latency experiments, CSV is convenient for spreadsheet charting, while JSONL is better for scripting and timeline reconstruction.

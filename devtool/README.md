# devtool: live view, calibration and click-target testing for MyoMouse

Separate from `myomouse/` and never moves the real mouse. It only borrows the BLE constants,
`bands.json` and the orientation maths from `myomouse`. No new dependencies.

```bash
python -m myomouse.identify            # once, if bands.json doesn't exist yet
python -m devtool --user dylan         # both bands
python -m devtool --user test --fake   # no hardware: hold L / R to squeeze
```

Close `myomouse.run` / `arena` first: a band only accepts one connection.

## Modes (keys 1 / 2 / 3 / 4)

**1 Live:** per band: connection status, battery, samples/s (green at 180–215, expect ~200),
packets/s, dropped packets, click count, 8 raw EMG traces (last 2 s; `G` = ×1/×4/×16 gain),
the activation trace with the threshold line, and a white line at every click.

**2 Calibrate:** SPACE, then 5 s rest (both arms), then 4 squeezes per hand.
Squeeze like a *normal click*, not max effort. Per channel it saves rest = median MAV at rest
(→ 0) and peak = 90th percentile MAV while squeezing (→ 1) to `data/devtool/calib/<user>.json`,
loaded automatically next time. It prints rest-noise p99 in activation units: keep the threshold
well above it. Re-run per person, and after taking the bands off.

**3 Test:** SPACE starts a session:
1. 20 s still, hands relaxed → false clicks/min at rest
2. 20 s moving the cursor without clicking → false clicks/min while moving
3. 20 targets; orange = LEFT fist, blue = RIGHT fist. 10 s timeout per target.

The summary appears on screen and in the terminal. Esc ends early (whatever was recorded is saved).

**4 Fatigue protocol:** SPACE, then cued squeezes (both hands, every 3.5 s, `--cue-interval`) for 8 min
(`--fatigue-min`), 3 min rest with arms still (`--fatigue-rest-min`; any click here is a false click),
then 1 min of cued squeezes again (`--fatigue-post-min`). Calibrate first. Everything is saved to
`<id>_fatigue_raw.npz` with phases and cue times. Squeeze like a normal click every time, even when
tired: don't squeeze harder to compensate.

```bash
python -m devtool --user emily                                       # then 2 (calibrate), 4 (fatigue), SPACE
python -m devtool.replay_fatigue data/devtool/sessions/<id>_fatigue_raw.npz   # fixed vs adaptive
python -m devtool.replay_fatigue <file> --models models/emily     # + her LDA, fixed vs adaptive
python -m devtool --fake --fake-fatigue 20                           # no hardware: fake bands that tire
```

`replay_fatigue` runs the fixed threshold and the fatigue-adaptive threshold (`devtool/fatigue.py`) on
the same raw EMG and prints, per minute, how many cued squeezes each caught, plus false clicks.
It saves `<id>_fatigue_replay.csv` and a chart `<id>_fatigue_replay.png`.

**How the adaptive threshold works (`devtool/fatigue.py`):** every squeeze burst (a click, or a weaker
near-miss that rose clearly above rest) is one measurement of the user's current squeeze strength.
A 1-D Kalman filter tracks that strength; the threshold scales with it (after 5 warm-up squeezes).
It never drops below 1.5 x the calibrated rest noise, and never rises above 1.5 x the base threshold.
`FatigueTracker.gain()` gives the same information as a factor to rescale raw EMG before LDA features.

## Settings

| Setting | Key | CLI | Meaning |
|---|---|---|---|
| click threshold | Up / Down | `--threshold 0.35` | click when activation ≥ this (rest 0, typical squeeze 1) |
| refractory | Left / Right | `--refractory 250` | min ms between clicks on one band; a crossing inside it is swallowed, not delayed |
| cursor smoothing | `[` `]` | `--smoothing 0.5` | exponential smoothing of cursor motion (IMU or mouse) |
| cursor source | M | `--cursor imu\|mouse` | `mouse` takes pointing out of the test, so you measure clicks only |

Settings are locked during a session and saved in every CSV row, so sessions can be compared.
Release needs activation < 0.6 × threshold (hysteresis). The cursor freezes for `FREEZE_MS`
after a click, like the real mouse. Gain and invert come from `myomouse/config.py`.

## Output: `data/devtool/sessions/`

- `<id>.csv`: one row per EMG sample around each click (−400…+200 ms by default,
  `--pre-ms/--post-ms`). Columns: click metadata (time, band, phase, outcome
  `hit / miss_outside / miss_wrong_band / false_rest`, target, cursor position at the click and
  at EMG onset, latencies, settings), then `sample_offset, sample_idx, t_sample_ms (= idx × 5),
  packet_idx, slot (0/1 = 1st/2nd sample of the BLE packet), valid, activation, ch0..ch7` (raw int8).
  Timed-out targets get one row with `event=timeout`. One row per click:
  `pd.read_csv(f).query("event=='click'").groupby("click_id").first()`.
- `<id>_summary.json`: the summary plus data quality (samples/s, dropped packets).
- `<id>_raw.npz`: the full raw EMG of the session, for replay.

```bash
python -m devtool.summary                    # table comparing all sessions
python -m devtool.replay data/devtool/sessions/<id>_raw.npz --refractory 150,200,250,300,350,400 --thresholds 0.3,0.35,0.4
```

`replay` re-runs the detector on the recorded EMG with other settings. One session can then
cover the whole 150–400 ms refractory sweep: it gives false clicks/min at rest, task clicks,
how many live hits would still fire, and double clicks (<500 ms apart).

## Metrics, defined

- **hit rate:** hits / all clicks in the target phase (wrong band inside the target = miss).
- **false clicks/min:** clicks during the still and moving rest phases.
- **time to target:** target shown → hitting click. Also split into shown → cursor enters, and enter → click.
- **click latency** = onset→detect + delivery.
  *onset→detect:* from EMG onset (25 ms envelope first above 0.5 × threshold) to the sample where
  the click fired, counted in samples × 5 ms. This is the detector's own delay.
  *delivery:* host time of the click event minus the estimated host time of that sample. Sample
  host time = t0 + idx × 5 ms, where t0 comes from the fastest-arriving recent packets, so this is
  BLE jitter + processing *above the minimum* BLE delay. The minimum (~1 connection interval)
  can't be measured without an external reference.
- **misses with cursor on target at onset:** the clench moved the cursor off the target
  before the click fired. If it's high, raise `FREEZE_MS` or smoothing.

## Timing details

Each notification = 2 samples. Samples are indexed and timestamped as idx × 5 ms. The Myo sends
its 4 EMG characteristics round-robin, so a skip = a dropped packet: the index advances 2 per
lost packet (`valid=0` rows) so later timestamps stay right. A skip is only filled if host time
confirms the time really passed. Otherwise it's counted as **out-of-order** (shown in yellow in
the header). If that counter climbs on your machine, the drop count isn't trustworthy there.

## Not covered

This tests a calibrated **threshold** detector, not the LDA classifier in `myomouse/detector.py`.
The numbers tell you how a threshold click behaves and give you labelled EMG windows. They
don't measure the trained model.

Tests: `pytest -q tests/test_devtool.py` (no hardware).

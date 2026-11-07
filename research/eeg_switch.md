# EEG safety switch: what was tried and the numbers

Owner: Juan · started Sat 7 Nov 2026, 16:45 · code: `eeg_switch/`

## Goal

An eyes-closed safety switch that works without a headset today and with the g.tec
Unicorn (8 ch, 250 Hz) tomorrow, by changing one line (`SOURCE` in
`eeg_switch/config.py`). Closing the eyes for about 2 s toggles the switch; blinks must not.

## Data

- **PhysioNet EEG Motor Movement/Imagery (eegmmidb)**, run 1 = 1 min eyes open,
  run 2 = 1 min eyes closed, 64 ch at 160 Hz, 109 people.
- Kept 8 channels matching the Unicorn montage: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8.
  Resampled to 250 Hz, in µV.
- **Why real data:** a synthetic sine wave makes any alpha detector look perfect.
  Real recordings bring blinks, drift and person-to-person differences in alpha.
  The synthetic generator (`eeg_switch/synthetic.py`) is used only for unit tests.
- **Splitting:** thresholds tuned on subjects 1–40 (dev); reported on subjects 71–109
  (test), never seen during tuning. Excluded 88, 89, 92, 100, 104 (known recording issues).

### Test scenarios

Each test person gets 3 timelines cut from their own two runs:
20 s eyes-open baseline, then 4 long closures (3–6 s, should toggle) and 4 short
closures (0.3–1.0 s, must not toggle), separated by 4–8 s of eyes open.
Pieces are joined with a 50 ms crossfade and offset matching.

**Limitation:** the open→closed transitions are spliced, not real. A real eye closure
starts with a blink artefact and alpha builds over ~0.5 s. Expect real delays to be a
little longer than these numbers. Long recordings reuse the same ~60 s of material.

## Detector (`eeg_switch/detector.py`)

1. Every 100 ms, take the last 1 s of the posterior channels (Pz, PO7, Oz, PO8).
2. Alpha power (8–12 Hz, Welch). Two options tested:
   - `rel`: alpha as a share of 2–30 Hz power (robust to broadband artefacts)
   - `abs`: absolute alpha power
3. Baseline = median over the first 20 s (eyes open, per person). Score = dB above baseline.
4. Score > ON_DB = closed; score < OFF_DB (= ON_DB − 2 dB) = open; in between keep state.
5. Closed for 2 s → one `safety_toggle` event, then disarm until eyes are open for 1 s.
6. Windows with posterior peak-to-peak > 200 µV are skipped (movement, cable knocks).

Events go on LSL stream `NeuroIntents` in the shared format:
`{"source": "eeg", "intent": "safety_toggle", "confidence": 0-1, "timestamp": LSL time}`.

## Results

### Pipeline check on synthetic data (NOT evidence, code check only)

Live LSL test (simulator → detector → listener): two injected 4 s closures toggled,
one 0.6 s closure did not. Offline on 10 synthetic test "subjects": 0 missed,
0 false toggles/min, median delay 2.28 s. Unit tests: 8 passed.

### Real data (PhysioNet, held-out subjects 71–109)

**PENDING:** the development sandbox can't reach physionet.org. Run on a laptop:

```
python -m eeg_switch.download
python -m eeg_switch.evaluate
```

Then copy the numbers from `research/eeg_switch_results/summary.json` here.

| Metric (held-out test) | Value |
|---|---|
| People | _ |
| Long closures tested | _ |
| Missed toggles | _ % |
| False toggles per minute | _ |
| Detection delay, median (incl. 2 s dwell) | _ s |
| Detection delay, 90th percentile | _ s |
| Chosen setting (from dev) | feature _, ON_DB _ |

Tuning table on dev: `research/eeg_switch_results/tuning.csv`.

### What to watch for in the real numbers

- **Low-alpha people:** roughly 5–10% of people show little eyes-closed alpha increase.
  Expect most misses to cluster in a few people (`test_per_subject.csv`), not spread evenly.
- **rel vs abs:** if `abs` wins on dev but `rel` is close, prefer `rel`: it is less
  sensitive to electrode contact changing during a session.

## Recommendation for dev

- **Use it as a deliberate switch, not a fast control.** The 2 s dwell is the point:
  expect ~2.3 s from closing the eyes to the toggle. Show a "closing … 1.2/2 s" progress
  indicator so users know to keep their eyes shut.
- **Always run the 20 s eyes-open calibration** at the start of each session, per person.
  Re-run it if the headset is re-seated.
- **Keep ON_DB / OFF_DB from `summary.json`** (tuned on dev subjects) as defaults, and
  offer a per-user "sensitivity" setting of ±1 dB.
- **Before demo day, verify the Unicorn's LSL stream name/type** with
  `python -m eeg_switch.listen --list` and update `STREAMS["unicorn"]` in `config.py`.
- **Fallback for people without a clear alpha response:** if calibration shows <3 dB
  difference in a short eyes-closed check, tell the user and offer the keypress/CV switch.
- The switch only toggles; consumers should keep their own ON/OFF state and show it.

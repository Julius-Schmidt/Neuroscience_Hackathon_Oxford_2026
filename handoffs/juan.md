# Juan: handoff

## Sat 7 Nov, 21:00

**Plan change:** EEG now serves Level 1 (brain–muscle divergence). The eyes-closed switch
(`eeg_switch/`) is **parked**: finished and tested, roadmap slide only. Gesture-mapper port not started.

**Done**
- `eeg_fatigue/` (Level 1), everything on one LSL clock:
  - `session`: cues + records Unicorn EEG + Myo EMG + markers into one file; autosaves after every
    block; checks the file at the end (FILE OK). `--sanity` = 3 min go/stop check.
  - `analyse`: EMG onset = trigger → EEG epochs → per-block average (20 squeezes) →
    ratio = C3 beta drop ÷ EMG peak, normalised to block 1, bootstrap 95% interval, verdict, figure.
    Also tests brain effort on its own (the ratio rises even if only the muscle tires).
  - `simulate`: offline file or live fake headset + arm over LSL for dry runs.
- Dry run on simulated data finds the planted effect, and none in the no-effect control (table in
  `research/eeg_fatigue.md`). Live LSL sanity check: FILE OK, GO.
- 16 unit tests pass. Run sheet: `research/eeg_fatigue_runsheet.md`.

**Next**
1. 21:30 dry run with Dylan (run sheet, "Before the first participant").
2. 22:00 headset sessions P01–P03; fill the real-results table in `research/eeg_fatigue.md`.

**Blocked**
- Unicorn LSL stream name/type still unverified (`python -m eeg_switch.listen --list` on the laptop).

## Sat 7 Nov, 17:25

**Done**
- `eeg_switch/` module (no headset needed), everything over LSL:
  - `simulator`: replays PhysioNet eyes-open/closed EEG as a Unicorn-like stream
    (`Unicorn`, EEG, 8 ch, 250 Hz). Keys: `c` = 4 s eyes closed, `b` = blink-length, `q` = quit.
  - `run_detector`: posterior alpha vs 20 s eyes-open baseline, 2 s dwell, hysteresis;
    publishes `safety_toggle` events on `NeuroIntents` in the shared format.
  - `evaluate`: tunes on subjects 1–40, reports on held-out 71–109
    (missed toggles, false toggles/min, delay).
  - `listen`: prints every intent event; `--list` shows all LSL streams.
- Live loop tested: simulator → detector → listener, toggles on 4 s closures, not on 0.6 s.
- 8 unit tests pass (`pytest -q tests/test_eeg_switch.py`).
- `research/eeg_switch.md` written; real-data numbers pending (see Blocked).

**Next**
1. Run `python -m eeg_switch.download` then `python -m eeg_switch.evaluate` on a laptop,
   paste numbers into `research/eeg_switch.md`.
2. Port gesture mapper to Windows and make it read `NeuroIntents` events.

**Blocked**
- Cloud sandbox can't reach physionet.org, so held-out numbers must be run locally.
- Need the real Unicorn LSL stream name/type to confirm `STREAMS["unicorn"]` in config.

**For Dylan:** event format used exactly as agreed. Stream `NeuroIntents`, type `Markers`,
one string channel, JSON payload; `source_id` = `neuro_intents_<source>`.

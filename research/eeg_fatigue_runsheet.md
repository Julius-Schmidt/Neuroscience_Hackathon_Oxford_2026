# Level 1 run sheet: headset sessions (22:00 – 01:00, cutoff 02:00)

Print this or keep it open on the recording laptop.

## Before the first participant (with Dylan, 21:30–22:00 dry run)

1. Unicorn Suite open, headset paired, **Unicorn LSL** app streaming (Open → Start).
2. `python -m eeg_switch.listen --list` → copy the Unicorn stream's exact **name** and **type**
   into `eeg_switch/config.py` → `STREAMS["unicorn"]`, and set `SOURCE = "unicorn"`.
   If the stream has more than 8 channels, the first 8 must be EEG (Fz C3 Cz C4 Pz PO7 Oz PO8).
3. Myo on the **right** forearm (the squeezing hand), `bands.json` present (`python -m myomouse.identify`).
4. Optional backup: LabRecorder open, recording all streams (Unicorn, MyoEMG, FatigueMarkers).
5. Dry run without people: `python -m eeg_fatigue.simulate --lsl` in one terminal,
   `python -m eeg_fatigue.session --participant DRY --sanity --emg lsl` in another → must print FILE OK and GO.
   (Set `SOURCE = "sim"` for this, then back to `"unicorn"`.)

## Per participant (~30 min each; fit the next one while analysing)

IDs only: P01, P02, P03 … (not Emily or Juan: they did the earlier fatigue session).

| Step | Command / action | Time |
|---|---|---|
| Fit | Headset on, **gel on C3, Cz, C4** (posterior channels dry is OK); Myo on right forearm | 8 min |
| Sanity | `python -m eeg_fatigue.session --participant P01 --sanity` | 3 min |
| Decide | **GO** → full session. **STOP** → re-gel / re-seat, rerun sanity once. STOP twice → next participant | 1 min |
| Session | `python -m eeg_fatigue.session --participant P01` | 15 min |
| File | Must print **FILE OK** before the headset comes off | 0 min |
| Analyse | `python -m eeg_fatigue.analyse data\fatigue\P01_session_*.npz` (while next person is fitted) | 2 min |

**Say to each participant:** "Eyes on this point. Jaw and face relaxed, don't clench your teeth.
When it says SQUEEZE, make a fist as hard as you can with your right hand until it says relax.
It will get tiring; keep giving maximum effort. You can stop at any time."

**Session = 15 min:** baseline 20 squeezes → 5 fatigue blocks of 20 (5 s between) → 3 min rest → 20 recovery squeezes.
Saves after every block; Ctrl+C keeps everything recorded so far.

## Reading the result

- `VERDICT: ratio RISES … DROPS after rest` + `brain effort up while muscle output down` = the Level 1 result.
- `ratio RISES … mostly muscle fatigue` = muscle got weaker but the brain signal did not change. Still a valid
  finding; report it as such, not as divergence.
- `no clear EEG response in block 1` = the signal was too noisy to measure. Counts against the midnight cutoff.
- After 2+ people: `python -m eeg_fatigue.analyse data\fatigue\P0*_session_*.npz` prints the group line.
- Figures: `research/eeg_fatigue_results/<participant>_beta_erd/figure.png`.
- Secondary measure: add `--measure mrcp` (motor potential). Report it as secondary; beta drop is the main measure.

## Cutoffs

- **Midnight:** "EEG response visible" must list blocks for at least one real participant. If not: stop,
  keep the files, use the limitations paragraph in `research/eeg_fatigue.md`.
- **02:00:** result or limitations paragraph. Then sleep.

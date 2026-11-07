# Level 1: brain–muscle divergence during fatigue

Owner: Juan · Sat 7 Nov 2026 · code: `eeg_fatigue/` · run sheet: `research/eeg_fatigue_runsheet.md`

## Question

During a fatiguing squeeze task, does brain effort rise relative to muscle output, and recover after rest?
Done when the ratio rises during fatigue and drops after rest in at least 2 people.

## Method

- **Recording:** g.tec Unicorn EEG (8 ch, 250 Hz, gel on C3/Cz/C4) + Myo EMG (8 ch, 200 Hz, right forearm)
  + cue markers, all on one LSL clock, one file per session (`eeg_fatigue/session.py`).
- **Task:** cued maximal squeezes (1.5 s squeeze, 2.5 ± 0.3 s relax), 20 per block:
  baseline → 5 fatigue blocks → 3 min rest → 1 recovery block (~15 min).
- **Analysis** (`eeg_fatigue/analyse.py`):
  1. EMG onsets are the trigger (high-pass 20 Hz, rectified, 50 ms smoothing, threshold crossing
     that lasts ≥ 0.3 s and follows a squeeze cue).
  2. EEG (common average reference) cut from −2 to +3 s around each onset; epochs over 150 µV
     peak-to-peak on C3/Cz/C4 dropped.
  3. Per block: **EEG effort** = drop in C3 beta (13–30 Hz) power from 0 to 1 s after onset,
     relative to −1.5 to −0.5 s (secondary: motor potential at C3, −0.5 to 0 s);
     **EMG output** = mean squeeze peak of the EMG envelope; EMG median frequency also reported.
  4. **Ratio** = (EEG effort ÷ block 1) ÷ (EMG peak ÷ block 1), 95% bootstrap interval over squeezes.
- **Verdict rules (set before seeing real data):** rises = last fatigue block ratio ≥ 1.10 and interval
  above 1; drops after rest = recovery ratio below that, interval not reaching it.
  **Brain effort on its own** must also rise (≥ 1.10, interval above 1) to call it a central component.

### Why brain effort is tested separately

On the simulated data, a weaker muscle with unchanged brain signal still raised the ratio to 1.36.
The ratio alone cannot tell muscle fatigue from central fatigue; the brain-effort trend can.

## Dry run (simulated, NOT evidence)

Simulated head + arm with a known effect put in. The analysis must find it, and find nothing when there is none.

| Simulated scenario | Ratio, last fatigue block | Recovery | Brain effort alone | Verdict |
|---|---|---|---|---|
| Divergence (brain ↑, muscle ↓) | 1.84 (1.69–2.01) | 1.38 | ×1.30 (1.23–1.40) | central component ✔ |
| Muscle only | 1.36 (1.23–1.51) | 1.22 | ×0.96 (0.90–1.04) | mostly muscle fatigue ✔ |
| No effect | 1.01 (0.91–1.11) | 1.00 | ×0.96 | no divergence ✔ |

Live LSL dry run (simulator → session → file check → analysis): see handoff for status.
Unit tests: 16 passed (`tests/test_eeg_fatigue.py`, `tests/test_eeg_switch.py`).

## Real results

_Fill in per participant from `research/eeg_fatigue_results/<P>_beta_erd/summary.json`._

| Participant | Sanity | Clean epochs | EEG response visible | Ratio last fatigue (95%) | Recovery | Brain effort alone | Verdict |
|---|---|---|---|---|---|---|---|
| P01 | | | | | | | |
| P02 | | | | | | | |
| P03 | | | | | | | |

## Limitations (use for the slide)

- 2–3 people, one session each, no repeat; healthy volunteers, not people with MS.
- Dry/hybrid 8-channel headset; C3 beta is small and easily masked by jaw and neck muscle activity
  during maximal squeezes.
- "Effort" is inferred from beta power drop, an indirect marker of motor cortex activity.
- EMG amplitude also changes with band position and sweat over 15 min; median frequency is reported as a check.
- The ratio rises whenever EMG falls, so it is only read together with the brain-effort trend.

**If the signal is too noisy (fallback paragraph):** "We recorded EEG and EMG together during a 15-minute
fatiguing squeeze task in N participants. With dry/hybrid electrodes, motor-cortex beta activity around
squeezes was not reliably visible above noise (EEG response not distinguishable from zero in block 1),
so the brain–muscle ratio could not be measured. The recording and analysis pipeline works end to end and
is ready for gel-based or higher-density EEG in a follow-up."

## Recommendation for dev

_After the sessions._

"""3-minute go/stop check, run automatically after `session --sanity`.

    python -m eeg_fatigue.sanity data/fatigue/P03_sanity_*.npz

Checks: no flat or wildly noisy channels; posterior alpha rises with eyes closed;
EMG finds the squeezes; C3 shows a power drop around the squeezes.
GO = all pass. STOP = re-gel / re-seat, then rerun. Twice STOP = drop Level 1.
"""

from __future__ import annotations

import sys

import numpy as np

from . import config as C
from . import signals as S
from .recording import load


def _alpha(rec, a, b):
    from eeg_switch.detector import alpha_db
    post = [rec.eeg_ch.index(c) for c in ("Pz", "PO7", "Oz", "PO8")]
    m = (rec.eeg_t >= a + 3) & (rec.eeg_t < b)
    x = S.reference(rec.eeg)[m][:, post]
    n = int(2 * C.EEG_SRATE)
    vals = [alpha_db(x[i:i + n], C.EEG_SRATE, "abs") for i in range(0, len(x) - n, n // 2)]
    return float(np.median(vals)) if vals else np.nan


def report(path) -> bool:
    rec = load(path)
    ok_all = True
    print(f"\n=== SANITY {rec.meta.get('participant', '?')}{'  [SIMULATED]' if rec.meta.get('simulated') else ''} ===")
    x = S.bandpass(rec.eeg - rec.eeg.mean(axis=0), 1, 40)
    sd = x.std(axis=0)
    bad = [f"{c} ({v:.1f} µV)" for c, v in zip(rec.eeg_ch, sd) if v < 1.0 or v > 60.0]
    print(f"[{'ok' if not bad else '!!'}] channels (1-40 Hz SD µV): " +
          " ".join(f"{c}={v:.0f}" for c, v in zip(rec.eeg_ch, sd)) + (f"  -> check {bad}" if bad else ""))
    motor_bad = [b for b in bad if b.split()[0] in ("C3", "Cz", "C4")]
    ok_all &= not motor_bad

    mk = list(zip(rec.markers, rec.marker_t))
    t_open = [t for m, t in mk if m == "eyes_open"]
    t_closed = [t for m, t in mk if m == "eyes_closed"]
    if t_open and t_closed:
        a_open = _alpha(rec, t_open[0], t_closed[0])
        a_closed = _alpha(rec, t_closed[0], t_closed[0] + C.SANITY_EYES_S)
        d = a_closed - a_open
        ok = d > 2.0
        print(f"[{'ok' if ok else '!!'}] eyes-closed alpha: +{d:.1f} dB (need > 2 dB)  <- headset is reading brain signal")
        ok_all &= ok

    from .analyse import per_squeeze
    sq = per_squeeze(rec, "beta_erd")
    n_cue = sq["n_cues"]
    ok = len(sq["onsets"]) >= 0.8 * n_cue
    print(f"[{'ok' if ok else '!!'}] EMG squeezes found: {len(sq['onsets'])}/{n_cue}")
    ok_all &= ok
    clean = sq["clean"]
    if clean.sum() >= 5:
        tr = sq["trials"][clean]
        eff = S.effort_from_trials(tr, sq["grid"], "beta_erd")
        rng = np.random.default_rng(0)
        bs = [S.effort_from_trials(tr[rng.integers(0, len(tr), len(tr))], sq["grid"], "beta_erd") for _ in range(300)]
        lo = np.percentile(bs, 2.5)
        ok = lo > 0
        print(f"[{'ok' if ok else '!!'}] {sq['channel']} beta drop during squeeze: {eff:.0f}% "
              f"(95% low end {lo:.0f}%, need > 0), {int(clean.sum())} clean epochs")
        ok_all &= ok
    else:
        print(f"[!!] only {int(clean.sum())} clean EEG epochs around squeezes (too noisy)")
        ok_all = False
    print(f"\n>>> {'GO: run the full session' if ok_all else 'STOP: re-gel C3/Cz/C4, check fit, rerun sanity'} <<<")
    return ok_all


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)

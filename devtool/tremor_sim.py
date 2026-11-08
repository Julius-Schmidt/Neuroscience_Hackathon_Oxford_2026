"""What does tremor do to the fist click, and does a "hold it" rule fix it? Simulation.

    python -m devtool.tremor_sim data/devtool/sessions/<id>_fatigue_raw.npz --models models/dylan

MS tremor is mostly postural / intention tremor at 2.5-7 Hz: the muscles switch on and off a
few times a second. A deliberate fist is one sustained contraction. This script adds simulated
tremor to a real dev tool mode 4 recording and replays the LDA click with different hold rules:

    tremor        bursts of muscle activity at 4-6 Hz (drifting), each ~40% of the cycle, with
                  the SAME channel pattern as your fist (worst case: it looks like a small fist)
                  and a size given as a fraction of your average squeeze. Present the whole
                  recording (rest too), which is harsher than intention tremor.
    hold rule     the debouncer needs the fist probability high for N decisions in a row before
                  it presses (config N_ON, 10 ms each). Current: 3 = 30 ms. Longer = tremor
                  bursts can't press, but every click comes later.

Reported per tremor size and hold: cued squeezes caught, false clicks per minute (outside cue
windows + the rest phase), broken squeezes (one cued squeeze -> 2+ clicks), and the median
delay from the cue to the click (includes reaction time; compare between rows, not to zero).

A simulation from one healthy recording: real tremor alternates between flexors and
extensors, varies with posture and effort, and comes with weakness and spasticity.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .emg import EMG_HZ
from .fatigue_features import squeeze_segment
from .replay import load
from .replay_fatigue import POST, PRE, load_models, score

SIZES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75)
HOLDS_MS = (30, 100, 200, 300)


def fist_profile(emg, idx, cues, side):
    """Per-channel std of the EMG during the cued squeezes (the 'shape' of your fist)."""
    env = np.convolve(np.abs(emg.astype(float)).mean(axis=1), np.ones(10) / 10, mode="same")
    segs = []
    for _ph, _t, ci in cues:
        i0 = int(np.searchsorted(idx, ci[side]))
        s = squeeze_segment(env, i0, min(i0 + 300, len(env)))
        if s:
            segs.append(emg[s[0]:s[1]].astype(float))
    return np.concatenate(segs).std(axis=0)


def tremor(n, profile, size, rng, f_lo=4.0, f_hi=6.0, duty=0.4):
    """Bursting EMG-like noise: on/off at a slowly drifting 4-6 Hz."""
    f = np.clip(5.0 + np.cumsum(rng.normal(0, 0.02, n)), f_lo, f_hi)
    phase = 2 * np.pi * np.cumsum(f) / EMG_HZ
    env = np.clip((np.cos(phase) - np.cos(np.pi * duty)) / (1 - np.cos(np.pi * duty)), 0, 1) ** 0.5
    return rng.normal(size=(n, len(profile))) * profile * size * env[:, None] * 1.6   # ~size of a squeeze when on


def lda_probs(x, model):
    from myomouse.features import sliding_features
    X, ends = sliding_features(x.astype(np.float32), model["window"], model["step"])
    return model["pipeline"].predict_proba(X)[:, 1], ends


def presses(p, ends, idx, n_on):
    from myomouse.detector import Debouncer
    deb = Debouncer(n_on=n_on)
    return np.array([int(idx[e - 1]) for e, pk in zip(ends, p) if deb.update(float(pk)) == "press"], dtype=np.int64)


def run(npz, models_dir, sizes=SIZES, holds=HOLDS_MS, plot=True, seed=0):
    bands, phases, _ev, _settings, _cals = load(npz)
    cues = json.loads(str(np.load(npz, allow_pickle=False)["cues"]))
    models = load_models(models_dir)
    if not models:
        raise SystemExit("needs a trained model: --models models/<user>")
    rng = np.random.default_rng(seed)
    rows = []
    for side, d in bands.items():
        if side not in models:
            continue
        model = models[side]
        step_ms = 1000 * model["step"] / EMG_HZ
        v = d["valid"]
        emg, idx = d["emg"][v], d["idx"][v]
        ph = {name: i[side] for name, _t, i in phases}
        end = int(d["idx"][-1]) + 1
        rest_range = (ph.get("recovery", end), ph.get("post", end))
        cue_idx = np.array([c[2][side] for c in cues])
        minutes = len(emg) / EMG_HZ / 60
        prof = fist_profile(emg, idx, cues, side)
        print(f"\n[{side}] {len(cue_idx)} cued squeezes, {minutes:.1f} min. Tremor size = fraction of your squeeze.")
        print(f"  {'tremor':>6s} {'hold':>6s} {'caught':>7s} {'false/min':>10s} {'broken':>7s} {'cue->click':>11s}")
        for size in sizes:
            x = emg if size == 0 else np.clip(np.round(emg + tremor(len(emg), prof, size, rng)), -128, 127)
            p, ends = lda_probs(x, model)
            for hold in holds:
                n_on = max(1, int(round(hold / step_ms)))
                pr = presses(p, ends, idx, n_on)
                caught, false = score(pr, cue_idx, rest_range)
                broken, delays = 0, []
                for c in cue_idx:
                    inwin = pr[(pr >= c - PRE) & (pr <= c + POST)]
                    broken += len(inwin) >= 2
                    if len(inwin):
                        delays.append((inwin[0] - c) * 1000 / EMG_HZ)
                row = dict(band=side, tremor=size, hold_ms=hold, caught_pct=round(100 * caught.mean(), 1),
                           false_per_min=round(len(false) / minutes, 2), broken=int(broken),
                           delay_ms=round(float(np.median(delays)), 0) if delays else None)
                rows.append(row)
                print(f"  {size:6.0%} {hold:4d}ms {row['caught_pct']:6.0f}% {row['false_per_min']:10.1f} "
                      f"{broken:7d} {row['delay_ms'] if row['delay_ms'] is not None else '-':>9}ms", flush=True)
    stem = Path(npz).name.replace("_raw.npz", "")
    out = Path(npz).parent / f"{stem}_tremor.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")
    if plot:
        try:
            _plot(rows, out.with_suffix(".png"))
        except Exception as e:
            print(f"(plot skipped: {e})")
    return rows


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for hold in sorted({r["hold_ms"] for r in rows}):
        rs = [r for r in rows if r["hold_ms"] == hold]
        t = [100 * r["tremor"] for r in rs]
        axes[0].plot(t, [r["caught_pct"] for r in rs], "o-", label=f"hold {hold} ms")
        axes[1].plot(t, [r["false_per_min"] for r in rs], "o-", label=f"hold {hold} ms")
        axes[2].plot(t, [r["broken"] for r in rs], "o-", label=f"hold {hold} ms")
    for a, title in zip(axes, ("cued squeezes caught (%)", "false clicks per minute", "broken squeezes (2+ clicks)")):
        a.set_title(title)
        a.set_xlabel("simulated tremor (% of squeeze size)")
        a.grid(alpha=0.3)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--models", required=True, help="folder with fist_right.joblib, e.g. models/dylan")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    run(a.npz, a.models, plot=not a.no_plot)


if __name__ == "__main__":
    main()

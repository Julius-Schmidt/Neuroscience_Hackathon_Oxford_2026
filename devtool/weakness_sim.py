"""How weak can a user be before the click stops working? A simulation on a real recording.

    python -m devtool.weakness_sim data/devtool/sessions/<id>_fatigue_raw.npz --models models/dylan

Takes a dev tool mode 4 recording and replays it with the muscle signal shrunk to a fraction
s of its real size (1.0 = as recorded, 0.2 = a user with a fifth of the strength), while the
sensor noise stays the same size:

    simulated = s * recorded + sqrt(1 - s^2) * noise

where `noise` is real band noise taken from quiet stretches of the rest phase, so the total
noise level stays constant while the muscle part shrinks. Result is rounded to int8 like the
Myo. Each detector is then scored on the same cues as replay_fatigue:

    fixed         threshold calibrated at FULL strength (the user got weaker since calibrating)
    adaptive      same calibration + the fatigue-adaptive threshold
    recalibrated  threshold calibrated AT the weaker strength (a weak user calibrating for themselves)
    lda           LDA trained at full strength
    lda adaptive  LDA with the fatigue tracker's gain

A simulation, not a patient: real weakness also changes which muscles fire and how steadily.
Use it as "how much margin does each method have", not as a clinical threshold.
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .emg import detect_offline
from .fatigue import detect_offline_adaptive, detect_offline_lda, gain_series
from .replay import load
from .replay_fatigue import load_models, score

LEVELS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08)


def noise_pool(emg, lo, hi, block=200):
    """Quiet blocks from the rest phase (the quieter half), concatenated."""
    seg = emg[lo:hi].astype(float)
    n = len(seg) // block
    if n < 4:
        raise SystemExit("rest phase too short to estimate the noise")
    blocks = seg[: n * block].reshape(n, block, -1)
    level = np.abs(blocks).mean(axis=(1, 2))
    quiet = blocks[level <= np.median(level)]
    return quiet.reshape(-1, seg.shape[1])


def make_noise(pool, n, rng, block=200):
    out = np.empty((n, pool.shape[1]))
    nb = len(pool) // block
    for i in range(0, n, block):
        k = rng.integers(nb)
        chunk = pool[k * block:(k + 1) * block]
        out[i:i + block] = chunk[: min(block, n - i)]
    return out - pool.mean(axis=0)


def simulate(emg, s, noise):
    x = s * emg.astype(float) + np.sqrt(max(0.0, 1 - s * s)) * noise
    return np.clip(np.round(x), -128, 127).astype(np.int8)


def run(npz, models_dir=None, levels=LEVELS, plot=True, seed=0):
    bands, phases, _ev, settings, cals = load(npz)
    z = np.load(npz, allow_pickle=False)
    cues = json.loads(str(z["cues"]))
    thr, ref = float(settings["threshold"]), float(settings["refractory_ms"])
    models = load_models(models_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for side, d in bands.items():
        v = d["valid"]
        emg, idx = d["emg"][v], d["idx"][v]
        ph = {name: i[side] for name, _t, i in phases}
        end = int(d["idx"][-1]) + 1
        rest_range = (ph.get("recovery", end), ph.get("post", end))
        lo, hi = (np.searchsorted(idx, r) for r in rest_range)
        pool = noise_pool(emg, lo, hi)
        noise = make_noise(pool, len(emg), rng)
        cue_idx = np.array([c[2][side] for c in cues])
        minutes = len(emg) / 200 / 60
        cal = cals[side]
        rest = np.asarray(cal.rest, float)
        print(f"\n[{side}] {len(cue_idx)} cued squeezes, {minutes:.1f} min, threshold {thr}")
        print(f"  {'strength':>8s}  " + "  ".join(f"{m:>17s}" for m in
              ("fixed", "adaptive", "recalibrated", "lda", "lda adaptive")) + "   (caught % / false per min)")
        for s in levels:
            x = simulate(emg, s, noise) if s < 1 else emg
            cal_s = replace(cal, peak=list(rest + s * (np.asarray(cal.peak, float) - rest)))
            res = {
                "fixed": detect_offline(x, idx, cal, thr, ref),
                "adaptive": detect_offline_adaptive(x, idx, cal, thr, ref)[0],
                "recalibrated": detect_offline(x, idx, cal_s, thr, ref),
            }
            if side in models:
                res["lda"] = detect_offline_lda(x, idx, models[side])
                res["lda adaptive"] = detect_offline_lda(x, idx, models[side], gain_series(x, idx, cal, thr))
            row = dict(band=side, strength=s)
            cells = []
            for name in ("fixed", "adaptive", "recalibrated", "lda", "lda adaptive"):
                if name not in res:
                    cells.append(f"{'-':>17s}")
                    continue
                caught, false = score(res[name], cue_idx, rest_range)
                c, f = 100 * caught.mean(), len(false) / minutes
                row[f"{name}_caught_pct"], row[f"{name}_false_per_min"] = round(c, 1), round(f, 2)
                cells.append(f"{c:7.0f}% {f:6.1f}/min")
            rows.append(row)
            print(f"  {s:8.0%}  " + "  ".join(cells), flush=True)
    stem = Path(npz).name.replace("_raw.npz", "")
    out = Path(npz).parent / f"{stem}_weakness.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "band", k != "strength", k)))
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
    names = [n for n in ("fixed", "adaptive", "recalibrated", "lda", "lda adaptive") if f"{n}_caught_pct" in rows[0]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for side in sorted({r["band"] for r in rows}):
        rs = [r for r in rows if r["band"] == side]
        s = [100 * r["strength"] for r in rs]
        for n in names:
            a1.plot(s, [r[f"{n}_caught_pct"] for r in rs], "o-", label=n)
            a2.plot(s, [r[f"{n}_false_per_min"] for r in rs], "o-", label=n)
    for a in (a1, a2):
        a.invert_xaxis()
        a.set_xlabel("simulated strength (% of the recorded squeeze)")
        a.grid(alpha=0.3)
    a1.set_ylabel("cued squeezes caught (%)")
    a1.set_ylim(0, 105)
    a2.set_ylabel("false clicks per minute")
    a1.legend(fontsize=8)
    a1.set_title("Detection as the user gets weaker")
    a2.set_title("False clicks")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--models", default=None, help="folder with fist_right.joblib (e.g. models/dylan)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    run(a.npz, a.models, plot=not a.no_plot)


if __name__ == "__main__":
    main()

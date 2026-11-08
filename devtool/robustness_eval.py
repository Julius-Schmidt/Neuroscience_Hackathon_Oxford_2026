"""Compare click models on a recording they were NOT trained on, under simulated weakness and
band rotation. Typical use: is `train --augment` actually better?

    python -m devtool.robustness_eval data/devtool/sessions/<id>_fatigue_raw.npz \\
        --models models/dylan models/dylan_aug

For each model: plain LDA and LDA + fatigue-adaptive gain, at strengths 100..12% (signal shrunk,
real noise kept, as in weakness_sim) and with the band rotated by one electrode either way.
Reports cued squeezes caught and false clicks per minute (outside cues + rest phase).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .fatigue import detect_offline_lda, gain_series
from .replay import load
from .replay_fatigue import load_models, score
from .weakness_sim import make_noise, noise_pool, simulate

LEVELS = (1.0, 0.5, 0.35, 0.25, 0.18, 0.12)
ROTATIONS = (0, -1, 1)


def evaluate(npz, model_dirs, levels=LEVELS, rotations=ROTATIONS, seed=0):
    bands, phases, _ev, settings, cals = load(npz)
    cues = json.loads(str(np.load(npz, allow_pickle=False)["cues"]))
    thr = float(settings["threshold"])
    rows = []
    for side, d in bands.items():
        v = d["valid"]
        emg, idx = d["emg"][v], d["idx"][v]
        ph = {name: i[side] for name, _t, i in phases}
        end = int(d["idx"][-1]) + 1
        rest_range = (ph.get("recovery", end), ph.get("post", end))
        lo, hi = (np.searchsorted(idx, r) for r in rest_range)
        noise = make_noise(noise_pool(emg, lo, hi), len(emg), np.random.default_rng(seed))
        cue_idx = np.array([c[2][side] for c in cues])
        minutes = len(emg) / 200 / 60
        cal = cals[side]
        models = {Path(mdir).name: load_models(mdir).get(side) for mdir in model_dirs}
        models = {k: v for k, v in models.items() if v is not None}
        for rot in rotations:
            for s in levels:
                x = simulate(emg, s, noise) if s < 1 else emg
                x = np.roll(x, rot, axis=1) if rot else x
                gain = gain_series(x, idx, cal, thr)          # slow part: once per condition
                for mname, m in models.items():
                    for name, g in (("lda", None), ("lda+adapt", gain)):
                        caught, false = score(detect_offline_lda(x, idx, m, g), cue_idx, rest_range)
                        rows.append(dict(band=side, model=mname, method=name, rotation=rot,
                                         strength=s, caught=round(100 * caught.mean(), 1),
                                         false_per_min=round(len(false) / minutes, 2)))
                print(f"  done rotation {rot:+d} strength {s:.0%}", flush=True)
    return rows


def print_table(rows):
    for (band, rot) in sorted({(r["band"], r["rotation"]) for r in rows}):
        print(f"\n[{band}] band rotation {rot:+d} electrode   caught % (false clicks / min)")
        sel = [r for r in rows if r["band"] == band and r["rotation"] == rot]
        cols = sorted({(r["model"], r["method"]) for r in sel})
        print("  strength  " + "  ".join(f"{m + ' ' + k:>22s}" for m, k in cols))
        for s in sorted({r["strength"] for r in sel}, reverse=True):
            cells = []
            for m, k in cols:
                r = next(x for x in sel if x["strength"] == s and x["model"] == m and x["method"] == k)
                cells.append(f"{r['caught']:>12.0f}% ({r['false_per_min']:4.1f})")
            print(f"  {s:7.0%}  " + "  ".join(f"{c:>22s}" for c in cells))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--models", nargs="+", required=True, help="model folders, e.g. models/dylan models/dylan_aug")
    a = ap.parse_args(argv)
    rows = evaluate(a.npz, a.models)
    print_table(rows)
    out = Path(a.npz).with_name(Path(a.npz).name.replace("_raw.npz", "_robustness.json"))
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

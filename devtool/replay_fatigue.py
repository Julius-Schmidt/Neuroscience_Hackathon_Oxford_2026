"""Fixed vs fatigue-adaptive click threshold on a recorded fatigue session (no bands needed).

    python -m devtool.replay_fatigue data/devtool/sessions/<id>_fatigue_raw.npz
    python -m devtool.replay_fatigue <file> --threshold 0.45 --refractory 250
    python -m devtool.replay_fatigue <file> --models models/emily     # + LDA, fixed vs adaptive

Record the session with dev tool mode 4 (fatigue protocol). This re-runs both detectors on the
SAME raw EMG, so the only difference is the detector. Scoring per cue:
    caught       a click within -0.3 s .. +1.5 s of the SQUEEZE cue
    missed       no click in that window
    false        a click outside every cue window (and every click during the rest phase)
With --models <dir> (holding fist_right.joblib / fist_left.joblib from `myomouse.train` for the
SAME person) it also runs the LDA fist detector, as-is and fatigue-adaptive (raw EMG rescaled by
the tracker's gain before features; the model is unchanged). Four conditions, same recording.
Output: a per-minute table, <id>_fatigue_replay.csv, and a chart <id>_fatigue_replay.png
(if matplotlib is installed): caught % per minute, fixed vs adaptive, plus tracked strength.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .emg import SAMPLE_MS, BandCal, detect_offline
from .fatigue import detect_offline_adaptive, detect_offline_lda, gain_series
from .replay import load

PRE = int(300 / SAMPLE_MS)
POST = int(1500 / SAMPLE_MS)


def score(clicks, cue_idx, rest_range):
    caught = np.array([np.any((clicks >= c - PRE) & (clicks <= c + POST)) for c in cue_idx], dtype=bool)
    in_cue = np.zeros(len(clicks), dtype=bool)
    for c in cue_idx:
        in_cue |= (clicks >= c - PRE) & (clicks <= c + POST)
    lo, hi = rest_range
    rest = (clicks >= lo) & (clicks < hi)
    false = (~in_cue) | rest
    return caught, clicks[false]


def load_models(models_dir):
    if not models_dir:
        return {}
    import joblib
    out = {}
    for side in ("right", "left"):
        f = Path(models_dir) / f"fist_{side}.joblib"
        if f.exists():
            out[side] = joblib.load(f)
    if not out:
        print(f"(no fist_right/left.joblib in {models_dir}: LDA skipped)")
    return out


def run(npz, threshold=None, refractory=None, out_dir=None, plot=True, models_dir=None):
    bands, phases, events, settings, cals = load(npz)
    z = np.load(npz, allow_pickle=False)
    if "cues" not in z.files:
        raise SystemExit("no cues in this file: record it with dev tool mode 4 (fatigue protocol)")
    cues = json.loads(str(z["cues"]))
    thr = float(threshold if threshold is not None else settings["threshold"])
    ref = float(refractory if refractory is not None else settings["refractory_ms"])
    stem = Path(npz).name.replace("_raw.npz", "")
    out_dir = Path(out_dir or Path(npz).parent)
    models = load_models(models_dir)
    rows, results = [], {}
    for side, d in bands.items():
        cal = cals.get(side, BandCal())
        v = d["valid"]
        emg, idx = d["emg"][v], d["idx"][v]
        end = int(d["idx"][-1]) + 1 if len(d["idx"]) else 0
        ph = {name: i[side] for name, _t, i in phases}
        rest_range = (ph.get("recovery", end), ph.get("post", end))
        cue_idx = np.array([c[2][side] for c in cues])
        cue_t = np.array([c[1] for c in cues])
        cue_phase = [c[0] for c in cues]
        fixed = detect_offline(emg, idx, cal, thr, ref)
        adapt, trace = detect_offline_adaptive(emg, idx, cal, thr, ref)
        cf, ff = score(fixed, cue_idx, rest_range)
        ca, fa = score(adapt, cue_idx, rest_range)
        lda = None
        if side in models:
            g = gain_series(emg, idx, cal, thr)
            cl, fl = score(detect_offline_lda(emg, idx, models[side]), cue_idx, rest_range)
            cla, fla = score(detect_offline_lda(emg, idx, models[side], g), cue_idx, rest_range)
            lda = dict(cl=cl, fl=fl, cla=cla, fla=fla)
        t0 = cue_t[0] if len(cue_t) else 0.0
        minute = np.floor((cue_t - t0) / 60).astype(int)
        results[side] = dict(cal=cal, cue_t=cue_t - t0, cue_phase=cue_phase, cf=cf, ca=ca, ff=ff, fa=fa,
                             trace=trace, idx0=cue_idx[0] if len(cue_idx) else 0, lda=lda)
        print(f"\n{side.upper()} band   threshold {thr}  refractory {ref:.0f} ms"
              + ("" if cal.calibrated else "   (UNCALIBRATED in this session)"))
        print(f"{'minute':>6s} {'phase':>8s} {'cues':>5s} {'fixed caught':>13s} {'adaptive caught':>16s}"
              + (f" {'LDA caught':>13s} {'LDA adaptive':>14s}" if lda else ""))
        for m in sorted(set(minute)):
            sel = minute == m
            phs = sorted({cue_phase[i] for i in np.where(sel)[0]})
            n = int(sel.sum())
            row = dict(band=side, minute=int(m), phase="+".join(phs), cues=n,
                       fixed_caught=int(cf[sel].sum()), adaptive_caught=int(ca[sel].sum()))
            extra = ""
            if lda:
                row.update(lda_caught=int(lda["cl"][sel].sum()), lda_adaptive_caught=int(lda["cla"][sel].sum()))
                extra = (f" {row['lda_caught']:>8d} ({row['lda_caught'] / n:4.0%})"
                         f" {row['lda_adaptive_caught']:>9d} ({row['lda_adaptive_caught'] / n:4.0%})")
            rows.append(row)
            print(f"{m:6d} {row['phase']:>8s} {n:5d} {row['fixed_caught']:>8d} ({row['fixed_caught'] / n:4.0%})"
                  f" {row['adaptive_caught']:>10d} ({row['adaptive_caught'] / n:4.0%})" + extra)
        n_rest_s = (rest_range[1] - rest_range[0]) * SAMPLE_MS / 1000
        print(f"total caught: fixed {cf.sum()}/{len(cf)}  adaptive {ca.sum()}/{len(ca)}")
        print(f"false clicks: fixed {len(ff)}  adaptive {len(fa)}   (rest phase {n_rest_s:.0f} s + clicks off-cue)")
        if lda:
            print(f"LDA:  caught {lda['cl'].sum()}/{len(cf)}  adaptive {lda['cla'].sum()}/{len(cf)}   "
                  f"false clicks {len(lda['fl'])} / {len(lda['fla'])}")
        if trace:
            print(f"tracked squeeze strength at the end: {trace[-1][3]:.0%} of start "
                  f"(threshold {trace[-1][2]:.2f})")
    csv_path = out_dir / f"{stem}_fatigue_replay.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["band"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {csv_path}")
    if plot:
        try:
            png = _plot(results, out_dir / f"{stem}_fatigue_replay.png", thr)
            if png:
                print(f"saved {png}")
        except Exception as e:  # never lose the table because the chart failed
            print(f"(chart failed: {e})")
    return results


def _plot(results, path, thr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not installed: no chart; pip install matplotlib)")
        return None
    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 6), squeeze=False, sharex=True)
    for k, (side, r) in enumerate(results.items()):
        t = r["cue_t"] / 60
        win = 10

        def roll(x):  # centred moving average that doesn't sag at the edges
            x = x.astype(float)
            k = np.ones(max(1, min(win, len(x))))
            return np.convolve(x, k, mode="same") / np.convolve(np.ones_like(x), k, mode="same")

        ph = np.array(r["cue_phase"])
        ax = axes[0][k]
        for name, ls in (("fatigue", "-"), ("post", "-")):
            m = ph == name
            if m.any():
                ax.plot(t[m], roll(r["cf"][m]) * 100, color="#888888", linestyle=ls,
                        label="fixed threshold" if name == "fatigue" else None)
                ax.plot(t[m], roll(r["ca"][m]) * 100, color="#1f6feb", linestyle=ls,
                        label="fatigue-adaptive" if name == "fatigue" else None)
                if r.get("lda"):
                    ax.plot(t[m], roll(r["lda"]["cl"][m]) * 100, color="#888888", linestyle=":",
                            label="LDA" if name == "fatigue" else None)
                    ax.plot(t[m], roll(r["lda"]["cla"][m]) * 100, color="#1f6feb", linestyle=":",
                            label="LDA, fatigue-adaptive" if name == "fatigue" else None)
        ax.set_ylim(0, 105)
        ax.set_ylabel("cued squeezes caught (%)")
        ax.set_title(f"{side} band")
        ax.legend(loc="lower left", frameon=False)
        ax2 = axes[1][k]
        if r["trace"]:
            ti = np.array([(x[0] - r["idx0"]) * SAMPLE_MS / 60000 for x in r["trace"]])
            ax2.plot(ti, [x[3] * 100 for x in r["trace"]], color="#1f6feb",
                     label="tracked squeeze strength (adaptive threshold scales with it)")
        ax2.axhline(100, color="#888888", linewidth=0.8)
        ax2.set_ylabel("% of starting strength")
        ax2.set_xlabel("minutes since first cue")
        ax2.legend(loc="upper right", frameon=False)
        if (ph == "fatigue").any() and (ph == "post").any():
            a0, a1 = t[ph == "fatigue"].max(), t[ph == "post"].min()
            for a in (ax, ax2):
                a.axvspan(a0, a1, color="#eeeeee", zorder=0)
            ax.text((a0 + a1) / 2, 50, "rest", ha="center", color="#888888")
        for a in (ax, ax2):
            for p in ("top", "right"):
                a.spines[p].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--threshold", type=float, default=None, help="default: the session's value")
    ap.add_argument("--refractory", type=float, default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--models", default=None,
                    help="folder with fist_right.joblib / fist_left.joblib (myomouse.train) for this person")
    a = ap.parse_args(argv)
    run(a.npz, a.threshold, a.refractory, plot=not a.no_plot, models_dir=a.models)


if __name__ == "__main__":
    main()

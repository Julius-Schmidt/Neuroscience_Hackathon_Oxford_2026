"""Which EMG feature tracks fist fatigue best? Offline, from a dev tool mode 4 recording.

    python -m devtool.fatigue_features data/devtool/sessions/<id>_fatigue_raw.npz

For every cued squeeze it finds the squeeze itself (where the activity is above half its
peak, within 1.5 s of the cue) and computes, on the most active channels:

    rms, mav      amplitude. Fatigue at a FIXED effort -> up; but people also squeeze softer
                  when tired, so amplitude mixes fatigue with effort.
    mnf, mdf      mean / median frequency. Fatigue -> down (spectrum shifts to low frequencies).
    dimitrov      FInsm5 = sum(P/f) / sum(P f^5). Fatigue -> up; reported as the most sensitive [10].
    low_frac      share of power in 15-45 Hz (tracks the median-frequency shift; Allison &
                  Fujiwara 2002 [11]) - fully inside the Myo's 0-100 Hz band. Fatigue -> up.
    zc            zero crossings per second, a time-domain frequency proxy. Fatigue -> down.

The Myo samples at 200 Hz, so everything above 100 Hz is invisible (and partly aliased):
the frequency features see only part of the shift. Treat them as candidates to test.

Ranking: each feature is normalised to its first 5 squeezes (= 100 %). The score is the
effect size between the first and last fifth of the fatigue phase (Cohen's d, sign flipped
so "moves the way fatigue should" is positive), plus whether it moves back during recovery.

Outputs <id>_features.csv (one row per squeeze) and <id>_features.png.
[n] = docs/REFERENCES.md.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .emg import EMG_HZ
from .replay import load

WIN_S = 1.5            # look for the squeeze this long after each cue
TOP_CH = 3             # use the channels that are most active during squeezes
NFFT = 256
EXPECT = {"rms": +1, "mav": +1, "mnf": -1, "mdf": -1, "dimitrov": +1, "low_frac": +1, "zc": -1}


def spectrum_features(x):
    """x: (n, ch) float, one squeeze. Returns dict of features averaged over channels."""
    x = x - x.mean(axis=0)
    n = len(x)
    w = np.hanning(n)[:, None]
    P = np.abs(np.fft.rfft(x * w, n=max(NFFT, n), axis=0)) ** 2           # (bins, ch)
    f = np.fft.rfftfreq(max(NFFT, n), 1 / EMG_HZ)
    keep = f >= 5                                                       # drop DC / motion
    P, f = P[keep], f[keep]
    tot = P.sum(axis=0) + 1e-12
    mnf = (f[:, None] * P).sum(axis=0) / tot
    cum = np.cumsum(P, axis=0) / tot
    mdf = f[np.argmax(cum >= 0.5, axis=0)]
    dim = (P / f[:, None]).sum(axis=0) / ((P * f[:, None] ** 5).sum(axis=0) + 1e-12)
    low = P[(f >= 15) & (f <= 45)].sum(axis=0) / tot
    zc = (np.diff(np.signbit(x), axis=0) != 0).sum(axis=0) / (n / EMG_HZ)
    return dict(rms=float(np.sqrt((x ** 2).mean())), mav=float(np.abs(x).mean()),
                mnf=float(mnf.mean()), mdf=float(mdf.mean()), dimitrov=float(dim.mean() * 1e10),
                low_frac=float(low.mean()), zc=float(zc.mean()))


def squeeze_segment(env, start, stop, min_len=20):
    """Indices [a, b) of the squeeze inside env[start:stop]: above half its peak, around the peak."""
    seg = env[start:stop]
    if len(seg) < min_len:
        return None
    k = int(np.argmax(seg))
    on = seg >= 0.5 * seg[k]
    a = k
    while a > 0 and on[a - 1]:
        a -= 1
    b = k
    while b < len(seg) and on[b]:
        b += 1
    return (start + a, start + b) if b - a >= min_len else None


def analyse(npz):
    bands, phases, _events, _settings, _cals = load(npz)
    z = np.load(npz, allow_pickle=False)
    if "cues" not in z.files:
        raise SystemExit("no cues in this file: record it with dev tool mode 4 (fatigue protocol)")
    cues = json.loads(str(z["cues"]))
    out = {}
    for side, d in bands.items():
        start = int(d["idx"][0])
        emg = d["emg"].astype(float)
        emg[~d["valid"]] = 0.0
        env = np.convolve(np.abs(emg).mean(axis=1), np.ones(10) / 10, mode="same")   # 50 ms
        segs = []
        for ph, t, idx in cues:
            i0 = int(idx[side]) - start
            s = squeeze_segment(env, max(i0, 0), min(i0 + int(WIN_S * EMG_HZ), len(env)))
            if s:
                segs.append((ph, t, s))
        if not segs:
            continue
        # most active channels across all squeezes
        act = np.mean([np.abs(emg[a:b]).mean(axis=0) for _p, _t, (a, b) in segs], axis=0)
        ch = np.argsort(act)[::-1][:TOP_CH]
        t0 = segs[0][1]
        rows = []
        for ph, t, (a, b) in segs:
            feats = spectrum_features(emg[a:b][:, ch])
            rows.append(dict(band=side, phase=ph, t_s=round(t - t0, 2), dur_ms=(b - a) * 1000 // EMG_HZ, **feats))
        out[side] = dict(rows=rows, channels=[int(c) for c in ch])
    return out


def rank(rows):
    """Per feature: % change over the fatigue phase, effect size, recovery."""
    fat = [r for r in rows if r["phase"] == "fatigue"]
    after = [r for r in rows if r["phase"] in ("post", "recovery")]
    res = {}
    if len(fat) < 10:
        return res
    k = max(3, len(fat) // 5)
    for name, sign in EXPECT.items():
        v = np.array([r[name] for r in fat])
        base = np.median(v[:5])
        early, late = v[:k], v[-k:]
        sd = np.sqrt((early.var(ddof=1) + late.var(ddof=1)) / 2) + 1e-12
        d = sign * (late.mean() - early.mean()) / sd
        change = 100 * (late.mean() / base - 1) if base else float("nan")
        rec = None
        if after:
            pv = np.mean([r[name] for r in after])
            span = late.mean() - early.mean()
            rec = 100 * (late.mean() - pv) / span if abs(span) > 1e-12 else None   # % of the shift undone
        res[name] = dict(change_pct=change, d=d, recovered_pct=rec)
    return res


def run(npz, plot=True):
    data = analyse(npz)
    stem = Path(npz).name.replace("_raw.npz", "")
    out_dir = Path(npz).parent
    allrows = [r for d in data.values() for r in d["rows"]]
    if not allrows:
        raise SystemExit("no squeezes found after the cues")
    csv_path = out_dir / f"{stem}_features.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    ranks = {}
    for side, d in data.items():
        r = rank(d["rows"])
        ranks[side] = r
        n_f = sum(1 for x in d["rows"] if x["phase"] == "fatigue")
        print(f"\n[{side}] {len(d['rows'])} squeezes ({n_f} in the fatigue phase), channels {d['channels']}")
        if not r:
            print("  too few squeezes in the fatigue phase to rank (need 10)")
            continue
        print(f"  {'feature':10s} {'fatigue change':>15s} {'effect size':>12s} {'recovered':>10s}   (effect > 0 = moves the expected way)")
        for name, v in sorted(r.items(), key=lambda kv: -kv[1]["d"]):
            rec = "-" if v["recovered_pct"] is None else f"{v['recovered_pct']:.0f}%"
            print(f"  {name:10s} {v['change_pct']:+14.1f}% {v['d']:+12.2f} {rec:>10s}")
    print(f"\nsaved {csv_path}")
    if plot:
        try:
            _plot(data, out_dir / f"{stem}_features.png")
        except Exception as e:  # never lose the numbers to a plotting problem
            print(f"(plot skipped: {e})")
    return data, ranks


def _plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(EXPECT)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.8 * len(names)), sharex=True)
    for side, d in data.items():
        rows = d["rows"]
        t = np.array([r["t_s"] for r in rows]) / 60
        for ax, name in zip(axes, names):
            v = np.array([r[name] for r in rows])
            base = np.median(v[:5]) or 1.0
            y = 100 * v / base
            ax.plot(t, y, ".", alpha=0.35, label=side)
            k = min(7, len(y))
            sm = np.convolve(y, np.ones(k), "same") / np.convolve(np.ones_like(y), np.ones(k), "same")
            ax.plot(t, sm, "-", lw=2)
            ax.set_ylabel(f"{name}\n% of start", fontsize=8)
            ax.axhline(100, color="grey", lw=0.5)
    for ax in axes:
        rows = next(iter(data.values()))["rows"]
        for ph, col in (("recovery", "#d8ecff"), ("post", "#ffe9cc")):
            ts = [r["t_s"] / 60 for r in rows if r["phase"] == ph]
            if ts:
                ax.axvspan(min(ts), max(ts), color=col, zorder=0)
    axes[-1].set_xlabel("minutes (blue = recovery, orange = post)")
    axes[0].set_title("Fatigue features per squeeze (normalised to the first 5)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    run(a.npz, plot=not a.no_plot)


if __name__ == "__main__":
    main()

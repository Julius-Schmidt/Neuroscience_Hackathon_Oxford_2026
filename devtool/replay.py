"""Re-run the click detector over a recorded session with other settings (no bands needed).

    python -m devtool.replay data/devtool/sessions/<id>_raw.npz
    python -m devtool.replay <id>_raw.npz --thresholds 0.25,0.35,0.45 --refractory 150,200,250,300,350,400

For each setting it reports, from the same raw EMG:
  false/min still, false/min move   clicks during the two rest phases
  task clicks                      clicks during the target phase
  hits kept                        live hits that would still fire within +-100 ms
  doubles                          task clicks < 500 ms after the previous one (bounces)
Hit/miss for new clicks can't be known (the cursor would have been elsewhere), so "hits kept"
and "doubles" are the proxies for what changing refractory/threshold does to real clicks.
"""

import argparse
import json

import numpy as np

from .emg import SAMPLE_MS, BandCal, detect_offline


def load(path):
    z = np.load(path, allow_pickle=False)
    phases = json.loads(str(z["phases"]))
    events = json.loads(str(z["events"]))
    settings = json.loads(str(z["settings"]))
    cals = {s: BandCal(**c) for s, c in json.loads(str(z["cal"])).items()}
    bands = {}
    for key in z.files:
        if key.endswith("_emg"):
            s = key[:-4]
            start = int(z[f"{s}_start_idx"])
            bands[s] = {"emg": z[key], "valid": z[f"{s}_valid"],
                        "idx": start + np.arange(len(z[key]))}
    return bands, phases, events, settings, cals


def phase_ranges(phases, side, end):
    out = {}
    for k, (name, _t, idx) in enumerate(phases):
        a = idx[side]
        b = phases[k + 1][2][side] if k + 1 < len(phases) else end
        out[name] = (a, b)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--thresholds", default=None, help="comma list; default = the session's value")
    ap.add_argument("--refractory", default="150,200,250,300,350,400")
    a = ap.parse_args(argv)
    bands, phases, events, settings, cals = load(a.npz)
    thrs = [float(v) for v in (a.thresholds or str(settings["threshold"])).split(",")]
    refs = [float(v) for v in a.refractory.split(",")]
    print(f"session settings: threshold {settings['threshold']}  refractory {settings['refractory_ms']} ms")
    head = f"{'band':6s} {'thr':>5s} {'refr':>5s} {'false/min still':>16s} {'false/min move':>15s} " \
           f"{'task clicks':>12s} {'hits kept':>10s} {'doubles':>8s}"
    print(head)
    print("-" * len(head))
    for s, d in bands.items():
        cal = cals.get(s, BandCal())
        if not cal.calibrated:
            print(f"{s}: note, band was uncalibrated in this session")
        v = d["valid"]
        emg, idx = d["emg"][v], d["idx"][v]
        end = int(d["idx"][-1]) + 1 if len(d["idx"]) else 0
        pr = phase_ranges(phases, s, end)
        live_hits = np.array([e["detect_idx"] for e in events
                              if e["event"] == "click" and e["band"] == s and e["outcome"] == "hit"])
        for thr in thrs:
            for ref in refs:
                c = detect_offline(emg, idx, cal, thr, ref)

                def in_phase(name):
                    if name not in pr:
                        return c[:0], 0.0
                    lo, hi = pr[name]
                    return c[(c >= lo) & (c < hi)], (hi - lo) * SAMPLE_MS / 1000

                still, still_s = in_phase("rest_still")
                move, move_s = in_phase("rest_move")
                task, _ = in_phase("targets")
                kept = sum(np.any(np.abs(task - h) <= 20) for h in live_hits) if len(live_hits) else 0
                doubles = int(np.sum(np.diff(task) * SAMPLE_MS < 500)) if len(task) > 1 else 0
                fs = f"{len(still) / still_s * 60:.1f}" if still_s else "-"
                fm = f"{len(move) / move_s * 60:.1f}" if move_s else "-"
                hk = f"{kept}/{len(live_hits)}" if len(live_hits) else "-"
                print(f"{s:6s} {thr:5.2f} {ref:5.0f} {fs:>16s} {fm:>15s} {len(task):>12d} {hk:>10s} {doubles:>8d}")


if __name__ == "__main__":
    main()

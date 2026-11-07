"""Held-out evaluation for the slides.

    python -m eeg_switch.evaluate                     tune on DEV, report on TEST (config.py)
    python -m eeg_switch.evaluate --quick             10 dev + 10 test subjects (~2 min)
    python -m eeg_switch.evaluate --synthetic         pipeline check only, no download

1. Tune: grid over feature (rel/abs) x ON_DB on DEV subjects. Pick the setting
   with the fewest missed closures, subject to false toggles <= --max-false per min.
2. Report: run that fixed setting once on TEST subjects (never seen while tuning).

Writes research/eeg_switch_results/{tuning.csv, test_per_subject.csv, summary.json}.
Each subject gets 3 random scenarios (4 long + 4 short closures each).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from . import config as C
from .detector import Params, run_offline
from .metrics import score, summarise

GRID_FEATURES = ["rel", "abs"]
GRID_ON_DB = [2.0, 3.0, 4.0, 5.0, 6.0]


def scenarios_for(subjects, n_rep, seed, synthetic=False):
    from .data import load_subject, make_scenario
    from .synthetic import synthetic_subject
    out = []
    for s in subjects:
        if s in C.EXCLUDE_SUBJECTS:
            continue
        try:
            sub = synthetic_subject(s) if synthetic else load_subject(s)
        except Exception as e:                          # missing file, bad EDF ...
            print(f"  skip subject {s}: {e}")
            continue
        rng = np.random.default_rng(seed * 1000 + s)
        out += [make_scenario(sub, rng) for _ in range(n_rep)]
    return out


def evaluate(scens, params: Params):
    rows = []
    for sc in scens:
        toggles, _ = run_offline(sc.eeg, params)
        rows.append(score(sc, toggles))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 dev + 10 test subjects")
    ap.add_argument("--synthetic", action="store_true", help="no real data: pipeline check only")
    ap.add_argument("--reps", type=int, default=3, help="scenarios per subject")
    ap.add_argument("--max-false", type=float, default=0.2, help="allowed false toggles per minute when tuning")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev, test = C.DEV_SUBJECTS, C.TEST_SUBJECTS
    if a.quick or a.synthetic:
        dev, test = dev[:10], test[:10]
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_synthetic" if a.synthetic else ""

    print(f"loading {len(dev)} dev subjects ...")
    dev_sc = scenarios_for(dev, a.reps, a.seed, a.synthetic)
    print(f"loading {len(test)} test subjects ...")
    test_sc = scenarios_for(test, a.reps, a.seed + 1, a.synthetic)

    base = Params()
    grid = []
    for feat in GRID_FEATURES:
        for on in GRID_ON_DB:
            p = replace(base, feature=feat, on_db=on, off_db=on - 2.0)
            s = summarise(evaluate(dev_sc, p))
            grid.append({"feature": feat, "on_db": on, "off_db": on - 2.0, **s})
            print(f"  dev  {feat:3s} on={on:3.1f}dB  missed={s['missed_rate']:.1%}  "
                  f"false/min={s['false_per_min']:.2f}  delay={s['delay_median_s']:.2f}s")
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(C.RESULTS_DIR / f"tuning{tag}.csv", index=False)

    ok = grid_df[grid_df["false_per_min"] <= a.max_false]
    pool = ok if len(ok) else grid_df
    best = pool.sort_values(["missed_rate", "false_per_min", "delay_median_s"]).iloc[0]
    chosen = replace(base, feature=best["feature"], on_db=float(best["on_db"]), off_db=float(best["off_db"]))
    print(f"\nchosen on DEV: feature={chosen.feature} ON_DB={chosen.on_db} OFF_DB={chosen.off_db}"
          + ("" if len(ok) else f"  (nothing met <= {a.max_false} false/min)"))

    rows = evaluate(test_sc, chosen)
    test_sum = summarise(rows)
    pd.DataFrame([{k: v for k, v in r.items() if k != "delays"} |
                  {"delay_median_s": float(np.median(r["delays"])) if r["delays"] else np.nan}
                  for r in rows]).to_csv(C.RESULTS_DIR / f"test_per_subject{tag}.csv", index=False)

    summary = {"data": "synthetic (NOT evidence)" if a.synthetic else "PhysioNet eegmmidb runs 1-2",
               "dev_subjects": sorted({sc.subject for sc in dev_sc}),
               "test_subjects": sorted({sc.subject for sc in test_sc}),
               "params": asdict(chosen), "test": test_sum}
    (C.RESULTS_DIR / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))

    t = test_sum
    print("\n=== HELD-OUT TEST ===")
    print(f"subjects: {t['subjects'] // a.reps}  closures (>=3 s): {t['closures']}  minutes: {t['minutes']:.1f}")
    print(f"missed toggles:      {t['missed_rate']:.1%}")
    print(f"false toggles / min: {t['false_per_min']:.2f}")
    print(f"detection delay:     median {t['delay_median_s']:.2f} s, 90th pct {t['delay_p90_s']:.2f} s "
          f"(includes the {chosen.dwell_s:.0f} s dwell)")
    print(f"results in {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()

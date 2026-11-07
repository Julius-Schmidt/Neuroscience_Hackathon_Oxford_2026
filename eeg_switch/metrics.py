"""Score toggles against a scenario's ground truth.

    hit         first toggle inside a long closure (from its start to 1.5 s after it ends)
    missed      a long closure with no toggle
    false       any other toggle (during open eyes, short closures, or a second toggle)
    delay       hit time - closure start (includes the 2 s dwell by design)
    false/min   false toggles per minute of post-baseline recording
"""

from __future__ import annotations

import numpy as np

from .data import Scenario

LATE_S = 1.5


def score(scenario: Scenario, toggles: list[tuple[float, float]]) -> dict:
    times = [t for t, _ in toggles]
    long_iv = [(a, b) for a, b, should in scenario.closed_intervals if should]
    used = set()
    delays = []
    for a, b in long_iv:
        cands = [i for i, t in enumerate(times) if a <= t <= b + LATE_S and i not in used]
        if cands:
            used.add(cands[0])
            delays.append(times[cands[0]] - a)
    n_false = len(times) - len(used)
    minutes = (len(scenario.eeg) / scenario.srate - scenario.baseline_s) / 60.0
    return {
        "subject": scenario.subject,
        "n_long": len(long_iv),
        "hits": len(delays),
        "missed": len(long_iv) - len(delays),
        "false": n_false,
        "minutes": minutes,
        "delays": delays,
    }


def summarise(rows: list[dict]) -> dict:
    n_long = sum(r["n_long"] for r in rows)
    missed = sum(r["missed"] for r in rows)
    false = sum(r["false"] for r in rows)
    minutes = sum(r["minutes"] for r in rows)
    delays = [d for r in rows for d in r["delays"]]
    per_subj_miss = [r["missed"] / max(1, r["n_long"]) for r in rows]
    return {
        "subjects": len(rows),
        "closures": n_long,
        "missed_rate": missed / max(1, n_long),
        "false_per_min": false / max(1e-9, minutes),
        "delay_median_s": float(np.median(delays)) if delays else float("nan"),
        "delay_p90_s": float(np.percentile(delays, 90)) if delays else float("nan"),
        "subjects_all_missed": int(sum(m == 1.0 for m in per_subj_miss)),
        "minutes": minutes,
    }

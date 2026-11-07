"""Summarise / compare click-target sessions from their CSVs.

    python -m devtool.summary                       # every session in data/devtool/sessions
    python -m devtool.summary path/to/session.csv   # one or more files
    python -m devtool.summary --user dylan          # only one user's sessions
"""

import argparse
from pathlib import Path

from .session import format_summary, load_events, summarize


def main(argv=None):
    from .app import DATA_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--user")
    ap.add_argument("--detail", action="store_true", help="full summary per session, not just the table")
    a = ap.parse_args(argv)
    files = [Path(p) for p in a.csv] or sorted((DATA_DIR / "sessions").glob("*.csv"))
    rows = []
    for f in files:
        s = summarize(load_events(f))
        if "rest" not in s or (a.user and s["user"] != a.user):
            continue
        if a.detail:
            print(format_summary(s), "\n")
        st, t, r = s["settings"], s["targets"], s["rest"]
        lat, tt = s["click_latency_ms"], s["time_to_target_s"]
        rows.append([s["session_id"], st["threshold"], st["refractory_ms"], st["smoothing"], st["cursor_source"],
                     f"{t['hit_rate'] * 100:.0f}%" if t["hit_rate"] is not None else "-",
                     f"{t['hits']}/{t['clicks']}", r["rest_still"]["false_per_min"], r["rest_move"]["false_per_min"],
                     tt["median"], lat["onset_to_detect_median"], lat["total_median"]])
    if not rows:
        print("no sessions found")
        return
    head = ["session", "thr", "refr", "smooth", "cursor", "hit%", "hits/clk", "false/min still",
            "false/min move", "t2target med s", "onset->det ms", "latency ms"]
    table = [head] + [[("-" if v is None else str(v)) for v in r] for r in rows]
    w = [max(len(r[i]) for r in table) for i in range(len(head))]
    for k, r in enumerate(table):
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(r)))
        if k == 0:
            print("  ".join("-" * x for x in w))


if __name__ == "__main__":
    main()

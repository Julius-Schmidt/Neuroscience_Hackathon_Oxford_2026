"""Click-target session logging: one long-format CSV per session, plus summary + raw stream.

CSV: one row per EMG sample in the window around each click (raw int8, every sample of every
BLE packet, with packet index and slot 0/1). Click metadata is repeated on each of its rows, so
`df.groupby("click_id").first()` gives one row per click. Targets that time out get a single
row with event=timeout and no EMG.

Files written to data/devtool/sessions/:
    <id>.csv            the session (the main output)
    <id>_summary.json   summary + data quality (also printed and shown on screen)
    <id>_raw.npz        full raw EMG for the session, for `python -m devtool.replay`
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .emg import SAMPLE_MS, find_onset, pct

META_COLS = [
    "session_id", "user", "event", "click_id", "phase", "band", "t_session_s", "t_wall",
    "outcome", "detect_idx", "onset_idx", "onset_to_detect_ms", "delivery_ms", "latency_ms",
    "target_id", "target_band", "target_x", "target_y", "target_r", "time_to_target_s",
    "time_to_enter_s", "cursor_x", "cursor_y", "cursor_x_onset", "cursor_y_onset",
    "inside_at_onset", "threshold", "refractory_ms", "smoothing", "cursor_source",
    "rest_still_s", "rest_move_s",
]
SAMPLE_COLS = ["sample_offset", "sample_idx", "t_sample_ms", "packet_idx", "slot", "valid",
               "activation"] + [f"ch{i}" for i in range(8)]
COLUMNS = META_COLS + SAMPLE_COLS


class Session:
    def __init__(self, out_dir: Path, user: str, bands: dict, settings: dict,
                 rest_still_s: float, rest_move_s: float, pre_ms=400, post_ms=200):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.id = time.strftime("%Y%m%d_%H%M%S") + f"_{user}"
        self.user, self.bands, self.settings = user, bands, dict(settings)
        self.rest_still_s, self.rest_move_s = rest_still_s, rest_move_s
        self.pre = int(round(pre_ms / SAMPLE_MS))
        self.post = int(round(post_ms / SAMPLE_MS))
        self.t_start = time.perf_counter()
        self.start_idx = {s: b.n for s, b in bands.items()}
        self.start_drops = {s: b.drops for s, b in bands.items()}
        self.start_ooo = {s: b.out_of_order for s, b in bands.items()}
        self.phases = []          # (name, t_session_s, {side: idx})
        self.pending = []
        self.events = []          # metadata of every click / timeout (for summary + replay)
        self.n_clicks = 0
        self.cues = []            # fatigue protocol: (phase, t_session_s, {side: sample idx})
        self.csv_path = self.out_dir / f"{self.id}.csv"
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)

    def t(self, host_time=None):
        return (host_time if host_time is not None else time.perf_counter()) - self.t_start

    def mark_phase(self, name):
        self.phases.append((name, self.t(), {s: b.n for s, b in self.bands.items()}))

    def mark_cue(self, phase):
        self.cues.append((phase, round(self.t(), 4), {s: b.n for s, b in self.bands.items()}))

    def _base(self, **kw):
        m = {c: "" for c in META_COLS}
        m.update(session_id=self.id, user=self.user, rest_still_s=self.rest_still_s,
                 rest_move_s=self.rest_move_s, t_wall=time.strftime("%H:%M:%S"),
                 threshold=self.settings["threshold"], refractory_ms=self.settings["refractory_ms"],
                 smoothing=self.settings["smoothing"], cursor_source=self.settings["cursor_source"])
        m.update(kw)
        return m

    def add_click(self, side, detect_idx, host_time, phase, outcome, cursor, cursor_at, target=None,
                  target_band="", t_target_shown=None, t_entered=None):
        """cursor_at(host_time) -> (x, y) from the cursor history, for the position at EMG onset."""
        b = self.bands[side]
        thr = self.settings["threshold"]
        onset = find_onset(b.fact, detect_idx, thr)
        t_det_host = b.host_time_of(detect_idx)
        delivery = (host_time - t_det_host) * 1000 if t_det_host is not None else float("nan")
        o2d = (detect_idx - onset) * SAMPLE_MS
        t_onset_host = b.host_time_of(onset)
        cxo, cyo = cursor_at(t_onset_host) if t_onset_host is not None else cursor
        inside_onset = ""
        if target is not None:
            tx, ty, r = target[:3]
            inside_onset = int((cxo - tx) ** 2 + (cyo - ty) ** 2 <= r * r)
        self.n_clicks += 1
        m = self._base(
            event="click", click_id=self.n_clicks, phase=phase, band=side,
            t_session_s=round(self.t(host_time), 4), outcome=outcome, detect_idx=detect_idx,
            onset_idx=onset, onset_to_detect_ms=round(o2d, 1), delivery_ms=round(delivery, 2),
            latency_ms=round(o2d + delivery, 2), cursor_x=round(cursor[0], 1),
            cursor_y=round(cursor[1], 1), cursor_x_onset=round(cxo, 1), cursor_y_onset=round(cyo, 1),
            inside_at_onset=inside_onset)
        if target is not None:
            tx, ty, r, tid = target
            m.update(target_id=tid, target_band=target_band, target_x=round(tx, 1),
                     target_y=round(ty, 1), target_r=r)
            if outcome == "hit":
                m["time_to_target_s"] = round(host_time - t_target_shown, 4)
            if t_entered is not None:
                m["time_to_enter_s"] = round(t_entered - t_target_shown, 4)
        self.events.append(m)
        self.pending.append((m, side, detect_idx, time.perf_counter()))
        return m

    def add_timeout(self, target, target_band, t_shown, t_entered):
        tx, ty, r, tid = target
        m = self._base(event="timeout", phase="targets", outcome="timeout", target_id=tid,
                       target_band=target_band, target_x=round(tx, 1), target_y=round(ty, 1),
                       target_r=r, t_session_s=round(self.t(), 4),
                       time_to_enter_s=round(t_entered - t_shown, 4) if t_entered else "")
        self.events.append(m)
        self._w.writerow([m[c] for c in META_COLS] + [""] * len(SAMPLE_COLS))
        self._f.flush()

    def poll(self, force=False):
        """Write clicks whose post-click window has arrived (or after 2 s, with what exists)."""
        keep = []
        for item in self.pending:
            m, side, di, t_add = item
            b = self.bands[side]
            if force or b.n > di + self.post or time.perf_counter() - t_add > 2.0:
                self._write_click(m, b, di)
            else:
                keep.append(item)
        self.pending = keep
        if not keep:
            self._f.flush()

    def _write_click(self, m, b, di):
        head = [m[c] for c in META_COLS]
        for off in range(-self.pre, self.post + 1):
            i = di + off
            if 0 <= i < b.n:
                v = bool(b.valid[i])
                row = [off, i, round(i * SAMPLE_MS, 1), int(b.pkt[i]) if v else "",
                       int(b.slot[i]) if v else "", int(v), round(float(b.act[i]), 4),
                       *[int(x) for x in b.emg[i]]]
            else:
                row = [off, i, round(i * SAMPLE_MS, 1), "", "", 0, "", *[""] * 8]
            self._w.writerow(head + row)

    def close(self, cals: dict):
        """Flush, save raw stream + summary. Returns (summary dict, text)."""
        self.poll(force=True)
        self._f.close()
        end_idx = {s: b.n for s, b in self.bands.items()}
        duration = self.t()
        quality = {}
        raw = {}
        for s, b in self.bands.items():
            a, z = self.start_idx[s], end_idx[s]
            n_valid = int(b.valid[a:z].sum())
            quality[s] = {"samples": n_valid, "samples_per_s": round(n_valid / max(duration, 1e-6), 1),
                          "dropped_packets": b.drops - self.start_drops[s]}
            for k in ("emg", "valid", "pkt", "slot", "arrival", "act"):
                raw[f"{s}_{k}"] = getattr(b, k)[a:z]
            raw[f"{s}_start_idx"] = np.int64(a)
        raw["phases"] = np.array(json.dumps(self.phases))
        raw["cues"] = np.array(json.dumps(self.cues))
        raw["events"] = np.array(json.dumps(self.events))
        raw["settings"] = np.array(json.dumps(self.settings))
        raw["cal"] = np.array(json.dumps({s: asdict(c) for s, c in cals.items()}))
        np.savez_compressed(self.out_dir / f"{self.id}_raw.npz", **raw)

        summary = summarize(self.events)
        summary["quality"] = quality
        summary["duration_s"] = round(duration, 1)
        summary["csv"] = str(self.csv_path)
        (self.out_dir / f"{self.id}_summary.json").write_text(json.dumps(summary, indent=2))
        return summary, format_summary(summary)


# --------------------------------------------------------------------------- summary
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_events(csv_path) -> list[dict]:
    """One dict per click / timeout from a session CSV."""
    seen, out = set(), []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["event"], row["click_id"], row["target_id"], row["t_session_s"])
            if key not in seen:
                seen.add(key)
                out.append({c: row[c] for c in META_COLS})
    return out


def summarize(events: list[dict]) -> dict:
    clicks = [e for e in events if e["event"] == "click"]
    if not events:
        return {"note": "no events"}
    e0 = events[0]
    still_s, move_s = _f(e0["rest_still_s"]), _f(e0["rest_move_s"])
    s = {"session_id": e0["session_id"], "user": e0["user"],
         "settings": {k: e0[k] for k in ("threshold", "refractory_ms", "smoothing", "cursor_source")}}

    # rest: every click is a false click
    rest = {}
    for phase, dur in (("rest_still", still_s), ("rest_move", move_s)):
        c = [e for e in clicks if e["phase"] == phase]
        by_band = defaultdict(int)
        for e in c:
            by_band[e["band"]] += 1
        rest[phase] = {"seconds": dur, "false_clicks": len(c),
                       "false_per_min": round(len(c) / dur * 60, 2) if dur > 0 else None,
                       "by_band": dict(by_band)}
    s["rest"] = rest

    task = [e for e in clicks if e["phase"] == "targets"]
    hits = [e for e in task if e["outcome"] == "hit"]
    timeouts = [e for e in events if e["event"] == "timeout"]
    per_band = {}
    for band in sorted({e["band"] for e in task}):
        bt = [e for e in task if e["band"] == band]
        bh = [e for e in bt if e["outcome"] == "hit"]
        per_band[band] = {"clicks": len(bt), "hits": len(bh),
                          "hit_rate": round(len(bh) / len(bt), 3) if bt else None}
    drift = [e for e in task if e["outcome"] != "hit" and str(e["inside_at_onset"]) == "1"
             and e["band"] == e["target_band"]]
    s["targets"] = {
        "targets_hit": len(hits), "targets_timed_out": len(timeouts),
        "clicks": len(task), "hits": len(hits),
        "miss_outside": sum(e["outcome"] == "miss_outside" for e in task),
        "miss_wrong_band": sum(e["outcome"] == "miss_wrong_band" for e in task),
        "hit_rate": round(len(hits) / len(task), 3) if task else None,
        "per_band": per_band,
        "misses_inside_at_onset": len(drift),
    }
    tt = [_f(e["time_to_target_s"]) for e in hits]
    te = [_f(e["time_to_enter_s"]) for e in hits]
    dwell = [a - b for a, b in zip(tt, te) if not math.isnan(b)]
    s["time_to_target_s"] = {"median": _r(pct(tt, 50), 3), "p25": _r(pct(tt, 25), 3),
                             "p75": _r(pct(tt, 75), 3), "mean": _r(np.mean(tt) if tt else math.nan, 3),
                             "median_time_to_enter": _r(pct(te, 50), 3),
                             "median_dwell_enter_to_click": _r(pct(dwell, 50), 3)}
    o2d = [_f(e["onset_to_detect_ms"]) for e in clicks]
    dlv = [_f(e["delivery_ms"]) for e in clicks]
    tot = [_f(e["latency_ms"]) for e in clicks]
    s["click_latency_ms"] = {"onset_to_detect_median": _r(pct(o2d, 50), 1),
                             "onset_to_detect_p90": _r(pct(o2d, 90), 1),
                             "delivery_median": _r(pct(dlv, 50), 1), "delivery_p90": _r(pct(dlv, 90), 1),
                             "total_median": _r(pct(tot, 50), 1), "total_p90": _r(pct(tot, 90), 1),
                             "n": len(clicks)}
    return s


def _r(v, n):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)


def format_summary(s: dict) -> str:
    if "rest" not in s:
        return "no clicks or targets recorded"
    st, r, t = s["settings"], s["rest"], s["targets"]
    tt, lat = s["time_to_target_s"], s["click_latency_ms"]
    hr = f"{t['hit_rate'] * 100:.0f}%" if t["hit_rate"] is not None else "-"
    lines = [
        f"session {s['session_id']}   threshold {st['threshold']}  refractory {st['refractory_ms']} ms  "
        f"smoothing {st['smoothing']}  cursor {st['cursor_source']}",
        f"hit rate {hr}  ({t['hits']}/{t['clicks']} clicks; {t['miss_outside']} outside, "
        f"{t['miss_wrong_band']} wrong band; {t['targets_timed_out']} targets timed out)",
        "  " + "   ".join(f"{b}: {v['hits']}/{v['clicks']}" for b, v in t["per_band"].items()),
        f"false clicks/min at rest: still {r['rest_still']['false_per_min']}  "
        f"({r['rest_still']['false_clicks']} in {r['rest_still']['seconds']:.0f} s), "
        f"moving arm {r['rest_move']['false_per_min']}  "
        f"({r['rest_move']['false_clicks']} in {r['rest_move']['seconds']:.0f} s)",
        f"time to target: median {tt['median']} s  (IQR {tt['p25']}-{tt['p75']})  "
        f"enter {tt['median_time_to_enter']} s + dwell {tt['median_dwell_enter_to_click']} s",
        f"click latency: onset->detect {lat['onset_to_detect_median']} ms (p90 {lat['onset_to_detect_p90']})"
        f"  + delivery {lat['delivery_median']} ms (p90 {lat['delivery_p90']})"
        f"  = {lat['total_median']} ms median",
        f"misses where the cursor was on target at EMG onset (clench moved it off): "
        f"{t['misses_inside_at_onset']}",
    ]
    if "quality" in s:
        lines.append("data: " + "   ".join(
            f"{b}: {q['samples_per_s']} samples/s, {q['dropped_packets']} dropped packets"
            for b, q in s["quality"].items()))
    return "\n".join(lines)

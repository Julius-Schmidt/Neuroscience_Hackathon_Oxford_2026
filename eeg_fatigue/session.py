"""Run the fatigue session: cues on screen, EEG + EMG + markers into ONE file.

    python -m eeg_fatigue.session --participant P03 --sanity        3 min check first
    python -m eeg_fatigue.session --participant P03                 full session (~15 min)
    python -m eeg_fatigue.session --participant SIM --emg lsl       dry run with eeg_fatigue.simulate --lsl

EEG comes from the Unicorn LSL stream (config: eeg_switch.config.SOURCE).
EMG comes straight from the Myo (--emg myo, default; band of config.HAND, needs bands.json)
or from an LSL stream named MyoEMG (--emg lsl). Every sample gets an LSL timestamp,
so EEG and EMG share one clock. Markers also go out on LSL, so LabRecorder can
record the same session as a backup.

The file is saved after every block (data/fatigue/<participant>_<time>.npz) and
checked at the end. Ctrl+C stops the session and keeps everything recorded so far.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from . import config as C
from .protocol import duration, sanity_steps, session_steps
from .recording import Recording, load, save

BOLD, RED, GREEN, DIM, RESET = "\033[1m", "\033[41;97m", "\033[32m", "\033[2m", "\033[0m"
CLEAR = "\r\033[K"


class Recorder:
    """Collects EEG (LSL inlet), EMG (Myo handler or LSL inlet) and markers in memory."""

    def __init__(self, emg_mode: str, side: str):
        import pylsl
        self.pylsl = pylsl
        self.lock = threading.Lock()
        self.eeg, self.eeg_t, self.emg, self.emg_t = [], [], [], []
        self.markers, self.marker_t = [], []
        self.running = True
        self.errors = []

        s = C.EEG_STREAM
        print(f"looking for EEG stream '{s['name']}' ({s['type']}) ...")
        found = [i for i in pylsl.resolve_byprop("name", s["name"], minimum=1, timeout=15) if i.type() == s["type"]]
        if not found:
            raise SystemExit(f"no EEG stream '{s['name']}'. Start Unicorn LSL (or eeg_fatigue.simulate --lsl). "
                             "Check names with: python -m eeg_switch.listen --list")
        flags = pylsl.proc_clocksync | pylsl.proc_dejitter
        self.eeg_inlet = pylsl.StreamInlet(found[0], max_buflen=60, processing_flags=flags)
        print(f"  EEG: {found[0].name()} {found[0].channel_count()} ch @ {found[0].nominal_srate():.0f} Hz")

        mk = C.MARKER_STREAM
        self.marker_out = pylsl.StreamOutlet(pylsl.StreamInfo(mk["name"], mk["type"], 1, 0, "string", "eeg_fatigue_markers"))
        e = C.EMG_STREAM
        self.band = None
        if emg_mode == "lsl":
            f2 = pylsl.resolve_byprop("name", e["name"], minimum=1, timeout=15)
            if not f2:
                raise SystemExit(f"no EMG stream '{e['name']}' on LSL")
            self.emg_inlet = pylsl.StreamInlet(f2[0], max_buflen=60, processing_flags=flags)
            print(f"  EMG: LSL {f2[0].name()} @ {f2[0].nominal_srate():.0f} Hz")
        else:
            self.emg_inlet = None
            # re-publish the Myo as an LSL stream too, so LabRecorder can back it up
            self.emg_out = pylsl.StreamOutlet(pylsl.StreamInfo(e["name"], e["type"], 8, e["srate"], "float32", f"myo_{side}"))
            from myomouse.device import open_band
            self.band = open_band(side)
            self.band.add_emg_handler(self._on_myo)
            print(f"  EMG: Myo ({side} arm) - connecting ...")
            self.band.start()

        self.threads = [threading.Thread(target=self._pull, args=(self.eeg_inlet, self.eeg, self.eeg_t, "EEG"), daemon=True)]
        if self.emg_inlet is not None:
            self.threads.append(threading.Thread(target=self._pull, args=(self.emg_inlet, self.emg, self.emg_t, "EMG"), daemon=True))
        for t in self.threads:
            t.start()

    def _on_myo(self, sample):
        ts = self.pylsl.local_clock()
        with self.lock:
            self.emg.append(sample)
            self.emg_t.append(ts)
        self.emg_out.push_sample([float(v) for v in sample], ts)

    def _pull(self, inlet, data, times, name):
        while self.running:
            try:
                chunk, ts = inlet.pull_chunk(timeout=0.2, max_samples=512)
            except Exception as e:                        # stream lost
                self.errors.append(f"{name}: {e}")
                time.sleep(0.5)
                continue
            if chunk:
                with self.lock:
                    data.extend(chunk)
                    times.extend(ts)

    def mark(self, value: str):
        ts = self.pylsl.local_clock()
        self.marker_out.push_sample([value], ts)
        with self.lock:
            self.markers.append(value)
            self.marker_t.append(ts)

    def snapshot(self, meta) -> Recording:
        with self.lock:
            eeg = np.asarray(self.eeg, float)[:, :len(C.EEG_CHANNELS)] if self.eeg else np.zeros((0, 8))
            emg = np.asarray(self.emg, float)[:, :8] if self.emg else np.zeros((0, 8))
            return Recording(eeg=eeg, eeg_t=np.asarray(self.eeg_t), emg=emg, emg_t=np.asarray(self.emg_t),
                             markers=list(self.markers), marker_t=np.asarray(self.marker_t), meta=dict(meta))

    def stats(self) -> str:
        with self.lock:
            return f"EEG {len(self.eeg_t)}  EMG {len(self.emg_t)}"

    def stop(self):
        self.running = False
        if self.band is not None:
            self.band.stop()
        for t in self.threads:
            t.join(1.0)


def show(step, left, total_left, stats):
    if step["kind"] == "squeeze":
        line = f"{RED}{BOLD} >>> {step['cue']:^30} <<< {RESET} {left:4.1f} s"
    elif step["kind"] == "relax":
        line = f"{GREEN}     {step['cue']:^30}     {RESET} {left:4.1f} s"
    else:
        line = f"{BOLD}     {step['cue']:^30}     {RESET} {left:5.1f} s"
    print(f"{CLEAR}{line}   {DIM}session left {total_left / 60:4.1f} min | {stats}{RESET}", end="", flush=True)


def run(steps, rec: Recorder, out_path, meta, beep=True):
    total = duration(steps)
    t_start = time.perf_counter()
    elapsed = 0.0
    for step in steps:
        if step["marker"]:
            rec.mark(step["marker"])
            if step["marker"].startswith("block_end:"):
                save(rec.snapshot(meta), out_path)        # autosave after every block
                print(f"{CLEAR}  {GREEN}saved{RESET} after {step['marker'].split(':')[1]}  ({rec.stats()})")
            if beep and step["marker"] in ("eyes_closed", "eyes_open", "rest_end"):
                print("\a", end="", flush=True)
        end = t_start + elapsed + step["dur"]
        while (now := time.perf_counter()) < end:
            show(step, end - now, total - (now - t_start), rec.stats())
            time.sleep(0.05)
        elapsed += step["dur"]
    print(CLEAR, end="")


def check_file(path, expect_s):
    """Reload what was saved and say plainly if anything is missing."""
    r = load(path)
    print(f"\n{BOLD}file check{RESET}: {path}\n  {r.summary()}")
    ok = True
    for name, t, rate in (("EEG", r.eeg_t, C.EEG_SRATE), ("EMG", r.emg_t, C.EMG_SRATE)):
        if len(t) < 2:
            print(f"  {RED} {name}: NO DATA {RESET}")
            ok = False
            continue
        got = len(t) / rate
        gaps = np.diff(t)
        if got < 0.9 * expect_s:
            print(f"  {RED} {name}: only {got:.0f} s of {expect_s:.0f} s {RESET}")
            ok = False
        if gaps.max() > 1.0:
            print(f"  {name}: largest gap {gaps.max():.1f} s ({(gaps > 1.0).sum()} gaps > 1 s)")
    flat = [c for c, v in zip(r.eeg_ch, r.eeg.std(axis=0)) if v < 0.5] if len(r.eeg) else []
    if flat:
        print(f"  {RED} flat EEG channels: {flat} {RESET}")
    print(f"  {GREEN}{BOLD}FILE OK{RESET}" if ok else f"  {RED}{BOLD} FILE INCOMPLETE - check before removing the headset {RESET}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="e.g. P03 (not a name)")
    ap.add_argument("--sanity", action="store_true", help="3 min check: eyes open/closed + 10 squeezes")
    ap.add_argument("--emg", choices=["myo", "lsl"], default="myo")
    ap.add_argument("--side", default=C.HAND, choices=["left", "right"])
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else int(time.time()) % 10000
    steps = sanity_steps(seed) if a.sanity else session_steps(seed)
    kind = "sanity" if a.sanity else "session"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = C.DATA_DIR / f"{a.participant}_{kind}_{stamp}.npz"
    meta = {"participant": a.participant, "kind": kind, "hand": a.side, "seed": seed, "started": stamp,
            "eeg_stream": C.EEG_STREAM["name"], "emg_mode": a.emg}

    rec = Recorder(a.emg, a.side)
    time.sleep(1.0)
    print(f"\n{BOLD}{kind} for {a.participant}{RESET}: {duration(steps) / 60:.1f} min, {a.side} hand. Saving to {out}")
    print("  Participant: eyes on one point, jaw and face relaxed, squeeze with the whole hand.")
    input(f"  {DIM}press Enter to start{RESET} ")
    try:
        run(steps, rec, out, meta)
    except KeyboardInterrupt:
        rec.mark("session_aborted")
        print(f"\n{RED} stopped early {RESET}")
    finally:
        save(rec.snapshot(meta), out)
        rec.stop()
        if rec.errors:
            print(f"stream errors: {rec.errors[:3]}")
    check_file(out, duration(steps))
    if a.sanity:
        from .sanity import report
        report(out)
    else:
        print(f"\nnext: python -m eeg_fatigue.analyse {out}")


if __name__ == "__main__":
    main()

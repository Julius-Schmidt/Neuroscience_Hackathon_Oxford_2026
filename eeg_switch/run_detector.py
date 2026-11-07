"""Live safety switch: EEG stream in, intent events out (both over LSL).

    python -m eeg_switch.run_detector                  uses config.SOURCE ("sim" or "unicorn")
    python -m eeg_switch.run_detector --source unicorn

Keep your eyes open and relaxed for the first 20 s (baseline). After that, close
your eyes for ~2 s to toggle the safety switch. Events go out on the "NeuroIntents"
stream as {"source": "eeg", "intent": "safety_toggle", "confidence": .., "timestamp": ..}.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import config as C
from .detector import AlphaSwitch, Params
from .events import EventOutlet


def open_eeg_inlet(source: str, timeout: float = 15.0):
    import pylsl
    s = C.STREAMS[source]
    print(f"looking for EEG stream name='{s['name']}' type='{s['type']}' ...")
    found = [i for i in pylsl.resolve_byprop("name", s["name"], minimum=1, timeout=timeout)
             if i.type() == s["type"]]
    if not found:
        raise SystemExit(f"no LSL stream '{s['name']}' ({s['type']}). Is the simulator / Unicorn LSL app running? "
                         f"List streams with: python -m eeg_switch.listen --list")
    info = found[0]
    if abs(info.nominal_srate() - s["srate"]) > 1:
        print(f"WARNING: stream rate {info.nominal_srate()} Hz, expected {s['srate']} Hz")
    if info.channel_count() < s["n_channels"]:
        raise SystemExit(f"stream has {info.channel_count()} channels, need {s['n_channels']}")
    print(f"connected: {info.name()} {info.channel_count()} ch @ {info.nominal_srate():.0f} Hz "
          f"(source_id={info.source_id() or '-'})")
    return pylsl.StreamInlet(info, max_buflen=10), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(C.STREAMS), default=C.SOURCE)
    ap.add_argument("--quiet", action="store_true", help="no live score line")
    a = ap.parse_args()

    inlet, info = open_eeg_inlet(a.source)
    params = Params(srate=info.nominal_srate() or C.SRATE)
    sw = AlphaSwitch(params)
    out = EventOutlet("eeg")
    print(f"publishing events on '{C.EVENT_STREAM['name']}'. Calibrating: eyes open for {params.baseline_s:.0f} s ...")

    state_on = False
    last_print = 0.0
    while True:
        chunk, ts = inlet.pull_chunk(timeout=0.2, max_samples=256)
        if not chunk:
            continue
        x = np.asarray(chunk, dtype=np.float64)[:, :len(C.EEG_CHANNELS)]
        for t_evt, conf in sw.push(x, np.asarray(ts)):
            state_on = not state_on
            ev = out.send(C.INTENT_TOGGLE, conf, t_evt)
            print(f"\n>>> TOGGLE  safety {'ON' if state_on else 'OFF'}  conf={ev['confidence']:.2f}  t={ev['timestamp']:.2f}")
        now = time.monotonic()
        if not a.quiet and now - last_print > 0.25:
            last_print = now
            if not sw.calibrated:
                print(f"\r  calibrating ... {sw.n_seen / params.srate:4.1f}/{params.baseline_s:.0f} s", end="")
            else:
                bar = "#" * int(max(0, min(30, (sw.score + 3) * 3)))
                print(f"\r  alpha {sw.score:+5.1f} dB |{bar:<30}| closed {sw.closed_t:3.1f}/{params.dwell_s:.0f}s "
                      f"{'armed' if sw.armed else 'wait-open'}  ", end="")


if __name__ == "__main__":
    main()

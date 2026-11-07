"""Replay real PhysioNet EEG as a Unicorn-like LSL stream (8 ch, 250 Hz, µV).

    python -m eeg_switch.simulator                      subject 1, scripted open/closed timeline
    python -m eeg_switch.simulator --mode open          eyes open only; you trigger closures
    python -m eeg_switch.simulator --subject 7 --loop
    python -m eeg_switch.simulator --synthetic          no download (unit-test signal, not real EEG)

Keys (Windows: just press; Mac/Linux: press then Enter):
    c   inject 4 s of this person's real eyes-closed EEG   -> should toggle
    b   inject 0.6 s of eyes-closed EEG (blink-length)     -> should NOT toggle
    q   quit

The stream uses the same name, type, channel count and rate as config.STREAMS["sim"],
which matches what the detector expects from the real headset.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import config as C
from .keys import Keys


def make_outlet(n_channels: int = 8):
    import pylsl
    s = C.STREAMS["sim"]
    info = pylsl.StreamInfo(s["name"], s["type"], n_channels, s["srate"], "float32", s["source_id"])
    chans = info.desc().append_child("channels")
    for name in C.EEG_CHANNELS:
        ch = chans.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    info.desc().append_child_value("simulated", "true")
    return pylsl.StreamOutlet(info, chunk_size=C.SIM_CHUNK), pylsl


class Player:
    """Plays a base timeline; injected segments play next, offset-matched to avoid steps."""

    def __init__(self, base: np.ndarray, closed: np.ndarray, rng, loop: bool):
        self.base, self.closed, self.rng, self.loop = base.astype(np.float64), closed, rng, loop
        self.pos = 0
        self.inject: list[dict] = []
        self.last = self.base[0].copy()

    def add_closed(self, seconds: float):
        n = int(seconds * C.SRATE)
        start = self.rng.integers(0, len(self.closed) - n)
        self.inject.append({"data": self.closed[start:start + n].astype(np.float64), "aligned": False})

    def next_chunk(self, n: int) -> np.ndarray | None:
        if self.inject:
            seg = self.inject[0]
            if not seg["aligned"]:                  # offset-match once, at its first sample
                seg["data"] += self.last - seg["data"][0]
                seg["aligned"] = True
            out, seg["data"] = seg["data"][:n], seg["data"][n:]
            if len(seg["data"]) == 0:
                self.inject.pop(0)
                # shift the base timeline so it continues from where the injection ended
                self.base = self.base + (out[-1] - self.base[self.pos % len(self.base)])
        else:
            if self.pos >= len(self.base):
                if not self.loop:
                    return None
                self.pos = 0
            out = self.base[self.pos:self.pos + n]
            self.pos += len(out)
        self.last = out[-1].copy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--mode", choices=["script", "open"], default="script",
                    help="script: real open/closed timeline; open: eyes-open only, keys add closures")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="unit-test signal instead of PhysioNet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.synthetic:
        from .synthetic import synthetic_subject
        sub = synthetic_subject(a.seed)
        print("WARNING: synthetic signal (unit tests only) - use real data for anything you measure")
    else:
        from .data import load_subject
        sub = load_subject(a.subject)
    if a.mode == "script":
        from .data import replay_sequence
        base = replay_sequence(sub, rng, a.minutes)
    else:
        base = sub.open
        a.loop = True

    outlet, pylsl = make_outlet()
    player = Player(base, sub.closed, rng, a.loop)
    keys = Keys()
    s = C.STREAMS["sim"]
    print(f"streaming '{s['name']}' ({s['type']}, 8 ch @ {s['srate']:.0f} Hz) from "
          f"{'synthetic' if a.synthetic else f'PhysioNet subject {a.subject}'}, mode={a.mode}")
    print("first 20 s are eyes open (detector calibration). keys: c = close eyes 4 s, b = blink-length, q = quit")

    dt = C.SIM_CHUNK / C.SRATE
    t_next = time.perf_counter()
    sent = 0
    while True:
        k = keys.get()
        if k == "q":
            break
        if k == "c":
            player.add_closed(C.INJECT_CLOSED_S)
            print(f"  [{sent / C.SRATE:7.1f}s] injected {C.INJECT_CLOSED_S:.0f} s eyes closed")
        elif k == "b":
            player.add_closed(0.6)
            print(f"  [{sent / C.SRATE:7.1f}s] injected 0.6 s eyes closed (should not toggle)")
        chunk = player.next_chunk(C.SIM_CHUNK)
        if chunk is None:
            print("end of recording (use --loop to repeat)")
            break
        outlet.push_chunk(chunk.astype(np.float32).tolist())
        sent += len(chunk)
        t_next += dt
        time.sleep(max(0.0, t_next - time.perf_counter()))


if __name__ == "__main__":
    main()

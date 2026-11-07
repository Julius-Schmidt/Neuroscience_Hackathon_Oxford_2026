"""Record calibration data for one arm (about 2 minutes).

    python -m myomouse.record --side right
    python -m myomouse.record --side left

Follow the cues in the terminal. The "rest" phases where you move your arm around
matter most: they teach the classifier that pointing is NOT a fist.
"""

import argparse
import time

import numpy as np

from . import config as C
from .device import open_band

REST, FIST, IGNORE = 0, 1, -1
TRANSITION_S = 0.25  # samples this soon after a cue change are left out of training


def protocol(side):
    move = "move your forearm around like you're steering a cursor" if side == C.POINTER_SIDE \
        else "move your arm around naturally"
    p = [("Relax your hand, arm still", REST, 8.0),
         (f"Relax your hand, {move}", REST, 15.0),
         ("Spread fingers / point / wave - anything EXCEPT a fist", REST, 8.0)]
    for _ in range(8):
        p += [("FIST - squeeze firmly", FIST, 1.5), ("relax", REST, 2.0)]
    for _ in range(5):
        p += [(f"FIST while you {move}", FIST, 1.5), (f"relax, keep moving", REST, 2.0)]
    p += [("Quick clicks: short FIST", FIST, 0.6), ("relax", REST, 1.2)] * 6
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--fake", action="store_true", help="simulated band, for testing")
    args = ap.parse_args()

    steps = protocol(args.side)
    bounds = np.cumsum([0.0] + [s[2] for s in steps])
    emg, labels, segs = [], [], []
    state = {"t0": None}

    def on_emg(sample):
        if state["t0"] is None:
            return
        t = time.perf_counter() - state["t0"]
        i = int(np.searchsorted(bounds, t, side="right") - 1)
        if i >= len(steps):
            return
        lab = steps[i][1] if t - bounds[i] >= TRANSITION_S else IGNORE
        emg.append(sample)
        labels.append(lab)
        segs.append(i)

    band = open_band(args.side, fake=args.fake)
    band.add_emg_handler(on_emg)
    band.start()

    print(f"\nRecording {args.side} arm, {bounds[-1]:.0f} s. Starting in 3...")
    time.sleep(3)
    state["t0"] = time.perf_counter()
    shown = -1
    try:
        while True:
            if not band.alive:
                raise SystemExit(f"band disconnected: {band.error}")
            t = time.perf_counter() - state["t0"]
            i = int(np.searchsorted(bounds, t, side="right") - 1)
            if i >= len(steps):
                break
            if i != shown:
                text, lab, dur = steps[i]
                mark = ">>>" if lab == FIST else "   "
                print(f"{mark} {text}  ({dur:.1f} s)", flush=True)
                shown = i
            time.sleep(0.01)
    finally:
        band.stop()

    C.DATA_DIR.mkdir(exist_ok=True)
    out = C.DATA_DIR / f"{args.side}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
    np.savez(out, emg=np.asarray(emg, dtype=np.int8), label=np.asarray(labels),
             seg=np.asarray(segs), side=args.side, fs=C.EMG_HZ)
    n = len(emg)
    print(f"\nsaved {out}: {n} samples ({n / C.EMG_HZ:.1f} s, expected ~{bounds[-1]:.0f} s)")
    if n < 0.8 * C.EMG_HZ * bounds[-1]:
        print("WARNING: fewer samples than expected - Bluetooth may be dropping packets.")


if __name__ == "__main__":
    main()

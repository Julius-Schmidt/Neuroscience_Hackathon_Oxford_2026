"""Record calibration data for one arm (about 2 minutes of recording).

    python -m myomouse.record --side right
    python -m myomouse.record --side left

The recording is split into short blocks. Before each block you read what to do and
press Enter; then a 3-2-1 countdown starts it. During a block a live line shows what to
do now and counts down to the next squeeze. Nothing is recorded while you're reading.

The "relax while moving your arm" parts matter most: they teach the classifier that
pointing is NOT a fist.
"""

import argparse
import sys
import time

import numpy as np

from . import config as C
from .device import open_band

REST, FIST, IGNORE = 0, 1, -1
TRANSITION_S = 0.25  # samples this soon after a cue change are left out of training

# terminal colours
BOLD, RED, GREEN, DIM, RESET = "\033[1m", "\033[41;97m", "\033[32m", "\033[2m", "\033[0m"
CLEAR_LINE = "\r\033[K"


def protocol(side):
    """List of blocks: (title, instructions, [(label, seconds, short cue), ...])."""
    if side == C.POINTER_SIDE:
        move = "move your forearm around like you're steering a cursor"
    else:
        move = "move your arm around naturally"

    def fists(n, squeeze, relax, cue_fist, cue_relax):
        steps = [(REST, 2.0, cue_relax)]  # lead-in so the first squeeze also gets a countdown
        for _ in range(n):
            steps += [(FIST, squeeze, cue_fist), (REST, relax, cue_relax)]
        return steps

    return [
        ("Rest, arm still", "Relax your hand completely and keep your arm still.",
         [(REST, 8.0, "relax, arm still")]),
        ("Rest, arm moving", f"Keep your hand relaxed and {move}.",
         [(REST, 15.0, "relax hand, keep moving")]),
        ("Other hand shapes", "Spread your fingers, point, wave - anything EXCEPT a fist.",
         [(REST, 8.0, "anything but a fist")]),
        ("Fists", "Arm still. SQUEEZE a firm fist when the line turns red, relax when it turns green.\n"
                  "  A countdown tells you when the next squeeze comes.",
         fists(8, 1.5, 2.5, "SQUEEZE", "relax")),
        ("Fists while moving", f"Keep moving the whole time ({move}).\n"
                               "  SQUEEZE when red, relax your hand (but keep moving) when green.",
         fists(5, 1.5, 2.5, "SQUEEZE (keep moving)", "relax, keep moving")),
        ("Quick clicks", "Arm still. Short, quick squeezes like a mouse click.",
         fists(6, 0.6, 1.8, "CLICK", "relax")),
    ]


def bar(frac, width=20):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "·" * (width - n)


def status_line(steps, i, t_in):
    """One live line: what to do now + countdown to the next squeeze."""
    label, dur, cue = steps[i]
    left = dur - t_in
    if label == FIST:
        return f"{RED}{BOLD} >>> {cue:^22} <<< {RESET}  {bar(left / dur)}  {left:4.1f} s"
    nxt = steps[i + 1] if i + 1 < len(steps) else None
    if nxt and nxt[0] == FIST:
        warn = f"{BOLD}{nxt[2]} in {left:3.1f} s{RESET}" if left < 1.5 else f"next: {nxt[2]} in {left:3.1f} s"
        return f"{GREEN}     {cue:^22}     {RESET}  {warn}"
    return f"{GREEN}     {cue:^22}     {RESET}  {bar(1 - t_in / dur)}  {left:4.1f} s left"


def countdown():
    for k in (3, 2, 1):
        print(f"{CLEAR_LINE}  starting in {BOLD}{k}{RESET} ...", end="", flush=True)
        time.sleep(1.0)
    print(CLEAR_LINE, end="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--fake", action="store_true", help="simulated band, for testing")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for Enter between blocks")
    args = ap.parse_args()

    blocks = protocol(args.side)
    emg, labels, segs = [], [], []
    # what the EMG handler should label incoming samples with; None = not recording
    state = {"steps": None, "bounds": None, "t0": None, "seg0": 0}

    def on_emg(sample):
        steps, bounds, t0 = state["steps"], state["bounds"], state["t0"]
        if steps is None:
            return
        t = time.perf_counter() - t0
        i = int(np.searchsorted(bounds, t, side="right") - 1)
        if not 0 <= i < len(steps):
            return
        lab = steps[i][0] if t - bounds[i] >= TRANSITION_S else IGNORE
        emg.append(sample)
        labels.append(lab)
        segs.append(state["seg0"] + i)

    band = open_band(args.side, fake=args.fake)
    band.add_emg_handler(on_emg)
    band.start()
    total = sum(d for _, _, steps in blocks for _, d, _ in steps)
    print(f"\n{BOLD}Recording {args.side} arm{RESET}: {len(blocks)} blocks, {total:.0f} s of recording.")

    seg0 = 0
    try:
        for b, (title, text, steps) in enumerate(blocks, 1):
            dur = sum(d for _, d, _ in steps)
            print(f"\n{BOLD}[{b}/{len(blocks)}] {title}{RESET}  ({dur:.0f} s)\n  {text}")
            if not args.no_wait:
                input(f"  {DIM}press Enter when ready{RESET} ")
            countdown()

            bounds = np.cumsum([0.0] + [d for _, d, _ in steps])
            state.update(steps=steps, bounds=bounds, seg0=seg0, t0=time.perf_counter())
            while True:
                if not band.alive:
                    raise SystemExit(f"\nband disconnected: {band.error}")
                t = time.perf_counter() - state["t0"]
                i = int(np.searchsorted(bounds, t, side="right") - 1)
                if i >= len(steps):
                    break
                print(CLEAR_LINE + status_line(steps, i, t - bounds[i]), end="", flush=True)
                time.sleep(0.03)
            state["steps"] = None  # stop recording while the next instructions are read
            seg0 += len(steps)
            print(f"{CLEAR_LINE}  {GREEN}done{RESET}")
    except KeyboardInterrupt:
        raise SystemExit("\naborted, nothing saved")
    finally:
        band.stop()

    C.DATA_DIR.mkdir(exist_ok=True)
    out = C.DATA_DIR / f"{args.side}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
    np.savez(out, emg=np.asarray(emg, dtype=np.int8), label=np.asarray(labels),
             seg=np.asarray(segs), side=args.side, fs=C.EMG_HZ)
    n = len(emg)
    print(f"\nsaved {out}: {n} samples ({n / C.EMG_HZ:.1f} s, expected ~{total:.0f} s)")
    if n < 0.8 * C.EMG_HZ * total:
        print("WARNING: fewer samples than expected - Bluetooth may be dropping packets.")
    print(f"next: python -m myomouse.train --side {args.side}")


if __name__ == "__main__":
    sys.exit(main())

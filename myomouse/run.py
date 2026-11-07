"""Live demo: one process per band, a mouse controller in the main process.

    python -m myomouse.run                  # both bands, LDA classifiers
    python -m myomouse.run --mode threshold # fallback: simple activation threshold
    python -m myomouse.run --dry-run        # print events instead of moving the mouse
    python -m myomouse.run --fake --dry-run # no hardware needed
    python -m myomouse.run --check          # before training: EMG rate, muscle activity, cursor only

Band R --> gyro/orientation -> cursor;  EMG -> fist -> left button (hold = drag)
Band L --> EMG -> fist -> right button
Ctrl+C to stop (always releases any held button).
"""

import argparse
import multiprocessing as mp
import queue
import time

from . import config as C


def band_worker(side, mode, pointer, fake, q, show, check):
    """Runs in its own process: reads one band, sends ('move'|'btn'|'p', ...) to the queue."""
    from collections import deque

    import numpy as np

    from .detector import FistDetector, load_model
    from .device import open_band
    from .features import envelope
    from .pointer import Pointer

    det = None if check else FistDetector(load_model(side), mode)
    recent = deque(maxlen=C.WINDOW)  # for the activity bar in --check mode
    band = open_band(side, fake=fake, seed=1 if side == "right" else 2)
    ptr = Pointer() if pointer else None
    last_show = [0.0]

    def on_emg(sample):
        if det is not None:
            ev = det.push(sample)
            if ev:
                q.put(("btn", side, ev, time.perf_counter()))
        else:
            recent.append(sample)
        if show and time.perf_counter() - last_show[0] > 0.1:
            last_show[0] = time.perf_counter()
            if det is not None:
                value = det.last_p
            else:  # raw muscle activity, scaled so a hard fist is close to 1
                value = min(1.0, envelope(np.array(recent)) / 4.0) if len(recent) > 1 else 0.0
            q.put(("p", side, value, 0.0))

    def on_imu(quat):
        dx, dy = ptr.update(quat)
        if dx or dy:
            q.put(("move", dx, dy, time.perf_counter()))

    band.add_emg_handler(on_emg)
    if ptr:
        band.add_imu_handler(on_imu)
    try:
        band.start()
        q.put(("ready", side, None, 0.0))
        t_rate, n_rate = time.perf_counter(), band.n_emg
        while band.alive:
            time.sleep(0.5)
            if show and time.perf_counter() - t_rate > 5:
                hz = (band.n_emg - n_rate) / (time.perf_counter() - t_rate)
                print(f"[{side}] EMG rate {hz:.0f} Hz (expect ~200)", flush=True)
                t_rate, n_rate = time.perf_counter(), band.n_emg
        print(f"[{side}] band stopped: {band.error}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        band.stop()


class Mouse:
    def __init__(self, dry):
        self.dry = dry
        if not dry:
            from pynput.mouse import Button, Controller
            self.ctl, self.Button = Controller(), Button
        self.held = set()
        self.frac = [0.0, 0.0]  # sub-pixel remainder so slow movements aren't lost

    def move(self, dx, dy):
        self.frac[0] += dx
        self.frac[1] += dy
        ix, iy = int(self.frac[0]), int(self.frac[1])
        if ix or iy:
            self.frac[0] -= ix
            self.frac[1] -= iy
            if not self.dry:
                self.ctl.move(ix, iy)

    def button(self, name, down):
        if down:
            self.held.add(name)
        else:
            self.held.discard(name)
        if self.dry:
            return
        b = getattr(self.Button, name)
        self.ctl.press(b) if down else self.ctl.release(b)

    def release_all(self):
        for name in list(self.held):
            self.button(name, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lda", "threshold"], default="lda")
    ap.add_argument("--sides", default="right,left", help="e.g. 'right' to run one band only")
    ap.add_argument("--dry-run", action="store_true", help="don't touch the real mouse")
    ap.add_argument("--fake", action="store_true", help="simulated bands")
    ap.add_argument("--show", action="store_true", help="print live fist probabilities")
    ap.add_argument("--check", action="store_true",
                    help="no classifier: show EMG rate + muscle activity and move the cursor (use before training)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = forever)")
    args = ap.parse_args()

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if args.check:
        args.show = True
    if not args.fake and not C.BANDS_FILE.exists():
        raise SystemExit("No bands.json yet. Run `python -m myomouse.identify` first.")
    if not args.check:
        from .detector import model_path
        missing = [s for s in sides if not model_path(s).exists()]
        if missing:
            raise SystemExit(
                f"No trained model for: {', '.join(missing)}.\n"
                "  Train first:  python -m myomouse.record --side <side>  then  python -m myomouse.train --side <side>\n"
                "  Or test the bands without a model:  python -m myomouse.run --check --dry-run")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=band_worker, daemon=True,
                         args=(s, args.mode, s == C.POINTER_SIDE, args.fake, q, args.show, args.check))
             for s in sides]
    for p in procs:
        p.start()

    mouse = Mouse(args.dry_run)
    freeze_until = 0.0
    probs = {s: 0.0 for s in sides}
    lat = []
    t_start = time.perf_counter()
    label = "check (no classifier, bars = muscle activity)" if args.check else args.mode
    print(f"mode={label} sides={sides} pointer={C.POINTER_SIDE} dry_run={args.dry_run}")
    try:
        while True:
            if args.seconds and time.perf_counter() - t_start > args.seconds:
                break
            if not any(p.is_alive() for p in procs):
                raise SystemExit("all band processes exited - see errors above")
            try:
                kind, a, b, t_sent = q.get(timeout=0.2)
            except queue.Empty:
                continue
            now = time.perf_counter()
            if kind == "move":
                if now >= freeze_until:
                    mouse.move(a, b)
            elif kind == "btn":
                name = C.BUTTON_FOR_SIDE[a]
                mouse.button(name, b == "press")
                freeze_until = now + C.FREEZE_MS / 1000
                lat.append((now - t_sent) * 1000)
                print(f"{a:5s} fist {b:7s} -> {name} button  (queue {lat[-1]:.2f} ms)", flush=True)
            elif kind == "p":
                probs[a] = b
                print("  ".join(f"{s}: {'#' * int(v * 20):20s} {v:.2f}" for s, v in probs.items()),
                      flush=True)
            elif kind == "ready":
                print(f"[{a}] ready")
    except KeyboardInterrupt:
        pass
    finally:
        mouse.release_all()
        for p in procs:
            p.terminate()
        if lat:
            print(f"{len(lat)} button events, max queue delay {max(lat):.2f} ms")


if __name__ == "__main__":
    main()

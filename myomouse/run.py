"""Live demo: one process per band, a mouse controller in the main process.

    python -m myomouse.run                  # both bands, LDA classifiers, playground window
    python -m myomouse.run --no-window      # same, terminal only
    python -m myomouse.run --mode threshold # fallback: simple activation threshold
    python -m myomouse.run --dry-run        # print events instead of moving the mouse
    python -m myomouse.run --fake --dry-run # no hardware needed
    python -m myomouse.run --check          # before training: EMG rate, muscle activity, cursor only

Band R --> gyro/orientation -> cursor;  EMG -> fist -> left button (hold = drag)
Band L --> EMG -> fist -> right button
The playground window (playground.py) shows the cursor trail, every click, and blocks to
recolour (left click) and drag (right button). The bands move the real mouse either way.
Ctrl+C (or Esc in the window) to stop - always releases any held button.
"""

import argparse
import multiprocessing as mp
import queue
import threading
import time
from collections import deque

from . import config as C


def band_worker(side, mode, pointer, fake, q, show, check):
    """Runs in its own process: reads one band, sends ('move'|'btn'|'p', ...) to the queue."""
    from collections import deque

    import numpy as np

    from .detector import FistDetector, load_model
    from .device import open_band
    from .features import envelope
    from .pointer import Pointer, roll_angle, yaw_pitch

    det = None if check else FistDetector(load_model(side), mode)
    recent = deque(maxlen=C.WINDOW)  # for the activity bar in --check mode
    band = open_band(side, fake=fake, seed=1 if side == "right" else 2)
    ptr = Pointer(pointer if isinstance(pointer, str) else None) if pointer else None
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
        # raw forearm angles too, so the arena can try other mappings live (run ignores these)
        yaw, pitch = yaw_pitch(quat)
        q.put(("ori", (yaw, pitch, roll_angle(quat)), None, time.perf_counter()))

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


def start_bands(sides, mode, fake, show, check, y_axis=None):
    """Check bands.json / models exist, then start one process per band. Returns (queue, procs)."""
    if not fake and not C.BANDS_FILE.exists():
        raise SystemExit("No bands.json yet. Run `python -m myomouse.identify` first.")
    if not check:
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
                         args=(s, mode, (y_axis or C.POINTER_Y_AXIS) if s == C.POINTER_SIDE else None,
                               fake, q, show, check))
             for s in sides]
    for p in procs:
        p.start()
    return q, procs


class Mouse:
    def __init__(self, dry):
        self.dry = dry
        if not dry:
            from pynput.mouse import Button, Controller
            self.ctl, self.Button = Controller(), Button
        self.held = set()
        self.frac = [0.0, 0.0]  # sub-pixel remainder so slow movements aren't lost
        self.vpos = (0.0, 0.0)  # dry run: a virtual cursor the playground draws
        self.vbounds = None

    def set_virtual(self, x, y, w, h):
        if x is not None:
            self.vpos = (x, y)
        self.vbounds = (w, h)

    def move(self, dx, dy):
        if self.dry and self.vbounds:
            w, h = self.vbounds
            self.vpos = (min(max(self.vpos[0] + dx, 0), w - 1), min(max(self.vpos[1] + dy, 0), h - 1))
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


class Controller:
    """Turns band messages into mouse actions. Also keeps what the playground window shows."""

    def __init__(self, mouse, sides, print_probs):
        self.mouse = mouse
        self.freeze_until = 0.0
        self.probs = {s: 0.0 for s in sides}
        self.print_probs = print_probs
        self.lat = []
        self.events = deque(maxlen=200)  # for the window: ('btn', side, 'press'|'release') / ('ready', side, None)

    def handle(self, kind, a, b, t_sent):
        now = time.perf_counter()
        if kind == "move":
            if now >= self.freeze_until:
                self.mouse.move(a, b)
        elif kind == "btn":
            name = C.BUTTON_FOR_SIDE[a]
            self.mouse.button(name, b == "press")
            self.freeze_until = now + C.FREEZE_MS / 1000
            self.lat.append((now - t_sent) * 1000)
            self.events.append(("btn", a, b))
            print(f"{a:5s} fist {b:7s} -> {name} button  (queue {self.lat[-1]:.2f} ms)", flush=True)
        elif kind == "p":
            self.probs[a] = b
            if self.print_probs:
                print("  ".join(f"{s}: {'#' * int(v * 20):20s} {v:.2f}" for s, v in self.probs.items()),
                      flush=True)
        elif kind == "ready":
            self.events.append(("ready", a, None))
            print(f"[{a}] ready")

    def drain(self):
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out


def add_pointer_args(ap):
    ap.add_argument("--y-axis", choices=["pitch", "roll"], default=None,
                    help="pitch = raise/lower the arm (default); roll = arm resting on a table, rotate the forearm")
    ap.add_argument("--pointer", choices=["right", "left"], default=None,
                    help="which band moves the cursor (default: config POINTER_SIDE)")
    ap.add_argument("--button", choices=["left", "right"], default=None,
                    help="one band only: which mouse button its fist presses")


def apply_pointer_args(args):
    """CLI overrides for config, applied in the main process (band processes get y_axis passed in)."""
    if args.pointer:
        C.POINTER_SIDE = args.pointer
    if args.button:
        sides = [s.strip() for s in args.sides.split(",") if s.strip()]
        if len(sides) == 1:
            C.BUTTON_FOR_SIDE = dict(C.BUTTON_FOR_SIDE, **{sides[0]: args.button})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lda", "threshold"], default="lda")
    ap.add_argument("--sides", default=None,
                    help="e.g. 'right' for one band; default = the bands in bands.json")
    ap.add_argument("--dry-run", action="store_true", help="don't touch the real mouse")
    ap.add_argument("--fake", action="store_true", help="simulated bands")
    ap.add_argument("--show", action="store_true", help="print live fist probabilities")
    ap.add_argument("--check", action="store_true",
                    help="no classifier: show EMG rate + muscle activity and move the cursor (use before training)")
    ap.add_argument("--no-window", action="store_true", help="no playground window, terminal only")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = forever)")
    add_pointer_args(ap)
    args = ap.parse_args()
    from .device import default_sides
    args.sides = args.sides or default_sides(args.fake)
    apply_pointer_args(args)

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if args.check:
        args.show = True
    window = not args.no_window
    # the window shows the probability bars, so the bands always send them then
    q, procs = start_bands(sides, args.mode, args.fake, args.show or window, args.check, args.y_axis)

    ctrl = Controller(Mouse(args.dry_run), sides, print_probs=args.show and not window)
    t_start = time.perf_counter()
    label = "check (no classifier, bars = muscle activity)" if args.check else args.mode
    print(f"mode={label} sides={sides} pointer={C.POINTER_SIDE} ({args.y_axis or C.POINTER_Y_AXIS}) "
          f"buttons={C.BUTTON_FOR_SIDE} dry_run={args.dry_run}")
    stop = threading.Event()

    def consume():
        while not stop.is_set():
            try:
                ctrl.handle(*q.get(timeout=0.1))
            except queue.Empty:
                continue

    try:
        if window:
            from .playground import run_window
            threading.Thread(target=consume, daemon=True).start()
            run_window(ctrl, procs, sides, args.dry_run, args.check, args.seconds)
        else:
            while True:
                if args.seconds and time.perf_counter() - t_start > args.seconds:
                    break
                if not any(p.is_alive() for p in procs):
                    raise SystemExit("all band processes exited - see errors above")
                try:
                    ctrl.handle(*q.get(timeout=0.2))
                except queue.Empty:
                    continue
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        time.sleep(0.15)  # let the consumer thread finish its last event
        ctrl.mouse.release_all()
        for p in procs:
            p.terminate()
        if ctrl.lat:
            print(f"{len(ctrl.lat)} button events, max queue delay {max(ctrl.lat):.2f} ms")


if __name__ == "__main__":
    main()

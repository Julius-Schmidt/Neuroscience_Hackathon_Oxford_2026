"""Fist -> a sound -> iPhone Switch Control. Drive a phone with the band, no extra hardware.

iPhone Switch Control can use a sound as a switch (Settings > Accessibility > Switch Control >
Switches > Add New Switch > Sound). This script plays a short sound from the laptop every
time the band detects a fist, so the phone hears a "switch press".

    python -m myomouse.sound_switch --test                 # no band: Enter plays the sound (tune volume first)
    python -m myomouse.sound_switch --user dylan           # band: every fist plays the sound
    python -m myomouse.sound_switch --user dylan --sound ee
    python -m myomouse.sound_switch --record mypop         # record your OWN pop/click (needs: pip install sounddevice)
    python -m myomouse.sound_switch --user dylan --sound mypop

Built-in sounds (synthesised): pop, click, ee. iOS listens for mouth sounds, so a synthetic one
may not trigger it; if not, record yourself making the sound you picked on the phone (--record)
and play that back - the recording is the most likely to work.

PHONE SETUP (do step 1 FIRST: with Switch Control on, normal touch gestures stop working)
  1. Settings > Accessibility > Accessibility Shortcut > tick Switch Control
     (then triple-click the side button turns it on/off - your way out).
  2. Settings > Accessibility > Switch Control > Switches > Add New Switch > Sound > pick the
     same sound (e.g. Pop) > action Select Item.
  3. One switch needs Auto Scanning (Switch Control > Scanning Style). For the claw: open the
     scanner menu and switch to point mode (crosshairs) - a line sweeps, a pop stops it.
  4. Put the phone near the laptop speaker, volume up, quiet-ish room.
"""

import argparse
import math
import struct
import sys
import time
import wave
from pathlib import Path

from . import config as C

RATE = 44100
SOUND_DIR = C.DATA_DIR / "sounds"


def _write(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max(1e-9, max(abs(s) for s in samples))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(32000 * s / peak)) for s in samples))


def _synth(name):
    import random
    rnd = random.Random(0)
    if name == "click":            # tongue-click-like: very short broadband transient
        n = int(0.012 * RATE)
        return [rnd.uniform(-1, 1) * math.exp(-t / (0.0015 * RATE)) for t in range(n)]
    if name == "pop":              # lip pop: low thump + short noise burst
        n = int(0.06 * RATE)
        out, lp = [], 0.0
        for t in range(n):
            lp += 0.25 * (rnd.uniform(-1, 1) - lp)                     # low-passed noise
            thump = math.sin(2 * math.pi * 320 * t / RATE) * math.exp(-t / (0.012 * RATE))
            out.append(0.6 * thump + 0.8 * lp * math.exp(-t / (0.004 * RATE)))
        return out
    if name == "ee":               # sustained "ee" vowel: 140 Hz voice through ee formants
        n, f0 = int(0.45 * RATE), 140.0
        formants = [(270, 60), (2290, 90), (3010, 120)]
        amps = []
        for k in range(1, int(4000 / f0)):
            f = k * f0
            amps.append((f, sum(math.exp(-((f - fc) / bw) ** 2) for fc, bw in formants) / k ** 0.3))
        out = []
        for t in range(n):
            env = min(1, t / (0.03 * RATE), (n - t) / (0.05 * RATE))
            out.append(env * sum(a * math.sin(2 * math.pi * f * t / RATE) for f, a in amps))
        return out
    raise SystemExit(f"unknown sound {name!r}: use pop, click, ee, or one you --record'ed")


def sound_path(name):
    p = SOUND_DIR / f"{name}.wav"
    if not p.exists():
        _write(p, _synth(name))
    return p


class Player:
    def __init__(self, path):
        self.path = str(path)
        self.ok = sys.platform == "win32"
        if not self.ok:
            print("(not Windows: sounds are only printed, not played)")

    def play(self):
        if self.ok:
            import winsound
            winsound.PlaySound(self.path, winsound.SND_FILENAME | winsound.SND_ASYNC)


def record(name, seconds=1.2):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("recording needs:  python -m pip install sounddevice")
    input(f"Press Enter, then make your sound once (recording {seconds:.1f} s)...")
    rec = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    x = [float(v) for v in rec[:, 0]]
    # trim to the loud part (+ a little margin)
    loud = [i for i, v in enumerate(x) if abs(v) > 0.2 * max(abs(u) for u in x)]
    a, b = max(0, loud[0] - int(0.03 * RATE)), min(len(x), loud[-1] + int(0.08 * RATE))
    _write(SOUND_DIR / f"{name}.wav", x[a:b])
    print(f"saved {SOUND_DIR / f'{name}.wav'} ({(b - a) / RATE * 1000:.0f} ms). Use it with --sound {name}")


def main():
    from .gestures import add_style_args
    from .users import add_user_arg, require_user
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_user_arg(ap, required=False)
    add_style_args(ap, move=False)
    ap.add_argument("--sound", default="pop", help="pop | click | ee | a name you recorded")
    ap.add_argument("--sides", default=None, help="default: the bands in bands.json")
    ap.add_argument("--mode", choices=["lda", "threshold"], default="lda")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--test", action="store_true", help="no band: press Enter to play the sound")
    ap.add_argument("--record", metavar="NAME", help="record your own sound as NAME")
    ap.add_argument("--min-gap", type=float, default=0.6,
                    help="ignore fists closer together than this (s): iOS needs a gap between switch presses")
    from .run import add_tune_args
    add_tune_args(ap)
    args = ap.parse_args()

    if args.record:
        record(args.record)
        return
    player = Player(sound_path(args.sound))
    if args.test:
        print(f"Enter = play '{args.sound}'. Ctrl+C to stop.")
        try:
            while True:
                input()
                player.play()
                print("  ♪", flush=True)
        except (KeyboardInterrupt, EOFError):
            return

    require_user(ap, args)
    from .device import default_sides
    from .run import start_bands, tune_from_args
    sides = [s.strip() for s in (args.sides or default_sides(args.fake)).split(",") if s.strip()]
    q, procs = start_bands(args.user, sides, args.mode, args.fake, False, False, args.click, C.MOVE, C.AXES,
                           tune_from_args(args))
    print(f"sound switch: every {args.click} on {', '.join(sides)} plays '{args.sound}'. Ctrl+C to stop.")
    last = 0.0
    n = 0
    try:
        while True:
            try:
                kind, a, b, _t = q.get(timeout=0.2)
            except Exception:
                continue
            if kind == "ready":
                print(f"[{a}] ready")
            elif kind == "btn" and b == "press":
                now = time.perf_counter()
                if now - last >= args.min_gap:
                    player.play()
                    n += 1
                    last = now
                    print(f"  {a} {args.click} -> '{args.sound}'  ({n})", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()

"""Live gesture readout from one band: shows which of the trained gestures you're doing.

    python -m myomouse.gestures11.live --user julius --side right
    python -m myomouse.gestures11.live --user julius --side right --reject 0.9   # stricter

Shows "Neutral" unless a gesture gets through all three filters (see gestures11/decide.py):
rest gate, rejection threshold, and N_ON decisions in a row. The line also shows the raw
classifier guess and why it was held back, so you can see what each filter does.

GestureClassifier is the streaming part, for use in other scripts:

    clf = GestureClassifier(joblib.load(model_path(user, side)))
    band.add_emg_handler(lambda s: (g := clf.push(s)) and print("now:", g))
"""

import argparse
import sys
import time
from collections import deque

import joblib
import numpy as np

from ..device import open_band
from ..features import envelope, window_features
from ..record import BOLD, CLEAR_LINE, DIM, RESET, bar
from ..users import add_user_arg
from . import GESTURES, IDLE_KEYS, N_OFF, N_ON, REJECT, STEP, model_path
from .decide import NONE, Switcher, raw_decision


class GestureClassifier:
    """Feed raw EMG samples; push() returns the gesture key when the shown gesture changes.
    "neutral" = no gesture (neutral, rest, too quiet or not sure enough)."""

    def __init__(self, model: dict, reject=None, n_on=None, n_off=None):
        self.model = model
        self.classes = model["classes"]
        # how often to decide and how many in a row are live settings, not part of the model,
        # so a faster setting in __init__.py works without retraining
        self.window, self.step = model["window"], STEP
        self.gate = model.get("gate")
        self.reject = model.get("reject", REJECT) if reject is None else reject
        self.switch = Switcher(N_ON if n_on is None else n_on, N_OFF if n_off is None else n_off)
        self.buf = deque(maxlen=self.window)
        self.since = 0
        self.proba = None  # latest probabilities, in model["classes"] order
        self.env = 0.0     # latest EMG level
        self.why = ""      # why the latest raw guess was held back

    @property
    def current(self):
        return self.switch.current

    def push(self, sample):
        self.buf.append(sample)
        self.since += 1
        if len(self.buf) < self.window or self.since < self.step:
            return None
        self.since = 0
        win = np.asarray(self.buf, dtype=np.int8)
        self.env = envelope(win)
        self.proba = self.model["pipeline"].predict_proba(window_features(win)[None, :])[0]
        d = raw_decision(self.proba, self.env, self.classes, self.gate, self.reject)
        best = self.classes[int(np.argmax(self.proba))]
        if d == NONE and best not in IDLE_KEYS:
            self.why = "quiet (rest gate)" if self.gate is not None and self.env < self.gate else "not sure enough"
        elif d != NONE and d != self.current:
            self.why = "confirming"
        else:
            self.why = ""
        return self.switch.update(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_user_arg(ap)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--reject", type=float, default=None,
                    help="rejection threshold (default: what train saved, usually 0.75)")
    ap.add_argument("--fake", action="store_true", help="simulated band, for testing")
    args = ap.parse_args(argv)

    p = model_path(args.user, args.side)
    if not p.exists():
        raise SystemExit(f"No gesture model at {p}. Run gestures11.record + gestures11.train first.")
    model = joblib.load(p)
    clf = GestureClassifier(model, args.reject)
    name = {g.key: g.name for g in GESTURES}
    band = open_band(args.side, fake=args.fake)
    band.add_emg_handler(clf.push)
    band.start()
    gate = "off (no rest recorded)" if clf.gate is None else f"{clf.gate:.2f}"
    print(f"{BOLD}{args.side} arm{RESET}: {len(clf.classes)} classes ({model['kind']}), reject < {clf.reject:.2f}, "
          f"rest gate {gate}. Ctrl+C to stop.\n")
    try:
        while band.alive:
            if clf.proba is not None:
                i = int(np.argmax(clf.proba))
                guess = f"guess: {name[clf.classes[i]]} {clf.proba[i]:.0%}"
                if clf.why:
                    guess += f" ({clf.why})"
                cur = clf.current
                pc = clf.proba[clf.classes.index(cur)] if cur in clf.classes and cur != NONE else 0.0
                print(f"{CLEAR_LINE}{BOLD}{name.get(cur, cur):^18}{RESET} {bar(pc, 12)} "
                      f"{DIM}level {clf.env:4.2f}   {guess}{RESET}", end="", flush=True)
            time.sleep(0.05)
        print(f"\nband disconnected: {band.error}")
    except KeyboardInterrupt:
        print()
    finally:
        band.stop()


if __name__ == "__main__":
    sys.exit(main())

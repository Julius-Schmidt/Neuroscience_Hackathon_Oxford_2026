"""Streaming click detector: ring buffer -> features -> model -> debounced press/release.

Works the same for both click gestures (fist / index lift); only the trained model differs.
"""

import math
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .features import envelope, window_features
from .users import model_dir


def model_path(user: str, side: str, click: str = "fist") -> Path:
    return model_dir(user) / f"{click}_{side}.joblib"


def load_model(user: str, side: str, click: str = "fist") -> dict:
    p = model_path(user, side, click)
    if not p.exists():
        raise FileNotFoundError(f"No {click} model for {user}'s {side} arm at {p}. Run record + train first.")
    return joblib.load(p)


class Debouncer:
    """Hysteresis + N-in-a-row so a click never flickers. Returns 'press', 'release' or None."""

    def __init__(self, p_on=C.P_ON, p_off=C.P_OFF, n_on=C.N_ON, n_off=C.N_OFF):
        self.p_on, self.p_off, self.n_on, self.n_off = p_on, p_off, n_on, n_off
        self.pressed = False
        self.count = 0

    def update(self, p: float):
        if not self.pressed:
            self.count = self.count + 1 if p >= self.p_on else 0
            if self.count >= self.n_on:
                self.pressed, self.count = True, 0
                return "press"
        else:
            self.count = self.count + 1 if p <= self.p_off else 0
            if self.count >= self.n_off:
                self.pressed, self.count = False, 0
                return "release"
        return None


class FistDetector:
    """Feed raw EMG samples one at a time; get press/release events back (fist or index model)."""

    def __init__(self, model: dict, mode: str = "lda", hold_ms=None, rescale=False):
        if mode not in ("lda", "threshold"):
            raise ValueError(mode)
        from .adapt import Rescaler, hold_to_n_on
        self.model, self.mode = model, mode
        self.window = model["window"]
        self.step = model["step"]
        self.buf = np.zeros((self.window, C.N_CHANNELS), dtype=np.int8)
        self.filled = 0
        self.since_last = 0
        n_on = hold_to_n_on(hold_ms, model["step"])
        self.debounce = Debouncer(n_on=n_on) if n_on else Debouncer()
        self.rescaler = Rescaler(model) if rescale and "threshold" in model else None
        self.gain = 1.0
        self.blocked = False      # set by the motion gate: no NEW presses while True
        self.last_p = 0.0

    def probability(self, win: np.ndarray) -> float:
        if self.mode == "threshold":
            t = self.model["threshold"]
            z = (envelope(win) - t["center"]) / t["scale"]
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        f = window_features(win)[None, :]
        return float(self.model["pipeline"].predict_proba(f)[0, 1])

    def push(self, sample):
        """sample: 8 ints. Returns 'press' / 'release' / None."""
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = sample
        self.filled = min(self.filled + 1, self.window)
        self.since_last += 1
        if self.filled < self.window or self.since_last < self.step:
            return None
        self.since_last = 0
        win = self.buf
        if self.rescaler is not None:
            self.gain = self.rescaler.feed(win)
            if self.gain != 1.0:
                win = np.clip(np.round(win.astype(np.float32) * self.gain), -128, 127).astype(np.int8)
        self.last_p = self.probability(win)
        if self.blocked and not self.debounce.pressed:
            self.debounce.count = 0
            return None
        return self.debounce.update(self.last_p)

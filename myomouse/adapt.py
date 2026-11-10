"""Per-user adaptations for the live click detector (all opt-in; defaults change nothing).

    hold_ms       how long the fist must be held before it clicks. Longer = tremor bursts can't
                  click (tremor_sim: 200 ms took false clicks from 77/min to ~0 at tremor half
                  the size of a squeeze), at the cost of each click landing later.
    rescale       follow the user's CURRENT squeeze strength and scale the signal back up before
                  the classifier, so a tired / weaker squeeze looks like the ones it was trained on.
                  Same idea as devtool/fatigue.py (Kalman-tracked burst peaks), measured on the
                  raw signal so near-misses that didn't click still count. Gain capped at 6x.
    motion_gate   ignore new clicks while the forearm is turning faster than this (deg/s, from
                  the band's gyro): arm movement is the main source of false clicks.

`StrengthTracker` is a 1-D Kalman filter (docs/REFERENCES.md [13]) on squeeze peaks: rest = 0, the trained squeeze = 1.
"""

from __future__ import annotations

import math

import numpy as np


class StrengthTracker:
    def __init__(self, q=0.0015, r=0.05, warmup=5, max_gain=6.0):
        self.q, self.r, self.warmup, self.max_gain = q, r, warmup, max_gain
        self.x = None          # current typical squeeze peak (1.0 = as trained)
        self.P = 0.0
        self._warm = []

    def update(self, peak: float):
        if not np.isfinite(peak) or peak <= 0:
            return
        if self.x is None:
            self._warm.append(float(peak))
            if len(self._warm) >= self.warmup:
                self.x = float(np.median(self._warm))
                self.P = float(np.var(self._warm)) or self.r
            return
        peak = min(peak, 3.0 * self.x)
        self.P += self.q
        k = self.P / (self.P + self.r)
        self.x += k * (peak - self.x)
        self.P *= 1 - k

    @property
    def strength(self) -> float:
        """current squeeze size relative to the trained squeeze (1.0 until warmed up)."""
        return 1.0 if self.x is None else self.x

    def gain(self) -> float:
        return float(np.clip(1.0 / max(self.strength, 1e-6), 1.0 / 1.5, self.max_gain))


class BurstWatcher:
    """Finds squeeze bursts on an activation stream (one value per decision). Returns a burst's
    peak when it ends. A burst starts above `start` and ends below 60% of that."""

    def __init__(self, min_len=4, max_len=600):
        self.min_len, self.max_len = min_len, max_len
        self.active, self.peak, self.len, self.level = False, 0.0, 0, 0.0

    def push(self, act: float, start: float):
        if not self.active:
            if act > start:
                self.active, self.peak, self.len, self.level = True, act, 1, start
            return None
        self.peak, self.len = max(self.peak, act), self.len + 1
        if act < 0.6 * self.level or self.len >= self.max_len:
            self.active = False
            return self.peak if self.len >= self.min_len else None
        return None


class Rescaler:
    """Live strength rescaling. feed(raw window) -> gain to multiply the window by."""

    def __init__(self, model: dict, floor: float = 0.07):
        t = model["threshold"]                         # log-RMS midpoint and spread from training
        self.rest = math.expm1(t["center"] - 3 * t["scale"])
        self.fist = math.expm1(t["center"] + 3 * t["scale"])
        self.rest_live = self.rest   # quiet level, followed live (training "rest" includes arm movement)
        self.floor = floor
        self.tracker = StrengthTracker()
        self.watch = BurstWatcher()

    def activation(self, win: np.ndarray) -> float:
        x = win.astype(np.float32)
        x = x - x.mean(axis=0)
        rms = float(np.sqrt((x * x).mean()))
        if rms < self.rest_live:      # quiet level: falls quickly, rises very slowly
            self.rest_live += 0.1 * (rms - self.rest_live)
        else:
            self.rest_live += 0.0005 * (rms - self.rest_live)
        return (rms - self.rest_live) / max(self.fist - self.rest_live, 1e-6)

    def feed(self, win: np.ndarray) -> float:
        act = self.activation(win)
        start = max(0.25 * self.tracker.strength, self.floor)   # half of the current click level
        peak = self.watch.push(act, start)
        if peak is not None:
            self.tracker.update(peak)
        return self.tracker.gain()


def hold_to_n_on(hold_ms: float | None, step: int, emg_hz: int = 200):
    """Hold time -> consecutive decisions needed (one decision every `step` samples)."""
    if not hold_ms:
        return None
    return max(1, int(round(hold_ms / (1000 * step / emg_hz))))

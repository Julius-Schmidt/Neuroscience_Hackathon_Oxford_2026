"""Eyes-closed safety switch from posterior alpha (8-12 Hz).

Every STEP_S (100 ms) the last WIN_S (1 s) of posterior EEG gives one alpha value
in dB. The first BASELINE_S of eyes-open EEG sets this person's baseline. Then:

    score = alpha_dB - baseline_dB
    score > ON_DB  -> looks closed     score < OFF_DB -> looks open    (between: keep state)

    eyes closed for DWELL_S   -> one "safety_toggle", then disarm
    eyes open for REARM_S     -> armed again

Blinks (< ~1 s) never reach DWELL_S; the hysteresis band stops flicker near the
threshold; windows with huge posterior artefacts are skipped, not counted.

The same AlphaSwitch class runs live (run_detector.py) and offline (evaluate.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import config as C


@dataclass
class Params:
    srate: float = C.SRATE
    feature: str = C.FEATURE
    win_s: float = C.WIN_S
    step_s: float = C.STEP_S
    baseline_s: float = C.BASELINE_S
    on_db: float = C.ON_DB
    off_db: float = C.OFF_DB
    dwell_s: float = C.DWELL_S
    rearm_s: float = C.REARM_S
    artifact_uv: float = C.ARTIFACT_UV


POST_IDX = [C.EEG_CHANNELS.index(c) for c in C.POSTERIOR]


def alpha_db(win: np.ndarray, srate: float, feature: str = "rel") -> float:
    """win: (n_samples, n_posterior). Alpha power in dB (absolute or relative to 2-30 Hz)."""
    f, p = signal.welch(win, fs=srate, nperseg=len(win), axis=0, detrend="linear", window="hann")
    a = p[(f >= C.ALPHA_BAND[0]) & (f <= C.ALPHA_BAND[1])].sum()
    if feature == "abs":
        return float(10 * np.log10(a + 1e-12))
    ref = p[(f >= C.REF_BAND[0]) & (f <= C.REF_BAND[1])].sum()
    return float(10 * np.log10(a / (ref + 1e-12) + 1e-12))


class AlphaSwitch:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.win_n = int(round(self.p.win_s * self.p.srate))
        self.step_n = max(1, int(round(self.p.step_s * self.p.srate)))
        self.buf = deque(maxlen=self.win_n)
        self.since = 0
        self.n_seen = 0
        self.baseline_vals: list[float] = []
        self.baseline = None
        self.closed_t = 0.0
        self.open_t = 0.0
        self.in_closed = False
        self.armed = True
        self.score = 0.0
        self.recent_scores = deque(maxlen=max(1, int(self.p.dwell_s / self.p.step_s)))
        self.n_artifacts = 0

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    def push(self, samples: np.ndarray, timestamps: np.ndarray | None = None) -> list[tuple[float, float]]:
        """Feed (n, 8) EEG in µV. Returns toggles as (timestamp, confidence)."""
        samples = np.atleast_2d(samples)
        if timestamps is None:
            timestamps = (self.n_seen + np.arange(len(samples))) / self.p.srate
        out = []
        for x, ts in zip(samples, timestamps):
            self.buf.append(x[POST_IDX])
            self.n_seen += 1
            self.since += 1
            if len(self.buf) == self.win_n and self.since >= self.step_n:
                self.since = 0
                ev = self._decide(float(ts))
                if ev:
                    out.append(ev)
        return out

    def _decide(self, ts: float):
        win = np.asarray(self.buf, dtype=np.float64)
        if np.ptp(win, axis=0).max() > self.p.artifact_uv:
            self.n_artifacts += 1
            return None                                   # hold state, don't count time
        val = alpha_db(win, self.p.srate, self.p.feature)

        if self.baseline is None:
            self.baseline_vals.append(val)
            if self.n_seen >= self.p.baseline_s * self.p.srate:
                self.baseline = float(np.median(self.baseline_vals))
            return None

        dt = self.p.step_s
        self.score = s = val - self.baseline
        self.recent_scores.append(s)
        if s > self.p.on_db or (self.in_closed and s >= self.p.off_db):
            self.in_closed = True
            self.closed_t += dt
            self.open_t = 0.0
        else:
            self.in_closed = False
            self.closed_t = 0.0
            if s < self.p.off_db:
                self.open_t += dt

        if not self.armed and self.open_t >= self.p.rearm_s:
            self.armed = True
        if self.armed and self.closed_t >= self.p.dwell_s - 1e-9:
            self.armed = False
            margin = float(np.median(self.recent_scores)) - self.p.on_db
            conf = 1.0 / (1.0 + np.exp(-margin / 2.0))
            return ts, float(conf)
        return None


def run_offline(eeg: np.ndarray, params: Params | None = None) -> tuple[list[tuple[float, float]], AlphaSwitch]:
    sw = AlphaSwitch(params)
    toggles = sw.push(eeg)
    return toggles, sw

"""Fatigue-adaptive clicking: track how strong this user's squeezes are *now*, and adapt to it.

As someone tires (or as MS fatigue sets in) their squeezes get weaker, so a fixed click
threshold starts missing clicks. FatigueTracker follows the user's current typical squeeze
strength with a 1-D Kalman filter and turns it into two things:

    threshold(base)  the click threshold, scaled with current strength   (threshold detector)
    gain()           a factor to multiply raw EMG by before features      (LDA: tired squeezes
                     are rescaled to look like fresh ones, the model itself is unchanged)

Measurements: the peak activation of every squeeze "burst" (rest = 0, calibrated typical
squeeze = 1). Bursts are found on the activation stream independently of clicks, so a weak
squeeze that *didn't* reach the threshold (a near-miss) still counts. Without that, the filter
would only ever see squeezes strong enough to click, and would never notice fatigue.

Safety: the threshold never drops below `floor` (rest noise + margin), the gain is capped,
and the first `warmup` bursts only set the starting strength (threshold stays fixed meanwhile).
Real fatigue can make EMG *rise* before it falls (Leeb et al. 2011), so the random walk lets
strength move both ways; the threshold is capped at `max_up` x base on the way up.
"""

from __future__ import annotations

import numpy as np

from .emg import ClickDetector, BandCal


class FatigueTracker:
    def __init__(self, q=0.0015, r=0.05, warmup=5, floor=0.12, max_gain=2.5, max_up=1.5):
        self.q, self.r = q, r            # process noise per burst, measurement noise (activation^2)
        self.warmup, self.floor, self.max_gain, self.max_up = warmup, floor, max_gain, max_up
        self.x = None                    # current typical squeeze peak (activation units)
        self.P = 0.0
        self.x0 = None                   # starting strength (mean of warm-up bursts)
        self._warm = []
        self.n = 0                       # measurements used

    @property
    def ready(self):
        return self.x0 is not None

    def update(self, peak: float):
        peak = float(peak)
        if not np.isfinite(peak) or peak <= 0:
            return
        if self.x0 is None:
            self._warm.append(peak)
            if len(self._warm) >= self.warmup:
                self.x0 = self.x = float(np.median(self._warm))
                self.P = float(np.var(self._warm)) or self.r
            return
        peak = min(peak, 3.0 * self.x)   # one huge squeeze shouldn't yank the estimate
        self.P += self.q
        k = self.P / (self.P + self.r)
        self.x += k * (peak - self.x)
        self.P *= (1 - k)
        self.n += 1

    @property
    def ratio(self):
        """current strength / starting strength (1.0 = fresh, 0.6 = squeezes 40% weaker)."""
        return 1.0 if not self.ready else self.x / self.x0

    def threshold(self, base: float) -> float:
        return float(min(max(base * self.ratio, self.floor), base * self.max_up))

    def gain(self) -> float:
        return float(np.clip(1.0 / max(self.ratio, 1e-6), 1.0 / self.max_up, self.max_gain))


class BurstWatcher:
    """Finds squeeze bursts on the activation stream. push() returns a burst's peak when it ends.

    start: activation > start_frac x threshold     end: activation < end_frac x start level
    Bursts shorter than min_len samples (40 ms) are ignored as noise.
    """

    def __init__(self, start_frac=0.5, end_frac=0.6, min_len=8, max_len=1200):
        self.start_frac, self.end_frac, self.min_len, self.max_len = start_frac, end_frac, min_len, max_len
        self.active, self.peak, self.len, self.level = False, 0.0, 0, 0.0

    def push(self, act: float, threshold: float, floor: float):
        start = max(self.start_frac * threshold, floor)
        if not self.active:
            if act > start:
                self.active, self.peak, self.len, self.level = True, act, 1, start
            return None
        self.peak, self.len = max(self.peak, act), self.len + 1
        if act < self.end_frac * self.level or self.len >= self.max_len:
            self.active = False
            return self.peak if self.len >= self.min_len else None
        return None


class AdaptiveClickDetector(ClickDetector):
    """ClickDetector whose threshold follows a FatigueTracker. Same push() interface."""

    def __init__(self, cal: BandCal, threshold=0.35, refractory_ms=250.0, tracker: FatigueTracker | None = None):
        super().__init__(cal, threshold, refractory_ms)
        self.base = threshold
        floor = max(1.5 * (cal.rest_act_p99 or 0.08), 0.12)
        self.tracker = tracker or FatigueTracker(floor=floor)
        self.watch = BurstWatcher()
        self.last_burst_peak = None

    def push(self, idx: int, sample) -> bool:
        fired = super().push(idx, sample)
        peak = self.watch.push(self.activation, self.threshold, self.tracker.floor)
        if peak is not None:
            self.last_burst_peak = peak
            self.tracker.update(peak)
            self.threshold = self.tracker.threshold(self.base)
        return fired


def detect_offline_adaptive(emg: np.ndarray, idx: np.ndarray, cal: BandCal, threshold, refractory_ms,
                            **tracker_kw):
    """Like emg.detect_offline, with the fatigue-adaptive threshold.

    Returns (click sample indices, trace) where trace is a list of
    (sample_idx, burst_peak, threshold_after, strength_ratio) per burst.
    """
    d = AdaptiveClickDetector(cal, threshold, refractory_ms,
                              FatigueTracker(floor=max(1.5 * (cal.rest_act_p99 or 0.08), 0.12), **tracker_kw))
    clicks, trace = [], []
    for i, s in zip(idx, emg):
        n_before = d.tracker.n + len(d.tracker._warm)
        if d.push(int(i), s):
            clicks.append(int(i))
        if d.tracker.n + len(d.tracker._warm) != n_before:
            trace.append((int(i), d.last_burst_peak, d.threshold, d.tracker.ratio))
    return np.array(clicks, dtype=np.int64), trace


# --------------------------------------------------------------------------- LDA (myomouse) path
def gain_series(emg: np.ndarray, idx: np.ndarray, cal: BandCal, threshold, **tracker_kw) -> np.ndarray:
    """Per-sample gain from the fatigue tracker, measured on the *unscaled* signal.

    The tracker watches the calibrated activation (like the threshold detector does) and
    gain() says how much to scale raw EMG so a tired squeeze looks like a fresh one."""
    floor = max(1.5 * (cal.rest_act_p99 or 0.08), 0.12)
    tr = FatigueTracker(floor=floor, **tracker_kw)
    act = ClickDetector(cal, threshold, 250.0)   # used only for its activation value
    watch = BurstWatcher()
    g = np.ones(len(emg), dtype=np.float32)
    for k, (i, s) in enumerate(zip(idx, emg)):
        act.push(int(i), s)
        peak = watch.push(act.activation, tr.threshold(threshold), tr.floor)
        if peak is not None:
            tr.update(peak)
        g[k] = tr.gain()
    return g


def detect_offline_lda(emg: np.ndarray, idx: np.ndarray, model: dict, gain: np.ndarray | None = None):
    """Run Julias's LDA fist detector (myomouse) over a recording, exactly as live:
    a window of model['window'] samples evaluated every model['step'] samples, then the same
    hysteresis debouncer. Returns sample indices where a press fired.

    gain: optional per-sample factor applied to the raw EMG first (fatigue-adaptive LDA).
    The model itself is never changed."""
    from myomouse.detector import Debouncer
    from myomouse.features import sliding_features
    x = emg.astype(np.float32)
    if gain is not None:
        x = np.clip(np.round(x * gain[:, None]), -128, 127)
    X, ends = sliding_features(x, model["window"], model["step"])
    if len(ends) == 0:
        return np.array([], dtype=np.int64)
    p = model["pipeline"].predict_proba(X)[:, 1]
    deb = Debouncer()
    presses = [int(idx[e - 1]) for e, pk in zip(ends, p) if deb.update(float(pk)) == "press"]
    return np.array(presses, dtype=np.int64)

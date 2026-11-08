"""Pure signal logic: envelope, per-user calibration, threshold click detector, cursor smoothing.

Nothing here touches Bluetooth or the screen, so it is shared by the live app and by replay.

Activation (what the threshold is compared against):
    per channel: MAV over the last ENV_WIN samples, scaled so rest = 0 and a typical
    contraction = 1 (from calibration), then averaged over the 8 channels.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

EMG_HZ = 200
SAMPLE_MS = 1000.0 / EMG_HZ          # 5 ms per sample
N_CH = 8
ENV_WIN = 20                         # samples (100 ms) for the click envelope
ONSET_WIN = 5                        # samples (25 ms) for the faster onset envelope
RELEASE_FRAC = 0.6                   # release when activation < threshold * this (hysteresis)

DEFAULT_REST = 2.0                   # used before a user is calibrated (int8 MAV units)
DEFAULT_PEAK = 30.0


# --------------------------------------------------------------------------- calibration
@dataclass
class BandCal:
    rest: list = field(default_factory=lambda: [DEFAULT_REST] * N_CH)
    peak: list = field(default_factory=lambda: [DEFAULT_PEAK] * N_CH)
    rest_act_p99: float | None = None    # 99th pct of activation at rest, after scaling
    peak_act_median: float | None = None
    calibrated: bool = False

    def arrays(self):
        rest = np.asarray(self.rest, dtype=np.float32)
        span = np.maximum(np.asarray(self.peak, dtype=np.float32) - rest, 1.0)  # never divide by ~0
        return rest, span


def calib_path(data_dir: Path, user: str) -> Path:
    return data_dir / "calib" / f"{user}.json"


def load_calibration(data_dir: Path, user: str) -> dict[str, BandCal]:
    p = calib_path(data_dir, user)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {side: BandCal(**{k: v for k, v in d.items() if k in BandCal.__dataclass_fields__})
            for side, d in raw.get("bands", {}).items()}


def save_calibration(data_dir: Path, user: str, cals: dict[str, BandCal]):
    p = calib_path(data_dir, user)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else {"bands": {}}
    for side, c in cals.items():  # keep the other band's calibration if only one was redone
        existing["bands"][side] = asdict(c)
    existing.update(user=user, saved=time.strftime("%Y-%m-%d %H:%M:%S"), env_win=ENV_WIN)
    p.write_text(json.dumps(existing, indent=2))
    return p


def mav_series(emg: np.ndarray, win: int = ENV_WIN) -> np.ndarray:
    """(n, 8) int8 -> (n, 8) trailing moving average of |x| (first win-1 rows use what exists)."""
    a = np.abs(emg.astype(np.float32))
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[win:] = c[win:] - c[:-win]
    n = np.minimum(np.arange(1, len(a) + 1), win)[:, None]
    return out / n


def compute_calibration(rest_emg: np.ndarray, squeeze_emg: np.ndarray) -> BandCal:
    """rest_emg: samples recorded at rest; squeeze_emg: samples from the squeeze cues only.

    rest = median MAV at rest, peak = 90th percentile of MAV during squeezes (typical, not max).
    """
    if len(rest_emg) < ENV_WIN * 2 or len(squeeze_emg) < ENV_WIN * 2:
        raise ValueError("not enough calibration data (band disconnected?)")
    m_rest = mav_series(rest_emg)[ENV_WIN:]
    m_sq = mav_series(squeeze_emg)[ENV_WIN:]
    rest = np.median(m_rest, axis=0)
    peak = np.percentile(m_sq, 90, axis=0)
    cal = BandCal(rest=[round(float(v), 3) for v in rest], peak=[round(float(v), 3) for v in peak],
                  calibrated=True)
    r, s = cal.arrays()
    act_rest = np.clip((m_rest - r) / s, 0, None).mean(axis=1)
    act_sq = np.clip((m_sq - r) / s, 0, None).mean(axis=1)
    cal.rest_act_p99 = round(float(np.percentile(act_rest, 99)), 3)
    cal.peak_act_median = round(float(np.median(act_sq)), 3)
    return cal


def activation_series(emg: np.ndarray, cal: BandCal, win: int = ENV_WIN) -> np.ndarray:
    r, s = cal.arrays()
    return np.clip((mav_series(emg, win) - r) / s, 0, None).mean(axis=1)


# --------------------------------------------------------------------------- detector
class ClickDetector:
    """Push one sample at a time. Returns True on the sample where a click (press) fires.

    press:   activation >= threshold, not already pressed, and >= refractory_ms since last press
             (a crossing inside the refractory period is swallowed, not delayed)
    release: activation < threshold * RELEASE_FRAC  (must release before the next press)
    Refractory is measured in sample time (sample index x 5 ms), not wall time.
    """

    def __init__(self, cal: BandCal, threshold=0.35, refractory_ms=250.0):
        self.threshold, self.refractory_ms = threshold, refractory_ms
        self.set_cal(cal)
        self.buf = np.zeros((ENV_WIN, N_CH), dtype=np.float32)
        self.sum = np.zeros(N_CH, dtype=np.float32)
        self.fast = np.zeros((ONSET_WIN, N_CH), dtype=np.float32)
        self.fsum = np.zeros(N_CH, dtype=np.float32)
        self.i = 0
        self.pressed = False
        self.last_press_idx = -10**9
        self.activation = 0.0
        self.fast_activation = 0.0

    def set_cal(self, cal: BandCal):
        self.cal = cal
        self.rest, self.span = cal.arrays()

    def push(self, idx: int, sample) -> bool:
        a = np.abs(np.asarray(sample, dtype=np.float32))
        k = self.i % ENV_WIN
        self.sum += a - self.buf[k]
        self.buf[k] = a
        kf = self.i % ONSET_WIN
        self.fsum += a - self.fast[kf]
        self.fast[kf] = a
        self.i += 1
        n = min(self.i, ENV_WIN)
        self.activation = float(np.clip((self.sum / n - self.rest) / self.span, 0, None).mean())
        nf = min(self.i, ONSET_WIN)
        self.fast_activation = float(np.clip((self.fsum / nf - self.rest) / self.span, 0, None).mean())
        if self.pressed:
            if self.activation < self.threshold * RELEASE_FRAC:
                self.pressed = False
            return False
        if self.activation >= self.threshold:
            # inside the refractory period a crossing is swallowed, and it still has to be released
            # before the next click (so a bounce can't fire late, once refractory ends)
            self.pressed = True
            if (idx - self.last_press_idx) * SAMPLE_MS >= self.refractory_ms:
                self.last_press_idx = idx
                return True
        return False


def detect_offline(emg: np.ndarray, idx: np.ndarray, cal: BandCal, threshold, refractory_ms):
    """Run the live detector over a recording. Returns array of sample indices where clicks fired."""
    d = ClickDetector(cal, threshold, refractory_ms)
    return np.array([i for i, s in zip(idx, emg) if d.push(int(i), s)], dtype=np.int64)


def find_onset(fast_act: np.ndarray, detect_pos: int, threshold: float, frac=0.5, max_back=80) -> int:
    """Walk back from the detection sample to where the fast (25 ms) activation was last below
    frac * threshold. Returns the position of the first sample above it (the EMG onset)."""
    level = threshold * frac
    lo = max(0, detect_pos - max_back)
    for p in range(detect_pos, lo - 1, -1):
        if fast_act[p] < level:
            return p + 1
    return lo


# --------------------------------------------------------------------------- cursor
class Cursor:
    """IMU quaternion -> pixel deltas, same maths as myomouse.pointer but smoothing is a parameter."""

    def __init__(self, gain_x, gain_y, invert_x=False, invert_y=False, deadzone=0.0015, smoothing=0.5,
                 y_axis="pitch"):
        self.y_axis = y_axis   # "pitch" = arm in the air, "roll" = forearm resting on a table
        self.gain_x, self.gain_y = gain_x, gain_y
        self.invert_x, self.invert_y = invert_x, invert_y
        self.deadzone, self.smoothing = deadzone, smoothing
        self.prev = None
        self.vx = self.vy = 0.0

    def update(self, quat):
        from myomouse.pointer import roll_angle, wrap, yaw_pitch
        yaw, pitch = yaw_pitch(quat)
        if self.y_axis == "roll":
            pitch = roll_angle(quat)
        if self.prev is None:
            self.prev = (yaw, pitch)
            return 0.0, 0.0
        dyaw, dpitch = wrap(yaw - self.prev[0]), wrap(pitch - self.prev[1])
        self.prev = (yaw, pitch)
        dyaw = 0.0 if abs(dyaw) < self.deadzone else dyaw
        dpitch = 0.0 if abs(dpitch) < self.deadzone else dpitch
        dx = -dyaw * self.gain_x * (-1 if self.invert_x else 1)
        dy = -dpitch * self.gain_y * (-1 if self.invert_y else 1)
        a = min(max(self.smoothing, 0.0), 0.98)
        self.vx = a * self.vx + (1 - a) * dx
        self.vy = a * self.vy + (1 - a) * dy
        return self.vx, self.vy


def smooth_point(prev, new, smoothing):
    """Exponential smoothing for the mouse-driven cursor, so the smoothing setting applies there too."""
    if prev is None:
        return new
    a = min(max(smoothing, 0.0), 0.98)
    return (a * prev[0] + (1 - a) * new[0], a * prev[1] + (1 - a) * new[1])


def pct(v, q):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.percentile(v, q)) if v else float("nan")

"""Classic time-domain EMG features (Hudgins set; docs/REFERENCES.md [1], [2]). Fast enough to run every 10 ms."""

import numpy as np

FEATURE_NAMES = ["MAV", "RMS", "WL", "ZC", "SSC"]
ZC_THRESH = 2.0  # ignore sign changes smaller than this (int8 units) to reject noise


def window_features(win: np.ndarray) -> np.ndarray:
    """win: (n_samples, n_channels) -> flat vector of len(FEATURE_NAMES) * n_channels."""
    x = win.astype(np.float32)
    x = x - x.mean(axis=0)  # remove per-window DC offset
    mav = np.abs(x).mean(axis=0)
    rms = np.sqrt((x * x).mean(axis=0))
    d = np.diff(x, axis=0)
    wl = np.abs(d).sum(axis=0) / len(x)
    zc = ((x[:-1] * x[1:] < 0) & (np.abs(d) >= ZC_THRESH)).mean(axis=0)
    ssc = ((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) >= ZC_THRESH) | (np.abs(d[1:]) >= ZC_THRESH))).mean(axis=0)
    # log-compress the amplitude features: makes them far more linear-separable across people
    return np.concatenate([np.log1p(mav), np.log1p(rms), np.log1p(wl), zc, ssc])


def sliding_features(emg: np.ndarray, window: int, step: int):
    """Feature matrix for every window ending at index `end` (exclusive). Returns (X, ends)."""
    ends = np.arange(window, len(emg) + 1, step)
    X = np.stack([window_features(emg[e - window:e]) for e in ends]) if len(ends) else np.empty((0, 0))
    return X, ends


def envelope(win: np.ndarray) -> float:
    """Single overall activation number, used by the threshold fallback."""
    x = win.astype(np.float32)
    x = x - x.mean(axis=0)
    return float(np.log1p(np.sqrt((x * x).mean())))

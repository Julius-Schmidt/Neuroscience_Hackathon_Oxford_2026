"""Signal steps: EMG onsets (the trigger), EMG peak and median frequency, EEG epochs,
band-power drop (ERD) and motor potential (MRCP)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import config as C


# ---------------- EMG ----------------

def emg_highpass(emg: np.ndarray, srate: float = C.EMG_SRATE) -> np.ndarray:
    sos = signal.butter(4, C.EMG_HP_HZ, "highpass", fs=srate, output="sos")
    return signal.sosfiltfilt(sos, emg - emg.mean(axis=0), axis=0)


def emg_envelope(emg: np.ndarray, srate: float = C.EMG_SRATE) -> np.ndarray:
    rect = np.abs(emg_highpass(emg, srate)).mean(axis=1)
    k = max(1, int(C.EMG_SMOOTH_S * srate))
    return np.convolve(rect, np.ones(k) / k, mode="same")


def detect_onsets(env: np.ndarray, t: np.ndarray, cue_t: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Squeeze onsets = rising threshold crossings that stay up >= ONSET_MIN_DUR_S.

    Threshold = max(rest median + K*MAD, FRACTION * 95th percentile). Onsets closer
    than ONSET_MIN_GAP_S to the previous one are merged; if squeeze cues exist, an
    onset must fall within CUE_MATCH_S of a cue (removes stray movements).
    """
    med = np.median(env)
    mad = np.median(np.abs(env - med)) * 1.4826
    thr = max(med + C.ONSET_K_MAD * mad, C.ONSET_FRACTION * np.percentile(env, 95))
    above = env > thr
    rises = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    onsets = []
    for i in rises:
        t_on = t[i]
        if onsets and t_on - onsets[-1] < C.ONSET_MIN_GAP_S:
            continue
        j = np.searchsorted(t, t_on + C.ONSET_MIN_DUR_S)
        if j <= i or not above[i:j].mean() > 0.8:
            continue
        onsets.append(t_on)
    onsets = np.array(onsets)
    if cue_t is not None and len(cue_t) and len(onsets):
        lag = onsets[:, None] - cue_t[None, :]
        ok = ((lag >= C.CUE_MATCH_S[0]) & (lag <= C.CUE_MATCH_S[1])).any(axis=1)
        onsets = onsets[ok]
    return onsets, float(thr)


def emg_peaks(env: np.ndarray, t: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    out = []
    for o in onsets:
        m = (t >= o + C.PEAK_WIN[0]) & (t <= o + C.PEAK_WIN[1])
        out.append(env[m].max() if m.any() else np.nan)
    return np.array(out)


def emg_median_freq(emg_hp: np.ndarray, t: np.ndarray, onsets: np.ndarray, srate: float = C.EMG_SRATE) -> np.ndarray:
    """Median frequency of the squeeze (0.2-1.2 s after onset). Falls with muscle fatigue."""
    out = []
    for o in onsets:
        m = (t >= o + 0.2) & (t <= o + 1.2)
        if m.sum() < 64:
            out.append(np.nan)
            continue
        f, p = signal.welch(emg_hp[m], fs=srate, nperseg=64, axis=0)
        p = p.mean(axis=1)
        c = np.cumsum(p)
        out.append(f[np.searchsorted(c, c[-1] / 2)])
    return np.array(out)


# ---------------- EEG ----------------

def reference(eeg: np.ndarray) -> np.ndarray:
    return eeg - eeg.mean(axis=1, keepdims=True) if C.REFERENCE == "car" else eeg


def bandpass(x: np.ndarray, lo: float, hi: float, srate: float = C.EEG_SRATE, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [lo, hi], "bandpass", fs=srate, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def epochs(x: np.ndarray, t: np.ndarray, onsets: np.ndarray, srate: float = C.EEG_SRATE):
    """Cut (n_onsets, n_times, n_ch) around each onset on a regular grid (interpolated
    from the LSL timestamps, so EEG/EMG clock differences are handled). Onsets whose
    epoch runs outside the recording or over a gap > 0.1 s are returned as invalid."""
    grid = np.arange(C.EPOCH[0], C.EPOCH[1], 1 / srate)
    out = np.full((len(onsets), len(grid), x.shape[1]), np.nan)
    valid = np.zeros(len(onsets), bool)
    for i, o in enumerate(onsets):
        a, b = o + C.EPOCH[0], o + C.EPOCH[1]
        lo, hi = np.searchsorted(t, a), np.searchsorted(t, b)
        if lo == 0 or hi >= len(t) or np.diff(t[lo - 1:hi + 1]).max() > 0.1:
            continue
        for c in range(x.shape[1]):
            out[i, :, c] = np.interp(o + grid, t, x[:, c])
        valid[i] = True
    return out, grid, valid


def power_epochs(eeg: np.ndarray, t: np.ndarray, onsets: np.ndarray, band) -> tuple:
    """Band power (Hilbert envelope squared) epochs on all channels."""
    bp = bandpass(reference(eeg), *band)
    pw = np.abs(signal.hilbert(bp, axis=0)) ** 2
    return epochs(pw, t, onsets)


def artifact_mask(eeg: np.ndarray, t: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """True = clean. Peak-to-peak of 1-40 Hz EEG on C3/Cz/C4 within the epoch."""
    x = bandpass(reference(eeg), 1.0, 40.0)
    ep, _, valid = epochs(x, t, onsets)
    idx = [C.EEG_CHANNELS.index(c) for c in ("C3", "Cz", "C4")]
    ptp = np.nanmax(ep[:, :, idx], axis=1) - np.nanmin(ep[:, :, idx], axis=1)
    return valid & (np.nanmax(ptp, axis=1) < C.ARTIFACT_UV)


def mrcp_epochs(eeg: np.ndarray, t: np.ndarray, onsets: np.ndarray):
    sos = signal.butter(2, [0.1, C.MRCP_LOWPASS], "bandpass", fs=C.EEG_SRATE, output="sos")
    x = signal.sosfiltfilt(sos, reference(eeg), axis=0)
    ep, grid, valid = epochs(x, t, onsets)
    base = (grid >= C.MRCP_BASE[0]) & (grid < C.MRCP_BASE[1])
    return ep - np.nanmean(ep[:, base], axis=1, keepdims=True), grid, valid


def erd_curve(power_trials: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """% change of trial-averaged power vs pre-movement baseline. Negative = drop (ERD)."""
    p = np.nanmean(power_trials, axis=0)
    base = (grid >= C.ERD_BASE[0]) & (grid < C.ERD_BASE[1])
    b = p[base].mean()
    return 100.0 * (p - b) / b


def effort_from_trials(trials: np.ndarray, grid: np.ndarray, measure: str) -> float:
    """One 'brain effort' number for a set of trials (bigger = more effort).

    beta_erd / mu_erd: % power drop in ERD_WIN (positive number)
    mrcp: -mean amplitude in MRCP_WIN (µV, positive = more negative potential)
    """
    if measure in ("beta_erd", "mu_erd"):
        curve = erd_curve(trials, grid)
        w = (grid >= C.ERD_WIN[0]) & (grid < C.ERD_WIN[1])
        return float(-curve[w].mean())
    w = (grid >= C.MRCP_WIN[0]) & (grid < C.MRCP_WIN[1])
    return float(-np.nanmean(trials[:, w]))

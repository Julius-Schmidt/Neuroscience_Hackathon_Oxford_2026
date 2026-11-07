"""Synthetic EEG for UNIT TESTS ONLY. Not evidence that the detector works.

1/f background, a posterior 10 Hz alpha rhythm that grows when eyes close,
frontal blink spikes while eyes are open, slow drift and 60 Hz mains hum.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .data import Scenario, SubjectEEG


def _pink(n, n_ch, rng):
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = (rng.standard_normal((len(f), n_ch)) + 1j * rng.standard_normal((len(f), n_ch))) / np.sqrt(f)[:, None]
    x = np.fft.irfft(spec, n=n, axis=0)
    return x / x.std(axis=0)


def _signal(seconds, closed, rng, srate=C.SRATE):
    n = int(seconds * srate)
    t = np.arange(n) / srate
    ch = C.EEG_CHANNELS
    x = 12.0 * _pink(n, len(ch), rng)
    post = np.array([c in C.POSTERIOR for c in ch], float)
    amp = 3.0 + (18.0 if closed else 0.0)                  # µV of alpha
    phase = rng.uniform(0, 2 * np.pi, len(ch))
    alpha_env = 1 + 0.3 * np.sin(2 * np.pi * 0.3 * t)[:, None]
    x += amp * alpha_env * post * np.sin(2 * np.pi * 10.0 * t[:, None] + phase)
    x += 4.0 * np.sin(2 * np.pi * 60.0 * t)[:, None]       # mains (PhysioNet is US)
    x += np.cumsum(rng.standard_normal((n, len(ch))), axis=0) * 0.05   # drift
    if not closed:                                         # blinks, frontal
        fz = ch.index("Fz")
        for b in rng.uniform(0, seconds, int(seconds / 4)):
            i = int(b * srate)
            k = np.arange(-25, 26)
            idx = np.clip(i + k, 0, n - 1)
            x[idx, fz] += 120 * np.exp(-0.5 * (k / 8) ** 2)
    return x.astype(np.float32)


def synthetic_subject(seed: int = 0, seconds: float = 61.0) -> SubjectEEG:
    rng = np.random.default_rng(seed)
    return SubjectEEG(subject=-seed - 1, open=_signal(seconds, False, rng), closed=_signal(seconds, True, rng))


def synthetic_scenario(seed: int = 0) -> Scenario:
    from .data import make_scenario
    return make_scenario(synthetic_subject(seed), np.random.default_rng(seed))

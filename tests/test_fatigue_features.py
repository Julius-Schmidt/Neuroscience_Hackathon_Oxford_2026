import numpy as np
from devtool.fatigue_features import spectrum_features


def _noise(cut_hz, n=300, seed=0):
    """Band-limited noise: 8 channels, low-passed at cut_hz (200 Hz sampling)."""
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.normal(size=(n, 8)), axis=0)
    f = np.fft.rfftfreq(n, 1 / 200)
    X[f > cut_hz] = 0
    return np.fft.irfft(X, n=n, axis=0) * 20


def test_spectral_features_follow_a_shift_to_low_frequencies():
    fresh, tired = spectrum_features(_noise(90)), spectrum_features(_noise(45))
    assert tired["mdf"] < fresh["mdf"] and tired["mnf"] < fresh["mnf"]
    assert tired["dimitrov"] > fresh["dimitrov"]
    assert tired["low_frac"] > fresh["low_frac"]
    assert tired["zc"] < fresh["zc"]


def test_weakness_sim_shrinks_signal_but_keeps_noise():
    from devtool.weakness_sim import simulate
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 3, size=(5000, 8))
    sig = noise.copy()
    sig[1000:2000] += rng.normal(0, 40, size=(1000, 8))      # a squeeze
    out = simulate(sig, 0.25, noise).astype(float)
    assert abs(out[3000:].std() / noise[3000:].std() - 1) < 0.15   # quiet part: same noise level
    assert abs(out[1000:2000].std() / sig[1000:2000].std() - 0.25) < 0.05

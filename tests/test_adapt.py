import numpy as np

from myomouse.adapt import Rescaler, StrengthTracker, hold_to_n_on


def test_hold_to_n_on():
    assert hold_to_n_on(None, 2) is None
    assert hold_to_n_on(200, 2) == 20      # one decision every 10 ms
    assert hold_to_n_on(5, 2) == 1


def test_tracker_follows_weaker_squeezes():
    t = StrengthTracker()
    for _ in range(5):
        t.update(1.0)
    assert t.gain() == 1.0
    for _ in range(200):
        t.update(0.4)
    assert 2.0 < t.gain() <= 6.0


def test_rescaler_gain_rises_for_weak_bursts():
    rng = np.random.default_rng(0)
    model = {"threshold": {"center": float(np.log1p(20)), "scale": 0.3}}
    r = Rescaler(model)
    g = 1.0
    for burst in range(30):
        amp = 30 if burst < 6 else 10          # strong squeezes, then a third of the size
        for k in range(60):
            a = amp if 10 <= k < 40 else 2
            win = np.clip(rng.normal(0, a, size=(30, 8)), -127, 127).astype(np.int8)
            g = r.feed(win)
    assert g > 1.8

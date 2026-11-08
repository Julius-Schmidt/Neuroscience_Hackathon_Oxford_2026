"""Hardware-free tests for the fatigue-adaptive threshold (devtool/fatigue.py)."""

import numpy as np

from devtool.emg import compute_calibration, detect_offline
from devtool.fatigue import FatigueTracker, detect_offline_adaptive


def noise(n, amp, rng):
    return np.clip(rng.normal(0, 1, (n, 8)) * amp, -128, 127).astype(np.int8)


def fatigue_recording(n_cues=80, start_amp=30.0, end_amp=9.0, seed=0):
    """Squeeze every 3.5 s (700 samples), 0.5 s long, getting weaker; rest noise in between."""
    rng = np.random.default_rng(seed)
    parts, cues, pos = [], [], 0
    for k in range(n_cues):
        amp = start_amp + (end_amp - start_amp) * k / (n_cues - 1)
        parts += [noise(200, 2, rng), noise(100, amp, rng), noise(400, 2, rng)]
        cues.append(pos + 200)
        pos += 700
    rest = noise(36000, 2, rng)                    # 3 min still at the end: every click here is false
    return np.concatenate(parts + [rest]), np.array(cues), pos


def score(clicks, cues, rest_start):
    hits = sum(np.any((clicks >= c - 60) & (clicks <= c + 300)) for c in cues)
    false_rest = int(np.sum(clicks >= rest_start))
    return hits, false_rest


def test_tracker_follows_weaker_squeezes_and_respects_floor():
    t = FatigueTracker(floor=0.2, warmup=3)
    for p in [1.0, 1.1, 0.9]:
        t.update(p)
    assert t.ready and abs(t.x0 - 1.0) < 1e-9
    for _ in range(200):
        t.update(0.5)
    assert 0.45 < t.ratio < 0.6
    assert abs(t.threshold(0.45) - 0.45 * t.ratio) < 1e-6
    assert t.threshold(0.2) == 0.2                 # never below the floor
    for _ in range(200):
        t.update(0.05)
    assert t.gain() == t.max_gain                  # gain is capped


def test_adaptive_keeps_clicking_as_squeezes_weaken_without_false_clicks():
    rng = np.random.default_rng(1)
    cal = compute_calibration(noise(2000, 2, rng), noise(2000, 30, rng))
    emg, cues, rest_start = fatigue_recording()
    idx = np.arange(len(emg))
    fixed = detect_offline(emg, idx, cal, 0.45, 250)
    adaptive, trace = detect_offline_adaptive(emg, idx, cal, 0.45, 250)
    hf, ff = score(fixed, cues, rest_start)
    ha, fa = score(adaptive, cues, rest_start)
    assert hf < 0.85 * len(cues)                   # fixed threshold loses the weak late squeezes
    assert ha >= 0.95 * len(cues)                  # adaptive keeps them
    assert fa == 0 and ff == 0                     # and adapting doesn't cause false clicks at rest
    assert trace and trace[-1][3] < 0.6            # tracked strength fell


def test_fatigue_protocol_end_to_end_with_fake_bands(tmp_path, monkeypatch):
    """Mode 4 records cues + raw EMG with tiring fake bands; replay_fatigue reads it back."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from devtool import app as A
    from devtool import replay_fatigue as R
    monkeypatch.setattr(A, "DATA_DIR", tmp_path)
    A.main(["--fake", "--fake-fatigue", "3", "--autotest-fatigue", "--user", "t", "--threshold", "0.5",
            "--fatigue-min", "0.15", "--fatigue-rest-min", "0.05", "--fatigue-post-min", "0.05",
            "--cue-interval", "1.0", "--seconds", "60"])
    raws = list((tmp_path / "sessions").glob("*_fatigue_raw.npz"))
    assert len(raws) == 1
    res = R.run(raws[0], threshold=0.6, plot=False)
    for side, r in res.items():
        assert len(r["cf"]) >= 8                       # cues were recorded
        assert r["ca"].sum() >= r["cf"].sum()          # adaptive never catches fewer than fixed here


def _toy_lda(rng):
    """A myomouse-style model dict trained on synthetic rest vs fresh-squeeze windows."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from myomouse import config as C
    from myomouse.features import sliding_features
    rest, sq = noise(4000, 2, rng), noise(4000, 30, rng)
    Xr, _ = sliding_features(rest, C.WINDOW, C.STEP)
    Xs, _ = sliding_features(sq, C.WINDOW, C.STEP)
    X, y = np.vstack([Xr, Xs]), np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X, y)
    return {"pipeline": pipe, "threshold": {"center": 0, "scale": 1}, "window": C.WINDOW, "step": C.STEP}


def test_batched_lda_matches_live_detector():
    from myomouse.detector import FistDetector
    from devtool.fatigue import detect_offline_lda
    rng = np.random.default_rng(3)
    model = _toy_lda(rng)
    emg, cues, _ = fatigue_recording(n_cues=12)
    idx = np.arange(len(emg))
    live = FistDetector(model, "lda")
    live_presses = [i for i, s in zip(idx, emg) if live.push(s) == "press"]
    assert list(detect_offline_lda(emg, idx, model)) == live_presses
    assert len(live_presses) >= 10


def test_lda_fatigue_replay_runs_and_adaptive_not_worse():
    from devtool.fatigue import detect_offline_lda, gain_series
    rng = np.random.default_rng(4)
    model = _toy_lda(rng)
    cal = compute_calibration(noise(2000, 2, rng), noise(2000, 30, rng))
    emg, cues, rest_start = fatigue_recording(end_amp=4.0)
    idx = np.arange(len(emg))
    g = gain_series(emg, idx, cal, 0.45)
    hf, ff = score(detect_offline_lda(emg, idx, model), cues, rest_start)
    ha, fa = score(detect_offline_lda(emg, idx, model, g), cues, rest_start)
    assert g.max() > 1.5 and g[:2000].max() == 1.0      # gain only rises once squeezes weaken
    assert ha >= hf and fa == 0

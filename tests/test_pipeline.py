"""Hardware-free tests: synthetic EMG through record-format -> train -> live detector."""

import math

import numpy as np
import pytest

from myomouse import config as C
from myomouse import train
from myomouse.detector import Debouncer, FistDetector, load_model
from myomouse.features import window_features
from myomouse.pointer import Pointer
from myomouse.record import FIST, IGNORE, TRANSITION_S, protocol


def synth_session(side, seed=0, motion_artifacts=True):
    """Fake a recording that follows the real protocol. Fist = strong EMG on channels 2-5."""
    rng = np.random.default_rng(seed)
    emg, lab, seg = [], [], []
    for i, (_, label, dur) in enumerate(protocol(side)):
        n = int(dur * C.EMG_HZ)
        gain = np.full(C.N_CHANNELS, 3.0)
        if label == FIST:
            gain[2:6] = 35.0
        x = rng.normal(0, 1, (n, C.N_CHANNELS)) * gain
        if motion_artifacts and label != FIST:
            # arm movement: slow drift + occasional bursts on other channels
            x[:, [0, 7]] += rng.normal(0, 8, (n, 2))
        emg.append(np.clip(x, -128, 127).astype(np.int8))
        k = int(TRANSITION_S * C.EMG_HZ)
        l = np.full(n, label)
        l[:k] = IGNORE
        lab.append(l)
        seg.append(np.full(n, i))
    return {"emg": np.concatenate(emg), "label": np.concatenate(lab), "seg": np.concatenate(seg)}


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(C, "MODEL_DIR", tmp_path / "models")
    C.DATA_DIR.mkdir()
    return tmp_path


def test_features_shape_and_order():
    rest = np.random.default_rng(0).normal(0, 2, (C.WINDOW, 8)).astype(np.int8)
    fist = np.random.default_rng(0).normal(0, 40, (C.WINDOW, 8)).astype(np.int8)
    fr, ff = window_features(rest), window_features(fist)
    assert fr.shape == (40,)
    assert (ff[:8] > fr[:8]).all()  # MAV bigger for a fist


def test_debouncer_needs_n_in_a_row_and_hysteresis():
    d = Debouncer(p_on=0.8, p_off=0.3, n_on=3, n_off=3)
    assert [d.update(p) for p in [0.9, 0.9, 0.1, 0.9, 0.9]] == [None] * 5  # interrupted
    assert d.update(0.9) == "press"
    assert [d.update(p) for p in [0.5, 0.5, 0.5]] == [None] * 3  # between thresholds: stays held
    assert [d.update(0.1) for _ in range(3)] == [None, None, "release"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_train_and_replay(tmp_dirs, side, monkeypatch):
    for sid in range(2):
        s = synth_session(side, seed=sid)
        np.savez(C.DATA_DIR / f"{side}_s{sid}.npz", **s, side=side, fs=C.EMG_HZ)
    monkeypatch.setattr("sys.argv", ["train", "--side", side])
    train.main()
    model = load_model(side)

    test = synth_session(side, seed=99)
    n_fists = len({g for g, l in zip(test["seg"], test["label"]) if l == FIST})
    for mode in ("lda", "threshold"):
        det = FistDetector(model, mode)
        events = [(i, e) for i, s in enumerate(test["emg"]) if (e := det.push(s))]
        presses = [i for i, e in events if e == "press"]
        assert len(presses) == n_fists, (mode, len(presses), n_fists)
        # press comes within 150 ms of the (synthetic, instant) fist onset
        onsets = [int(np.argmax(test["seg"] == g)) for g in sorted({g for g, l in zip(test["seg"], test["label"]) if l == FIST})]
        delays = [(p - o) / C.EMG_HZ * 1000 for p, o in zip(presses, onsets)]
        assert max(delays) < 150, (mode, delays)


def test_pointer_turn_left_moves_cursor_left():
    p = Pointer()
    total = 0.0
    for k in range(50):
        yaw = 0.2 * k / 50  # rotate forearm 0.2 rad to the left
        q = (math.cos(yaw / 2) * 16384, 0, 0, math.sin(yaw / 2) * 16384)
        dx, dy = p.update(q)
        total += dx
    assert total < -0.15 * C.GAIN_X
    assert abs(dy) < 1e-6


def test_pointer_ignores_wrist_roll():
    p = Pointer()
    moved = 0.0
    for k in range(50):
        roll = 1.0 * k / 50  # rotate around the forearm's own axis (band x-axis)
        q = (math.cos(roll / 2) * 16384, math.sin(roll / 2) * 16384, 0, 0)
        dx, dy = p.update(q)
        moved += abs(dx) + abs(dy)
    assert moved < 1.0

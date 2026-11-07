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
    steps = [st for _, _, block in protocol(side) for st in block]
    for i, (label, dur, _) in enumerate(steps):
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


def test_arena_hits_and_misses():
    from myomouse.arena import Arena

    a = Arena(800, 600, ["left"])
    tx, ty, r, want = a.target
    a.move(tx - a.x, ty - a.y, now=0.0)          # cursor onto the target
    a.button("right", True, now=1.0)              # wrong button -> miss
    a.button("right", False, now=1.0)
    a.button(want, True, now=2.0)                 # right button -> hit, new target
    assert (a.hits, a.misses) == (1, 1)
    a.inv_x, a.gain = True, 2.0
    x0 = a.x
    a.move(10, 0, now=5.0)
    assert a.x == max(0, x0 - 20)                 # flipped and doubled (clamped at the edge)
    a.button("left", True, now=6.0)
    x1 = a.x
    a.move(10, 0, now=6.05)                       # frozen right after a click
    assert a.x == x1


def fake_camera(sess, delay=0.3, late=0.2, fps=30, stray=(3.0, 4.0), blind=None):
    """Camera that sees the hand close `delay` s after each fist cue and open `late` s after it ends."""
    t_emg = np.arange(len(sess["emg"])) / C.EMG_HZ
    closed = []
    for g in np.unique(sess["seg"][sess["label"] == FIST]):
        idx = np.flatnonzero(sess["seg"] == g)
        closed.append((t_emg[idx[0]] + delay, t_emg[idx[-1]] + late))
    closed.append(stray)  # a fist during the first "relax" block
    cam_t = np.arange(0, t_emg[-1], 1 / fps)
    ratio = np.full(len(cam_t), 2.0)
    for a, b in closed:
        ratio[(cam_t >= a) & (cam_t < b)] = 1.0
    if blind:
        ratio[(cam_t >= blind[0]) & (cam_t < blind[1])] = np.nan
    return t_emg, cam_t, ratio


def test_closure_ratio_open_vs_fist():
    from myomouse.vision import closure_ratio
    pts = np.zeros((21, 3))
    for k, (mcp, tip) in enumerate(zip((5, 9, 13, 17), (8, 12, 16, 20))):
        pts[mcp] = (k * 0.02 - 0.03, 0.08, 0)
        pts[tip] = (k * 0.02 - 0.03, 0.16, 0)   # straight finger: tip twice as far as knuckle
    open_r = closure_ratio(pts)
    for k, tip in enumerate((8, 12, 16, 20)):
        pts[tip] = (k * 0.02 - 0.03, 0.07, 0.03)  # curled back towards the palm
    assert open_r == pytest.approx(2.0, rel=0.05)
    assert closure_ratio(pts) < 1.2


def test_vision_labels_follow_the_hand_not_the_cue():
    from myomouse.vision import vision_labels
    s = synth_session("right")
    t, cam_t, ratio = fake_camera(s)
    lab, report = vision_labels(t, s["label"], s["seg"], cam_t, ratio, lag=0.0, margin=0.08)
    assert any("timed by the camera" in r for r in report), report

    g = int(s["seg"][np.argmax(s["label"] == FIST)])        # first cued fist
    idx = np.flatnonzero(s["seg"] == g)
    t0, t1 = t[idx[0]], t[idx[-1]]
    assert (lab[(t > t0 - 0.25) & (t < t0 + 0.2)] == 0).all()          # cue said fist, hand still open
    assert (lab[(t > t0 + 0.2) & (t < t0 + 0.38)] != FIST).all()      # closing: left out
    assert (lab[(t > t0 + 0.45) & (t < t1)] == FIST).all()          # hand visibly closed
    assert (lab[(t > t1 + 0.05) & (t < t1 + 0.2 - 0.1)] == FIST).all()  # still closed after the cue
    assert (lab[(t > t1 + 0.35) & (t < t1 + 0.8)] == 0).all()         # opened again
    assert (lab[(t > 3.1) & (t < 3.9)] == IGNORE).all()               # stray fist while "relaxing"


def test_vision_labels_fall_back_to_cue_without_a_hand():
    from myomouse.vision import vision_labels
    s = synth_session("right")
    t, cam_t, ratio = fake_camera(s)
    lab, report = vision_labels(t, s["label"], s["seg"], cam_t, np.full_like(ratio, np.nan))
    assert (lab == s["label"]).all() and "cue timing only" in report[0]
    # hand out of view for one fist: that fist keeps its cue labels
    g = int(s["seg"][np.argmax(s["label"] == FIST)])
    idx = np.flatnonzero(s["seg"] == g)
    t, cam_t, ratio = fake_camera(s, blind=(t[idx[0]] - 0.5, t[idx[-1]] + 1.0))
    lab, report = vision_labels(t, s["label"], s["seg"], cam_t, ratio, lag=0.0)
    assert (lab[idx] == s["label"][idx]).all()
    assert any("out of view" in r for r in report), report


def test_train_with_camera_labels(tmp_dirs, monkeypatch):
    from myomouse.vision import vision_labels
    for sid in range(2):
        s = synth_session("right", seed=sid)
        t, cam_t, ratio = fake_camera(s, delay=0.0, late=0.0, stray=(0, 0))
        lab, _ = vision_labels(t, s["label"], s["seg"], cam_t, ratio, lag=0.0)
        np.savez(C.DATA_DIR / f"right_s{sid}.npz", emg=s["emg"], label=lab, label_cue=s["label"],
                 seg=s["seg"], t=t, cam_t=cam_t, cam_ratio=ratio, side="right", fs=C.EMG_HZ)
    for labels in ("camera", "cue"):
        monkeypatch.setattr("sys.argv", ["train", "--side", "right", "--labels", labels])
        train.main()
        assert load_model("right")["pipeline"] is not None


def test_playground_left_recolours_right_drags():
    from myomouse.playground import Playground
    pg = Playground(900, 800)
    b = pg.blocks[0]
    x, y = b["rect"][0] + 10, b["rect"][1] + 10
    c0 = b["color"]
    pg.press("left", x, y, 0.0)
    pg.release("left", x, y, 0.1)
    assert b["color"] == c0 + 1
    pg.press("right", x, y, 1.0)             # grab
    pg.motion(x + 300, y + 50)               # drag
    pg.release("right", x + 300, y + 50, 1.5)
    pg.motion(x + 500, y + 300)              # released: block stays
    assert b["rect"][:2] == [b["rect"][0], b["rect"][1]] and b["rect"][0] == x - 10 + 300
    assert b["rect"][1] == y - 10 + 50 and b["color"] == c0 + 1
    pg.press("left", 5, 5, 2.0)              # empty space: just a click
    assert pg.clicks == {"left": 2, "right": 1}

"""Hardware-free tests for the multi-gesture path: synthetic recordings -> train -> live classifier."""

from itertools import combinations

import joblib
import numpy as np
import pytest

from myomouse import config as C
from myomouse.gestures11 import GESTURES, IDLE, IGNORE, KEYS, NEUTRAL, REST, model_path, parse_gestures
from myomouse.gestures11 import train
from myomouse.gestures11.decide import NONE, Switcher, raw_decision
from myomouse.gestures11.live import GestureClassifier
from myomouse.gestures11.record import protocol, step_label

USER = "tester"
PAIRS = list(combinations(range(C.N_CHANNELS), 2))[::2]  # a different channel pair per gesture


def all_steps(reps, seed):
    return [st for _, _, block in protocol(reps, seed=seed) for st in block]


def synth_session(seed, reps=2):
    """Follows the real protocol; each gesture = strong EMG on its own channel pair,
    neutral = some posture effort, rest = nearly silent."""
    rng = np.random.default_rng(seed)
    emg, lab, seg = [], [], []
    for i, (g, dur, _) in enumerate(all_steps(reps, seed)):
        n = int(dur * C.EMG_HZ)
        gain = np.full(C.N_CHANNELS, 1.0 if g == REST else 3.0)
        if g not in IDLE:
            gain[list(PAIRS[g])] = 30.0
        emg.append(np.clip(rng.normal(0, 1, (n, C.N_CHANNELS)) * gain, -128, 127).astype(np.int8))
        lab.append(np.array([step_label(g, dur, k / C.EMG_HZ) for k in range(n)]))
        seg.append(np.full(n, i))
    return {"emg": np.concatenate(emg), "label": np.concatenate(lab), "seg": np.concatenate(seg),
            "gestures": np.asarray(KEYS)}


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(C, "MODEL_DIR", tmp_path / "models")
    (C.DATA_DIR / USER).mkdir(parents=True)


def test_protocol_rest_blocks_and_every_gesture_each_rep():
    blocks = protocol(reps=3, seed=1, move_reps=0)
    assert len(blocks) == 5                                # rest, 3 repetitions, rest again
    for _, _, steps in (blocks[0], blocks[-1]):
        assert all(g == REST for g, _, _ in steps)
    for _, _, steps in blocks[1:-1]:
        held = sorted(g for g, _, _ in steps if g != NEUTRAL)
        assert held == [g for g in range(len(GESTURES)) if g not in IDLE]
        assert all(steps[k][0] == NEUTRAL for k in range(0, len(steps), 2))  # neutral between holds


def test_protocol_moving_part_like_the_fist_protocol():
    from myomouse.gestures11.record import MOVING
    blocks = protocol(reps=2, seed=1, move_reps=3)
    names = [b[0] for b in blocks]
    assert names[3] == "Neutral while moving" and names[4:7] == [f"Moving repetition {i}" for i in (1, 2, 3)]
    assert all(st[2] == MOVING for _, _, steps in blocks[3:7] for st in steps)   # every moving step marked
    assert not any(st[2] == MOVING for _, _, steps in blocks[:3] for st in steps)
    for _, _, steps in blocks[4:7]:
        assert sorted(g for g, _, _ in steps if g != NEUTRAL) == [g for g in range(len(GESTURES)) if g not in IDLE]


def test_step_label_trims_edges():
    assert step_label(3, 3.0, 0.1) == IGNORE
    assert step_label(3, 3.0, 1.5) == 3
    assert step_label(3, 3.0, 2.95) == IGNORE


def test_parse_gestures_keeps_neutral_and_rest():
    assert parse_gestures("power,chuck_grip") == sorted([NEUTRAL, REST, KEYS.index("power_grip"),
                                                         KEYS.index("chuck_grip")])
    with pytest.raises(SystemExit):
        parse_gestures("open_hand")                     # dropped from the set
    with pytest.raises(SystemExit):
        parse_gestures("jazz_hands")


def test_raw_decision_gate_and_reject():
    classes = ["neutral", "power_grip", "rest"]
    sure, unsure = np.array([0.05, 0.9, 0.05]), np.array([0.3, 0.6, 0.1])
    assert raw_decision(sure, env=2.0, classes=classes, gate=1.0, reject=0.75) == "power_grip"
    assert raw_decision(sure, env=0.5, classes=classes, gate=1.0, reject=0.75) == NONE   # too quiet
    assert raw_decision(unsure, env=2.0, classes=classes, gate=1.0, reject=0.75) == NONE  # not sure
    assert raw_decision(np.array([0.1, 0.0, 0.9]), 2.0, classes, None, 0.5) == NONE      # rest = no gesture


def test_switcher_is_slow_in_fast_out():
    sw = Switcher(n_on=3, n_off=2)
    assert [sw.update(d) for d in ["open", "open", NONE, "open", "open"]] == [None] * 5  # interrupted
    assert sw.update("open") == "open"
    assert sw.update("power") is None and sw.update("power") is None  # another gesture also needs 3
    assert sw.update(NONE) is None and sw.update(NONE) == NONE        # back to neutral after 2


def test_train_and_live(tmp_dirs, capsys):
    for sid in range(2):
        np.savez(C.DATA_DIR / USER / f"gestures_right_s{sid}.npz", **synth_session(sid), side="right")
    train.main(["--user", USER, "--side", "right"])
    out = capsys.readouterr().out
    assert "new session" in out and "confusion matrix" in out and "idle -> gesture" in out

    model = joblib.load(model_path(USER, "right"))
    assert model["classes"] == KEYS
    assert model["gate"] is None and model["reject"] == 0.6   # rest gate off by default
    test = synth_session(99, reps=1)
    clf = GestureClassifier(model)
    shown = np.array([clf.push(s) or clf.current for s in test["emg"]])
    # judge the middle of every hold: the live output must be that gesture, or neutral when idle
    lab = test["label"]
    want = np.array([NONE if c in IDLE else KEYS[c] for c in np.clip(lab, 0, None)], dtype=object)
    idx = np.flatnonzero(lab != IGNORE)[::20]
    assert np.mean(shown[idx] == want[idx]) > 0.9
    idle = idx[np.isin(lab[idx], IDLE)]
    assert np.mean(shown[idle] != NONE) < 0.02   # almost never a gesture while idle


def test_train_on_subset(tmp_dirs):
    np.savez(C.DATA_DIR / USER / "gestures_left_s0.npz", **synth_session(3), side="left")
    train.main(["--user", USER, "--side", "left", "--gestures", "power,chuck", "--rest-gate"])
    model = joblib.load(model_path(USER, "left"))
    assert model["classes"] == ["neutral", "power_grip", "chuck_grip", "rest"]
    assert model["gate"] is not None


def test_old_recordings_with_dropped_gestures_still_load(tmp_dirs):
    """A recording made with the old 10-gesture list: dropped gestures are skipped, the rest map by name."""
    old_keys = ["neutral", "radial_deviation", "ulnar_deviation", "power_grip", "rest"]
    lab = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, -1])
    np.savez(C.DATA_DIR / USER / "gestures_right_old.npz", emg=np.zeros((10, 8), np.int8), label=lab,
             seg=np.arange(10), gestures=np.asarray(old_keys))
    s = train.load_sessions(USER, "right")[0]
    assert list(s["label"]) == [NEUTRAL, NEUTRAL, IGNORE, IGNORE, KEYS.index("ulnar_deviation"),
                                KEYS.index("ulnar_deviation"), KEYS.index("power_grip"),
                                KEYS.index("power_grip"), REST, IGNORE]


# --------------------------------------------------------------------------- camera labels

def synth_camera(reps=3, seed=5, delay=0.4, late=0.3, wrong_hold=None, hidden_hold=None):
    """Recording timeline + camera frames of someone who reacts `delay` s after each cue,
    lets go `late` s after it ends, does gesture wrong_hold[1] in hold wrong_hold[0], and
    has the hand out of view during hold hidden_hold."""
    from myomouse.gestures11.record import protocol, step_label
    rng = np.random.default_rng(seed)
    steps = all_steps(reps, seed)
    seg_label = [st[0] for st in steps]
    bounds = np.cumsum([0.0] + [st[1] for st in steps])
    emg_t = np.arange(0, bounds[-1], 1 / C.EMG_HZ)
    seg = np.searchsorted(bounds, emg_t, side="right") - 1
    cue = np.array([step_label(seg_label[s], steps[s][1], t - bounds[s]) for t, s in zip(emg_t, seg)])

    neutral = rng.normal(0, 0.05, (21, 3))
    poses = [neutral] + [neutral + np.vstack([np.zeros((1, 3)), rng.normal(0, 0.03, (20, 3))])
                         for _ in range(len(GESTURES) - 1)]
    cam_t = np.arange(0, bounds[-1], 1 / 30)
    pts = np.empty((len(cam_t), 21, 3))
    for i, t in enumerate(cam_t):
        doing = NEUTRAL
        for s, (g, _, _) in enumerate(steps):
            if g not in IDLE and bounds[s] + delay <= t < bounds[s + 1] + late:
                doing = wrong_hold[1] if wrong_hold and s == wrong_hold[0] else g
        pts[i] = poses[doing] + rng.normal(0, 0.002, (21, 3))
        if hidden_hold is not None and bounds[hidden_hold] - 0.5 <= t < bounds[hidden_hold + 1] + 1.0:
            pts[i] = np.nan
    return dict(emg_t=emg_t, cue=cue, seg=seg, seg_label=seg_label, cam_t=cam_t, cam_pts=pts, bounds=bounds)


def test_camera_labels_follow_the_hand():
    from myomouse.gestures11.vision import gesture_vision_labels
    d = synth_camera()
    labels, report = gesture_vision_labels(d["emg_t"], d["cue"], d["seg"], d["seg_label"],
                                           d["cam_t"], d["cam_pts"], lag=0.0)
    assert any("timed by the camera" in r for r in report)
    holds = [s for s, g in enumerate(d["seg_label"]) if g not in IDLE]
    starts, ends = [], []
    for s in holds:
        g = d["seg_label"][s]
        t0, t1 = d["bounds"][s], d["bounds"][s + 1]
        on = d["emg_t"][labels == g]
        on = on[(on > t0 - 0.5) & (on < t1 + 1.0)]
        starts.append(on[0] - t0)
        ends.append(on[-1] - t1)
        # before the hand moved it's neutral (or left out), never the gesture
        early = (d["emg_t"] >= t0) & (d["emg_t"] < t0 + 0.3)
        assert not (labels[early] == g).any()
    assert 0.35 < np.median(starts) < 0.6       # reaction time picked up (cue labels would start at 0.6)
    assert 0.1 < np.median(ends) < 0.35         # late release picked up (cue labels end 0.2 s early)


def test_camera_drops_wrong_gesture_and_keeps_cue_when_hidden():
    from myomouse.gestures11.vision import gesture_vision_labels
    d0 = synth_camera()
    holds = [s for s, g in enumerate(d0["seg_label"]) if g not in IDLE]
    wrong_s, hidden_s = holds[2], holds[7]
    did = d0["seg_label"][holds[3]]
    if did == d0["seg_label"][wrong_s]:
        did = d0["seg_label"][holds[4]]
    d = synth_camera(wrong_hold=(wrong_s, did), hidden_hold=hidden_s)
    labels, report = gesture_vision_labels(d["emg_t"], d["cue"], d["seg"], d["seg_label"],
                                           d["cam_t"], d["cam_pts"], lag=0.0)
    g = d["seg_label"][wrong_s]
    z = (d["emg_t"] > d["bounds"][wrong_s] - 0.5) & (d["emg_t"] < d["bounds"][wrong_s + 1] + 1.0)
    assert not (labels[z] == g).any()
    assert any("looked like" in r for r in report)
    hz = d["seg"] == hidden_s
    assert (labels[hz] == d["cue"][hz]).all()   # out of view: cue timing kept
    assert any("out of view" in r for r in report)


# --------------------------------------------------------------------------- pick test

def test_pick_game_rounds():
    import random
    from myomouse.gestures11.pick import SHOW_S, TIMEOUT_S, PickGame
    names = ["power_grip", "chuck_grip", "pinch_grip"]
    game = PickGame(names, rng=random.Random(0))
    game.update(0.0, NONE)                          # first pair right away
    assert game.state == "choose" and len(set(game.pair)) == 2
    first = game.pair
    other = next(g for g in names if g not in game.pair)
    game.update(1.0, other)                         # a gesture that isn't on screen
    assert game.results[-1]["outcome"] == "wrong gesture" and game.state == "choose"
    picked = game.pair[1]
    game.update(1.5, picked)                        # picks the right card -> "Predicted"
    assert game.state == "shown" and game.picked == 1 and game.results[-1]["outcome"] == "picked"
    game.update(1.5 + SHOW_S + 0.01, picked)        # next pair straight away, still holding
    assert game.state == "choose" and set(game.pair) != set(first)
    n = len(game.results)
    game.update(2.5, picked)                        # held gesture doesn't count again ...
    assert len(game.results) == n and game.state == "choose"
    game.update(2.6, NONE)                          # ... until it's let go
    if picked in game.pair:
        game.update(2.7, picked)
        assert game.state == "shown"
    else:
        game.update(game.t_state + TIMEOUT_S + 0.1, NONE)   # nothing done
        assert game.results[-1]["outcome"] == "nothing"
    counts, med = game.score()
    assert counts["wrong gesture"] == 1 and counts["picked"] >= 1 and med is not None


def test_pick_game_target_mode():
    import random
    from myomouse.gestures11.pick import PickGame
    game = PickGame(["power_grip", "chuck_grip"], target_mode=True, rng=random.Random(1))
    game.update(0.0, NONE)
    game.update(1.0, game.pair[1 - game.target])    # did the other card
    assert game.results[-1]["outcome"] == "other card"

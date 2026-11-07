"""EEG switch unit tests (synthetic signal, no download, no LSL network needed).

    pytest -q tests/test_eeg_switch.py
"""

import numpy as np
import pytest

from eeg_switch import config as C
from eeg_switch.data import make_scenario
from eeg_switch.detector import AlphaSwitch, Params, alpha_db, run_offline
from eeg_switch.events import decode, encode, make_event
from eeg_switch.metrics import score
from eeg_switch.synthetic import synthetic_scenario, synthetic_subject


def test_event_format_is_the_shared_one():
    ev = make_event("eeg", "safety_toggle", 1.7, 12.5)
    assert set(ev) == {"source", "intent", "confidence", "timestamp"}
    assert ev["confidence"] == 1.0                      # clipped to 0-1
    assert decode(encode(ev)) == ev
    with pytest.raises(ValueError):
        make_event("eog", "x", 0.5, 0.0)


def test_alpha_feature_rises_when_eyes_close():
    sub = synthetic_subject(1)
    post = [C.EEG_CHANNELS.index(c) for c in C.POSTERIOR]
    w = int(C.SRATE)
    a_open = alpha_db(sub.open[:w, post], C.SRATE)
    a_closed = alpha_db(sub.closed[:w, post], C.SRATE)
    assert a_closed - a_open > 3.0


def test_ground_truth_lines_up_with_samples():
    sc = synthetic_scenario(3)
    n = len(sc.eeg)
    for a, b, _ in sc.closed_intervals:
        assert 0 < a < b < n / sc.srate


def test_long_closures_toggle_and_short_ones_do_not():
    sc = synthetic_scenario(2)
    toggles, sw = run_offline(sc.eeg)
    r = score(sc, toggles)
    assert sw.calibrated
    assert r["missed"] == 0
    assert r["false"] == 0
    assert all(C.DWELL_S <= d <= C.DWELL_S + 1.5 for d in r["delays"])


def test_no_toggle_while_eyes_open():
    sub = synthetic_subject(4)
    toggles, _ = run_offline(np.vstack([sub.open, sub.open]))
    assert toggles == []


def test_one_toggle_per_closure_until_eyes_reopen():
    sub = synthetic_subject(5)
    eeg = np.vstack([sub.open[:int(25 * C.SRATE)], sub.closed[:int(15 * C.SRATE)]])
    toggles, _ = run_offline(eeg)
    assert len(toggles) == 1                             # 15 s closed = still one toggle


def test_artifact_windows_are_skipped():
    sub = synthetic_subject(6)
    eeg = sub.open.copy()
    eeg[int(30 * C.SRATE):int(32 * C.SRATE), C.EEG_CHANNELS.index("Oz")] += 1000.0
    toggles, sw = run_offline(eeg)
    assert toggles == [] and sw.n_artifacts > 0


def test_streaming_in_chunks_matches_offline():
    sc = synthetic_scenario(7)
    offline, _ = run_offline(sc.eeg)
    sw = AlphaSwitch(Params())
    live = []
    for i in range(0, len(sc.eeg), 10):                  # like 40 ms LSL chunks
        live += sw.push(sc.eeg[i:i + 10])
    assert [round(t, 3) for t, _ in live] == [round(t, 3) for t, _ in offline]

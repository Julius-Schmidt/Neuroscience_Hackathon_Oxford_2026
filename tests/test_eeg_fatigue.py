"""Level 1 analysis tests on simulated recordings (no hardware, no LSL).

    pytest -q tests/test_eeg_fatigue.py
"""

import numpy as np
import pytest

from eeg_fatigue import config as C
from eeg_fatigue import signals as S
from eeg_fatigue.analyse import analyse
from eeg_fatigue.recording import load, save
from eeg_fatigue.simulate import offline


@pytest.fixture(scope="module")
def recs():
    return {s: offline(s, seed=1) for s in ("divergence", "muscle", "null")}


def test_emg_onsets_find_every_cued_squeeze(recs):
    r = recs["divergence"]
    env = S.emg_envelope(r.emg)
    cues = r.marker_times("cue_squeeze")
    onsets, _ = S.detect_onsets(env, r.emg_t, cues)
    assert len(onsets) == len(cues)
    lag = onsets - cues[np.searchsorted(cues, onsets) - 1]
    assert np.all((lag > 0) & (lag < 0.6))                     # onset follows its cue


def test_blocks_parsed_from_markers(recs):
    kinds = [b["kind"] for b in recs["divergence"].blocks()]
    assert kinds == ["baseline"] + ["fatigue"] * C.FATIGUE_BLOCKS + ["recovery"] * C.RECOVERY_BLOCKS


def test_divergence_detected(recs):
    v = analyse(recs["divergence"], n_boot=200)["verdict"]
    assert v["rises"] and v["drops_after_rest"] and v["brain_effort_rises"]


def test_muscle_only_is_not_called_central(recs):
    v = analyse(recs["muscle"], n_boot=200)["verdict"]
    assert v["rises"] and not v["brain_effort_rises"]          # ratio alone would mislead


def test_null_gives_no_rise(recs):
    v = analyse(recs["null"], n_boot=200)["verdict"]
    assert not v["rises"]


def test_ratio_block1_is_one(recs):
    df = analyse(recs["divergence"], n_boot=50)["blocks"]
    assert df.loc[0, "ratio"] == pytest.approx(1.0)


def test_no_eeg_response_is_reported_not_faked(recs):
    r = recs["divergence"]
    rng = np.random.default_rng(0)
    r2 = type(r)(**{**r.__dict__, "eeg": rng.normal(0, 10, r.eeg.shape)})   # pure noise EEG
    res = analyse(r2, n_boot=100)
    assert res["verdict"]["rises"] in (None, False)


def test_save_load_roundtrip(tmp_path, recs):
    p = tmp_path / "x.npz"
    save(recs["null"], p)
    r = load(p)
    assert r.eeg.shape == recs["null"].eeg.shape and r.markers == recs["null"].markers

import numpy as np

from myomouse.augment import augmented_sessions, shrink


def _session(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    emg = rng.normal(0, 3, size=(n, 8))
    label = np.zeros(n, dtype=int)
    label[1000:2000] = 1
    emg[1000:2000, 2:5] += rng.normal(0, 40, size=(1000, 3))
    return {"emg": np.clip(np.round(emg), -128, 127).astype(np.int8), "label": label,
            "seg": np.zeros(n, dtype=int), "file": "x"}


def test_augmented_copies_keep_labels_and_noise_floor():
    s = _session()
    out = augmented_sessions(s, rest_value=0)
    assert [o["aug"] for o in out] == ["strength 0.60", "strength 0.40", "strength 0.25", "rotated -1", "rotated +1"]
    for o in out:
        assert o["emg"].shape == s["emg"].shape and o["emg"].dtype == np.int8
        assert o["label"] is s["label"]
    weak = out[2]["emg"].astype(float)
    assert abs(weak[3000:].std() / s["emg"][3000:].astype(float).std() - 1) < 0.2     # noise floor kept
    assert weak[1000:2000, 2:5].std() < 0.4 * s["emg"][1000:2000, 2:5].astype(float).std()  # fist shrunk
    assert np.array_equal(out[4]["emg"][:, 3], s["emg"][:, 2])                        # rolled by one electrode


def test_shrink_is_identity_at_full_strength():
    s = _session()
    assert np.array_equal(shrink(s["emg"], 1.0, np.zeros((len(s["emg"]), 8))), s["emg"])

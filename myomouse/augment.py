"""Training-data augmentation for the click model (opt-in: train --augment).

Two things that break an EMG classifier in daily use, and the standard cheap fixes:

  weaker / tired squeezes   add copies of each recording with the muscle signal shrunk
                            (s * x + sqrt(1 - s^2) * real band noise, so the noise floor stays
                            the same - what a weaker user actually looks like), so the model
                            learns a fist at many effort levels, not just the one recorded.
  band put on rotated       add copies with the 8 channels rolled by one electrode either way
                            (the Myo's electrodes sit in a ring, 45 degrees apart).

Labels are unchanged: a shrunk fist is still a fist, a shrunk rest still rest.
"""

import numpy as np

STRENGTHS = (0.6, 0.4, 0.25)
ROLLS = (-1, 1)


def noise_like(emg, label, rest_value, rng, block=100):
    """Real band noise: quiet blocks from the rest parts of this recording, shuffled."""
    x = emg.astype(np.float32)
    rest = x[label == rest_value]
    n = len(rest) // block
    if n < 4:
        return rng.normal(0, max(1.0, float(x.std()) * 0.1), size=x.shape).astype(np.float32)
    blocks = rest[: n * block].reshape(n, block, -1)
    level = np.abs(blocks).mean(axis=(1, 2))
    quiet = blocks[level <= np.median(level)]
    idx = rng.integers(0, len(quiet), size=int(np.ceil(len(x) / block)))
    out = np.concatenate([quiet[i] for i in idx])[: len(x)]
    return out - out.mean(axis=0)


def shrink(emg, s, noise):
    x = s * emg.astype(np.float32) + np.sqrt(max(0.0, 1 - s * s)) * noise
    return np.clip(np.round(x), -128, 127).astype(np.int8)


def augmented_sessions(sess, rest_value, strengths=STRENGTHS, rolls=ROLLS, seed=0):
    """Extra copies of one recording dict (same labels/segments, different 'emg')."""
    rng = np.random.default_rng(seed)
    out = []
    noise = noise_like(sess["emg"], sess["label"], rest_value, rng)
    for s in strengths:
        out.append(dict(sess, emg=shrink(sess["emg"], s, noise), aug=f"strength {s:.2f}"))
    for r in rolls:
        out.append(dict(sess, emg=np.roll(sess["emg"], r, axis=1), aug=f"rotated {r:+d}"))
    return out

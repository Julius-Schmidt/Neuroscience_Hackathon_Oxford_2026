"""PhysioNet eegmmidb: load eyes-open / eyes-closed runs as Unicorn-like arrays.

Run 1 = 1 min eyes-open rest, run 2 = 1 min eyes-closed rest (64 ch, 160 Hz, µV).
We keep the 8 channels nearest the Unicorn montage and resample to 250 Hz.

A "scenario" splices pieces of the two runs into one timeline with ground truth:
eyes-open baseline, then open stretches, long closures (should toggle) and short
closures (must not toggle). Real blinks, drift and person-to-person differences
come with the recordings; only the open<->closed transitions are spliced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class SubjectEEG:
    subject: int
    open: np.ndarray        # (n_samples, 8) µV at 250 Hz, eyes open
    closed: np.ndarray      # (n_samples, 8) µV at 250 Hz, eyes closed
    srate: float = C.SRATE


@dataclass
class Scenario:
    eeg: np.ndarray                         # (n_samples, 8) µV
    srate: float
    baseline_s: float                       # first part is eyes-open calibration
    closed_intervals: list = field(default_factory=list)   # [(start_s, end_s, should_toggle)]
    subject: int = 0


def download(subjects, path=C.DATA_DIR) -> None:
    from mne.datasets import eegbci
    path.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        eegbci.load_data(s, [C.RUN_EYES_OPEN, C.RUN_EYES_CLOSED], path=str(path), update_path=False,
                         verbose="error")
        print(f"  subject {s:3d} ok")


def _load_run(subject: int, run: int, path=C.DATA_DIR) -> np.ndarray:
    import mne
    from mne.datasets import eegbci
    fname = eegbci.load_data(subject, [run], path=str(path), update_path=False, verbose="error")[0]
    raw = mne.io.read_raw_edf(fname, preload=True, verbose="error")
    eegbci.standardize(raw)                       # "Fz.." -> "Fz", "Po7." -> "PO7"
    raw.pick(C.EEG_CHANNELS)
    raw.resample(C.SRATE, verbose="error")
    return (raw.get_data(picks=C.EEG_CHANNELS).T * 1e6).astype(np.float32)   # V -> µV


def load_subject(subject: int, path=C.DATA_DIR) -> SubjectEEG:
    return SubjectEEG(subject, _load_run(subject, C.RUN_EYES_OPEN, path),
                      _load_run(subject, C.RUN_EYES_CLOSED, path))


FADE_S = 0.05


def _crossfade_concat(pieces: list[np.ndarray], srate: float, fade_s: float = FADE_S) -> np.ndarray:
    """Join pieces with a short crossfade and matched offsets, so splices don't add steps."""
    n_f = int(fade_s * srate)
    out = pieces[0].astype(np.float64).copy()
    for p in pieces[1:]:
        p = p.astype(np.float64) - p[:n_f].mean(axis=0) + out[-n_f:].mean(axis=0)
        w = np.linspace(0, 1, n_f)[:, None]
        out[-n_f:] = out[-n_f:] * (1 - w) + p[:n_f] * w
        out = np.vstack([out, p[n_f:]])
    return out.astype(np.float32)


def make_scenario(sub: SubjectEEG, rng: np.random.Generator, baseline_s: float = C.BASELINE_S,
                  n_long: int = 4, n_short: int = 4, long_s=(3.0, 6.0), short_s=(0.3, 1.0),
                  open_s=(4.0, 8.0)) -> Scenario:
    """Baseline (open) + shuffled long and short closures separated by open stretches.

    Pieces are cut at random offsets from the subject's own runs and may repeat
    material (each run is only ~60 s).
    """
    sr = sub.srate

    def cut(src, dur):
        n = int(dur * sr)
        start = rng.integers(0, len(src) - n)
        return src[start:start + n]

    kinds = ["long"] * n_long + ["short"] * n_short
    rng.shuffle(kinds)
    pieces = [sub.open[:int(baseline_s * sr)]]
    closed_piece = []                      # (piece index, should_toggle)
    rest_open = sub.open[int(baseline_s * sr):]
    for k in kinds:
        d_open = rng.uniform(*open_s)
        pieces.append(cut(rest_open if len(rest_open) > d_open * sr + 1 else sub.open, d_open))
        d = rng.uniform(*(long_s if k == "long" else short_s))
        closed_piece.append((len(pieces), k == "long"))
        pieces.append(cut(sub.closed, d))
    pieces.append(cut(sub.open, rng.uniform(*open_s)))
    eeg = _crossfade_concat(pieces, sr)

    # ground truth from actual sample positions (each join overlaps FADE samples;
    # a closed piece counts from the middle of its fade-in)
    n_f = int(FADE_S * sr)
    starts = np.cumsum([0] + [len(p) - n_f for p in pieces[:-1]])
    intervals = []
    for i, should in closed_piece:
        a = (starts[i] + n_f / 2) / sr
        b = (starts[i] + len(pieces[i]) - n_f / 2) / sr
        intervals.append((float(a), float(b), should))
    return Scenario(eeg=eeg, srate=sr, baseline_s=baseline_s, closed_intervals=intervals,
                    subject=sub.subject)


def replay_sequence(sub: SubjectEEG, rng: np.random.Generator, minutes: float = 10.0) -> np.ndarray:
    """Long open/closed timeline for the simulator (mostly open, closures 3-6 s)."""
    sc = make_scenario(sub, rng, baseline_s=C.BASELINE_S,
                       n_long=int(minutes * 3), n_short=int(minutes * 2), open_s=(8.0, 15.0))
    return sc.eeg

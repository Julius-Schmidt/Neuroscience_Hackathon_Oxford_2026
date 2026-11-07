"""One recording = EEG + EMG + markers on the same (LSL) clock.

Saved by session.py as .npz. Recordings made with LabRecorder (.xdf) load too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C


@dataclass
class Recording:
    eeg: np.ndarray            # (n, 8) µV
    eeg_t: np.ndarray          # (n,) s, LSL clock
    emg: np.ndarray            # (n, 8) raw Myo units
    emg_t: np.ndarray
    markers: list[str]
    marker_t: np.ndarray
    eeg_ch: list[str] = field(default_factory=lambda: list(C.EEG_CHANNELS))
    meta: dict = field(default_factory=dict)

    def marker_times(self, prefix: str) -> np.ndarray:
        return np.array([t for m, t in zip(self.markers, self.marker_t) if m.startswith(prefix)])

    def blocks(self) -> list[dict]:
        """[{name, kind, start, end}] from block_start / block_end markers."""
        out, open_ = [], {}
        for m, t in zip(self.markers, self.marker_t):
            if m.startswith("block_start:"):
                _, name, kind = m.split(":")
                open_[name] = {"name": name, "kind": kind, "start": float(t)}
            elif m.startswith("block_end:"):
                name = m.split(":")[1]
                if name in open_:
                    b = open_.pop(name)
                    b["end"] = float(t)
                    out.append(b)
        for b in open_.values():                         # session stopped mid-block
            b["end"] = float(max(self.eeg_t[-1], self.emg_t[-1]))
            out.append(b)
        return sorted(out, key=lambda b: b["start"])

    def summary(self) -> str:
        dur_eeg = self.eeg_t[-1] - self.eeg_t[0] if len(self.eeg_t) else 0
        dur_emg = self.emg_t[-1] - self.emg_t[0] if len(self.emg_t) else 0
        return (f"EEG {len(self.eeg_t)} samples ({dur_eeg:.0f} s, {len(self.eeg_t) / max(dur_eeg, 1e-9):.0f} Hz) | "
                f"EMG {len(self.emg_t)} samples ({dur_emg:.0f} s, {len(self.emg_t) / max(dur_emg, 1e-9):.0f} Hz) | "
                f"{len(self.markers)} markers, {len(self.blocks())} blocks, "
                f"{len(self.marker_times('cue_squeeze'))} squeeze cues")


def save(rec: Recording, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, eeg=rec.eeg.astype(np.float32), eeg_t=rec.eeg_t, emg=rec.emg.astype(np.float32),
             emg_t=rec.emg_t, markers=np.array(rec.markers, dtype=str), marker_t=rec.marker_t,
             eeg_ch=np.array(rec.eeg_ch, dtype=str), meta=json.dumps(rec.meta))
    tmp.replace(path)                                    # never leaves a half-written file


def load(path: str | Path) -> Recording:
    path = Path(path)
    if path.suffix.lower() == ".xdf":
        return load_xdf(path)
    d = np.load(path, allow_pickle=False)
    return Recording(eeg=d["eeg"].astype(np.float64), eeg_t=d["eeg_t"], emg=d["emg"].astype(np.float64),
                     emg_t=d["emg_t"], markers=[str(m) for m in d["markers"]], marker_t=d["marker_t"],
                     eeg_ch=[str(c) for c in d["eeg_ch"]], meta=json.loads(str(d["meta"])))


def load_xdf(path: Path) -> Recording:
    """LabRecorder file: picks the EEG, MyoEMG and FatigueMarkers streams by name/type."""
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=True)

    def find(pred, what):
        for s in streams:
            info = s["info"]
            if pred(info["name"][0], info["type"][0]):
                return s
        raise ValueError(f"{path.name}: no {what} stream; found "
                         f"{[(s['info']['name'][0], s['info']['type'][0]) for s in streams]}")

    eeg = find(lambda n, t: n == C.EEG_STREAM["name"] or t.upper() == "EEG", "EEG")
    emg = find(lambda n, t: n == C.EMG_STREAM["name"] or t.upper() == "EMG", "EMG")
    mk = find(lambda n, t: n == C.MARKER_STREAM["name"], "marker")
    return Recording(eeg=np.asarray(eeg["time_series"], float)[:, :len(C.EEG_CHANNELS)],
                     eeg_t=np.asarray(eeg["time_stamps"]),
                     emg=np.asarray(emg["time_series"], float)[:, :8], emg_t=np.asarray(emg["time_stamps"]),
                     markers=[str(m[0]) for m in mk["time_series"]], marker_t=np.asarray(mk["time_stamps"]),
                     meta={"source_file": path.name})

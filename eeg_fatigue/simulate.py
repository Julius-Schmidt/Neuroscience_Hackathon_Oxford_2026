"""Simulated head + arm for the dry run. NOT evidence: it shows the analysis can find
an effect that was put in on purpose, and finds nothing when none was put in.

    python -m eeg_fatigue.simulate --out data/fatigue/SIM_divergence.npz          offline, instant
    python -m eeg_fatigue.simulate --out data/fatigue/SIM_null.npz --scenario null
    python -m eeg_fatigue.simulate --lsl                 live: fake Unicorn + fake MyoEMG over LSL;
                                                         reacts to the session's squeeze cues

Model (per squeeze, f = fatigue 0..1, grows with every squeeze, partly recovers in rest):
    EEG  C3 beta/mu drop during the squeeze = 35% x (1 + G*f), weaker at Cz/C4; beta rebound after;
         motor potential before onset at C3/Cz = -6 µV x (1 + G*f) (offline only);
         posterior alpha x3.5 with eyes closed; frontal blinks; 1/f background; drift
    EMG  burst amplitude x (1 - M*f), spectrum shifts lower with f (median frequency falls)
Scenarios: divergence (G=0.8, M=0.35), muscle (G=0, M=0.35), null (G=0, M=0).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import signal

from . import config as C
from .protocol import sanity_steps, session_steps
from .recording import Recording, save

SCENARIOS = {"divergence": (0.8, 0.35), "muscle": (0.0, 0.35), "null": (0.0, 0.0)}
FATIGUE_STEP = 1 / 90          # per squeeze in baseline/fatigue blocks
RECOVERY = 0.7                 # fraction of fatigue removed by the rest period
REACTION = (0.25, 0.08)        # mean, sd of reaction time after the cue (s)


class Body:
    """Holds squeeze events and fatigue; produces EEG and EMG chunk by chunk."""

    def __init__(self, scenario="divergence", seed=0):
        self.G, self.M = SCENARIOS[scenario]
        self.rng = np.random.default_rng(seed)
        self.f = 0.0
        self.counting = False
        self.eyes_closed = False
        self.squeezes: list[tuple[float, float, float]] = []      # (onset, effort, emg_amp)
        self.closed_spans: list[list[float]] = []
        ch = C.EEG_CHANNELS
        self.w_motor = np.array([{"C3": 1.0, "Cz": 0.6, "C4": 0.4}.get(c, 0.0) for c in ch])
        self.w_mrcp = np.array([{"C3": 1.0, "Cz": 0.8, "C4": 0.3}.get(c, 0.0) for c in ch])
        self.w_alpha = np.array([1.0 if c in ("Pz", "PO7", "Oz", "PO8") else 0.25 for c in ch])
        self.phase = self.rng.uniform(0, 2 * np.pi, (3, len(ch)))
        self.b_pink, self.a_pink = [0.049922035, -0.095993537, 0.050612699, -0.004408786], \
                                   [1, -2.494956002, 2.017265875, -0.522189400]
        self.zi_eeg = np.zeros((3, len(ch)))
        self.zi_emg_lp = None
        self.drift = np.zeros(len(ch))
        self.next_blink = 2.0

    # --- events -------------------------------------------------------------------
    def on_marker(self, m: str, t: float):
        if m.startswith("block_start:"):
            self.counting = m.split(":")[2] in ("baseline", "fatigue", "recovery")
        elif m == "rest_end":
            self.f *= (1 - RECOVERY)
        elif m == "eyes_closed":
            self.closed_spans.append([t, np.inf])
        elif m == "eyes_open" and self.closed_spans and self.closed_spans[-1][1] == np.inf:
            self.closed_spans[-1][1] = t
        elif m == "cue_squeeze":
            onset = t + max(0.12, self.rng.normal(*REACTION))
            effort = (1 + self.G * self.f) * self.rng.normal(1, 0.1)
            amp = 45.0 * (1 - self.M * self.f) * self.rng.normal(1, 0.08)
            self.squeezes.append((onset, effort, amp, self.f))
            if self.counting:
                self.f = min(1.0, self.f + FATIGUE_STEP)

    # --- signals --------------------------------------------------------------------
    def eeg(self, t: np.ndarray) -> np.ndarray:
        n, nch = len(t), len(C.EEG_CHANNELS)
        dt = t[1] - t[0] if n > 1 else 1 / C.EEG_SRATE
        white = self.rng.standard_normal((n, nch))
        pink, self.zi_eeg = signal.lfilter(self.b_pink, self.a_pink, white, axis=0, zi=self.zi_eeg)
        x = 25.0 * pink
        # motor rhythm modulation
        mod = np.ones(n)
        mrcp = np.zeros(n)
        for onset, eff, _, _ in self.squeezes:
            if t[0] - 4 > onset or t[-1] + 2 < onset:
                continue
            u = t - onset
            depth = min(0.9, 0.35 * eff)
            win = np.clip((u + 0.3) / 0.3, 0, 1) * np.clip((C.SQUEEZE_S - u) / 0.3, 0, 1)
            reb = np.exp(-0.5 * ((u - C.SQUEEZE_S - 0.7) / 0.35) ** 2) * 0.3
            mod *= (1 - depth * win) * (1 + reb)
            mrcp += -6.0 * eff * np.where((u > -0.8) & (u <= 0), (u + 0.8) / 0.8,
                                          np.where((u > 0) & (u < 0.5), 1 - u / 0.5, 0))
        closed = np.zeros(n, bool)
        for a, b in self.closed_spans:
            closed |= (t >= a) & (t < b)
        rates = [10.0, 11.0, 20.0]
        amps = [4.0, 6.0, 4.0]
        for k, (fr, am) in enumerate(zip(rates, amps)):
            ph = self.phase[k] + 2 * np.pi * fr * dt * np.arange(1, n + 1)[:, None]
            self.phase[k] = ph[-1] % (2 * np.pi)
            if k == 0:
                env = np.where(closed, 3.5, 1.0)[:, None] * self.w_alpha
            else:
                env = (mod[:, None] * self.w_motor + (1 - self.w_motor))
            x += am * env * np.sin(ph)
        x += mrcp[:, None] * self.w_mrcp
        self.drift += self.rng.normal(0, 0.3, nch)
        x += self.drift
        fz = C.EEG_CHANNELS.index("Fz")
        while self.next_blink < t[-1]:
            if not np.any((self.next_blink >= np.array([a for a, _ in self.closed_spans] or [np.inf])) &
                          (self.next_blink < np.array([b for _, b in self.closed_spans] or [np.inf]))):
                x[:, fz] += 90 * np.exp(-0.5 * ((t - self.next_blink) / 0.08) ** 2)
            self.next_blink += self.rng.exponential(4.0)
        return x

    def emg(self, t: np.ndarray) -> np.ndarray:
        n = len(t)
        env = np.full(n, 2.0)
        low_mix = np.zeros(n)
        for onset, _, amp, f_then in self.squeezes:
            u = t - onset
            on = np.clip(u / 0.05, 0, 1) * np.clip((C.SQUEEZE_S - 0.25 - u) / 0.1, 0, 1)
            env = np.maximum(env, amp * on)
            low_mix = np.maximum(low_mix, on * (0.2 + 0.6 * f_then * (1 if self.M > 0 else 0)))
        white = self.rng.standard_normal((n, 8))
        b, a = signal.butter(2, 35, fs=C.EMG_SRATE)
        if self.zi_emg_lp is None:
            self.zi_emg_lp = np.zeros((2, 8))
        low, self.zi_emg_lp = signal.lfilter(b, a, white, axis=0, zi=self.zi_emg_lp)
        low /= 0.45                                                    # ~unit variance
        mix = (1 - low_mix)[:, None] * white + low_mix[:, None] * low
        gains = np.array([1.0, 0.9, 0.7, 0.5, 0.4, 0.5, 0.7, 0.9])
        return np.clip(np.round(env[:, None] * gains * mix), -128, 127)


def offline(scenario="divergence", seed=0, sanity=False) -> Recording:
    steps = sanity_steps(seed) if sanity else session_steps(seed)
    body = Body(scenario, seed)
    rng = np.random.default_rng(seed + 1)
    t0 = 1000.0
    markers, mt = [], []
    t = t0 + 2.0
    for s in steps:
        if s["marker"]:
            markers.append(s["marker"])
            mt.append(t)
            body.on_marker(s["marker"], t)
        t += s["dur"]
    t_end = t + 2.0
    # generate in 1 s chunks (same code path as live)
    eeg_t = np.arange(t0, t_end, 1 / C.EEG_SRATE)
    emg_t = np.arange(t0, t_end, 1 / C.EMG_SRATE)
    eeg = np.vstack([body.eeg(eeg_t[i:i + 250]) for i in range(0, len(eeg_t), 250)])
    emg = np.vstack([body.emg(emg_t[i:i + 200]) for i in range(0, len(emg_t), 200)])
    # Bluetooth realism: arrival jitter and ~1% lost packets (2 samples each)
    keep = np.ones(len(emg_t), bool)
    for i in rng.choice(len(emg_t) // 2, size=len(emg_t) // 200, replace=False):
        keep[2 * i:2 * i + 2] = False
    emg_t = emg_t + rng.uniform(0, 0.012, len(emg_t))
    return Recording(eeg=eeg, eeg_t=eeg_t, emg=emg[keep], emg_t=emg_t[keep], markers=markers,
                     marker_t=np.array(mt), meta={"participant": f"SIM_{scenario}", "kind": "sanity" if sanity else "session",
                                                  "simulated": True, "scenario": scenario, "hand": C.HAND})


def live(scenario="divergence", seed=0):
    """Publish fake Unicorn EEG + fake MyoEMG; react to squeeze cues from eeg_fatigue.session."""
    import pylsl
    body = Body(scenario, seed)
    s = C.EEG_STREAM
    eeg_out = pylsl.StreamOutlet(pylsl.StreamInfo(s["name"], s["type"], 8, C.EEG_SRATE, "float32", "eeg_fatigue_sim_eeg"))
    e = C.EMG_STREAM
    emg_out = pylsl.StreamOutlet(pylsl.StreamInfo(e["name"], e["type"], 8, C.EMG_SRATE, "float32", "eeg_fatigue_sim_emg"))
    print(f"simulating '{s['name']}' EEG + '{e['name']}' EMG ({scenario}). Waiting for session markers ... Ctrl+C to stop")
    inlet = None
    t0 = pylsl.local_clock()
    n_eeg = n_emg = 0
    last_look = 0.0
    while True:
        now = pylsl.local_clock()
        if inlet is None and now - last_look > 2:
            last_look = now
            found = pylsl.resolve_byprop("name", C.MARKER_STREAM["name"], minimum=1, timeout=0.1)
            if found:
                inlet = pylsl.StreamInlet(found[0], processing_flags=pylsl.proc_clocksync)
                print("  connected to session markers")
        if inlet is not None:
            while True:
                m, ts = inlet.pull_sample(timeout=0.0)
                if m is None:
                    break
                body.on_marker(m[0], ts)
        k_eeg = int((now - t0) * C.EEG_SRATE) - n_eeg
        if k_eeg > 0:
            tt = t0 + (n_eeg + np.arange(k_eeg)) / C.EEG_SRATE
            eeg_out.push_chunk(body.eeg(tt).astype(np.float32).tolist(), tt.tolist())
            n_eeg += k_eeg
        k_emg = int((now - t0) * C.EMG_SRATE) - n_emg
        if k_emg > 0:
            tt = t0 + (n_emg + np.arange(k_emg)) / C.EMG_SRATE
            emg_out.push_chunk(body.emg(tt).astype(np.float32).tolist(), tt.tolist())
            n_emg += k_emg
        time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="divergence")
    ap.add_argument("--out", help="write an offline recording (.npz)")
    ap.add_argument("--sanity", action="store_true", help="offline: simulate the 3 min sanity check instead")
    ap.add_argument("--lsl", action="store_true", help="live fake headset + arm over LSL")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.lsl:
        live(a.scenario, a.seed)
        return
    out = a.out or str(C.DATA_DIR / f"SIM_{a.scenario}{'_sanity' if a.sanity else ''}.npz")
    rec = offline(a.scenario, a.seed, a.sanity)
    from pathlib import Path
    save(rec, Path(out))
    print(f"wrote {out}\n  {rec.summary()}")


if __name__ == "__main__":
    main()

"""python -m devtool.live_eval <raw.npz> <model dir> <strength> <rescale 0|1> <hold ms> [tremor]

Run the LIVE FistDetector (sample by sample, as on the band) over the fatigue recording,
with simulated weakness, plain vs --rescale, and report caught / false per min."""
import json, sys
import numpy as np
from devtool.replay import load
from devtool.replay_fatigue import load_models, score
from devtool.weakness_sim import make_noise, noise_pool, simulate
from myomouse.detector import FistDetector

npz, mdir, s, rescale, hold = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5])
bands, phases, _e, _s, _c = load(npz)
cues = json.loads(str(np.load(npz, allow_pickle=False)["cues"]))
side = "right"; d = bands[side]; v = d["valid"]; emg, idx = d["emg"][v], d["idx"][v]
ph = {n: i[side] for n, _t, i in phases}; end = int(d["idx"][-1]) + 1
rest_range = (ph.get("recovery", end), ph.get("post", end))
lo, hi = (np.searchsorted(idx, r) for r in rest_range)
noise = make_noise(noise_pool(emg, lo, hi), len(emg), np.random.default_rng(0))
x = simulate(emg, s, noise) if s < 1 else emg
trem = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
if trem:
    from devtool.tremor_sim import fist_profile, tremor
    prof = fist_profile(emg, idx, cues, side)
    x = np.clip(np.round(x.astype(float) + tremor(len(x), prof, trem, np.random.default_rng(1))), -128, 127).astype(np.int8)
det = FistDetector(load_models(mdir)[side], "lda", hold_ms=hold, rescale=rescale)
presses, gains = [], []
for k, smp in enumerate(x):
    if det.push(smp) == "press":
        presses.append(int(idx[k]))
    if k % 2000 == 0:
        gains.append(round(det.gain, 2))
caught, false = score(np.array(presses, dtype=np.int64), np.array([c[2][side] for c in cues]), rest_range)
mins = len(emg) / 200 / 60
print(json.dumps(dict(tremor=trem, model=mdir, strength=s, rescale=rescale, hold=hold, caught=round(100 * caught.mean(), 1),
                      false_per_min=round(len(false) / mins, 2), gain_end=gains[-1], gains=gains[::6])), flush=True)

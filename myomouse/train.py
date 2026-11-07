"""Train the fist classifier for one arm from every recording in data/.

    python -m myomouse.train --side right
    python -m myomouse.train --side left

Saves models/fist_<side>.joblib with both an LDA model and a threshold fallback,
and prints how many clicks it would have caught / falsely fired on the recording.
"""

import argparse

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .detector import FistDetector, model_path
from .features import envelope, sliding_features
from .record import FIST, IGNORE


def load_sessions(side):
    files = sorted(C.DATA_DIR.glob(f"{side}_*.npz"))
    if not files:
        raise SystemExit(f"No recordings for {side} in {C.DATA_DIR}. Run myomouse.record first.")
    return [dict(np.load(f)) | {"file": f.name} for f in files]


def windows(sess, window, step):
    X, ends = sliding_features(sess["emg"], window, step)
    lab = sess["label"]
    # keep only windows that lie entirely inside one clean (non-transition) phase
    keep = np.array([(lab[e - window:e] == lab[e - 1]).all() and lab[e - 1] != IGNORE for e in ends])
    env = np.array([envelope(sess["emg"][e - window:e]) for e in ends])
    groups = sess["seg"][ends - 1] + 1000 * sess["sid"]
    return X[keep], lab[ends - 1][keep], groups[keep], env[keep]


def event_report(model, mode, sess):
    """Replay the recording through the live detector and score it like a demo would."""
    det = FistDetector(model, mode)
    lab, seg = sess["label"], sess["seg"]
    presses = []
    for i, s in enumerate(sess["emg"]):
        if det.push(s) == "press":
            presses.append(i)
    fist_segs = sorted({int(g) for g, l in zip(seg, lab) if l == FIST})
    seg_start = {int(g): int(np.argmax(seg == g)) for g in np.unique(seg)}
    hit, lat = set(), []
    false = 0
    for i in presses:
        g = int(seg[i])
        # a press up to 0.4 s into the following relax phase still belongs to that fist
        if g not in fist_segs and (g - 1) in fist_segs and (i - seg_start[g]) < 0.4 * C.EMG_HZ:
            g -= 1
        if g in fist_segs:
            if g not in hit:
                lat.append((i - seg_start[g]) / C.EMG_HZ * 1000)
            hit.add(g)
        else:
            false += 1
    lat_s = f", median {np.median(lat):.0f} ms after the cue (includes your reaction time)" if lat else ""
    return (f"{mode:9s}: caught {len(hit)}/{len(fist_segs)} fists, "
            f"{false} false clicks, {len(presses) - len(hit) - false} double clicks{lat_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    args = ap.parse_args()

    sessions = load_sessions(args.side)
    parts = []
    for sid, s in enumerate(sessions):
        s["sid"] = sid
        parts.append(windows(s, C.WINDOW, C.STEP))
        print(f"{s['file']}: {len(s['emg']) / C.EMG_HZ:.0f} s")
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    env = np.concatenate([p[3] for p in parts])
    print(f"{len(y)} windows, {int((y == FIST).sum())} fist / {int((y != FIST).sum())} rest")

    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    pred = cross_val_predict(pipe, X, y, groups=groups, cv=GroupKFold(n_splits=5))
    print(f"\ncross-validated window accuracy (balanced): {balanced_accuracy_score(y, pred):.3f}")
    print("confusion [rest, fist] rows=true:\n", confusion_matrix(y, pred))
    pipe.fit(X, y)

    rest_med, fist_med = np.median(env[y != FIST]), np.median(env[y == FIST])
    if fist_med <= rest_med:
        print("WARNING: fist isn't stronger than rest in this data - check the band is on and tight.")
    threshold = {"center": float((rest_med + fist_med) / 2), "scale": float(max(fist_med - rest_med, 1e-3) / 6)}

    model = {"pipeline": pipe, "threshold": threshold, "window": C.WINDOW, "step": C.STEP, "side": args.side}
    C.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, model_path(args.side))
    print(f"\nsaved {model_path(args.side)}")

    print("\nreplay of the recordings (on training data, so slightly optimistic):")
    for s in sessions:
        for mode in ("lda", "threshold"):
            print(" ", s["file"], event_report(model, mode, s))


if __name__ == "__main__":
    main()

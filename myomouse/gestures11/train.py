"""Train one user's multi-gesture classifier for one arm from data/<user>/gestures_<side>_*.npz.

    python -m myomouse.gestures11.train --user julius --side right
    python -m myomouse.gestures11.train --user julius --side right --model svm
    python -m myomouse.gestures11.train --user julius --side right --gestures power,open,flex,ext

Prints two accuracies:
  - same session: tested on held-out repetitions of recordings it also trained on (optimistic)
  - new session:  tested on a whole recording it never saw (needs 2+ recordings). This is the
                  number that tells you how it'll do after taking the band off and on.
both per 200 ms window. Then a confusion matrix (rows = what you did, columns = what it said,
in %) and the gesture pairs it mixes up most.

Then the live pipeline (gestures11/decide.py: rejection + switching, + rest gate with --rest-gate) is replayed
on the held-out data for a few rejection thresholds:
  - idle -> gesture:  how often a gesture shows while you're in neutral / rest (want: low)
  - gesture right:    how often the right gesture shows while you hold one
  - wrong gesture:    how often a DIFFERENT gesture shows while you hold one (want: low)
The model is saved with --reject (default REJECT in gestures11/__init__.py); live can override it.

--gestures trains on a subset (neutral + rest always included), e.g. to drop pairs it confuses.
--labels cue ignores the camera labels (where recorded), to compare the two.
"""

import argparse

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import config as C
from ..features import envelope, sliding_features
from ..users import add_user_arg
from . import (GESTURES, IDLE, IGNORE, KEYS, N_OFF, N_ON, REJECT, REST, REST_GATE, STEP, WINDOW, model_path,
               parse_gestures, recordings)
from .decide import gate_from_rest, simulate


def load_sessions(user, side, labels="camera"):
    files = recordings(user, side)
    if not files:
        raise SystemExit(f"No gesture recordings for {user}'s {side} arm.\n"
                         f"  Record first:  python -m myomouse.gestures11.record --user {user} --side {side}")
    out = []
    for f in files:
        s = dict(np.load(f))
        s.setdefault("label_cue", s["label"])
        s["camera"] = "cam_t" in s
        if labels == "cue":
            s["label"] = s["label_cue"]
        # map the file's label numbers to today's gesture list by name
        names = [str(k) for k in s["gestures"]]
        remap = np.array([KEYS.index(k) if k in KEYS else IGNORE for k in names])
        for key in ("label", "label_cue", "seg_label"):   # every label array in the file
            if key in s:
                lab = s[key]
                s[key] = np.where(lab >= 0, remap[np.clip(lab, 0, None)], IGNORE)
        s["file"] = f.name
        out.append(s)
    return out


def moving_steps(sess):
    """Step ids of "rest while moving the arm" (more EMG than a still, limp hand)."""
    if "seg_moving" in sess:
        return set(np.flatnonzero(sess["seg_moving"]).tolist())
    # older recordings: the rest block was [still, moving], so the second rest step moved
    rest_steps = sorted({int(g) for g, l in zip(sess["seg"], sess["label_cue"]) if l == REST})
    return {rest_steps[1]} if len(rest_steps) > 1 else set()


def windows(sess, window=WINDOW, step=STEP):
    """Features of every window that lies entirely inside one labelled hold.
    Returns X, y, hold id, envelope (overall EMG level, for the rest gate)."""
    X, ends = sliding_features(sess["emg"], window, step)
    lab, seg = sess["label"], sess["seg"]
    keep = np.array([lab[e - 1] != IGNORE and (lab[e - window:e] == lab[e - 1]).all() for e in ends], dtype=bool)
    if not keep.any():
        return X[:0], lab[:0], seg[:0], np.zeros(0)
    env = np.array([envelope(sess["emg"][e - window:e]) for e in ends[keep]])
    return X[keep], lab[ends - 1][keep], seg[ends - 1][keep], env


def make_model(kind):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", probability=True))
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


def live_metrics(y, shown):
    """(idle -> gesture, gesture right, wrong gesture) rates of the live output `shown` (keys)."""
    idle = np.isin(y, IDLE)
    truth = np.array([KEYS[c] for c in y], dtype=object)
    false_on = float(np.mean(shown[idle] != "neutral")) if idle.any() else float("nan")
    g = ~idle
    right = float(np.mean(shown[g] == truth[g])) if g.any() else float("nan")
    wrong = float(np.mean((shown[g] != truth[g]) & (shown[g] != "neutral"))) if g.any() else float("nan")
    return false_on, right, wrong


def live_report(y, proba, env, holds, classes, gate, chosen, gate_available=None):
    """Table: the live pipeline at a few rejection thresholds (gate = the one saved, or None),
    plus a comparison row with the rest gate if it's available but not used (or the other way round)."""
    keys = [KEYS[c] for c in classes]
    rows = [("no filters (plain classifier)", dict(gate=None, reject=0.0, n_on=1, n_off=1))]
    for r in sorted({0.6, 0.75, 0.9, chosen}):
        rows.append((f"reject < {r:.2f}" + ("  <- saved" if r == chosen else ""), dict(gate=gate, reject=r)))
    if gate is not None:
        rows.append((f"reject < {chosen:.2f}, no rest gate", dict(gate=None, reject=chosen)))
    elif gate_available is not None:
        rows.append((f"reject < {chosen:.2f}, + rest gate", dict(gate=gate_available, reject=chosen)))
    lines = [f"{'':34s}{'idle -> gesture':>16s}{'gesture right':>15s}{'wrong gesture':>15s}"]
    for name, kw in rows:
        f, r, w = live_metrics(y, simulate(proba, env, holds, keys, **kw))
        lines.append(f"{name:34s}{f:16.1%}{r:15.1%}{w:15.1%}")
    return "\n".join(lines)


def confusion_report(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes).astype(float)
    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    shorts = [GESTURES[c].short for c in classes]
    lines = ["true \\ said".ljust(17) +"".join(f"{s:>6}" for s in shorts) + "   recall"]
    for i, c in enumerate(classes):
        row = "".join(f"{v:6.0f}" if v >= 0.5 else f"{'.':>6}" for v in pct[i])
        lines.append(f"{GESTURES[c].name:17s}" + row + f"   {pct[i, i]:5.0f}%")
    # most confused pairs (both directions added up)
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            pairs.append((pct[i, j] + pct[j, i], i, j))
    pairs.sort(reverse=True)
    worst = [(v, i, j) for v, i, j in pairs[:3] if v >= 5]
    if worst:
        lines.append("\nmost mixed up:")
        for v, i, j in worst:
            lines.append(f"  {GESTURES[classes[i]].name} <-> {GESTURES[classes[j]].name}: "
                         f"{pct[i, j]:.0f}% + {pct[j, i]:.0f}%")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_user_arg(ap)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--model", choices=["lda", "svm"], default="lda",
                    help="lda = fast, robust (default); svm = sometimes better on the hard pairs")
    ap.add_argument("--labels", choices=["camera", "cue"], default="camera",
                    help="camera-timed labels (default, where recorded) or plain cue timing")
    ap.add_argument("--reject", type=float, default=REJECT,
                    help=f"rejection threshold saved with the model (default {REJECT})")
    ap.add_argument("--rest-gate", action=argparse.BooleanOptionalAction, default=REST_GATE,
                    help="also hide gestures quieter than your recorded rest (default: off)")
    ap.add_argument("--gestures", default=None,
                    help="only these gestures (keys or short names, comma separated), e.g. power,open,flex,ext")
    args = ap.parse_args(argv)

    keep_classes = parse_gestures(args.gestures)
    sessions = load_sessions(args.user, args.side, args.labels)
    Xs, ys, holds, sids, envs, stills, movings = [], [], [], [], [], [], []
    for sid, s in enumerate(sessions):
        X, y, h, e = windows(s)
        m = np.isin(y, keep_classes)
        Xs.append(X[m]); ys.append(y[m]); holds.append(h[m] + 100000 * sid); sids.append(np.full(m.sum(), sid))
        envs.append(e[m])
        stills.append((y[m] == REST) & ~np.isin(h[m], list(moving_steps(s))))
        movings.append(np.isin(h[m], list(moving_steps(s))) & (y[m] != REST))   # gestures / neutral, arm moving
        src = "camera labels" if s["camera"] and args.labels == "camera" else "cue labels"
        print(f"{s['file']}: {len(s['emg']) / C.EMG_HZ:.0f} s, {m.sum()} windows, {src}")
    X, y, holds, sids, env, still, moving = map(np.concatenate, (Xs, ys, holds, sids, envs, stills, movings))
    classes = [c for c in keep_classes if (y == c).any()]
    missing = [GESTURES[c].name for c in keep_classes if c not in classes]
    if missing:
        print(f"WARNING: no data for {', '.join(missing)}"
              + (" (record again: recordings now start with a rest block)" if "Rest" in missing else ""))
    if len(classes) < 2:
        raise SystemExit("need at least two gestures with data")
    counts = ", ".join(f"{GESTURES[c].short} {int((y == c).sum())}" for c in classes)
    print(f"{len(y)} windows ({counts})\n")

    pipe = make_model(args.model)
    # whole holds are held out, spread so every gesture has holds left to train on in each fold
    fewest = min(len(np.unique(holds[y == c])) for c in classes)
    if fewest < 2:
        raise SystemExit("every gesture needs at least 2 repetitions to test on held-out ones")
    cv = StratifiedGroupKFold(n_splits=min(5, fewest), shuffle=True, random_state=0)
    within_p = cross_val_predict(pipe, X, y, groups=holds, cv=cv, method="predict_proba")
    within = np.array(classes)[within_p.argmax(axis=1)]
    print(f"same session  (held-out repetitions): window {balanced_accuracy_score(y, within):.1%}")
    shown, shown_p, what = within, within_p, "same session"
    if len(sessions) >= 2:
        cross_p = cross_val_predict(pipe, X, y, groups=sids, cv=LeaveOneGroupOut(), method="predict_proba")
        cross = np.array(classes)[cross_p.argmax(axis=1)]
        print(f"new session   (held-out recording):   window {balanced_accuracy_score(y, cross):.1%}")
        for sid, s in enumerate(sessions):
            m = sids == sid
            print(f"    tested on {s['file']}: {balanced_accuracy_score(y[m], cross[m]):.1%}")
        shown, shown_p, what = cross, cross_p, "new session"
    else:
        print("new session: record a second time (band off and on in between) to see this - "
              "it's the realistic number")

    print(f"\nconfusion matrix ({what}, per window, % of each row):")
    print(confusion_report(y, shown, classes))
    if moving.any() and len(np.unique(y[moving])) > 1:
        print(f"\nwhile MOVING the arm ({what}, {moving.sum()} windows): window {balanced_accuracy_score(y[moving], shown[moving]):.1%}"
              f"   arm still: {balanced_accuracy_score(y[~moving], shown[~moving]):.1%}")
        print(confusion_report(y[moving], shown[moving], [c for c in classes if (y[moving] == c).any()]))
    else:
        print("\nwhile moving the arm: no data yet - record again (recordings now include moving repetitions)")

    # rest gate: EMG as quiet as your limp hand -> neutral
    # only the still rest: moving the arm adds EMG and would push the gate above the gestures
    gate_available = gate_from_rest(env[still])
    gate = gate_available if args.rest_gate else None
    if gate_available is None:
        print("\nrest gate: off (no rest recorded)")
    else:
        gated = np.mean(env[~np.isin(y, IDLE)] < gate_available)
        print(f"\nrest gate {'ON' if gate is not None else 'off (--rest-gate turns it on)'}: "
              f"EMG level below {gate_available:.2f} would count as neutral "
              f"(still rest median {np.median(env[still]):.2f}, neutral {np.median(env[y == IDLE[0]]):.2f}); "
              f"it would hide {gated:.0%} of gesture windows")
        if gated > 0.1 and gate is not None:
            print("  WARNING: some gestures are barely stronger than rest - do them more firmly when recording")
    print(f"\nlive output ({what}; switching: {N_ON} decisions in a row into a gesture, {N_OFF} back to neutral):")
    print(live_report(y, shown_p, env, holds, classes, gate, args.reject, gate_available))

    pipe.fit(X, y)
    model = {"pipeline": pipe, "classes": [KEYS[c] for c in pipe.classes_], "window": WINDOW, "step": STEP,
             "gate": gate, "reject": args.reject, "n_on": N_ON, "n_off": N_OFF, "kind": args.model, "side": args.side, "user": args.user,
             "trained_on": [s["file"] for s in sessions]}
    out = model_path(args.user, args.side)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"\nsaved {out}")
    print(f"next: python -m myomouse.gestures11.live --user {args.user} --side {args.side}")


if __name__ == "__main__":
    main()

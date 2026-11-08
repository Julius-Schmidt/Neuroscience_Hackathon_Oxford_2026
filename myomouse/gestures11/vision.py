"""Camera labels for the gestures (recording only; live recognition is Myo-only).

Same idea as myomouse.vision for the fist: the cue says when you *should* do a gesture,
the camera sees when you actually did. Every gesture moves the hand away from neutral,
either the fingers (grips, open hand) or the whole hand (wrist bends, forearm turns), so
one measure works for all of them:

    pose change = RMS distance (cm) of the 21 MediaPipe landmarks from your neutral pose,
                  measured relative to the wrist (so moving the arm around doesn't count,
                  but rotating the hand does)

Per hold, the camera decides when the gesture started and ended. With 3+ repetitions it
also compares each hold's pose with the other repetitions of the same gesture: a hold that
looks more like a different gesture (you did the wrong one) is left out of training.
Like the fist version it only makes labels more cautious; where it can't tell, it keeps
the cue timing or leaves samples out.
"""

from pathlib import Path

import numpy as np

from .. import config as C
from ..vision import HandCamera, _hysteresis
from . import CAM_MIN_GAP_CM, CAM_STRAY_FRAC, GESTURES, IDLE, IGNORE, NEUTRAL

WRIST = 0
ASSETS = Path(__file__).resolve().parent / "assets"
_pictures = {}


def picture(key):
    """Example photo of a gesture (assets/<key>.png) as a BGR array, or None if missing."""
    if key not in _pictures:
        import cv2
        p = ASSETS / f"{key}.png"
        _pictures[key] = cv2.imread(str(p)) if p.exists() else None
    return _pictures[key]


def relative(pts):
    """(..., 21, 3) world landmarks -> wrist-relative, in cm."""
    p = np.asarray(pts, dtype=np.float64)
    return (p - p[..., WRIST:WRIST + 1, :]) * 100.0


def pose_change(pts, ref):
    """RMS landmark distance (cm) between pose(s) `pts` (..., 21, 3) and reference `ref` (21, 3)."""
    d = relative(pts) - relative(ref)
    return np.sqrt(np.mean(np.sum(d * d, axis=-1), axis=-1))


class GestureCamera(HandCamera):
    """HandCamera that keeps every frame's landmarks and shows 'change from neutral'."""

    def __init__(self, side, t_ref, **kw):
        super().__init__(side, t_ref, click="fist", keep_points=True, **kw)
        self.neutral = None  # set by calibrate_neutral()

    def calibrate_neutral(self, n=15):
        """Use the last n frames as the neutral pose (for the live meter only). True if it worked."""
        recent = np.asarray(self.points[-n:]) if self.points else np.empty((0, 21, 3))
        ok = np.isfinite(recent).all(axis=(1, 2)) if len(recent) else np.zeros(0, bool)
        if ok.sum() < n // 2:
            return False
        self.neutral = np.median(recent[ok], axis=0)
        return True

    def change(self):
        p = self.last_points
        if self.neutral is None or p is None or not np.isfinite(p).all():
            return np.nan
        return float(pose_change(p, self.neutral))

    def meter(self, ratio):
        d = self.change()
        if np.isnan(d):
            return 0.0, "hand seen"
        return float(np.clip(d / 5.0, 0, 1)), f"change from neutral {d:.1f} cm"

    def status(self):
        if self.last is None:
            return "cam: -"
        if self.last_points is None or not np.isfinite(self.last_points).all():
            return f"cam: no {self.side} hand"
        d = self.change()
        return "cam: hand seen" if np.isnan(d) else f"cam: change {d:4.1f} cm"


def gesture_vision_labels(emg_t, cue, seg, seg_label, cam_t, cam_pts, lag=None, margin=None,
                          early=None, spill=None, min_gap=CAM_MIN_GAP_CM):
    """Refine the cue labels of a gesture recording with the camera.

    emg_t:     arrival time (s) of every EMG sample
    cue:       cue label per sample (gesture index, IGNORE near step edges)
    seg:       step id per sample;  seg_label[step id] = gesture of that step
    cam_t:     time of every camera frame;  cam_pts: (frames, 21, 3) world landmarks, nan = no hand
    Returns (labels, report_lines).
    """
    lag = C.VISION_LAG_S if lag is None else lag
    margin = C.VISION_MARGIN_S if margin is None else margin
    early = C.VISION_EARLY_S if early is None else early
    spill = C.VISION_SPILL_S if spill is None else spill
    emg_t, cue, seg = np.asarray(emg_t, float), np.asarray(cue), np.asarray(seg)
    seg_label = np.asarray(seg_label)
    cam_t = np.asarray(cam_t, float) - lag
    cam_pts = np.asarray(cam_pts, float).reshape(len(cam_t), 21, 3) if len(cam_t) else np.empty((0, 21, 3))
    labels = cue.copy()
    seen = np.isfinite(cam_pts).all(axis=(1, 2)) if len(cam_t) else np.zeros(0, bool)
    if seen.sum() < 20:
        return labels, ["camera: no hand seen - using the cue timing only"]

    # each EMG sample -> latest camera frame before it (if recent enough)
    k = np.clip(np.searchsorted(cam_t, emg_t, side="right") - 1, 0, None)
    fresh = (np.searchsorted(cam_t, emg_t, side="right") > 0) & (emg_t - cam_t[k] < C.VISION_MAX_GAP_S)
    visible = fresh & seen[k]

    # neutral pose = median of the frames during the (trimmed) neutral phases
    nf = np.unique(k[visible & (cue == NEUTRAL)])
    if len(nf) < 20:
        return labels, ["camera: hand not visible enough while neutral - using the cue timing only"]
    neutral = np.median(cam_pts[nf], axis=0)
    d_frame = np.full(len(cam_t), np.nan)
    d_frame[seen] = pose_change(cam_pts[seen], neutral)
    d = np.where(visible, d_frame[k], np.nan)
    neutral_lvl = float(np.median(d_frame[nf]))

    holds = [int(g) for g in np.unique(seg) if seg_label[int(g)] not in IDLE]  # rest blocks keep cue labels
    zone_all = np.zeros(len(cue), bool)
    timed, delays, lengths = [], [], []
    missed, blind, flat = [], [], []
    for g in holds:
        gest = int(seg_label[g])
        idx = np.flatnonzero(seg == g)
        t0, t1 = emg_t[idx[0]], emg_t[idx[-1]]
        z = (emg_t >= t0 - early) & (emg_t <= t1 + spill)
        zone_all |= z
        if (z & visible).sum() < 0.3 * z.sum():
            blind.append(gest)                       # mostly out of view: keep the cue labels
            continue
        mid = visible & (seg == g) & (cue == gest)
        if mid.sum() < 10:
            blind.append(gest)
            continue
        gap = float(np.percentile(d[mid], 75)) - neutral_lvl
        if gap < min_gap:
            flat.append(gest)                        # camera can't see this one: keep the cue labels
            continue

        # active / not per camera frame (hysteresis), then per EMG sample
        frames = np.flatnonzero((cam_t >= t0 - early) & (cam_t <= t1 + spill))
        amount = (d_frame[frames] - neutral_lvl) / gap
        states = _hysteresis(amount, on=C.VISION_ON, off=C.VISION_OFF)
        st = np.full(len(cam_t), np.nan)
        st[frames] = [np.nan if s is None else float(s) for s in states]
        active_s = z & visible & (st[k] == 1.0)
        known = z & visible & np.isfinite(st[k])
        switch_t = np.array([cam_t[frames[i]] for i in range(1, len(states))
                             if states[i] is not None and states[i - 1] is not None and states[i] != states[i - 1]])
        near = np.zeros(len(cue), bool)
        if len(switch_t):
            near = z & (np.min(np.abs(emg_t[:, None] - switch_t[None, :]), axis=1) < margin)

        if not active_s.any():
            missed.append(gest)                      # never seen doing it: don't guess
            labels[z] = np.where(known, NEUTRAL, IGNORE)[z]
            labels[z & (seg == g)] = IGNORE
            continue
        zl = np.where(active_s, gest, NEUTRAL)
        zl[~known | near] = IGNORE
        labels[z] = zl[z]
        ta = emg_t[active_s]
        delays.append((ta[0] - t0) * 1000)
        lengths.append((ta[-1] - ta[0]) * 1000)
        act_frames = np.unique(k[active_s])
        timed.append((g, gest, np.median(cam_pts[act_frames], axis=0), z))

    # wrong gesture? compare each hold with the other repetitions of every gesture
    wrong = []
    by_gest = {}
    for g, gest, pose, z in timed:
        by_gest.setdefault(gest, []).append(pose)
    templates = {gest: np.median(p, axis=0) for gest, p in by_gest.items() if len(p) >= 3}
    if len(templates) >= 2:
        for g, gest, pose, z in timed:
            if gest not in templates:
                continue
            dist = {o: float(pose_change(pose, t)) for o, t in templates.items()}
            nearest = min(dist, key=dist.get)
            if nearest != gest and dist[gest] > min_gap:
                labels[z & (labels == gest)] = IGNORE
                wrong.append((gest, nearest))

    # outside the holds you should be neutral: clearly moved hands are left out
    stray = ~zone_all & visible & (d > neutral_lvl + CAM_STRAY_FRAC * min_gap)
    labels[stray & (cue == NEUTRAL)] = IGNORE
    n_stray = int(np.sum(np.diff(np.concatenate([[0], stray.astype(int)])) == 1))

    name = lambda gi: GESTURES[gi].name  # noqa: E731
    rep = [f"camera: hand visible {visible.mean() * 100:.0f}% of the time"]
    if delays:
        rep.append(f"camera: {len(delays)}/{len(holds)} holds timed by the camera - you started "
                   f"{np.median(delays):.0f} ms after the cue (median), held {np.median(lengths):.0f} ms")
    if flat:
        rep.append(f"camera: can't see a difference from neutral for {len(flat)} holds "
                   f"({', '.join(sorted({name(x) for x in flat}))}) - cue timing used")
    if missed:
        rep.append(f"camera: {len(missed)} holds never showed on camera "
                   f"({', '.join(sorted({name(x) for x in missed}))}) - left out of training")
    if blind:
        rep.append(f"camera: {len(blind)} holds out of view - cue timing used")
    for gest, other in wrong:
        rep.append(f"camera: one {name(gest)} looked like {name(other)} - left out of training")
    if n_stray:
        rep.append(f"camera: {n_stray} moments where the hand moved while it should be neutral - left out")
    rep.append(f"camera: changed {np.mean(labels != cue) * 100:.0f}% of the labels")
    return labels, rep

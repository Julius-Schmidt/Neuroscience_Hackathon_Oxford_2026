"""Webcam hand tracking, used ONLY while recording training data (the live demo is Myo-only).

The cue timing in `record` says when you *should* click; the camera sees when you
actually did. MediaPipe tracks the hand and we measure, per frame (3D world coordinates,
so it doesn't depend on distance to the camera or hand rotation):

    --click fist:   ratio = mean over the 4 fingers of |fingertip - wrist| / |knuckle - wrist|
                    (~2 with an open hand, ~1 with a fist; LOWER = clicking)
    --click index:  angle (radians) between the index finger and the middle + ring fingers
                    (small when resting on the table, big when the index is lifted; HIGHER = clicking)

After the recording, `vision_labels` uses that to move each click label to when the
hand really did it. It only ever makes labels more cautious: wherever camera and
cue disagree in a way it can't resolve, the samples are left out of training.
"""

import threading
import time
from pathlib import Path

import numpy as np

from . import config as C
from .record import FIST, IGNORE, REST

MODEL_FILE = Path(__file__).resolve().parents[1] / "assets" / "hand_landmarker.task"   # shared with gesture_mapper

WRIST, MCP, TIP = 0, (5, 9, 13, 17), (8, 12, 16, 20)
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def closure_ratio(pts) -> float:
    """pts: 21 (x, y, z) points. ~2 = open hand, ~1 = fist."""
    p = np.asarray(pts, dtype=np.float64)
    tip = np.linalg.norm(p[list(TIP)] - p[WRIST], axis=1)
    mcp = np.linalg.norm(p[list(MCP)] - p[WRIST], axis=1)
    return float(np.mean(tip / np.maximum(mcp, 1e-6)))


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


def index_lift_angle(pts) -> float:
    """pts: 21 (x, y, z) points. Angle between the index finger (knuckle -> tip) and the
    middle + ring fingers. ~0 with all fingers resting, grows as the index lifts off the table."""
    p = np.asarray(pts, dtype=np.float64)
    index = _unit(p[8] - p[5])
    others = _unit(_unit(p[12] - p[9]) + _unit(p[16] - p[13]))
    return float(np.arccos(np.clip(index @ others, -1.0, 1.0)))


# per click gesture: camera measure, whether a LOWER value means "clicking", minimum
# rest-vs-click difference to trust the camera, and words for messages
METRIC = {
    "fist": {"fn": closure_ratio, "low_is_active": True, "rest": "open hand", "active": "fist"},
    "index": {"fn": index_lift_angle, "low_is_active": False, "rest": "index down", "active": "index up"},
}


# --------------------------------------------------------------------------- labels

def _hysteresis(closed_amount, on, off):
    """Per camera frame: True = closed, False = open, None = no hand. Hysteresis between on/off."""
    state, out = None, []
    for c in closed_amount:
        if np.isnan(c):
            state = None
        elif state is None:
            state = bool(c >= (on + off) / 2)
        elif state and c <= off:
            state = False
        elif not state and c >= on:
            state = True
        out.append(state)
    return out


def vision_labels(emg_t, cue, seg, cam_t, cam_ratio, lag=None, margin=None, early=None, spill=None,
                  click="fist"):
    """Refine cue labels with the camera.

    emg_t:   arrival time (s) of every EMG sample      cue: cue label per sample (REST/FIST/IGNORE)
    seg:     cue segment id per sample                 cam_t, cam_ratio: per camera frame (nan = no hand)
    click:   which gesture cam_ratio measures (see METRIC)
    Returns (labels, report_lines).
    """
    m = METRIC[click]
    min_gap = C.VISION_MIN_GAP if click == "fist" else C.VISION_MIN_GAP_INDEX
    sign = -1.0 if m["low_is_active"] else 1.0  # work on "higher = clicking" internally
    lag = C.VISION_LAG_S if lag is None else lag
    margin = C.VISION_MARGIN_S if margin is None else margin
    early = C.VISION_EARLY_S if early is None else early
    spill = C.VISION_SPILL_S if spill is None else spill
    emg_t, cue, seg = np.asarray(emg_t, float), np.asarray(cue), np.asarray(seg)
    cam_t, cam_ratio = np.asarray(cam_t, float) - lag, sign * np.asarray(cam_ratio, float)
    labels = cue.copy()
    if len(cam_t) < 10 or np.isfinite(cam_ratio).sum() < 10:
        return labels, ["camera: no hand seen - using the cue timing only"]

    # each EMG sample -> latest camera frame before it (if recent enough)
    k = np.searchsorted(cam_t, emg_t, side="right") - 1
    fresh = (k >= 0) & (emg_t - cam_t[np.clip(k, 0, None)] < C.VISION_MAX_GAP_S)
    ratio = np.where(fresh, cam_ratio[np.clip(k, 0, None)], np.nan)
    visible = np.isfinite(ratio)

    fist_segs = sorted({int(g) for g in np.unique(seg[cue == FIST])})
    zone = np.zeros(len(cue), bool)  # around cued fists (reaction time + late release)
    zones = []
    for g in fist_segs:
        idx = np.flatnonzero(seg == g)
        t0, t1 = emg_t[idx[0]], emg_t[idx[-1]]
        z = (emg_t >= t0 - early) & (emg_t <= t1 + spill)
        zone |= z
        zones.append((g, t0, z))

    # calibrate resting / clicking hand shape from this recording
    rest_r = ratio[visible & (cue == REST) & ~zone]
    fist_r = ratio[visible & (cue == FIST)]
    if len(rest_r) < 50 or len(fist_r) < 50:
        return labels, ["camera: hand not visible enough during the recording - using the cue timing only"]
    rest_lvl, fist_lvl = np.median(rest_r), np.percentile(fist_r, 75)
    shown = f"{m['rest']} {sign * rest_lvl:.2f}, {m['active']} {sign * fist_lvl:.2f}"
    if fist_lvl - rest_lvl < min_gap:
        return labels, [f"camera: {m['rest']} and {m['active']} look too similar ({shown}) "
                        "- using the cue timing only. Is the hand clearly in view?"]

    # clicking or not per camera frame, then per EMG sample
    amount = (cam_ratio - rest_lvl) / (fist_lvl - rest_lvl)  # 0 = resting, 1 = clicking
    states = _hysteresis(amount, on=C.VISION_ON, off=C.VISION_OFF)
    st = np.array([np.nan if s is None else float(s) for s in states])
    closed = np.where(fresh, st[np.clip(k, 0, None)], np.nan)
    visible = np.isfinite(closed)
    closed = visible & (closed == 1.0)

    # don't trust samples right around a visual open<->close switch
    switch_t = np.array([cam_t[i] for i in range(1, len(states))
                         if states[i] is not None and states[i - 1] is not None and states[i] != states[i - 1]])
    near_switch = np.zeros(len(cue), bool)
    if len(switch_t):
        j = np.searchsorted(switch_t, emg_t)
        d_prev = np.abs(emg_t - switch_t[np.clip(j - 1, 0, None)])
        d_next = np.abs(switch_t[np.clip(j, None, len(switch_t) - 1)] - emg_t)
        near_switch = np.minimum(d_prev, d_next) < margin

    vis_label = np.where(closed, FIST, REST)
    vis_label[near_switch] = IGNORE

    # around cued fists: the camera decides the timing
    delays, holds, missed, blind = [], [], 0, 0
    for g, t0, z in zones:
        zc = z & visible & closed
        if (z & visible).sum() < 0.3 * z.sum():
            blind += 1                       # hand mostly out of view: keep cue labels
            continue
        if not zc.any():
            missed += 1                      # camera never saw it close: don't guess
            labels[z & (cue == FIST)] = IGNORE
            labels[z & visible & (cue != FIST)] = vis_label[z & visible & (cue != FIST)]
            continue
        labels[z] = np.where(visible[z], vis_label[z], IGNORE)
        tc = emg_t[zc]
        delays.append((tc[0] - t0) * 1000)
        holds.append((tc[-1] - tc[0]) * 1000)

    # everywhere else you were told to relax: a fist the camera sees there is left out
    stray = ~zone & visible & (closed | near_switch)
    labels[stray & (cue != IGNORE)] = IGNORE
    n_stray = int(np.sum(np.diff(np.concatenate([[0], (~zone & closed).astype(int)])) == 1))

    word = "fist" if click == "fist" else "index lift"
    rep = [f"camera: hand visible {visible.mean() * 100:.0f}% of the time ({shown})"]
    if delays:
        rep.append(f"camera: {len(delays)}/{len(zones)} {word}s timed by the camera - you did it "
                   f"{np.median(delays):.0f} ms after the cue (median), held {np.median(holds):.0f} ms")
    if missed:
        rep.append(f"camera: {missed} cued {word}s never showed on camera - left out of training")
    if blind:
        rep.append(f"camera: {blind} cued {word}s out of view - using the cue timing for them")
    if n_stray:
        rep.append(f"camera: {n_stray} {word}-like moments during relax phases - left out of training")
    changed = np.mean(labels != cue) * 100
    rep.append(f"camera: changed {changed:.0f}% of the labels "
               f"({int((labels == FIST).sum())} {word} samples, cue had {int((cue == FIST).sum())})")
    return labels, rep


# --------------------------------------------------------------------------- camera

def list_cameras():
    """[(index, name)] of the connected cameras (needs cv2-enumerate-cameras), else []."""
    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
        return [(c.index, c.name) for c in enumerate_cameras(cv2.CAP_AVFOUNDATION)]
    except Exception:
        return []


def pick_camera_index(want=None) -> int:
    """Which camera to open. `want` (or config VISION_CAMERA if None): a number = that index,
    text = first camera whose name contains it (e.g. "web camera"), empty = built-in camera.
    Picking by name matters: indices shift when an iPhone shows up as Continuity Camera."""
    want = C.VISION_CAMERA if want is None else want
    want = str(want).strip()
    if want.isdigit():
        return int(want)
    cams = list_cameras()
    if want:
        for idx, name in cams:
            if want.lower() in name.lower():
                return idx
        names = ", ".join(f"{i}: {n}" for i, n in cams) or "none found"
        raise SystemExit(f"no camera matching '{want}' (cameras: {names})")
    for idx, name in cams:
        if "facetime" in name.lower() or "built-in" in name.lower():
            return idx
    for idx, name in cams:
        if not any(k in name.lower() for k in ("iphone", "ipad", "continuity")):
            return idx
    return cams[0][0] if cams else 0


class HandCamera:
    """Background thread: webcam -> MediaPipe -> click measure (METRIC[click]) of the `side` hand per frame.

    show() must be called from the main thread (macOS only allows windows there).
    """

    def __init__(self, side, t_ref, index=None, swap_hands=False, preview=True, click="fist", keep_points=False):
        try:
            import cv2  # noqa: F401
            import mediapipe  # noqa: F401
        except ImportError:
            raise SystemExit("camera needs: pip install mediapipe opencv-python   (or use --no-camera)")
        self.side, self.t_ref = side, t_ref
        self.click, self.metric = click, METRIC[click]["fn"]
        self.index = pick_camera_index(index)
        cams = dict(list_cameras())
        print(f"camera {self.index}: {cams.get(self.index, '?')}  (other: python -m myomouse.record --list-cameras)")
        self.swap, self.preview = swap_hands, preview
        self.times, self.ratios = [], []
        self.keep_points = keep_points
        self.points = []            # with keep_points: (21, 3) world landmarks per frame, nan = no hand
        self.last_points = None
        self.last = None            # (frame, landmarks or None, ratio, hands_seen)
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    def start(self, timeout=15.0):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout) or self.error:
            raise SystemExit(f"camera failed: {self.error or 'timeout'} "
                             "(System Settings -> Privacy & Security -> Camera, or use --no-camera)")

    def _run(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        try:
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                raise RuntimeError(f"could not open camera {self.index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
                running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self._ready.set()
            return
        self._ready.set()
        other = {"left": "right", "right": "left"}
        last_ms = -1
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                t = time.perf_counter() - self.t_ref
                if not ok:
                    time.sleep(0.01)
                    continue
                # mirrored like a selfie, which is nicer to watch
                frame = cv2.flip(frame, 1)
                ms = max(int(t * 1000), last_ms + 1)
                last_ms = ms
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = landmarker.detect_for_video(img, ms)
                ratio, lms, seen = np.nan, None, []
                pts = np.full((21, 3), np.nan, dtype=np.float32)
                for i, hand_lms in enumerate(res.hand_landmarks or []):
                    name = res.handedness[i][0].category_name.lower() if res.handedness else "?"
                    # on the mirrored frame MediaPipe names each hand the opposite way (tested
                    # live); --swap-hands undoes this for a camera that's mirrored already
                    if not self.swap:
                        name = other.get(name, name)
                    seen.append(name)
                    if name == self.side and lms is None and res.hand_world_landmarks:
                        w = res.hand_world_landmarks[i]
                        pts = np.array([(p.x, p.y, p.z) for p in w], dtype=np.float32)
                        ratio = self.metric(pts)
                        lms = hand_lms
                self.times.append(t)
                self.ratios.append(ratio)
                if self.keep_points:
                    self.points.append(pts)
                self.last_points = pts
                self.last = (frame, lms, ratio, seen)
        finally:
            cap.release()
            landmarker.close()

    def status(self):
        """Short text for the terminal line."""
        if self.last is None:
            return "cam: -"
        r = self.last[2]
        if np.isnan(r):
            return f"cam: no {self.side} hand"
        if self.click == "index":
            return f"cam: index {'up  ' if r > 0.35 else 'down'} {np.degrees(r):3.0f} deg"
        return f"cam: {'closed' if r < 1.45 else 'open  '} {r:4.2f}"

    def meter(self, ratio):
        """(0..1 bar, text) for the bottom of the window. Rough, just for watching."""
        if self.click == "index":
            return np.clip(ratio / 0.8, 0, 1), f"index lifted {np.degrees(ratio):.0f} deg"
        return np.clip(2.0 - ratio, 0, 1), f"hand closed {ratio:.2f}"

    def show(self, ui=None, wait_ms=1):
        """Draw the latest frame plus instructions (main thread only). Returns the key pressed, or -1.

        ui: dict with any of
            title, lines  - instruction panel at the top
            cue, color    - big banner in the middle ("red" = squeeze, "green" = relax)
            sub           - smaller line under the banner (countdowns)
            progress      - 0..1 bar under the banner
            footer        - hint at the bottom, e.g. "SPACE = start"
            image         - picture (BGR array) shown bottom right, e.g. what the gesture looks like
            image_label   - caption above that picture
        """
        if not self.preview or self.last is None:
            time.sleep(wait_ms / 1000)
            return -1
        import cv2

        ui = ui or {}
        frame, lms, ratio, seen = self.last
        h0, w0 = frame.shape[:2]
        scale = 1.5
        img = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        h, w = img.shape[:2]
        font, white = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255)

        def shade(x0, y0, x1, y1, color=(0, 0, 0), alpha=0.6):
            roi = img[max(0, y0):y1, max(0, x0):x1]
            roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

        def centered(text, y, size, thick, color=white):
            tw = cv2.getTextSize(text, font, size, thick)[0][0]
            cv2.putText(img, text, ((w - tw) // 2, y), font, size, color, thick, cv2.LINE_AA)

        # hand skeleton + how closed it is
        if lms is not None:
            pts = [(int(p.x * w), int(p.y * h)) for p in lms]
            for a, b in HAND_LINES:
                cv2.line(img, pts[a], pts[b], (80, 220, 80), 3, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(img, pt, 4, white, -1, cv2.LINE_AA)
            shade(0, h - 50, w, h)
            closed, what = self.meter(ratio)
            cv2.rectangle(img, (15, h - 35), (15 + int(250 * closed), h - 15), (60, 60, 230), -1)
            cv2.rectangle(img, (15, h - 35), (265, h - 15), white, 1)
            cv2.putText(img, what, (280, h - 18), font, 0.6, white, 1, cv2.LINE_AA)
        else:
            shade(0, h - 50, w, h, (0, 0, 120), 0.7)
            msg = f"{self.side.upper()} hand not found" + (f"  (seeing: {', '.join(seen)})" if seen else "")
            cv2.putText(img, msg, (15, h - 18), font, 0.7, white, 2, cv2.LINE_AA)
        if ui.get("footer"):
            fw = cv2.getTextSize(ui["footer"], font, 0.7, 2)[0][0]
            cv2.putText(img, ui["footer"], (w - fw - 15, h - 18), font, 0.7, (120, 255, 255), 2, cv2.LINE_AA)

        # instruction panel, wrapped to the window width
        lines = []
        for raw in ui.get("lines", []):
            words, cur = raw.split(), ""
            for word in words:
                nxt = f"{cur} {word}".strip()
                if cv2.getTextSize(nxt, font, 0.7, 2)[0][0] > w - 40 and cur:
                    lines.append(cur)
                    cur = word
                else:
                    cur = nxt
            lines.append(cur)
        panel_h = 0
        if ui.get("title") or lines:
            panel_h = 20 + (45 if ui.get("title") else 0) + 32 * len(lines)
            shade(0, 0, w, panel_h, alpha=0.7)
            y = 15
            if ui.get("title"):
                cv2.putText(img, ui["title"], (20, y + 30), font, 1.0, white, 2, cv2.LINE_AA)
                y += 45
            for line in lines:
                cv2.putText(img, line, (20, y + 25), font, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
                y += 32

        # big cue banner, right under the panel so the hand stays visible
        if ui.get("cue"):
            col = {"red": (40, 40, 210), "green": (60, 160, 40)}.get(ui.get("color"), (60, 60, 60))
            y0 = panel_h
            shade(0, y0, w, y0 + 130, col, 0.8)
            centered(ui["cue"], y0 + 65, 1.8, 4)
            if ui.get("sub"):
                centered(ui["sub"], y0 + 108, 0.9, 2)
            if ui.get("progress") is not None:
                pw = int((w - 80) * float(np.clip(ui["progress"], 0, 1)))
                cv2.rectangle(img, (40, y0 + 120), (40 + pw, y0 + 126), white, -1)

        # example picture, bottom right above the status bar
        pic = ui.get("image")
        if pic is not None:
            top_free = panel_h + (130 if ui.get("cue") else 0) + 10
            bottom = h - 60
            pw = int(w * 0.4)
            ph = int(pic.shape[0] * pw / pic.shape[1])
            cap_h = 34 if ui.get("image_label") else 0
            if ph + cap_h > bottom - top_free:  # tall picture: fit it into the free height
                ph = max(40, bottom - top_free - cap_h)
                pw = int(pic.shape[1] * ph / pic.shape[0])
            x0, y1 = w - pw - 15, bottom
            y0 = y1 - ph
            img[y0:y1, x0:x0 + pw] = cv2.resize(pic, (pw, ph), interpolation=cv2.INTER_AREA)
            cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + pw + 1, y1 + 1), white, 2)
            if cap_h:
                shade(x0 - 2, y0 - cap_h, x0 + pw + 2, y0 - 2, alpha=0.75)
                cv2.putText(img, ui["image_label"], (x0 + 8, y0 - 10), font, 0.75, white, 2, cv2.LINE_AA)

        cv2.imshow("myomouse camera", img)
        return cv2.waitKey(wait_ms)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self.preview:
            try:
                import cv2
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:  # noqa: BLE001
                pass

"""
Joystick debug visualiser — serves live annotated frames via MJPEG.

    python src/cv/joystick_debug.py
    → open http://localhost:5555 in browser

Per hand shows:
  - MediaPipe hand skeleton
  - Wrist dot
  - Calibration neutral (crosshair)
  - Deadzone ring (inner) + max-range ring (outer)
  - Vector arrow from neutral to wrist
  - Text: offset, scaled output, dist, frac, power-curved frac
  - Sensitivity power-bar

Close a fist to anchor that stick where your hand currently is; it stays
anchored across releases until an open palm explicitly lets go (same as
run.py). Press 'q' to stop.
"""

import sys
import os
import time
import logging
import threading
import cv2
import mediapipe as mp
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joystick import HandJoystick, DEADZONE, MAX_RANGE, CURVE
from camera import index_for_name
from recognizer import recognize
from keypress import KeyListener

# joystick.py logs calibrate/release_anchor via logging.getLogger("neuratect");
# without a handler configured those INFO messages go nowhere. run.py sets up
# a file+console handler; standalone here we just need console.
logging.basicConfig(level=logging.INFO, format="%(message)s")

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

PORT = 5555

# Per-hand colors (BGR)
COLORS = {
    "left":    (50,  160, 255),   # orange
    "right":   (50,  220,  80),   # green
    "unknown": (200, 200, 200),
}

# MediaPipe hand skeleton connection pairs
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Global JPEG frame buffer ─────────────────────────────────────────────────
_frame_lock = threading.Lock()
_latest_jpeg: bytes = b""


def _set_frame(bgr_frame) -> None:
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with _frame_lock:
            _latest_jpeg = bytes(buf)


def _get_frame() -> bytes:
    with _frame_lock:
        return _latest_jpeg


# ── MJPEG HTTP server ────────────────────────────────────────────────────────
_HTML = b"""<!doctype html>
<html><head><title>Joystick Debug</title>
<style>
  body { background: #111; display: flex; flex-direction: column;
         align-items: center; padding: 20px; font-family: monospace; color: #aaa; }
  img  { image-rendering: pixelated; width: 640px; height: 480px;
         border: 1px solid #333; }
  p    { margin: 8px 0; font-size: 12px; }
</style></head>
<body>
  <img src="/stream">
  <p>Orange = LEFT hand &nbsp;|&nbsp; Green = RIGHT hand</p>
  <p>Inner ring = deadzone &nbsp;|&nbsp; Outer ring = max range &nbsp;|&nbsp; Bar = sensitivity output</p>
  <p>Close a fist to anchor neutral point; open palm lets go</p>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence per-request logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = _get_frame()
                    if jpg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpg + b"\r\n"
                        )
                        self.wfile.flush()
                    time.sleep(0.033)   # ~30 fps cap
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self.end_headers()


def _run_server(port: int) -> None:
    HTTPServer(("localhost", port), _Handler).serve_forever()


# ── Drawing ──────────────────────────────────────────────────────────────────
def _draw_hand_overlay(
    frame,
    h: int, w: int,
    wrist_x: float, wrist_y: float,
    stick: HandJoystick,
    vec,
    hand: str,
    gesture_name: str,
    confidence: float,
) -> None:
    color = COLORS.get(hand, COLORS["unknown"])
    ax = int(wrist_x * w)   # anchor x (middle finger MCP, landmark 9)
    ay = int(wrist_y * h)

    # Anchor dot (middle finger base)
    cv2.circle(frame, (ax, ay), 6, color, -1)
    cv2.circle(frame, (ax, ay), 7, (255, 255, 255), 1)

    # Text block anchored to top (left) or bottom (right)
    ty = 18 if hand != "right" else h - 72

    neutral = stick._neutral
    if neutral is None:
        cv2.putText(frame,
                    f"{hand.upper()}: close fist to anchor",
                    (8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return

    nx, ny = neutral
    n_px = (int(nx * w), int(ny * h))

    # Neutral crosshair
    cs = 9
    cv2.line(frame, (n_px[0] - cs, n_px[1]), (n_px[0] + cs, n_px[1]), color, 1)
    cv2.line(frame, (n_px[0], n_px[1] - cs), (n_px[0], n_px[1] + cs), color, 1)

    dz   = stick._dz
    mr   = stick._max
    mcurve = stick._mouse_curve

    # Deadzone ring (bright)
    dz_r = int(dz * w)
    cv2.circle(frame, n_px, dz_r, color, 1)

    # Max-range ring (dim)
    mr_r = int(mr * w)
    dim = tuple(int(c * 0.4) for c in color)
    cv2.circle(frame, n_px, mr_r, dim, 1)

    # Vector arrow from neutral to anchor
    cv2.arrowedLine(frame, n_px, (ax, ay), color, 2, tipLength=0.25)

    # Compute joystick values using this stick's config
    ox = -(wrist_x - nx)
    oy = wrist_y - ny
    dist  = float(np.hypot(ox, oy))
    in_dz = dist < dz
    frac   = 0.0 if in_dz else min(1.0, (dist - dz) / (mr - dz))
    curved = frac ** mcurve

    # Sensitivity power-bar (sits just outside deadzone ring)
    bar_w = 48
    bar_h = 5
    bx = n_px[0] + dz_r + 4
    by = n_px[1] - bar_h // 2
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
    fill = int(curved * bar_w)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), color, -1)
    cv2.putText(frame, f"{curved:.2f}",
                (bx + bar_w + 3, by + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Text block
    dz_tag = " [DZ]" if in_dz else ""
    lines = [
        f"{hand.upper()} | {gesture_name} ({confidence:.2f}){dz_tag}",
        f"  offset  x={ox:+.3f}  y={oy:+.3f}",
        f"  mouse   x={vec.raw_x:+.3f}  y={vec.raw_y:+.3f}  (curve={mcurve})",
        f"  dist={dist:.3f}  frac={frac:.2f}  curved={curved:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (8, ty + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


# ── Main camera loop ─────────────────────────────────────────────────────────
def main():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # Start MJPEG server in background
    threading.Thread(target=_run_server, args=(PORT,), daemon=True).start()
    print(f"  → http://localhost:{PORT}")
    print("  Close a fist to anchor neutral. Press 'q' to stop.")

    idx = index_for_name()
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    left_stick  = HandJoystick(name="left")
    right_stick = HandJoystick(name="right", deadzone=0.02, mouse_curve=1.0, smooth=0.6)

    try:
        with KeyListener() as keys:
            while not keys.should_quit:
                ok, frame = cap.read()
                if not ok:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                # Draw skeleton for all detected hands
                if result.hand_landmarks:
                    for lms in result.hand_landmarks:
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                        for a, b in _CONNECTIONS:
                            cv2.line(frame, pts[a], pts[b], (70, 70, 70), 1)
                        for p in pts:
                            cv2.circle(frame, p, 2, (140, 140, 140), -1)

                # Per-hand calibration + overlay
                if result.hand_landmarks:
                    for i, lms in enumerate(result.hand_landmarks):
                        hand = "unknown"
                        if result.handedness and i < len(result.handedness):
                            hand = result.handedness[i][0].category_name.lower()

                        anchor = lms[9]  # base of middle finger (MCP)
                        match = recognize(lms)
                        name, conf = match if match else ("none", 1.0)

                        stick = left_stick if hand == "left" else right_stick
                        if name == "open_palm":
                            stick.release_anchor()
                        vec = stick.update(name == "fist", anchor.x, anchor.y)

                        _draw_hand_overlay(
                            frame, h, w,
                            anchor.x, anchor.y,
                            stick, vec, hand, name, conf,
                        )

                # Legend strip at bottom (show both sticks' config)
                def _cfg(s):
                    return f"DZ={s._dz:.3f} MAX={s._max:.2f} mcurve={s._mouse_curve}"
                cv2.putText(
                    frame,
                    f"L: {_cfg(left_stick)}   R: {_cfg(right_stick)}",
                    (4, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA,
                )

                _set_frame(frame)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
        print("Stopped.")


if __name__ == "__main__":
    main()

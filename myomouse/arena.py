"""Practice arena: move an on-screen cursor and hit targets with clicks (fists, or index lifts).

The arena draws its own cursor, so the real mouse is never touched: no Accessibility
permission needed and no stray clicks on your desktop. It uses the same band processes
as `myomouse.run`, so what works here works in the live demo.

    python -m myomouse.arena --user dylan                 # the bands in bands.json, dylan's classifiers
    python -m myomouse.arena --user julius --sides right  # pointer band only (left-click targets only)
    python -m myomouse.arena --check                      # no models yet: cursor only, Space/Enter click
    python -m myomouse.arena --user julius --fake         # no hardware
    python -m myomouse.arena --user julius --move air     # the original 3D pointing + fist
    python -m myomouse.arena --user julius --click index  # experimental: index lift

Targets:  blue = left click (right hand),  orange = right click (left hand).
Keys:  M  control: band (the --move pointer the live demo uses: table / air / slide) /
          relative (angle change moves the cursor) / absolute (angle from origin =
          cursor position) / joystick (angle from origin = cursor SPEED, small movements)
       A  axes: yaw+pitch (arm in the air) / yaw+roll (arm on a table) / roll+pitch
       K  calibrate: spend 10 s moving around your COMFORTABLE range (can be lopsided, e.g.
          easy inwards, hard outwards). Its middle becomes the origin and its edges the
          screen edges, so sensitivity is set for you. Best with M = absolute or joystick.
       V  show/hide the angle map: what K recorded (dots), the fitted comfortable region,
          where each screen corner sits in arm angles, and where your arm is now
       P  claw machine (point scanning, ONE gesture, no arm movement): a line sweeps across,
          fist stops it; a second line sweeps down, fist stops it and clicks there.
          , / .  slower / faster sweep     ; / '  less / more reaction-time compensation
       O  set the rotation origin = how you're holding your arm right now (absolute, joystick)
       Left/Right  x sensitivity   Up/Down  y sensitivity   +/- both   [ ] joystick dead zone
       X/Y flip direction   C re-centre   R reset score   F fullscreen   Esc quit
       Space / Enter = left / right click from the keyboard
"""

import math

import argparse
import queue
import random
import time
from collections import deque

from . import config as C
from .pointer import wrap
from .gestures import WORD, add_style_args
from .run import start_bands
from .users import add_user_arg, require_user

MODES = ["band", "relative", "absolute", "joystick"]
AXES = [("yaw", "pitch"), ("yaw", "roll"), ("roll", "pitch")]
IDX = {"yaw": 0, "pitch": 1, "roll": 2}


class Claw:
    """Point scanning: x line sweeps (ping-pong), fist fixes x; y line sweeps, fist fixes y + clicks.
    Stopping is backed up by `lag` seconds of travel, to undo the fist-detection + reaction delay."""

    def __init__(self):
        self.on = False
        self.sweep_s = 3.0      # seconds for one full pass across the screen
        self.lag = 0.08         # reaction-time compensation; 80 ms felt right in testing (0.18 and 0.30 overshot)
        self.reset()

    def reset(self):
        self.stage = "x"
        self.pos = 0.0          # 0..1 along the current axis
        self.dir = 1
        self.x = self.y = None
        self.pause_until = 0.0

    def update(self, now, dt):
        if not self.on or now < self.pause_until:
            return
        self.pos += self.dir * dt / self.sweep_s
        if self.pos >= 1.0:
            self.pos, self.dir = 2.0 - self.pos, -1
        elif self.pos <= 0.0:
            self.pos, self.dir = -self.pos, 1

    def stop(self, now):
        """A fist. Returns True when it completes a click (second stop)."""
        if now < self.pause_until:
            return False
        back = self.pos - self.dir * self.lag / self.sweep_s
        back = min(max(back, 0.0), 1.0)
        if self.stage == "x":
            self.x, self.stage, self.pos, self.dir = back, "y", 0.0, 1
            self.pause_until = now + 0.35          # a beat so the same fist can't also stop y
            return False
        self.y = back
        return True


class Mapping:
    """Forearm angles (yaw, pitch, roll) -> cursor, computed here so it can be changed live."""

    def __init__(self, y_axis):
        self.mode = "band"                 # band = use the band's own --move pointer ('move' messages)
        self.axes = ("yaw", "roll") if y_axis == "roll" else ("yaw", "pitch")
        self.gx, self.gy = 1.0, 1.0       # multipliers on the base gains below
        self.deadzone = 0.06               # rad (~3.5 deg), joystick only
        self.origin = None
        self.prev = None
        self.sm = None                     # smoothed angles
        self.span = None                   # calibrated comfortable range per angle (rad), or None
        self.calib = None                  # samples while calibrating
        self.calib_until = 0.0
        self.w, self.h = 1280, 800
        self.cal_pts = None                # calibration samples relative to the origin (N x 3)
        self._fit = {}                     # axes -> (W, r): whitening fit of the reachable shape

    def base(self, axis, along_x=True):
        span = self.span[IDX[axis]] if self.span else None
        length = self.w if along_x else self.h
        if self.mode == "joystick":        # px per second per radian beyond the dead zone
            if span:                       # full tilt crosses the screen in ~0.8 s
                return (length / 0.8) / max(span / 2 - self.deadzone, 0.02)
            return 2500.0
        if span:                           # whole comfortable range = whole screen
            return length / span
        return {"yaw": C.GAIN_X, "pitch": C.GAIN_Y, "roll": C.GAIN_ROLL}[axis]

    def start_calibration(self, now, seconds=10.0):
        self.calib, self.calib_until = [], now + seconds

    def calibrating(self, now):
        if self.calib is not None and now >= self.calib_until:
            self.finish_calibration()
        return self.calib is not None

    def finish_calibration(self):
        import numpy as np
        samples, self.calib = self.calib, None
        if len(samples) < 50:
            return False
        ref = samples[0]
        d = np.array([[wrap(v - r0) for v, r0 in zip(smp, ref)] for smp in samples])
        lo, hi = np.percentile(d, 5, axis=0), np.percentile(d, 95, axis=0)
        self.span = tuple(max(float(x), math.radians(4)) for x in hi - lo)
        # absolute mode: centre on the middle of the swept cloud and keep its shape
        c = d.mean(axis=0)
        self.origin = tuple(wrap(r0 + ci) for r0, ci in zip(ref, c))
        self.cal_pts = d - c
        self._fit = {}
        self.gx = self.gy = 1.0
        self.prev = None
        print("K: %d samples. 5-95%% range from start pose (deg): " % len(samples) + ", ".join(
            f"{n} {math.degrees(a):+.0f}..{math.degrees(b):+.0f}" for n, a, b in zip(("yaw", "pitch", "roll"), lo, hi)))
        return True

    def set_origin(self):
        self.origin = self.sm

    def shape_fit(self):
        """Whitening transform for the current axes: the comfortable region (any tilted /
        lopsided ellipse, e.g. outward rotation and roll limiting each other) -> unit circle."""
        import numpy as np
        if self.axes not in self._fit:
            pts = self.cal_pts[:, [IDX[self.axes[0]], IDX[self.axes[1]]]]
            lam, V = np.linalg.eigh(np.cov(pts.T) + np.eye(2) * math.radians(1) ** 2)
            W = V @ np.diag(1 / np.sqrt(lam)) @ V.T
            r = float(np.percentile(np.linalg.norm(pts @ W.T, axis=1), 90))
            self._fit[self.axes] = (W, max(r, 1e-6))
        return self._fit[self.axes]

    def update(self, ang, dt, w, h):
        """ang = (yaw, pitch, roll). Returns ('delta', dx, dy) or ('abs', x, y) or None."""
        a = C.SMOOTHING
        self.sm = ang if self.sm is None else tuple(
            s + (1 - a) * wrap(v - s) for s, v in zip(self.sm, ang))
        if self.origin is None:
            self.origin = self.sm
        self.w, self.h = w, h
        if self.calib is not None:
            self.calib.append(self.sm)
            return None
        if self.mode == "band":
            return None
        ax, ay = IDX[self.axes[0]], IDX[self.axes[1]]
        sx = -1.0  # turning left / raising / rolling one way moves the cursor left / up
        if self.mode == "relative":
            if self.prev is None:
                self.prev = self.sm
                return None
            dx = wrap(self.sm[ax] - self.prev[ax])
            dy = wrap(self.sm[ay] - self.prev[ay])
            # hysteresis, not discard: slow movements accumulate until they pass the dead zone
            prev = list(self.prev)
            if abs(dx) < C.DEADZONE_RAD:
                dx = 0.0
            else:
                prev[ax] = self.sm[ax]
            if abs(dy) < C.DEADZONE_RAD:
                dy = 0.0
            else:
                prev[ay] = self.sm[ay]
            self.prev = tuple(prev)
            return ("delta", sx * dx * self.base(self.axes[0]) * self.gx,
                    sx * dy * self.base(self.axes[1], False) * self.gy)
        ex = wrap(self.sm[ax] - self.origin[ax])
        ey = wrap(self.sm[ay] - self.origin[ay])
        self.prev = self.sm
        if self.mode == "absolute" and self.cal_pts is not None:
            # unit circle of comfortable reach -> circle through the screen corners,
            # so every corner sits on the edge of what you can reach comfortably
            W, r = self.shape_fit()
            vx = (W[0, 0] * ex + W[0, 1] * ey) / r * self.gx
            vy = (W[1, 0] * ex + W[1, 1] * ey) / r * self.gy
            return ("abs", w / 2 + sx * vx * (w / 2) * math.sqrt(2),
                    h / 2 + sx * vy * (h / 2) * math.sqrt(2))
        if self.mode == "absolute":
            return ("abs", w / 2 + sx * ex * self.base(self.axes[0]) * self.gx,
                    h / 2 + sx * ey * self.base(self.axes[1], False) * self.gy)
        dz = self.deadzone

        def rate(e):
            return 0.0 if abs(e) < dz else (abs(e) - dz) * (1 if e > 0 else -1)
        return ("delta", sx * rate(ex) * self.base(self.axes[0]) * self.gx * dt,
                sx * rate(ey) * self.base(self.axes[1], False) * self.gy * dt)

BG = (18, 20, 26)
FG = (225, 228, 235)
DIM = (120, 126, 140)
BUTTON_COLOR = {"left": (70, 140, 255), "right": (255, 150, 60)}
SIDE_FOR_BUTTON = {b: s for s, b in C.BUTTON_FOR_SIDE.items()}
HIT, MISS = (80, 210, 120), (230, 80, 80)


class Arena:
    def __init__(self, w, h, buttons):
        self.w, self.h = w, h
        self.buttons = buttons  # which buttons targets can ask for
        self.x, self.y = w / 2, h / 2
        self.gain = 1.0
        self.inv_x = self.inv_y = False
        self.held = set()
        self.freeze_until = 0.0
        self.trail = deque(maxlen=40)
        self.flash = None  # (color, until)
        self.reset_score()
        self.new_target()

    def reset_score(self):
        self.hits = self.misses = 0
        self.times = []

    def new_target(self):
        m = 80
        r = random.choice([28, 40, 55])
        # never spawn right under the cursor
        for _ in range(50):
            tx, ty = random.uniform(m, self.w - m), random.uniform(m + 60, self.h - m - 100)
            if (tx - self.x) ** 2 + (ty - self.y) ** 2 > (4 * r) ** 2:
                break
        self.target = (tx, ty, r, random.choice(self.buttons))
        self.t_target = time.perf_counter()

    def resize(self, w, h):
        self.w, self.h = w, h
        self.x, self.y = min(self.x, w - 1), min(self.y, h - 1)
        self.new_target()

    def move(self, dx, dy, now):
        if now < self.freeze_until:
            return
        dx *= self.gain * (-1 if self.inv_x else 1)
        dy *= self.gain * (-1 if self.inv_y else 1)
        self.x = min(max(self.x + dx, 0), self.w - 1)
        self.y = min(max(self.y + dy, 0), self.h - 1)
        self.trail.append((self.x, self.y))

    def place(self, x, y, now):
        if now < self.freeze_until:
            return
        self.x = min(max(x if not self.inv_x else self.w - x, 0), self.w - 1)
        self.y = min(max(y if not self.inv_y else self.h - y, 0), self.h - 1)
        self.trail.append((self.x, self.y))

    def button(self, name, down, now):
        self.freeze_until = now + C.FREEZE_MS / 1000
        if not down:
            self.held.discard(name)
            return
        self.held.add(name)
        tx, ty, r, want = self.target
        inside = (self.x - tx) ** 2 + (self.y - ty) ** 2 <= r * r
        if inside and name == want:
            self.hits += 1
            self.times.append(now - self.t_target)
            self.flash = (HIT, now + 0.25)
            self.new_target()
        else:
            self.misses += 1
            self.flash = (MISS, now + 0.25)


def main():
    ap = argparse.ArgumentParser()
    add_user_arg(ap, required=False)
    add_style_args(ap)
    ap.add_argument("--mode", choices=["lda", "threshold"], default="lda")
    ap.add_argument("--sides", default=None,
                    help="e.g. 'right' for one band; default = the bands in bands.json")
    ap.add_argument("--fake", action="store_true", help="simulated bands")
    ap.add_argument("--check", action="store_true", help="no classifier: cursor only, click with Space/Enter")
    ap.add_argument("--size", default="1280x800")
    args = ap.parse_args()
    from .device import default_sides
    args.sides = args.sides or default_sides(args.fake)
    if not args.check:
        require_user(ap, args)

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if C.POINTER_SIDE not in sides:
        print(f"note: the pointer band ({C.POINTER_SIDE}) isn't running, so the cursor won't move")
    q, procs = start_bands(args.user, sides, args.mode, args.fake, True, args.check, args.click, args.move, args.axes)
    word = WORD[args.click]
    table = args.move in ("table", "slide")
    gain_x = C.TABLE_GAIN_TURN if table else C.GAIN_X
    status = {s: "connecting..." for s in sides}
    probs = {s: 0.0 for s in sides}

    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame  # imported here so the band processes (spawned) don't load it

    pygame.init()
    w, h = (int(v) for v in args.size.split("x"))
    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    pygame.key.set_repeat(300, 60)       # hold an arrow key to keep changing sensitivity
    show_map = [True]
    claw = Claw()
    if args.check:
        claw.lag = 0.0   # Space has no EMG delay to undo; compensating would make the line jump back
    last_frame = [time.perf_counter()]

    def claw_fist(name, now):
        if claw.stop(now):
            arena.x, arena.y = claw.x * arena.w, claw.y * arena.h
            arena.freeze_until = 0.0
            arena.button(name, True, now)
            arena.button(name, False, now)
            fx, fy = claw.x, claw.y
            claw.reset()
            claw.last = (fx, fy)
            claw.pause_until = now + 0.6
        elif claw.x is not None:   # first stop (a fist during a pause is ignored)
            arena.x = claw.x * arena.w

    def draw_map():
        """Arm angles from the origin, in degrees (+-60 across the box)."""
        import numpy as np
        size, deg = 300, 60.0
        x0, y0 = arena.w - size - 16, arena.h - size - 50
        box = pygame.Rect(x0, y0, size, size)
        pygame.draw.rect(screen, (24, 26, 34), box)
        pygame.draw.rect(screen, (70, 76, 92), box, 1)
        cx, cy = x0 + size / 2, y0 + size / 2
        pygame.draw.line(screen, (50, 54, 66), (x0, cy), (x0 + size, cy))
        pygame.draw.line(screen, (50, 54, 66), (cx, y0), (cx, y0 + size))
        ax, ay = IDX[mapping.axes[0]], IDX[mapping.axes[1]]

        def pt(ex, ey):  # radians from origin -> box pixels (clamped to the box)
            px = cx + math.degrees(ex) / deg * size / 2
            py = cy - math.degrees(ey) / deg * size / 2
            return (min(max(px, x0), x0 + size), min(max(py, y0), y0 + size))
        if mapping.cal_pts is not None:
            for q in mapping.cal_pts[::4]:
                pygame.draw.circle(screen, (90, 110, 150), pt(q[ax], q[ay]), 1)
            if mapping.mode == "absolute":
                W, rr = mapping.shape_fit()
                Wi = np.linalg.inv(W)
                ring = [pt(*(Wi @ (rr * np.array([math.cos(t), math.sin(t)])))) for t in np.linspace(0, 2 * math.pi, 60)]
                pygame.draw.lines(screen, (120, 200, 140), True, ring, 1)
                # screen corners in arm-angle space (inverse of the absolute mapping)
                for sx_, sy_ in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    v = np.array([-sx_ / math.sqrt(2) / mapping.gx, -sy_ / math.sqrt(2) / mapping.gy])
                    e = Wi @ (rr * v)
                    pygame.draw.circle(screen, (240, 200, 80), pt(*e), 5, 1)
        if mapping.sm is not None and mapping.origin is not None:
            e = [wrap(v - o) for v, o in zip(mapping.sm, mapping.origin)]
            pygame.draw.circle(screen, (255, 255, 255), pt(e[ax], e[ay]), 5)
        text(f"x = {mapping.axes[0]}   y = {mapping.axes[1]}   (+-{deg:.0f} deg)  "
             "blue: K sweep  green: fit  yellow: screen corners", (x0 - 160, y0 - 22), DIM)
    pygame.display.set_caption("MyoMouse arena")
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 18)
    big = pygame.font.SysFont("Menlo,Consolas,monospace", 26, bold=True)
    clock = pygame.time.Clock()

    # with --check (or one band) only offer targets for buttons that can actually be clicked
    buttons = ["left", "right"] if (args.check and len(sides) > 1) else [C.BUTTON_FOR_SIDE[s] for s in sides]
    mapping = Mapping("roll" if table else "pitch")

    def scale(fx, fy):
        if mapping.mode == "band":          # the band's pointer: one overall gain, like before
            arena.gain *= fx * fy
        else:
            mapping.gx *= fx
            mapping.gy *= fy
    last_ori = [None]
    arena = Arena(w, h, buttons)

    def text(s, pos, color=FG, f=font):
        screen.blit(f.render(s, True, color), pos)

    running = True
    try:
        while running:
            now = time.perf_counter()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    arena.resize(ev.w, ev.h)
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        scale(1.15, 1.15) if mapping.mode != "band" else scale(1.15, 1.0)
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        scale(1 / 1.15, 1 / 1.15) if mapping.mode != "band" else scale(1 / 1.15, 1.0)
                    elif k == pygame.K_RIGHT:
                        scale(1.15, 1.0)
                    elif k == pygame.K_LEFT:
                        scale(1 / 1.15, 1.0)
                    elif k == pygame.K_UP:
                        scale(1.0, 1.15)
                    elif k == pygame.K_DOWN:
                        scale(1.0, 1 / 1.15)
                    elif k == pygame.K_m:
                        mapping.mode = MODES[(MODES.index(mapping.mode) + 1) % len(MODES)]
                        mapping.gx = mapping.gy = 1.0
                        mapping.set_origin()
                        mapping.prev = None
                    elif k == pygame.K_a:
                        mapping.axes = AXES[(AXES.index(mapping.axes) + 1) % len(AXES)] \
                            if mapping.axes in AXES else AXES[0]
                        mapping.set_origin()
                        mapping.prev = None
                    elif k == pygame.K_v:
                        show_map[0] = not show_map[0]
                    elif k == pygame.K_p:
                        claw.on = not claw.on
                        claw.reset()
                        show_map[0] = not claw.on and show_map[0]
                    elif k == pygame.K_COMMA:
                        claw.sweep_s *= 1.15
                    elif k == pygame.K_PERIOD:
                        claw.sweep_s /= 1.15
                    elif k == pygame.K_SEMICOLON:
                        claw.lag = max(0.0, claw.lag - 0.02)
                    elif k == pygame.K_QUOTE:
                        claw.lag += 0.02
                    elif k == pygame.K_k:
                        mapping.start_calibration(now)
                    elif k == pygame.K_o:
                        mapping.set_origin()
                        arena.x, arena.y = arena.w / 2, arena.h / 2
                    elif k == pygame.K_LEFTBRACKET:
                        mapping.deadzone = max(0.0, mapping.deadzone - 0.01)
                    elif k == pygame.K_RIGHTBRACKET:
                        mapping.deadzone += 0.01
                    elif k == pygame.K_x:
                        arena.inv_x = not arena.inv_x
                    elif k == pygame.K_y:
                        arena.inv_y = not arena.inv_y
                    elif k == pygame.K_c:
                        arena.x, arena.y = arena.w / 2, arena.h / 2
                        arena.trail.clear()
                    elif k == pygame.K_r:
                        arena.reset_score()
                        arena.new_target()
                    elif k == pygame.K_f:
                        pygame.display.toggle_fullscreen()
                        arena.resize(*screen.get_size())
                    elif k == pygame.K_SPACE and claw.on:
                        claw_fist("left", now)
                    elif k == pygame.K_SPACE:
                        arena.button("left", True, now)
                    elif k == pygame.K_RETURN:
                        arena.button("right", True, now)
                elif ev.type == pygame.KEYUP:
                    if ev.key == pygame.K_SPACE and not claw.on:
                        arena.button("left", False, now)
                    elif ev.key == pygame.K_RETURN:
                        arena.button("right", False, now)

            # drain everything the bands sent since the last frame
            while True:
                try:
                    kind, a, b, _t = q.get_nowait()
                except queue.Empty:
                    break
                if kind == "ori":
                    dt = 0.02 if last_ori[0] is None else min(max(_t - last_ori[0], 0.0), 0.1)
                    last_ori[0] = _t
                    out = mapping.update(a, dt, arena.w, arena.h)
                    if claw.on or mapping.mode == "band":
                        pass
                    elif out and out[0] == "delta":
                        arena.move(out[1], out[2], now)
                    elif out:
                        arena.place(out[1], out[2], now)
                elif kind == "move":
                    if mapping.mode == "band" and not claw.on:
                        arena.move(a, b, now)
                elif kind == "btn" and claw.on:
                    if b == "press":
                        claw_fist(C.BUTTON_FOR_SIDE[a], now)
                elif kind == "btn":
                    arena.button(C.BUTTON_FOR_SIDE[a], b == "press", now)
                elif kind == "p":
                    probs[a] = b
                elif kind == "ready":
                    status[a] = "ready"
            for s, p in zip(sides, procs):
                if not p.is_alive():
                    status[s] = "stopped (see terminal)"

            # --- draw ---
            screen.fill(BG)
            if arena.flash and now < arena.flash[1]:
                pygame.draw.rect(screen, arena.flash[0], screen.get_rect(), 6)

            tx, ty, r, want = arena.target
            col = BUTTON_COLOR[want]
            pygame.draw.circle(screen, col, (tx, ty), r, 4)
            pygame.draw.circle(screen, col, (tx, ty), 5)
            text(f"{want} click ({SIDE_FOR_BUTTON[want]} {word})", (tx - r, ty + r + 6), col)

            if len(arena.trail) > 1:
                pygame.draw.lines(screen, (70, 76, 92), False, list(arena.trail), 2)
            frozen = now < arena.freeze_until
            ccol = DIM if frozen and not arena.held else FG
            for b in arena.held:
                ccol = BUTTON_COLOR[b]
            claw.update(now, min(now - last_frame[0], 0.1))
            last_frame[0] = now
            if claw.on:
                gold, dim_gold = (255, 200, 60), (120, 95, 40)
                if claw.stage == "x":
                    lx = claw.pos * arena.w
                    pygame.draw.line(screen, gold, (lx, 0), (lx, arena.h), 3)
                    pygame.draw.polygon(screen, gold, [(lx - 14, 0), (lx + 14, 0), (lx, 22)])
                    arena.x = lx
                else:
                    lx, ly = claw.x * arena.w, claw.pos * arena.h
                    pygame.draw.line(screen, dim_gold, (lx, 0), (lx, arena.h), 2)
                    pygame.draw.line(screen, gold, (0, ly), (arena.w, ly), 3)
                    # the claw descending
                    pygame.draw.line(screen, gold, (lx - 12, ly - 4), (lx - 4, ly + 12), 3)
                    pygame.draw.line(screen, gold, (lx + 12, ly - 4), (lx + 4, ly + 12), 3)
                    arena.x, arena.y = lx, ly
                text(f"CLAW: fist to stop the {'across' if claw.stage == 'x' else 'down'} sweep   "
                     f"pass {claw.sweep_s:.1f}s (, .)   compensation {claw.lag * 1000:.0f} ms (; ')   P to exit",
                     (16, 96), gold)
            cx, cy = arena.x, arena.y
            if show_map[0]:
                draw_map()
            pygame.draw.circle(screen, ccol, (cx, cy), 10, 0 if arena.held else 2)
            pygame.draw.line(screen, ccol, (cx - 18, cy), (cx - 12, cy), 2)
            pygame.draw.line(screen, ccol, (cx + 12, cy), (cx + 18, cy), 2)
            pygame.draw.line(screen, ccol, (cx, cy - 18), (cx, cy - 12), 2)
            pygame.draw.line(screen, ccol, (cx, cy + 12), (cx, cy + 18), 2)

            avg = sum(arena.times) / len(arena.times) if arena.times else 0
            last = arena.times[-1] if arena.times else 0
            text(f"hits {arena.hits}   misses {arena.misses}   avg {avg:.2f} s   last {last:.2f} s",
                 (16, 12), FG, big)
            if mapping.mode == "band":
                text(f"[M] band: --move {args.move}, axes {args.axes} | gain x{arena.gain:.2f} "
                     f"(= {gain_x * arena.gain:.0f} px/rad, +/- or arrows)   "
                     f"flip x: {'on' if arena.inv_x else 'off'}  y: {'on' if arena.inv_y else 'off'}",
                     (16, 48), DIM)
            unit = "px/s per rad" if mapping.mode == "joystick" else "px/rad"
            gx_abs = mapping.base(mapping.axes[0]) * mapping.gx
            gy_abs = mapping.base(mapping.axes[1], False) * mapping.gy
            if mapping.mode != "band": text(f"[M] {mapping.mode}   [A] x={mapping.axes[0]} y={mapping.axes[1]}   "
                 f"x {gx_abs:.0f} / y {gy_abs:.0f} {unit} (arrows)"
                 + (f"   dead zone {math.degrees(mapping.deadzone):.1f} deg ([ ])" if mapping.mode == "joystick" else "")
                 + f"   flip x: {'on' if arena.inv_x else 'off'}  y: {'on' if arena.inv_y else 'off'}",
                 (16, 48), DIM)
            if mapping.calibrating(now):
                left = mapping.calib_until - now
                text(f"CALIBRATING {left:4.1f}s: sweep slowly around the edges of what's comfortable "
                     f"(all directions), then relax", (16, 96), (255, 210, 90))
            elif mapping.span:
                text("calibrated range:  " + "   ".join(
                    f"{n} {math.degrees(v):.0f} deg" for n, v in zip(("yaw", "pitch", "roll"), mapping.span))
                    + "   (K to redo)", (16, 96), DIM)
            if mapping.sm is not None and mapping.origin is not None:
                e = [math.degrees(wrap(v - o)) for v, o in zip(mapping.sm, mapping.origin)]
                text(f"from origin [O]:  yaw {e[0]:+5.1f}   pitch {e[1]:+5.1f}   roll {e[2]:+5.1f} deg",
                     (16, 72), DIM)

            y = arena.h - 30 - 26 * len(sides)
            label = "activity" if args.check else f"{args.click} p"
            for s in sides:
                bar_w = 200
                bcol = BUTTON_COLOR[C.BUTTON_FOR_SIDE[s]]
                text(f"{s:5s} {label}", (16, y))
                pygame.draw.rect(screen, (50, 54, 64), (150, y + 3, bar_w, 14))
                pygame.draw.rect(screen, bcol, (150, y + 3, bar_w * probs[s], 14))
                if not args.check:  # press threshold marker
                    px = 150 + bar_w * C.P_ON
                    pygame.draw.line(screen, FG, (px, y), (px, y + 20), 1)
                text(status[s], (150 + bar_w + 12, y), DIM)
                y += 26
            text("M mode  A axes  K calibrate  O origin  arrows/+- sensitivity  [ ] dead zone  X/Y flip  C centre  "
                 "R reset  F fullscreen  Space/Enter click  Esc quit", (16, arena.h - 26), DIM)

            pygame.display.flip()
            clock.tick(120)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        for p in procs:
            p.terminate()
        print(f"hits {arena.hits}, misses {arena.misses}")
        if mapping.span:
            print("calibrated comfortable range (deg): " + ", ".join(
                f"{n} {math.degrees(v):.0f}" for n, v in zip(("yaw", "pitch", "roll"), mapping.span)))
        if mapping.mode == "band":
            if arena.gain != 1.0 or arena.inv_x or arena.inv_y:
                print("To keep these settings, put this in myomouse/config.py:")
                if table:
                    y_gain = ("TABLE_GAIN_SLIDE", C.TABLE_GAIN_SLIDE) if args.move == "slide" else ("TABLE_GAIN_Y", C.TABLE_GAIN_Y)
                    print(f"  TABLE_GAIN_TURN = {C.TABLE_GAIN_TURN * arena.gain:.0f}\n"
                          f"  {y_gain[0]} = {y_gain[1] * arena.gain:.0f}")
                    print(f"  TABLE_INVERT_X = {arena.inv_x != C.TABLE_INVERT_X}\n"
                          f"  TABLE_INVERT_Y = {arena.inv_y != C.TABLE_INVERT_Y}")
                else:
                    print(f"  GAIN_X = {C.GAIN_X * arena.gain:.0f}\n  GAIN_Y = {C.GAIN_Y * arena.gain:.0f}")
                    print(f"  INVERT_X = {arena.inv_x != C.INVERT_X}\n  INVERT_Y = {arena.inv_y != C.INVERT_Y}")
        else:
            print(f"last setting: mode={mapping.mode} axes x={mapping.axes[0]} y={mapping.axes[1]}  "
                  f"x {mapping.base(mapping.axes[0]) * mapping.gx:.0f}  y {mapping.base(mapping.axes[1], False) * mapping.gy:.0f}"
                  f"{' px/s per rad, dead zone %.1f deg' % math.degrees(mapping.deadzone) if mapping.mode == 'joystick' else ' px/rad'}"
                  f"  flip x={arena.inv_x} y={arena.inv_y}")
            if mapping.mode == "relative" and mapping.axes in (("yaw", "pitch"), ("yaw", "roll")):
                print("To use it in the live demo, put this in myomouse/config.py:")
                print(f"  POINTER_Y_AXIS = {mapping.axes[1]!r}")
                print(f"  GAIN_X = {C.GAIN_X * mapping.gx:.0f}")
                if mapping.axes[1] == "roll":
                    print(f"  GAIN_ROLL = {C.GAIN_ROLL * mapping.gy:.0f}\n  INVERT_ROLL = {arena.inv_y != C.INVERT_ROLL}")
                else:
                    print(f"  GAIN_Y = {C.GAIN_Y * mapping.gy:.0f}\n  INVERT_Y = {arena.inv_y != C.INVERT_Y}")
                print(f"  INVERT_X = {arena.inv_x != C.INVERT_X}")
            else:
                print("(absolute / joystick / roll+pitch are arena-only for now: tell Claude which one felt best)")


if __name__ == "__main__":
    main()

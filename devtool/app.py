"""MyoMouse dev tool: live view, per-user calibration, click-target test with CSV logging.

    python -m devtool --user dylan                  # both bands (needs bands.json from myomouse.identify)
    python -m devtool --user dylan --sides right    # one band
    python -m devtool --fake                        # no hardware: hold L / R to squeeze

Modes:   1 live view    2 calibrate    3 click-target test    4 fatigue protocol (raw EMG for replay)
Keys:    Up/Down threshold    Left/Right refractory    [ ] cursor smoothing
         M cursor source (IMU / real mouse)    C re-centre cursor    G trace gain
         Space start (calibration / test)    Esc stop session or quit
"""

from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from collections import deque

import numpy as np

from myomouse import config as MC

from .ble import BandState, BleManager, FakeManager
from .emg import (BandCal, Cursor, compute_calibration, load_calibration, save_calibration,
                  smooth_point)
from .session import Session

DATA_DIR = MC.ROOT / "data" / "devtool"

BG, PANEL, FG, DIM = (16, 18, 24), (26, 29, 37), (226, 229, 236), (120, 126, 140)
OK, WARN, BAD = (90, 205, 120), (240, 190, 70), (235, 85, 85)
SIDE_COL = {"left": (255, 150, 60), "right": (70, 140, 255)}
CH_COL = [(230, 90, 90), (240, 150, 70), (230, 210, 80), (130, 210, 90),
          (80, 200, 190), (90, 150, 240), (160, 110, 240), (230, 110, 200)]
TRACE_S = 2.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="default", help="calibration + session files are saved per user")
    ap.add_argument("--sides", default=None,
                    help="e.g. 'right' for one band; default = the bands in bands.json")
    ap.add_argument("--fake", action="store_true", help="simulated bands; hold L / R keys to squeeze")
    ap.add_argument("--fake-drop", type=float, default=0.0, help="fake packet loss rate, e.g. 0.01")
    ap.add_argument("--fake-fatigue", type=float, default=0.0,
                    help="fake bands tire: seconds of squeezing to get tired (e.g. 40); 0 = never")
    ap.add_argument("--fatigue-min", type=float, default=8.0, help="fatigue protocol: minutes of cued squeezes")
    ap.add_argument("--fatigue-rest-min", type=float, default=3.0, help="fatigue protocol: minutes of rest (arms still)")
    ap.add_argument("--fatigue-post-min", type=float, default=1.0, help="fatigue protocol: minutes of squeezes after rest")
    ap.add_argument("--cue-interval", type=float, default=3.5, help="fatigue protocol: s between squeeze cues")
    ap.add_argument("--cue-on", type=float, default=0.8, help="fatigue protocol: s the SQUEEZE cue is shown")
    ap.add_argument("--autotest-fatigue", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--threshold", type=float, default=0.35, help="click when activation >= this (rest=0, typical squeeze=1)")
    ap.add_argument("--refractory", type=float, default=250, help="ms between clicks on one band (try 150-400)")
    ap.add_argument("--smoothing", type=float, default=MC.SMOOTHING, help="cursor smoothing 0..0.95")
    ap.add_argument("--cursor", choices=["imu", "mouse"], default=None,
                    help=f"imu = {MC.POINTER_SIDE} band orientation (default), mouse = real mouse (isolates clicks)")
    ap.add_argument("--y-axis", choices=["pitch", "roll"], default=None,
                    help="IMU cursor: pitch = raise/lower the arm; roll = forearm on a table, rotate it")
    ap.add_argument("--pointer", choices=["right", "left"], default=None, help="which band moves the cursor")
    ap.add_argument("--targets", type=int, default=20, help="targets per test session")
    ap.add_argument("--target-timeout", type=float, default=10.0)
    ap.add_argument("--rest-still", type=float, default=20.0, help="s of still rest before targets")
    ap.add_argument("--rest-move", type=float, default=20.0, help="s of moving-but-not-clicking before targets")
    ap.add_argument("--pre-ms", type=float, default=400, help="EMG logged before each click")
    ap.add_argument("--post-ms", type=float, default=200, help="EMG logged after each click")
    ap.add_argument("--cal-rest", type=float, default=5.0)
    ap.add_argument("--cal-reps", type=int, default=4)
    ap.add_argument("--size", default="1400x860")
    ap.add_argument("--seconds", type=float, default=0, help="quit after this long (testing)")
    ap.add_argument("--autotest", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    from myomouse.device import default_sides
    a.sides = a.sides or default_sides(a.fake)
    return a


class App:
    def __init__(self, args):
        self.a = args
        if args.pointer:
            MC.POINTER_SIDE = args.pointer
        y_axis = args.y_axis or MC.POINTER_Y_AXIS
        self.sides = [s.strip() for s in args.sides.split(",") if s.strip()]
        self.cals = load_calibration(DATA_DIR, args.user)
        self.threshold, self.refractory, self.smoothing = args.threshold, args.refractory, args.smoothing
        self.bands = {s: BandState(s, self.cals.get(s, BandCal()), self.threshold, self.refractory)
                      for s in self.sides}
        self.cursor_source = args.cursor or ("mouse" if args.fake or MC.POINTER_SIDE not in self.sides else "imu")
        if y_axis == "roll":
            self.imu = Cursor(MC.GAIN_X, MC.GAIN_ROLL, MC.INVERT_X, MC.INVERT_ROLL, MC.DEADZONE_RAD,
                              self.smoothing, y_axis="roll")
        else:
            self.imu = Cursor(MC.GAIN_X, MC.GAIN_Y, MC.INVERT_X, MC.INVERT_Y, MC.DEADZONE_RAD, self.smoothing)
        self._acc = [0.0, 0.0]
        self._acc_lock = threading.Lock()
        self.freeze_until = 0.0
        if MC.POINTER_SIDE in self.bands:
            self.bands[MC.POINTER_SIDE].on_imu = self._on_imu
        if args.fake:
            self.mgr = FakeManager(self.bands, drop_rate=args.fake_drop, fatigue_tau=args.fake_fatigue).start()
        else:
            from myomouse.device import load_bands
            addrs = load_bands()
            missing = [s for s in self.sides if s not in addrs]
            if missing:
                raise SystemExit(f"bands.json has no {missing}. Run `python -m myomouse.identify`.")
            self.mgr = BleManager(self.bands, addrs, MC.POINTER_SIDE).start()
        self.mode = "live"
        self.trace_gain = 1.0
        self.rate = {s: (0.0, 0.0) for s in self.sides}     # samples/s, packets/s
        self._rate_ref = {s: (time.perf_counter(), 0) for s in self.sides}
        self.click_count = {s: 0 for s in self.sides}
        self.flash = {s: 0.0 for s in self.sides}
        self.msg, self.msg_until = "", 0.0
        # test state
        self.session: Session | None = None
        self.phase = "idle"
        self.summary_text = ""
        self.cal_state = None

    # ------------------------------------------------------------------ cursor
    def _on_imu(self, quat):  # BLE thread
        dx, dy = self.imu.update(quat)
        if time.perf_counter() < self.freeze_until:
            return
        with self._acc_lock:
            self._acc[0] += dx
            self._acc[1] += dy

    def cursor_at(self, t):
        if t is None:
            return self.cx, self.cy
        for tt, x, y in reversed(self.hist):
            if tt <= t:
                return x, y
        return (self.hist[0][1], self.hist[0][2]) if self.hist else (self.cx, self.cy)

    def update_cursor(self, pygame, area):
        now = time.perf_counter()
        x0, y0, w, h = area
        if self.cursor_source == "imu":
            with self._acc_lock:
                dx, dy = self._acc
                self._acc = [0.0, 0.0]
            self.cx = min(max(self.cx + dx, x0), x0 + w - 1)
            self.cy = min(max(self.cy + dy, y0), y0 + h - 1)
        else:
            if now >= self.freeze_until:
                mx, my = pygame.mouse.get_pos()
                self._mouse_s = smooth_point(getattr(self, "_mouse_s", None), (mx, my), self.smoothing)
                self.cx = min(max(self._mouse_s[0], x0), x0 + w - 1)
                self.cy = min(max(self._mouse_s[1], y0), y0 + h - 1)
        self.hist.append((now, self.cx, self.cy))

    # ------------------------------------------------------------------ settings
    def settings(self):
        return {"threshold": round(self.threshold, 3), "refractory_ms": round(self.refractory),
                "smoothing": round(self.smoothing, 3), "cursor_source": self.cursor_source}

    def note(self, text, secs=2.5):
        self.msg, self.msg_until = text, time.perf_counter() + secs

    def change(self, what, delta):
        if self.session is not None:
            self.note("settings are locked during a session (Esc to stop it)")
            return
        if what == "threshold":
            self.threshold = round(min(max(self.threshold + delta, 0.05), 2.0), 3)
            for b in self.bands.values():
                b.det.threshold = self.threshold
        elif what == "refractory":
            self.refractory = min(max(self.refractory + delta, 50), 1000)
            for b in self.bands.values():
                b.det.refractory_ms = self.refractory
        elif what == "smoothing":
            self.smoothing = round(min(max(self.smoothing + delta, 0.0), 0.95), 3)
            self.imu.smoothing = self.smoothing

    # ------------------------------------------------------------------ calibration
    def cal_steps(self):
        steps = [("Relax both arms, hands loose, arms still", "ready", None, 3.0),
                 ("REST - relax both hands, don't move", "rest", None, self.a.cal_rest)]
        for s in self.sides:
            steps.append((f"Next: {s.upper()} hand. Squeeze like a normal click, not max effort", "ready", s, 3.0))
            for k in range(self.a.cal_reps):
                steps.append((f"SQUEEZE {s.upper()} ({k + 1}/{self.a.cal_reps})", "squeeze", s, 1.5))
                steps.append((f"relax {s}", "relax", s, 2.0))
        return steps

    def start_calibration(self):
        self.cal_state = {"steps": self.cal_steps(), "i": 0, "t": time.perf_counter(),
                          "start_n": {s: b.n for s, b in self.bands.items()}, "spans": [], "result": None}

    def tick_calibration(self):
        cs = self.cal_state
        if cs is None or cs["result"] is not None:
            return
        label, kind, side, dur = cs["steps"][cs["i"]]
        if time.perf_counter() - cs["t"] < dur:
            return
        cs["spans"].append((kind, side, cs["start_n"], {s: b.n for s, b in self.bands.items()}))
        cs["i"] += 1
        cs["t"] = time.perf_counter()
        cs["start_n"] = {s: b.n for s, b in self.bands.items()}
        if cs["i"] >= len(cs["steps"]):
            self.finish_calibration()

    def finish_calibration(self):
        cs = self.cal_state
        skip = int(0.3 * 200)  # ignore reaction time at the start of each cue
        out, lines = {}, []
        for s, b in self.bands.items():
            rest = [b.emg[a[s] + skip:z[s]][b.valid[a[s] + skip:z[s]]]
                    for k, _, a, z in cs["spans"] if k == "rest"]
            sq = [b.emg[a[s] + skip:z[s]][b.valid[a[s] + skip:z[s]]]
                  for k, side, a, z in cs["spans"] if k == "squeeze" and side == s]
            try:
                cal = compute_calibration(np.concatenate(rest), np.concatenate(sq))
            except ValueError as e:
                lines.append(f"{s}: FAILED - {e}")
                continue
            out[s] = cal
            b.det.set_cal(cal)
            self.cals[s] = cal
            weak = [i for i, (r, p) in enumerate(zip(cal.rest, cal.peak)) if p < r * 2]
            lines.append(f"{s}: rest {[round(v, 1) for v in cal.rest]}")
            lines.append(f"{' ' * len(s)}  peak {[round(v, 1) for v in cal.peak]}")
            lines.append(f"{' ' * len(s)}  activation at rest p99 = {cal.rest_act_p99}, "
                         f"typical squeeze median = {cal.peak_act_median}"
                         + (f"   weak channels (peak < 2x rest): {weak}" if weak else ""))
            if cal.rest_act_p99 is not None and cal.rest_act_p99 >= self.threshold * 0.7:
                lines.append(f"{' ' * len(s)}  WARNING: rest noise is close to the threshold {self.threshold}")
        if out:
            p = save_calibration(DATA_DIR, self.a.user, out)
            lines.append(f"saved {p}")
        cs["result"] = lines
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------ test session
    def start_session(self, area):
        self.session = Session(DATA_DIR / "sessions", self.a.user, self.bands, self.settings(),
                               self.a.rest_still, self.a.rest_move, self.a.pre_ms, self.a.post_ms)
        self.summary_text = ""
        self.targets_done = 0
        self.target = None
        self.area = area
        self.set_phase("countdown")

    def set_phase(self, name):
        self.phase, self.phase_t = name, time.perf_counter()
        if self.session and name in ("rest_still", "rest_move", "targets"):
            self.session.mark_phase(name)
        if name == "targets":
            self.new_target()

    def new_target(self):
        x0, y0, w, h = self.area
        r = random.choice([30, 42, 56])
        for _ in range(100):
            tx, ty = random.uniform(x0 + 80, x0 + w - 80), random.uniform(y0 + 80, y0 + h - 80)
            if math.hypot(tx - self.cx, ty - self.cy) > max(4 * r, 200):
                break
        self.targets_done_id = getattr(self, "targets_done_id", 0) + 1
        self.target = (tx, ty, r, self.targets_done_id)
        self.target_band = random.choice([s for s in self.sides if self.bands[s].connected] or self.sides)
        self.t_shown, self.t_entered = time.perf_counter(), None

    def tick_session(self):
        if self.session is None:
            return
        el = time.perf_counter() - self.phase_t
        if self.phase == "countdown" and el >= 3:
            self.set_phase("rest_still" if self.a.rest_still > 0 else "rest_move")
        elif self.phase == "rest_still" and el >= self.a.rest_still:
            self.set_phase("rest_move" if self.a.rest_move > 0 else "targets")
        elif self.phase == "rest_move" and el >= self.a.rest_move:
            self.set_phase("targets")
        elif self.phase == "targets":
            tx, ty, r, _ = self.target
            if self.t_entered is None and math.hypot(self.cx - tx, self.cy - ty) <= r:
                self.t_entered = time.perf_counter()
            if time.perf_counter() - self.t_shown > self.a.target_timeout:
                self.session.add_timeout(self.target, self.target_band, self.t_shown, self.t_entered)
                self.next_target()
        self.session.poll()

    def next_target(self):
        self.targets_done += 1
        if self.targets_done >= self.a.targets:
            self.end_session()
        else:
            self.new_target()

    def end_session(self):
        if self.session is None:
            return
        s = self.session
        self.session = None
        self.phase = "done"
        self.target = None
        fatigue = self.phase in FATIGUE_PHASES or self.mode == "fatigue"
        _, text = s.close(self.cals)
        if fatigue:
            text = fatigue_summary(s)
        self.summary_text = text + f"\nsaved {s.csv_path}"
        print("\n" + self.summary_text, flush=True)

    def handle_click(self, side, idx, host_t):
        self.click_count[side] += 1
        self.flash[side] = time.perf_counter() + 0.25
        self.freeze_until = host_t + MC.FREEZE_MS / 1000
        if self.session is not None and self.phase in FATIGUE_PHASES:
            outcome = "false_rest" if self.phase == "recovery" else "cue_click"
            self.session.add_click(side, idx, host_t, self.phase, outcome, (self.cx, self.cy), self.cursor_at)
            return
        if self.session is None or self.phase not in ("rest_still", "rest_move", "targets"):
            return
        cur = self.cursor_at(host_t)
        if self.phase != "targets":
            self.session.add_click(side, idx, host_t, self.phase, "false_rest", cur, self.cursor_at)
            return
        tx, ty, r, _ = self.target
        inside = math.hypot(cur[0] - tx, cur[1] - ty) <= r
        outcome = "hit" if inside and side == self.target_band else (
            "miss_wrong_band" if inside else "miss_outside")
        self.session.add_click(side, idx, host_t, "targets", outcome, cur, self.cursor_at, self.target,
                               self.target_band, self.t_shown, self.t_entered)
        if outcome == "hit":
            self.next_target()

    # ------------------------------------------------------------------ main loop
    def run(self):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.init()
        W, H = (int(v) for v in self.a.size.split("x"))
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        pygame.display.set_caption(f"MyoMouse dev tool - {self.a.user}")
        self.font = pygame.font.SysFont("Consolas,Menlo,DejaVu Sans Mono,monospace", 15)
        self.big = pygame.font.SysFont("Consolas,Menlo,DejaVu Sans Mono,monospace", 30, bold=True)
        self.cx, self.cy = W / 2, H / 2
        self.hist = deque(maxlen=1200)
        clock = pygame.time.Clock()
        t_start = time.perf_counter()
        running = True
        try:
            while running:
                W, H = self.screen.get_size()
                test_area = (0, 60, W, H - 60 - 110)
                for ev in pygame.event.get():
                    running = self.on_event(ev, test_area) and running
                if self.a.seconds and time.perf_counter() - t_start > self.a.seconds:
                    running = False
                if self.a.autotest or self.a.autotest_fatigue:
                    self.autotest(test_area)
                self.update_cursor(pygame, test_area)
                for s, b in self.bands.items():
                    while b.clicks:
                        self.handle_click(*b.clicks.popleft())
                self.update_rates()
                self.tick_calibration()
                self.tick_session()
                self.tick_fatigue()
                self.draw(W, H, test_area)
                pygame.display.flip()
                clock.tick(60)
        except KeyboardInterrupt:
            pass
        finally:
            self.end_session()
            self.mgr.stop()
            pygame.quit()

    def on_event(self, ev, test_area):
        pg = self.pg
        if ev.type == pg.QUIT:
            return False
        if ev.type in (pg.KEYDOWN, pg.KEYUP) and self.a.fake and ev.key in (pg.K_l, pg.K_r):
            side = "left" if ev.key == pg.K_l else "right"
            if side in self.mgr.squeeze:
                self.mgr.squeeze[side] = ev.type == pg.KEYDOWN
            return True
        if ev.type != pg.KEYDOWN:
            return True
        k = ev.key
        if k == pg.K_ESCAPE:
            if self.session is not None:
                self.end_session()
                return True
            if self.cal_state and self.cal_state["result"] is None:
                self.cal_state = None
                return True
            return False
        if k == pg.K_q:
            return False
        if k in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
            if self.session is not None:
                self.note("finish or Esc the session first")
            else:
                self.mode = {pg.K_1: "live", pg.K_2: "calibrate", pg.K_3: "test", pg.K_4: "fatigue"}[k]
                self.cal_state = None
                self.phase = "idle" if self.mode == "test" else self.phase
        elif k == pg.K_SPACE:
            if self.mode == "calibrate" and (self.cal_state is None or self.cal_state["result"] is not None):
                self.start_calibration()
            elif self.mode == "test" and self.session is None:
                self.start_session(test_area)
            elif self.mode == "fatigue" and self.session is None:
                self.start_fatigue()
        elif k == pg.K_UP:
            self.change("threshold", 0.02)
        elif k == pg.K_DOWN:
            self.change("threshold", -0.02)
        elif k == pg.K_RIGHT:
            self.change("refractory", 25)
        elif k == pg.K_LEFT:
            self.change("refractory", -25)
        elif k == pg.K_RIGHTBRACKET:
            self.change("smoothing", 0.05)
        elif k == pg.K_LEFTBRACKET:
            self.change("smoothing", -0.05)
        elif k == pg.K_m:
            if self.session is not None:
                self.note("settings are locked during a session")
            elif MC.POINTER_SIDE not in self.sides and self.cursor_source == "mouse":
                self.note(f"IMU cursor needs the {MC.POINTER_SIDE} band")
            else:
                self.cursor_source = "mouse" if self.cursor_source == "imu" else "imu"
        elif k == pg.K_c:
            x0, y0, w, h = test_area
            self.cx, self.cy = x0 + w / 2, y0 + h / 2
        elif k == pg.K_g:
            self.trace_gain = {1.0: 4.0, 4.0: 16.0}.get(self.trace_gain, 1.0)
        return True

    def update_rates(self):
        now = time.perf_counter()
        for s, b in self.bands.items():
            t, n = self._rate_ref[s]
            if now - t >= 1.0:
                dp = b.n_packets - n
                self.rate[s] = (2 * dp / (now - t), dp / (now - t))
                self._rate_ref[s] = (now, b.n_packets)

    # ------------------------------------------------------------------ drawing
    def text(self, s, pos, col=FG, font=None):
        surf = (font or self.font).render(str(s), True, col)
        self.screen.blit(surf, pos)
        return surf.get_width()

    def draw(self, W, H, test_area):
        self.screen.fill(BG)
        # top bar
        cal_txt = ", ".join(f"{s}: {'calibrated' if self.cals.get(s, BandCal()).calibrated else 'UNCALIBRATED'}"
                            for s in self.sides)
        self.text(f"[1] live  [2] calibrate  [3] test  [4] fatigue      mode: {self.mode.upper()}      user: {self.a.user}"
                  f"   ({cal_txt})", (12, 8), FG)
        st = self.settings()
        self.text(f"threshold {st['threshold']:.2f} (Up/Down)   refractory {st['refractory_ms']} ms (Left/Right)   "
                  f"smoothing {st['smoothing']:.2f} ([ ])   cursor {st['cursor_source']} (M)   "
                  f"trace gain x{self.trace_gain:g} (G)" + ("   fake: hold L / R to squeeze" if self.a.fake else ""),
                  (12, 30), DIM)
        if self.mode == "live":
            self.draw_live(W, H)
        elif self.mode == "calibrate":
            self.draw_calibrate(W, H)
        elif self.mode == "fatigue":
            self.draw_fatigue(W, H)
        else:
            self.draw_test(W, H, test_area)
        if time.perf_counter() < self.msg_until:
            self.text(self.msg, (12, H - 24), WARN)

    def band_header(self, s, x, y):
        b = self.bands[s]
        sps, pps = self.rate[s]
        col = OK if b.connected else BAD
        w = self.text(f"{s.upper():5s}", (x, y), SIDE_COL[s])
        w += self.text(f" {b.status}", (x + w, y), col)
        bat = f"{b.battery}%" if b.battery is not None else "?"
        rate_col = OK if 180 <= sps <= 215 else (WARN if sps > 0 else BAD)
        self.text(f"battery {bat}", (x, y + 20), WARN if (b.battery or 100) < 20 else FG)
        self.text(f"{sps:5.0f} samples/s", (x + 130, y + 20), rate_col)
        self.text(f"{pps:4.0f} pkts/s   dropped {b.drops}   clicks {self.click_count[s]}"
                  + (f"   out-of-order {b.out_of_order}" if b.out_of_order else ""),
                  (x + 290, y + 20), WARN if b.out_of_order else DIM)

    def draw_live(self, W, H):
        pg = self.pg
        n = len(self.sides)
        pw = (W - 12 * (n + 1)) // n
        for k, s in enumerate(self.sides):
            b = self.bands[s]
            x = 12 + k * (pw + 12)
            y = 60
            ph = H - y - 36
            pg.draw.rect(self.screen, PANEL, (x, y, pw, ph), border_radius=6)
            if time.perf_counter() < self.flash[s]:
                pg.draw.rect(self.screen, SIDE_COL[s], (x, y, pw, ph), 3, border_radius=6)
            self.band_header(s, x + 10, y + 8)
            N = int(TRACE_S * 200)
            end = b.n
            a = max(0, end - N)
            emg = b.emg[a:end].astype(np.float32)
            valid = b.valid[a:end]
            act = b.act[a:end]
            ty = y + 56
            act_h = 110
            strip = (ph - 56 - act_h - 46) / 8
            xs = x + 50 + np.arange(len(emg)) * (pw - 60) / N
            for c in range(8):
                cy = ty + strip * (c + 0.5)
                self.text(f"ch{c}", (x + 10, cy - 8), DIM)
                pg.draw.line(self.screen, (44, 48, 58), (x + 50, cy), (x + pw - 10, cy))
                if len(emg) > 1:
                    v = np.clip(emg[:, c] * self.trace_gain / 128.0, -1, 1) * (strip * 0.48)
                    v[~valid] = 0
                    pts = np.column_stack([xs, cy - v])
                    pg.draw.lines(self.screen, CH_COL[c], False, pts.tolist(), 1)
            # activation strip
            ay = ty + strip * 8 + 12
            pg.draw.rect(self.screen, (20, 22, 28), (x + 50, ay, pw - 60, act_h))
            scale = act_h / 1.5
            thr_y = ay + act_h - self.threshold * scale
            pg.draw.line(self.screen, WARN, (x + 50, thr_y), (x + pw - 10, thr_y))
            one_y = ay + act_h - 1.0 * scale
            pg.draw.line(self.screen, (60, 64, 76), (x + 50, one_y), (x + pw - 10, one_y))
            self.text("act", (x + 10, ay + 4), DIM)
            self.text("thr", (x + 10, thr_y - 8), WARN)
            self.text("1.0", (x + 10, one_y - 8), DIM)
            if len(act) > 1:
                pts = np.column_stack([xs, ay + act_h - np.clip(act, 0, 1.5) * scale])
                pg.draw.lines(self.screen, SIDE_COL[s], False, pts.tolist(), 2)
            for ci in list(b.click_idx):
                if a <= ci < end:
                    cx = x + 50 + (ci - a) * (pw - 60) / N
                    pg.draw.line(self.screen, FG, (cx, ty), (cx, ay + act_h), 1)
            self.text(f"activation {b.det.activation:.2f}   (rest = 0, typical squeeze = 1; white lines = clicks)",
                      (x + 50, ay + act_h + 6), DIM)

    def draw_calibrate(self, W, H):
        pg = self.pg
        cs = self.cal_state
        if cs is None:
            lines = ["Calibration for user: " + self.a.user,
                     f"{self.a.cal_rest:.0f} s rest (both arms), then {self.a.cal_reps} squeezes per hand.",
                     "Squeeze like a normal click, not maximum effort: that sets activation = 1.",
                     "", "Press SPACE to start."]
            for i, l in enumerate(lines):
                self.text(l, (60, 120 + 36 * i), FG, self.big if i == 0 else None)
        elif cs["result"] is None:
            label, kind, side, dur = cs["steps"][cs["i"]]
            el = time.perf_counter() - cs["t"]
            col = BAD if kind == "squeeze" else (OK if kind in ("rest", "relax") else FG)
            self.text(label, (60, 140), col, self.big)
            pg.draw.rect(self.screen, PANEL, (60, 200, W - 120, 16))
            pg.draw.rect(self.screen, col, (60, 200, (W - 120) * min(el / dur, 1), 16))
            self.text(f"step {cs['i'] + 1}/{len(cs['steps'])}   Esc to cancel", (60, 226), DIM)
        else:
            self.text("Calibration done (SPACE to redo)", (60, 110), OK, self.big)
            for i, l in enumerate(cs["result"]):
                self.text(l, (60, 160 + 22 * i), FG if "WARN" not in l and "FAIL" not in l else WARN)
        self.draw_band_strip(W, H)

    def draw_band_strip(self, W, H):
        pg = self.pg
        y0 = H - 104
        n = len(self.sides)
        pw = (W - 12 * (n + 1)) // n
        for k, s in enumerate(self.sides):
            b = self.bands[s]
            x = 12 + k * (pw + 12)
            pg.draw.rect(self.screen, PANEL, (x, y0, pw, 78), border_radius=6)
            if time.perf_counter() < self.flash[s]:
                pg.draw.rect(self.screen, SIDE_COL[s], (x, y0, pw, 78), 3, border_radius=6)
            self.band_header(s, x + 10, y0 + 6)
            bw = pw - 20
            pg.draw.rect(self.screen, (20, 22, 28), (x + 10, y0 + 52, bw, 16))
            pg.draw.rect(self.screen, SIDE_COL[s], (x + 10, y0 + 52, bw * min(b.det.activation / 1.5, 1), 16))
            tx = x + 10 + bw * self.threshold / 1.5
            pg.draw.line(self.screen, WARN, (tx, y0 + 48), (tx, y0 + 72), 2)
            ox = x + 10 + bw / 1.5
            pg.draw.line(self.screen, DIM, (ox, y0 + 48), (ox, y0 + 72), 1)

    def draw_test(self, W, H, area):
        pg = self.pg
        x0, y0, w, h = area
        pg.draw.rect(self.screen, (20, 22, 29), area)
        el = time.perf_counter() - getattr(self, "phase_t", 0)
        if self.session is None and self.phase != "done":
            lines = ["Click-target test", "",
                     f"1. {self.a.rest_still:.0f} s: keep still, hands relaxed (counts false clicks)",
                     f"2. {self.a.rest_move:.0f} s: move the cursor around, DON'T click (false clicks while moving)",
                     f"3. {self.a.targets} targets: click each with the band its colour says "
                     f"(orange = LEFT, blue = RIGHT)",
                     "", f"cursor: {self.cursor_source}    Press SPACE to start, Esc to stop early."]
            for i, l in enumerate(lines):
                self.text(l, (x0 + 60, y0 + 60 + 32 * i), FG, self.big if i == 0 else None)
        elif self.phase == "done":
            self.text("Session summary   (SPACE for another session)", (x0 + 40, y0 + 30), OK, self.big)
            for i, l in enumerate(self.summary_text.split("\n")):
                self.text(l, (x0 + 40, y0 + 80 + 24 * i), FG)
        else:
            prompts = {"countdown": f"Starting in {max(0, 3 - el):.0f}...  relax",
                       "rest_still": f"REST: keep still, hands relaxed   {max(0, self.a.rest_still - el):.0f} s",
                       "rest_move": f"MOVE the cursor around, don't click   {max(0, self.a.rest_move - el):.0f} s",
                       "targets": f"target {self.targets_done + 1}/{self.a.targets}"}
            self.text(prompts.get(self.phase, ""), (x0 + 20, y0 + 14), FG, self.big)
            n_false = sum(e["outcome"] == "false_rest" for e in self.session.events)
            hits = sum(e["outcome"] == "hit" for e in self.session.events)
            n_task = sum(e["phase"] == "targets" and e["event"] == "click" for e in self.session.events)
            self.text(f"false clicks at rest {n_false}    hits {hits}/{n_task} clicks", (x0 + 20, y0 + 52), DIM)
            if self.target and self.phase == "targets":
                tx, ty, r, _ = self.target
                col = SIDE_COL[self.target_band]
                pg.draw.circle(self.screen, col, (tx, ty), r, 4)
                pg.draw.circle(self.screen, col, (tx, ty), 4)
                self.text(f"{self.target_band.upper()} fist", (tx - r, ty + r + 6), col)
        # cursor
        frozen = time.perf_counter() < self.freeze_until
        cc = DIM if frozen else FG
        pg.draw.circle(self.screen, cc, (self.cx, self.cy), 9, 2)
        pg.draw.line(self.screen, cc, (self.cx - 16, self.cy), (self.cx - 11, self.cy), 2)
        pg.draw.line(self.screen, cc, (self.cx + 11, self.cy), (self.cx + 16, self.cy), 2)
        pg.draw.line(self.screen, cc, (self.cx, self.cy - 16), (self.cx, self.cy - 11), 2)
        pg.draw.line(self.screen, cc, (self.cx, self.cy + 11), (self.cx, self.cy + 16), 2)
        self.draw_band_strip(W, H)

    # ------------------------------------------------------------------ fatigue protocol
    def start_fatigue(self):
        """Cued squeezes for --fatigue-min, still rest for --fatigue-rest-min, cued squeezes again
        for --fatigue-post-min. Every sample is saved to <id>_raw.npz with phases and cue times,
        for `python -m devtool.replay_fatigue` (fixed vs fatigue-adaptive threshold)."""
        self.session = Session(DATA_DIR / "sessions", self.a.user + "_fatigue", self.bands, self.settings(),
                               0.0, 0.0, self.a.pre_ms, self.a.post_ms)
        self.summary_text = ""
        self.cue_until, self.next_cue, self.n_cues = 0.0, None, 0
        self.phase, self.phase_t = "f_countdown", time.perf_counter()

    def _fatigue_phase(self, name):
        self.phase, self.phase_t = name, time.perf_counter()
        self.session.mark_phase(name)
        self.cue_until = 0.0
        self.next_cue = self.phase_t + 1.0 if name in ("fatigue", "post") else None

    def tick_fatigue(self):
        if self.session is None or self.phase not in FATIGUE_PHASES + ("f_countdown",):
            return
        now = time.perf_counter()
        el = now - self.phase_t
        if self.phase == "f_countdown":
            if el >= 3:
                self._fatigue_phase("fatigue")
        elif self.phase == "fatigue" and el >= self.a.fatigue_min * 60:
            self._fatigue_phase("recovery")
        elif self.phase == "recovery" and el >= self.a.fatigue_rest_min * 60:
            self._fatigue_phase("post")
        elif self.phase == "post" and el >= self.a.fatigue_post_min * 60:
            self.end_session()
            return
        if self.next_cue is not None and now >= self.next_cue:
            self.session.mark_cue(self.phase)
            self.n_cues += 1
            self.cue_until = now + self.a.cue_on
            self.next_cue += self.a.cue_interval
        self.session.poll()

    def draw_fatigue(self, W, H):
        if self.session is None:
            lines = ["Fatigue protocol for user: " + self.a.user,
                     f"{self.a.fatigue_min:g} min: squeeze BOTH hands every time the screen says SQUEEZE "
                     f"(every {self.a.cue_interval:g} s), like a normal click.",
                     "Keep going even when it gets hard - that is the point. Don't squeeze harder to compensate.",
                     f"Then {self.a.fatigue_rest_min:g} min rest, arms still (any click here is a false click),",
                     f"then {self.a.fatigue_post_min:g} min of squeezes again (recovery).",
                     "Calibrate first (mode 2). Everything is saved for replay.", "",
                     "Press SPACE to start.   Esc stops early (what was recorded is saved)."]
            for i, l in enumerate(lines):
                self.text(l, (60, 100 + 34 * i), FG, self.big if i == 0 else None)
            if self.summary_text:
                for i, l in enumerate(self.summary_text.splitlines()):
                    self.text(l, (60, 400 + 22 * i), OK)
        else:
            now = time.perf_counter()
            el = now - self.phase_t
            total = {"f_countdown": 3, "fatigue": self.a.fatigue_min * 60, "recovery": self.a.fatigue_rest_min * 60,
                     "post": self.a.fatigue_post_min * 60}[self.phase]
            title = {"f_countdown": "get ready", "fatigue": "FATIGUE: squeeze on every cue",
                     "recovery": "REST: arms still, hands relaxed", "post": "AFTER REST: squeeze on every cue"}
            self.text(title[self.phase], (60, 100), FG, self.big)
            self.text(f"{max(total - el, 0):5.0f} s left in this part     cues so far {self.n_cues}     "
                      f"clicks " + "  ".join(f"{s} {self.click_count[s]}" for s in self.sides), (60, 145), DIM)
            if now < self.cue_until:
                self.pg.draw.rect(self.screen, BAD, (60, 190, W - 120, 220), border_radius=12)
                self.text("SQUEEZE", (W // 2 - 70, 285), FG, self.big)
            elif self.phase in ("fatigue", "post"):
                self.pg.draw.rect(self.screen, PANEL, (60, 190, W - 120, 220), border_radius=12)
                self.text("relax", (W // 2 - 40, 285), DIM, self.big)
        self.draw_band_strip(W, H)

    # ------------------------------------------------------------------ self-test (fake bands)
    def autotest(self, area):
        """Drives the UI with fake bands: calibrate, then a session where the fake 'user' moves
        the cursor to each target and squeezes the right band. Used by tests only."""
        now = time.perf_counter()
        if self.a.autotest_fatigue:
            if not getattr(self, "_auto_f", False):
                self._auto_f = True
                self.mode = "fatigue"
                self.start_fatigue()
            elif self.session is None:
                self.a.seconds = 0.01
            else:
                for s in self.mgr.squeeze:
                    self.mgr.squeeze[s] = now < self.cue_until - 0.2   # ~0.6 s squeeze per cue
            return
        st = getattr(self, "_auto", None)
        if st is None:
            self._auto = st = {"stage": "cal", "next": now}
            self.cursor_source = "imu"  # no IMU in fake mode, so the cursor stays where autotest puts it
            self.mode = "calibrate"
            self.start_calibration()
        if st["stage"] == "cal":
            cs = self.cal_state
            label, kind, side, _ = cs["steps"][min(cs["i"], len(cs["steps"]) - 1)]
            for s in self.mgr.squeeze:
                self.mgr.squeeze[s] = cs["result"] is None and kind == "squeeze" and side == s
            if cs["result"] is not None:
                st["stage"] = "test"
                self.mode = "test"
                self.start_session(area)
        elif st["stage"] == "test":
            if self.session is None:
                st["stage"] = "done"
                self.a.seconds = 0.01
                return
            if self.phase == "rest_move":
                self.cx += 3
            if self.phase == "targets" and self.target:
                tx, ty, r, tid = self.target
                self.cx, self.cy = tx, ty
                want = self.target_band
                # every 4th target: squeeze the wrong band first, to exercise misses
                if st.get("tid") != tid:
                    st["tid"], st["t0"] = tid, now
                k = now - st["t0"]
                wrong = [s for s in self.sides if s != want]
                squeeze_side = wrong[0] if (tid % 4 == 0 and wrong and 0.3 < k < 0.6) else want
                for s in self.mgr.squeeze:
                    self.mgr.squeeze[s] = (s == squeeze_side) and (0.3 < k < 0.6 or 0.9 < k < 1.2)


FATIGUE_PHASES = ("fatigue", "recovery", "post")


def fatigue_summary(s: Session) -> str:
    cues = {}
    for phase, _t, _i in s.cues:
        cues[phase] = cues.get(phase, 0) + 1
    clicks = {}
    for e in s.events:
        if e["event"] == "click":
            key = (e["phase"], e["band"])
            clicks[key] = clicks.get(key, 0) + 1
    bands = sorted({b for _p, b in clicks} | set(s.bands))
    lines = [f"fatigue session {s.id}: {cues.get('fatigue', 0)} cues during fatigue, "
             f"{cues.get('post', 0)} after rest"]
    for b in bands:
        lines.append(f"  {b}: clicks during fatigue {clicks.get(('fatigue', b), 0)}, "
                     f"false clicks during rest {clicks.get(('recovery', b), 0)}, "
                     f"after rest {clicks.get(('post', b), 0)}")
    lines.append(f"compare fixed vs adaptive:  python -m devtool.replay_fatigue {s.out_dir / (s.id + '_raw.npz')}")
    return "\n".join(lines)


def main(argv=None):
    App(parse_args(argv)).run()


if __name__ == "__main__":
    main()

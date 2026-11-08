"""System-wide claw (point scanning): click anywhere on the real screen with ONE gesture.

A see-through, click-through, always-on-top overlay over the whole screen:

    1st press   a vertical line starts sweeping left <-> right
    2nd press   it stops (backed up by the reaction-time compensation); a horizontal line
                starts sweeping up <-> down
    3rd press   it stops, the real mouse moves there and clicks (left / right / double)

Driven by the Blocks action "claw step" (any input: a Myo fist, a BCI marker, a camera pose).
ClawState is the pure logic (tested without a screen); ClawOverlay draws it with Qt.
"""

import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

import theme


class ClawState:
    """Point scanning with an optional refine pass (like iOS Switch Control):

        press 1  coarse line sweeps the whole width
        press 2  stops -> a narrow ribbon around it; a slow line sweeps just that ribbon
        press 3  fixes x; coarse line sweeps the height
        press 4  stops -> slow line in a ribbon
        press 5  fixes y and clicks
    (refine off: 3 presses, as before.)

    Drive mode (two switches, e.g. left fist / right fist): HOLD one to glide the crosshair
    along x, the other along y - slow at first, faster the longer you hold, bouncing at the
    edges; let go to stop. A quick TAP of either clicks where the crosshair is.
    """
    IDLE, X, XF, Y, YF, DRIVE = "idle", "x", "x-fine", "y", "y-fine", "drive"

    def __init__(self, sweep_s=3.0, lag_s=0.08, refine=True, band=0.16):
        self.sweep_s, self.lag_s, self.refine, self.band = sweep_s, lag_s, refine, band
        self.stage = self.IDLE
        self.pos, self.dir = 0.0, 1           # absolute 0..1 along the current axis
        self.lo, self.hi = 0.0, 1.0           # current sweep range
        self.x = self.y = None
        self.pause_until = 0.0
        # drive mode
        self.cx, self.cy = 0.5, 0.5
        self.driving = None                   # "x" / "y" while a switch is held
        self.drive_dir = {"x": 1, "y": 1}
        self.held_since = 0.0
        self.v0, self.vmax, self.ramp_s, self.tap_s = 0.04, 0.6, 1.6, 0.3

    # ── scanning ──
    def _speed(self):
        return (self.hi - self.lo) / self.sweep_s

    def tick(self, now, dt):
        if self.stage == self.DRIVE and self.driving:
            held = now - self.held_since
            v = self.v0 + (self.vmax - self.v0) * min(1.0, max(0.0, held - 0.15) / self.ramp_s)
            ax = self.driving
            val = (self.cx if ax == "x" else self.cy) + self.drive_dir[ax] * v * dt
            if val >= 1.0:
                val, self.drive_dir[ax] = 2.0 - val, -1
            elif val <= 0.0:
                val, self.drive_dir[ax] = -val, 1
            if ax == "x":
                self.cx = val
            else:
                self.cy = val
            return
        if self.stage in (self.IDLE, self.DRIVE) or now < self.pause_until:
            return
        self.pos += self.dir * self._speed() * dt
        if self.pos >= self.hi:
            self.pos, self.dir = 2 * self.hi - self.pos, -1
        elif self.pos <= self.lo:
            self.pos, self.dir = 2 * self.lo - self.pos, 1

    def _stopped(self):
        back = self.pos - self.dir * self.lag_s * self._speed()
        return min(max(back, self.lo), self.hi)

    def _ribbon(self, centre):
        half = self.band / 2
        lo, hi = max(0.0, centre - half), min(1.0, centre + half)
        return lo, hi

    def step(self, now):
        """One switch press. Returns (fx, fy) in 0..1 when it completes a click, else None."""
        if now < self.pause_until:
            return None
        if self.stage in (self.IDLE, self.DRIVE):
            self.stage, self.lo, self.hi, self.pos, self.dir = self.X, 0.0, 1.0, 0.0, 1
            self.pause_until = now + 0.25
            return None
        stop = self._stopped()
        self.pause_until = now + 0.3
        if self.stage == self.X and self.refine:
            self.stage = self.XF
            self.lo, self.hi = self._ribbon(stop)
            self.pos, self.dir = self.lo, 1
            return None
        if self.stage in (self.X, self.XF):
            self.x = stop
            self.stage, self.lo, self.hi, self.pos, self.dir = self.Y, 0.0, 1.0, 0.0, 1
            return None
        if self.stage == self.Y and self.refine:
            self.stage = self.YF
            self.lo, self.hi = self._ribbon(stop)
            self.pos, self.dir = self.lo, 1
            return None
        self.y = stop
        self.stage = self.IDLE
        self.pause_until = now + 0.4
        return self.x, self.y

    def cancel(self):
        self.stage = self.IDLE
        self.driving = None

    # ── drive mode ──
    def drive(self, axis, down, now):
        """Returns (fx, fy) when a quick tap means 'click here', else None."""
        if self.stage != self.DRIVE:
            self.stage = self.DRIVE
        if down:
            self.driving, self.held_since = axis, now
            self._start = (self.cx, self.cy)
            return None
        if self.driving != axis:
            return None
        self.driving = None
        if now - self.held_since < self.tap_s:     # a tap, not a glide: undo the nudge, click
            self.cx, self.cy = self._start
            return self.cx, self.cy
        return None


class ClawOverlay(QWidget):
    """Transparent full-screen overlay. step(params) is called on the GUI thread."""

    def __init__(self, click_fn):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.click_fn = click_fn            # (x_px, y_px, button) in OS screen pixels
        self.state = ClawState()
        self.button = "left"
        self.restart = False
        self.last = None
        self._t = time.perf_counter()
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.status = ""

    def _ensure_shown(self):
        if not self.isVisible():
            scr = QApplication.primaryScreen()
            self.setGeometry(scr.geometry())
            self.show()
            self.raise_()
            self.timer.start()
        self._last_activity = time.perf_counter()

    def step(self, params: dict):
        self.state.sweep_s = float(params.get("sweep_s", self.state.sweep_s))
        self.state.lag_s = float(params.get("lag_ms", self.state.lag_s * 1000)) / 1000.0
        self.state.refine = bool(params.get("refine", True))
        self.button = params.get("button", "left")
        self.restart = bool(params.get("restart", False))
        self._ensure_shown()
        now = time.perf_counter()
        done = self.state.step(now)
        if done:
            fx, fy = done
            self.last = (fx, fy, now)
            self._click(fx, fy)
            if self.restart:
                QTimer.singleShot(600, lambda: self.state.step(time.perf_counter()))
        self.update()

    def drive(self, params: dict):
        self.button = params.get("button", "left")
        self._ensure_shown()
        now = time.perf_counter()
        done = self.state.drive(params.get("axis", "x"), bool(params.get("down")), now)
        if done:
            self.last = (done[0], done[1], now)
            self._click(*done)
        self.update()

    def cancel(self):
        self.state.cancel()
        self.update()

    def _click(self, fx, fy):
        scr = QApplication.primaryScreen()
        g = scr.geometry()
        x, y = g.x() + fx * g.width(), g.y() + fy * g.height()
        import sys
        if sys.platform == "win32":                 # Qt logical px -> physical px for SetCursorPos
            r = scr.devicePixelRatio()
            x, y = x * r, y * r
        self.click_fn(x, y, self.button)

    def _tick(self):
        now = time.perf_counter()
        self.state.tick(now, min(now - self._t, 0.1))
        self._t = now
        idle_drive = (self.state.stage == ClawState.DRIVE and not self.state.driving
                      and now - getattr(self, "_last_activity", now) > 6.0)
        if (self.state.stage == ClawState.IDLE and (not self.last or now - self.last[2] > 1.2)) or idle_drive:
            if idle_drive:
                self.state.stage = ClawState.IDLE
            if not self.restart:
                self.timer.stop()
                self.hide()
                return
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        s = self.state
        col = QColor(theme.ORANGE)
        dim = QColor(theme.ORANGE)
        dim.setAlpha(110)
        ribbon = QColor(theme.ORANGE)
        ribbon.setAlpha(40)
        if s.stage in (ClawState.X, ClawState.XF):
            if s.stage == ClawState.XF:
                p.fillRect(QRectF(s.lo * w, 0, (s.hi - s.lo) * w, h), ribbon)
            x = s.pos * w
            p.setPen(QPen(col, 4 if s.stage == ClawState.X else 2))
            p.drawLine(int(x), 0, int(x), h)
            self._label(p, "stop to fine-tune \u2194" if s.stage == ClawState.X and s.refine
                        else "stop \u2194", x + 12, 40)
        elif s.stage in (ClawState.Y, ClawState.YF):
            x = s.x * w
            if s.stage == ClawState.YF:
                p.fillRect(QRectF(0, s.lo * h, w, (s.hi - s.lo) * h), ribbon)
            y = s.pos * h
            p.setPen(QPen(dim, 3))
            p.drawLine(int(x), 0, int(x), h)
            p.setPen(QPen(col, 4 if s.stage == ClawState.Y else 2))
            p.drawLine(0, int(y), w, int(y))
            p.setBrush(QColor(theme.ORANGE))
            p.drawEllipse(QRectF(x - 9, y - 9, 18, 18))
            self._label(p, ("stop to fine-tune \u2195" if s.stage == ClawState.Y and s.refine
                            else f"stop to {self.button} click \u2195"), x + 14, y - 14)
        elif s.stage == ClawState.DRIVE:
            x, y = s.cx * w, s.cy * h
            p.setPen(QPen(col if s.driving == "x" else dim, 4 if s.driving == "x" else 2))
            p.drawLine(int(x), 0, int(x), h)
            p.setPen(QPen(col if s.driving == "y" else dim, 4 if s.driving == "y" else 2))
            p.drawLine(0, int(y), w, int(y))
            p.setBrush(QColor(theme.ORANGE))
            p.drawEllipse(QRectF(x - 9, y - 9, 18, 18))
            self._label(p, "hold to glide \u2194 / \u2195  \u00b7  tap to click", x + 14, y - 14)
        if self.last and time.perf_counter() - self.last[2] < 1.2:
            fx, fy, _ = self.last
            p.setPen(QPen(QColor(theme.GREEN), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(fx * w - 22, fy * h - 22, 44, 44))
        p.end()

    @staticmethod
    def _label(p, text, x, y):
        f = QFont()
        f.setPointSize(13)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 160))
        p.drawText(int(x) + 1, int(y) + 1, text)
        p.setPen(QColor("white"))
        p.drawText(int(x), int(y), text)

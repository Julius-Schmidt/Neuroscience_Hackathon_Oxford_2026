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
    IDLE, SWEEP_X, SWEEP_Y = "idle", "x", "y"

    def __init__(self, sweep_s=3.0, lag_s=0.08):
        self.sweep_s, self.lag_s = sweep_s, lag_s
        self.stage = self.IDLE
        self.pos, self.dir = 0.0, 1
        self.x = self.y = None
        self.pause_until = 0.0

    def tick(self, now, dt):
        if self.stage == self.IDLE or now < self.pause_until:
            return
        self.pos += self.dir * dt / self.sweep_s
        if self.pos >= 1.0:
            self.pos, self.dir = 2.0 - self.pos, -1
        elif self.pos <= 0.0:
            self.pos, self.dir = -self.pos, 1

    def step(self, now):
        """One switch press. Returns (fx, fy) in 0..1 when it completes a click, else None."""
        if now < self.pause_until:
            return None
        if self.stage == self.IDLE:
            self.stage, self.pos, self.dir = self.SWEEP_X, 0.0, 1
            self.pause_until = now + 0.25
            return None
        back = min(max(self.pos - self.dir * self.lag_s / self.sweep_s, 0.0), 1.0)
        if self.stage == self.SWEEP_X:
            self.x, self.stage, self.pos, self.dir = back, self.SWEEP_Y, 0.0, 1
            self.pause_until = now + 0.3
            return None
        self.y = back
        self.stage = self.IDLE
        self.pause_until = now + 0.4
        return self.x, self.y

    def cancel(self):
        self.stage = self.IDLE


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

    def step(self, params: dict):
        self.state.sweep_s = float(params.get("sweep_s", self.state.sweep_s))
        self.state.lag_s = float(params.get("lag_ms", self.state.lag_s * 1000)) / 1000.0
        self.button = params.get("button", "left")
        self.restart = bool(params.get("restart", False))
        if not self.isVisible():
            scr = QApplication.primaryScreen()
            self.setGeometry(scr.geometry())
            self.show()
            self.raise_()
            self.timer.start()
        now = time.perf_counter()
        done = self.state.step(now)
        if done:
            fx, fy = done
            self.last = (fx, fy, now)
            self._click(fx, fy)
            if self.restart:
                QTimer.singleShot(600, lambda: self.state.step(time.perf_counter()))
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
        if self.state.stage == ClawState.IDLE and (not self.last or now - self.last[2] > 1.2):
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
        if s.stage == ClawState.SWEEP_X:
            x = s.pos * w
            p.setPen(QPen(col, 4))
            p.drawLine(int(x), 0, int(x), h)
            self._label(p, "fist to stop  ↔", x + 12, 40)
        elif s.stage == ClawState.SWEEP_Y:
            x = s.x * w
            y = s.pos * h
            p.setPen(QPen(dim, 3))
            p.drawLine(int(x), 0, int(x), h)
            p.setPen(QPen(col, 4))
            p.drawLine(0, int(y), w, int(y))
            p.setBrush(QColor(theme.ORANGE))
            p.drawEllipse(QRectF(x - 9, y - 9, 18, 18))
            self._label(p, f"fist to {self.button} click  ↕", x + 14, y - 14)
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

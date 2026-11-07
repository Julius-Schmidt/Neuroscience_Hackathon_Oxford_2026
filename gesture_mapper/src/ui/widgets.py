"""
Reusable building blocks: cards, chips, segmented control, sidebar items,
and the live CameraView that every page shares.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, Signal, QSize
from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap)
from PySide6.QtCore import QRect
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

import theme
import labels

_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class FlowLayout(QLayout):
    """Left-to-right wrap layout (the standard Qt FlowLayout recipe)."""

    def __init__(self, parent=None, spacing: int = 12):
        super().__init__(parent)
        self._items = []
        self._space = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(0, 0, width, apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect.x(), rect.y(), rect.width(), apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _do_layout(self, x0, y0, width, apply):
        x, y, row_h = x0, y0, 0
        for it in self._items:
            hint = it.sizeHint()
            if x + hint.width() > x0 + width and row_h > 0:
                x = x0
                y += row_h + self._space
                row_h = 0
            if apply:
                it.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x += hint.width() + self._space
            row_h = max(row_h, hint.height())
        return y + row_h - y0


def hline() -> QFrame:
    ln = QFrame()
    ln.setFixedHeight(1)
    ln.setStyleSheet(f"background: {theme.CARD_BORDER}; border: none;")
    return ln


def make_label(text: str, cls: str | None = None, size: int | None = None,
               weight: QFont.Weight | None = None, color: str | None = None) -> QLabel:
    lb = QLabel(text)
    if cls:
        lb.setProperty("class", cls)
    style = []
    if size:
        style.append(f"font-size: {size}px;")
    if weight:
        f = lb.font()
        f.setWeight(weight)
        lb.setFont(f)
    if color:
        style.append(f"color: {color};")
    if style:
        lb.setStyleSheet(" ".join(style))
    return lb


class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")


class Chip(QWidget):
    """Small rounded pill: optional colored dot + text."""

    def __init__(self, text: str = "", dot: str | None = None,
                 bg: str = theme.CARD, fg: str = theme.TEXT_DIM, parent=None):
        super().__init__(parent)
        self._text, self._dot, self._bg, self._fg = text, dot, bg, fg
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set(self, text: str | None = None, dot: str | None = "keep",
            bg: str | None = None, fg: str | None = None):
        if text is not None:
            self._text = text
        if dot != "keep":
            self._dot = dot
        if bg:
            self._bg = bg
        if fg:
            self._fg = fg
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        w = fm.horizontalAdvance(self._text) + 24 + (14 if self._dot else 0)
        return QSize(w, 24)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._bg))
        p.drawRoundedRect(r, 12, 12)
        x = 12
        if self._dot:
            p.setBrush(QColor(self._dot))
            p.drawEllipse(QPointF(x, self.height() / 2), 3.5, 3.5)
            x += 12
        p.setPen(QColor(self._fg))
        f = p.font()
        f.setPointSize(11)
        f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        p.drawText(QRectF(x, 0, self.width() - x - 8, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, self._text)
        p.end()


class SegmentedControl(QWidget):
    """macOS-style segmented picker."""

    changed = Signal(int)

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self._buttons: list[QPushButton] = []
        self.setStyleSheet(f"""
            QWidget#seg {{ background: {theme.FIELD}; border-radius: 9px; }}
            QPushButton {{
                background: transparent; border: none; border-radius: 7px;
                padding: 5px 16px; color: {theme.TEXT_DIM}; font-weight: 500;
            }}
            QPushButton:checked {{
                background: #3a3a41; color: {theme.TEXT}; font-weight: 600;
            }}
        """)
        inner = QWidget(self)
        inner.setObjectName("seg")
        lay = QHBoxLayout(inner)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        for i, text in enumerate(items):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, idx=i: self.set_index(idx, emit=True))
            lay.addWidget(b)
            self._buttons.append(b)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        self._index = -1
        self.set_index(0)

    def set_index(self, idx: int, emit: bool = False):
        if idx == self._index:
            for i, b in enumerate(self._buttons):
                b.setChecked(i == idx)
            return
        self._index = idx
        for i, b in enumerate(self._buttons):
            b.setChecked(i == idx)
        if emit:
            self.changed.emit(idx)

    def index(self) -> int:
        return self._index


class SidebarItem(QPushButton):
    def __init__(self, text: str, symbol: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._symbol = symbol
        self._refresh_icon()
        self.toggled.connect(lambda _: self._refresh_icon())
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none; border-radius: 8px;
                padding: 8px 12px; text-align: left; font-size: 13px;
                font-weight: 500; color: {theme.TEXT_DIM};
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.05); color: {theme.TEXT}; }}
            QPushButton:checked {{
                background: rgba(10,132,255,0.18); color: {theme.TEXT}; font-weight: 600;
            }}
        """)

    def _refresh_icon(self):
        color = theme.ACCENT if self.isChecked() else theme.TEXT_DIM
        self.setIcon(theme.sf_icon(self._symbol, 15, color))


class CameraView(QWidget):
    """Live camera preview with landmark-skeleton overlay and gesture chips.
    Reads the detector's stashed last_frame/last_hands/last_gestures — it
    never touches the camera itself, so it works identically whether the
    preview pump or the live control loop is doing the reading."""

    def __init__(self, engine, radius: int = 14, show_chips: bool = True, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._radius = radius
        self._show_chips = show_chips
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)

    def showEvent(self, e):
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), self._radius, self._radius)
        p.setClipPath(clip)

        det = self._engine.detector
        frame = det.last_frame if det else None
        if frame is None:
            self._paint_placeholder(p, w, h)
            p.end()
            return

        fh, fw = frame.shape[:2]
        # cover-crop the (mirrored) frame into the widget
        scale = max(w / fw, h / fh)
        sw, sh = fw * scale, fh * scale
        ox, oy = (w - sw) / 2, (h - sh) / 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, fw, fh, 3 * fw, QImage.Format.Format_RGB888).copy()
        img = img.mirrored(True, False)   # selfie view
        p.drawImage(QRectF(ox, oy, sw, sh), img)

        # subtle bottom gradient so chips stay readable
        g = QLinearGradient(0, h * 0.55, 0, h)
        g.setColorAt(0, QColor(0, 0, 0, 0))
        g.setColorAt(1, QColor(0, 0, 0, 130))
        p.fillRect(QRectF(0, 0, w, h), g)

        hands = det.last_hands
        gestures = det.last_gestures

        def to_px(lm) -> QPointF:
            return QPointF(ox + (1.0 - lm.x) * sw, oy + lm.y * sh)

        for i, (hand, lms) in enumerate(hands):
            color = QColor(theme.HAND_LEFT_COLOR if hand == "left"
                           else theme.HAND_RIGHT_COLOR)
            pen = QPen(color, 2.2, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            faint = QColor(color)
            faint.setAlpha(210)
            pen.setColor(faint)
            p.setPen(pen)
            for a, b in _BONES:
                p.drawLine(to_px(lms[a]), to_px(lms[b]))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("white"))
            for lm in lms:
                p.drawEllipse(to_px(lm), 2.2, 2.2)

            # pinch highlight: thumb tip → any fingertip that's nearly closed
            if i < len(gestures):
                g_ = gestures[i]
                ratios = {8: g_.pinch_index, 12: g_.pinch_middle,
                          16: g_.pinch_ring, 20: g_.pinch_pinky}
                for tip, ratio in ratios.items():
                    if ratio < 0.30:
                        alpha = int(255 * min(1.0, (0.30 - ratio) / 0.14))
                        c = QColor(theme.ACCENT)
                        c.setAlpha(alpha)
                        p.setPen(QPen(c, 3.2, Qt.PenStyle.SolidLine,
                                      Qt.PenCapStyle.RoundCap))
                        p.drawLine(to_px(lms[4]), to_px(lms[tip]))

        if self._show_chips:
            x = 14
            for i, (hand, lms) in enumerate(hands):
                if i >= len(gestures):
                    break
                g_ = gestures[i]
                name = labels.gesture_label(g_.name) if g_.name != "none" else "—"
                hand_lb = "L" if hand == "left" else "R" if hand == "right" else "?"
                text = f"{hand_lb} · {name}"
                if g_.name != "none":
                    text += f"  {g_.confidence:.0%}"
                x = self._draw_chip(p, x, h - 40, text,
                                    theme.HAND_LEFT_COLOR if hand == "left"
                                    else theme.HAND_RIGHT_COLOR)
            if not hands:
                self._draw_chip(p, 14, h - 40, "No hands in view", theme.TEXT_FAINT)
        p.end()

    def _draw_chip(self, p: QPainter, x: float, y: float, text: str,
                   dot: str) -> float:
        f = p.font()
        f.setPointSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(text)
        w = tw + 34
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 23, 200))
        p.drawRoundedRect(QRectF(x, y, w, 28), 14, 14)
        p.setBrush(QColor(dot))
        p.drawEllipse(QPointF(x + 14, y + 14), 3.5, 3.5)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(x + 24, y, w - 24, 28),
                   Qt.AlignmentFlag.AlignVCenter, text)
        return x + w + 8

    def _paint_placeholder(self, p: QPainter, w: int, h: int):
        p.fillRect(0, 0, w, h, QColor("#111113"))
        pm = theme.sf_symbol("camera.fill", 30, theme.TEXT_FAINT)
        pw = pm.width() / pm.devicePixelRatio()
        ph = pm.height() / pm.devicePixelRatio()
        p.drawPixmap(int((w - pw) / 2), int(h / 2 - ph - 6), pm)
        p.setPen(QColor(theme.TEXT_FAINT))
        f = p.font()
        f.setPointSize(13)
        p.setFont(f)
        msg = self._engine.camera_status_text()
        p.drawText(QRectF(0, h / 2 + 4, w, 44), Qt.AlignmentFlag.AlignHCenter, msg)

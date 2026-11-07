"""
First-run onboarding: welcome → hand roles → preloaded gestures → modes →
permissions → live try-it. Content is generated from the real gesture library
and mapping tables, so the tour never drifts from the truth.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

import theme
import labels
import handicon
from widgets import Card, CameraView, FlowLayout, make_label


def _v(widget, margins=(0, 0, 0, 0), spacing=8) -> QVBoxLayout:
    lay = QVBoxLayout(widget)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


class OnboardingView(QWidget):
    finished = Signal()

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("page")

        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 28)
        root.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_welcome())
        self.stack.addWidget(self._page_hands())
        self.stack.addWidget(self._page_gestures())
        self.stack.addWidget(self._page_modes())
        self.stack.addWidget(self._page_permissions())
        self.stack.addWidget(self._page_try())
        root.addWidget(self.stack, stretch=1)

        nav = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.setProperty("class", "ghost")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.clicked.connect(lambda: self._go(self.stack.currentIndex() - 1))
        nav.addWidget(self.back_btn)
        nav.addStretch()

        self.dots = make_label("", None, 14, None, theme.TEXT_FAINT)
        nav.addWidget(self.dots)
        nav.addStretch()

        self.next_btn = QPushButton("Get Started")
        self.next_btn.setProperty("class", "primary")
        self.next_btn.setMinimumSize(150, 38)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        self._perm_timer = QTimer(self)
        self._perm_timer.setInterval(1000)
        self._perm_timer.timeout.connect(self._refresh_perms)

        self._go(0)

    # ── navigation ───────────────────────────────────────────────────────────

    def _go(self, idx: int):
        idx = max(0, min(idx, self.stack.count() - 1))
        self.stack.setCurrentIndex(idx)
        n = self.stack.count()
        self.dots.setText("  ".join("●" if i == idx else "○" for i in range(n)))
        self.back_btn.setVisible(idx > 0)
        self.next_btn.setText(
            "Get Started" if idx == 0 else
            "Open Neuratect" if idx == n - 1 else "Continue")
        if idx >= 4:
            # Permissions page: opening the camera here is what triggers the
            # macOS prompt at the moment the user expects it.
            self.engine.start_camera()
            self._refresh_perms()
            self._perm_timer.start()
        else:
            self._perm_timer.stop()

    def _next(self):
        if self.stack.currentIndex() == self.stack.count() - 1:
            self.finished.emit()
        else:
            self._go(self.stack.currentIndex() + 1)

    # ── pages ────────────────────────────────────────────────────────────────

    def _page_welcome(self) -> QWidget:
        page = QWidget()
        lay = _v(page, spacing=10)
        lay.addStretch(3)
        icon = QLabel()
        icon.setPixmap(handicon.gesture_pixmap("open_palm", 130))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)
        title = make_label("Welcome to Neuratect", None, 34, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("Control your Mac with your hands.\n"
                         "No mouse, no keyboard — a camera and two hands.",
                         "dim", 16)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addStretch(4)
        return page

    def _page_hands(self) -> QWidget:
        page = QWidget()
        lay = _v(page, spacing=14)
        lay.addStretch(1)
        title = make_label("Two hands, two jobs", None, 28, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("The split stays the same everywhere, so it becomes muscle memory.",
                         "dim", 15)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(14)

        row = QHBoxLayout()
        row.setSpacing(16)
        row.addStretch()
        # mirrored like your own hands: left hand's card on the left
        row.addWidget(self._hand_card(
            "Left hand", "The commander", theme.HAND_LEFT_COLOR,
            "keyboard.fill",
            ["Poses fire commands",
             "Point to open Spotlight",
             "In games: fist becomes a walk stick",
             "Pinches jump, crouch, switch tools"]))
        row.addWidget(self._hand_card(
            "Right hand", "The pointer", theme.HAND_RIGHT_COLOR,
            "cursorarrow.motionlines",
            ["Move your hand — the cursor follows",
             "Pinch index finger to click",
             "Pinch middle finger to right-click",
             "Open your palm to lift the “mouse”"]))
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch(2)
        return page

    def _hand_card(self, hand: str, role: str, color: str, symbol: str,
                   bullets: list[str]) -> Card:
        card = Card()
        card.setFixedWidth(330)
        lay = _v(card, (22, 20, 22, 20), 8)
        ic = QLabel()
        ic.setPixmap(theme.sf_symbol(symbol, 26, color))
        lay.addWidget(ic)
        lay.addWidget(make_label(hand, None, 12, QFont.Weight.DemiBold, color))
        lay.addWidget(make_label(role, None, 20, QFont.Weight.Bold))
        lay.addSpacing(6)
        for b in bullets:
            lay.addWidget(make_label(f"·  {b}", "dim", 14))
        return card

    def _page_gestures(self) -> QWidget:
        page = QWidget()
        lay = _v(page, spacing=10)
        lay.addStretch(1)
        title = make_label("Gestures come preloaded", None, 28, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("Every icon is a real recorded pose. Record your own later — "
                         "they work the moment you save them.", "dim", 15)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(16)

        grid_row = QHBoxLayout()
        grid_row.addStretch()
        host = QWidget()
        host.setFixedWidth(640)
        flow = FlowLayout(host, spacing=14)
        names = [n for n in self.engine.gesture_names() if n != "pinch"]
        for name in names + ["pinch_index"]:
            tile = Card()
            tile.setFixedSize(146, 138)
            tlay = _v(tile, (10, 12, 10, 12), 4)
            ic = QLabel()
            ic.setPixmap(handicon.gesture_pixmap(name, 76))
            ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tlay.addWidget(ic)
            label = ("Pinch = click" if name == "pinch_index"
                     else labels.gesture_label(name))
            tlay.addWidget(make_label(label, None, 13, QFont.Weight.DemiBold),
                           alignment=Qt.AlignmentFlag.AlignHCenter)
            flow.addWidget(tile)
        grid_row.addWidget(host)
        grid_row.addStretch()
        lay.addLayout(grid_row)
        lay.addStretch(2)
        return page

    def _page_modes(self) -> QWidget:
        page = QWidget()
        lay = _v(page, spacing=10)
        lay.addStretch(1)
        title = make_label("It knows what app you’re in", None, 28, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("The same gesture can do different things in different places. "
                         "Neuratect switches automatically.", "dim", 15)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(16)

        row = QHBoxLayout()
        row.setSpacing(14)
        row.addStretch()
        highlights = {
            "default":   [("Point (left)", "Spotlight"),
                          ("Hand moves", "Cursor follows"),
                          ("Pinch", "Click & drag")],
            "spotlight": [("Fist / palm (left)", "Arrow keys"),
                          ("Thumbs up (right)", "Return"),
                          ("Peace (right)", "Tab")],
            "minecraft": [("Fist (left)", "Walk · sprint"),
                          ("Fist (right)", "Camera look"),
                          ("Pinches", "Mine · jump · hotbar")],
        }
        for ctx in labels.CONTEXT_ORDER:
            card = Card()
            card.setFixedWidth(240)
            clay = _v(card, (18, 16, 18, 16), 6)
            ic = QLabel()
            ic.setPixmap(theme.sf_symbol(labels.context_symbol(ctx), 22, theme.ACCENT))
            clay.addWidget(ic)
            clay.addWidget(make_label(labels.context_label(ctx), None, 17,
                                      QFont.Weight.Bold))
            clay.addSpacing(4)
            for gesture, action in highlights[ctx]:
                r = QHBoxLayout()
                r.addWidget(make_label(gesture, "dim", 13))
                r.addStretch()
                r.addWidget(make_label(action, None, 13, QFont.Weight.Medium))
                clay.addLayout(r)
            row.addWidget(card)
        row.addStretch()
        lay.addLayout(row)

        note = make_label("Fully remappable in Mappings — per app, per hand, per gesture.",
                          "faint", 13)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addSpacing(10)
        lay.addWidget(note)
        lay.addStretch(2)
        return page

    def _page_permissions(self) -> QWidget:
        page = QWidget()
        lay = _v(page, spacing=10)
        lay.addStretch(1)
        title = make_label("Two permissions", None, 28, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("macOS asks once. Both are granted to the app you launched "
                         "Neuratect from.", "dim", 15)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(18)

        col_row = QHBoxLayout()
        col_row.addStretch()
        col_host = QWidget()
        col_host.setFixedWidth(560)
        col = _v(col_host, spacing=12)

        self._cam_card = self._perm_card(
            "camera.fill", "Camera", "Sees your hands. Nothing is recorded or leaves "
            "this Mac.")
        self._input_card = self._perm_card(
            "accessibility", "Input control",
            "Lets gestures press keys and move the mouse (Accessibility).",
            grant=self.engine.request_input_permission)
        col.addWidget(self._cam_card)
        col.addWidget(self._input_card)
        col_row.addWidget(col_host)
        col_row.addStretch()
        lay.addLayout(col_row)
        lay.addStretch(2)
        return page

    def _perm_card(self, symbol: str, title: str, detail: str, grant=None) -> Card:
        card = Card()
        lay = QHBoxLayout(card)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(14)
        ic = QLabel()
        ic.setPixmap(theme.sf_symbol(symbol, 22, theme.ACCENT))
        ic.setFixedWidth(34)
        lay.addWidget(ic)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(make_label(title, None, 15, QFont.Weight.DemiBold))
        d = make_label(detail, "dim", 13)
        d.setWordWrap(True)
        col.addWidget(d)
        lay.addLayout(col, stretch=1)
        status = make_label("", None, 13, QFont.Weight.DemiBold)
        lay.addWidget(status)
        card.status_label = status
        if grant:
            btn = QPushButton("Grant…")
            btn.clicked.connect(lambda: (grant(), self._refresh_perms()))
            lay.addWidget(btn)
            card.grant_btn = btn
        return card

    def _refresh_perms(self):
        cam = self.engine.camera_permission()
        cam_ok = cam is True
        self._cam_card.status_label.setText("✓ granted" if cam_ok else "waiting…")
        self._cam_card.status_label.setStyleSheet(
            f"color: {theme.GREEN if cam_ok else theme.ORANGE};")
        ok = self.engine.input_permission()
        self._input_card.status_label.setText("✓ granted" if ok else "not yet")
        self._input_card.status_label.setStyleSheet(
            f"color: {theme.GREEN if ok else theme.ORANGE};")
        if hasattr(self._input_card, "grant_btn"):
            self._input_card.grant_btn.setVisible(not ok)

    def _page_try(self) -> QWidget:
        page = QWidget()
        lay = _v(page, spacing=10)
        title = make_label("Try it", None, 28, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("Show a hand — make a fist, point, pinch. Watch it light up. "
                         "Nothing is controlled until you press Start on the Home screen.",
                         "dim", 15)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(8)
        cam = CameraView(self.engine, radius=16)
        lay.addWidget(cam, stretch=1)
        return page

    def showEvent(self, e):
        self._refresh_perms()
        super().showEvent(e)

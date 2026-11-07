"""
Settings: pointer + pinch tunables (persisted to data/settings.json, the same
file run.py reads), permissions, and onboarding replay.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSlider, QVBoxLayout, QWidget)

import run
import theme
from widgets import Card, Chip, hline, make_label


class SliderRow(QWidget):
    committed = Signal(float)

    def __init__(self, title: str, hint: str, lo: float, hi: float,
                 value: float, fmt: str = "{:.2f}", parent=None):
        super().__init__(parent)
        self._lo, self._hi, self._fmt = lo, hi, fmt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.addWidget(make_label(title, None, 13, QFont.Weight.DemiBold))
        top.addStretch()
        self.value_label = make_label(fmt.format(value), None, 13,
                                      QFont.Weight.DemiBold, theme.ACCENT)
        top.addWidget(self.value_label)
        lay.addLayout(top)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._to_int(value))
        self.slider.valueChanged.connect(
            lambda v: self.value_label.setText(self._fmt.format(self._to_float(v))))
        self.slider.sliderReleased.connect(
            lambda: self.committed.emit(self.value()))
        lay.addWidget(self.slider)
        lay.addWidget(make_label(hint, "faint"))

    def _to_int(self, v: float) -> int:
        return round((v - self._lo) / (self._hi - self._lo) * 1000)

    def _to_float(self, i: int) -> float:
        return self._lo + (self._hi - self._lo) * i / 1000

    def value(self) -> float:
        return self._to_float(self.slider.value())

    def set_value(self, v: float):
        self.slider.setValue(self._to_int(v))


class SettingsPage(QWidget):
    replay_onboarding = Signal()

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("page")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 24)
        root.setSpacing(14)

        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(make_label("Settings", "h1"))
        col.addWidget(make_label("Tunables apply the next time control starts.",
                                 "dim"))
        root.addLayout(col)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 4, 0, 12)
        lay.setSpacing(14)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        s = engine.get_settings()

        # Pointer
        card = Card()
        clay = QVBoxLayout(card)
        clay.setContentsMargins(18, 14, 18, 16)
        clay.setSpacing(14)
        clay.addWidget(make_label("Pointer", "h2"))
        self.sens = SliderRow(
            "Sensitivity",
            "Cursor travel per hand-width of movement. Higher = faster cursor.",
            0.2, 2.0, s.get("gui_sensitivity", run.GUI_SENSITIVITY))
        self.sens.committed.connect(
            lambda v: engine.update_settings(gui_sensitivity=round(v, 3)))
        clay.addWidget(self.sens)
        lay.addWidget(card)

        # Pinch
        card = Card()
        clay = QVBoxLayout(card)
        clay.setContentsMargins(18, 14, 18, 16)
        clay.setSpacing(14)
        clay.addWidget(make_label("Pinch click", "h2"))
        self.engage = SliderRow(
            "Engage threshold",
            "Thumb-to-fingertip distance (as a fraction of hand size) that presses "
            "the button. Lower = a firmer pinch is needed.",
            0.05, 0.30, s.get("pinch_engage", run.PINCH_ENGAGE))
        self.release = SliderRow(
            "Release threshold",
            "Opening past this releases. Kept above engage so a click can’t flicker.",
            0.10, 0.50, s.get("pinch_release", run.PINCH_RELEASE))
        self.engage.committed.connect(self._commit_pinch)
        self.release.committed.connect(self._commit_pinch)
        clay.addWidget(self.engage)
        clay.addWidget(self.release)
        lay.addWidget(card)

        # Setup
        card = Card()
        clay = QVBoxLayout(card)
        clay.setContentsMargins(18, 14, 18, 16)
        clay.setSpacing(10)
        clay.addWidget(make_label("Setup", "h2"))

        cam_row = QHBoxLayout()
        cam_row.addWidget(make_label("Camera", None, 13, QFont.Weight.Medium))
        cam_row.addStretch()
        cam_row.addWidget(make_label(engine.camera_name(), "dim"))
        clay.addLayout(cam_row)
        clay.addWidget(hline())

        perm_row = QHBoxLayout()
        perm_row.addWidget(make_label("Input control (Accessibility)", None, 13,
                                      QFont.Weight.Medium))
        perm_row.addStretch()
        self.perm_chip = Chip("checking…", dot=theme.TEXT_FAINT)
        perm_row.addWidget(self.perm_chip)
        self.grant_btn = QPushButton("Grant…")
        self.grant_btn.clicked.connect(self._grant)
        perm_row.addWidget(self.grant_btn)
        clay.addLayout(perm_row)
        clay.addWidget(hline())

        ob_row = QHBoxLayout()
        ob_row.addWidget(make_label("Welcome tour", None, 13, QFont.Weight.Medium))
        ob_row.addStretch()
        replay = QPushButton("Replay")
        replay.clicked.connect(self.replay_onboarding.emit)
        ob_row.addWidget(replay)
        clay.addLayout(ob_row)
        lay.addWidget(card)
        lay.addStretch()

        self._refresh_permission()

    def _commit_pinch(self, _v: float):
        engage = round(self.engage.value(), 3)
        release = round(self.release.value(), 3)
        if release < engage + 0.04:      # hysteresis must stay open
            release = engage + 0.04
            self.release.set_value(release)
        self.engine.update_settings(pinch_engage=engage, pinch_release=release)

    def _grant(self):
        self.engine.request_input_permission()
        self._refresh_permission()

    def _refresh_permission(self):
        ok = self.engine.input_permission()
        self.perm_chip.set(text="granted" if ok else "not granted",
                           dot=theme.GREEN if ok else theme.ORANGE)
        self.grant_btn.setVisible(not ok)

    def showEvent(self, e):
        self._refresh_permission()
        super().showEvent(e)

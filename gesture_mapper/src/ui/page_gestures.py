"""
Gestures: the library as a card grid (icons drawn from the stored templates),
plus the guided recording flow that replaces manage_gestures.py.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

import theme
import labels
import handicon
from widgets import Card, CameraView, FlowLayout, make_label

# Poses run.py's interaction model depends on directly (joysticks, clutch,
# pinch templates) — deletable only with a strong warning.
_CORE = {"fist", "open_palm", "pinch"}


class GestureCard(Card):
    delete_requested = Signal(str)

    def __init__(self, engine, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setFixedSize(176, 188)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.addStretch()
        rm = QPushButton()
        rm.setProperty("class", "ghost")
        rm.setIcon(theme.sf_icon("trash", 12, theme.TEXT_FAINT))
        rm.setFixedSize(24, 24)
        rm.setToolTip("Delete gesture")
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.clicked.connect(lambda: self.delete_requested.emit(self.name))
        top.addWidget(rm)
        lay.addLayout(top)

        icon = QLabel()
        icon.setPixmap(handicon.gesture_pixmap(name, 88))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)

        lay.addWidget(make_label(labels.gesture_label(name), None, 14,
                                 QFont.Weight.DemiBold),
                      alignment=Qt.AlignmentFlag.AlignHCenter)

        n = len(engine.bindings_for_gesture(name))
        sub = (f"{n} binding{'s' if n != 1 else ''}" if n
               else "part of the pointer" if name in _CORE else "unbound")
        lay.addWidget(make_label(sub, "faint"),
                      alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch()


class PinchCard(Card):
    def __init__(self, engine, name: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(176, 188)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(4)
        lay.addSpacing(24)
        icon = QLabel()
        icon.setPixmap(handicon.gesture_pixmap(name, 88))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)
        lay.addWidget(make_label(labels.gesture_label(name), None, 14,
                                 QFont.Weight.DemiBold),
                      alignment=Qt.AlignmentFlag.AlignHCenter)
        n = len(engine.bindings_for_gesture(name))
        lay.addWidget(make_label(f"{n} binding{'s' if n != 1 else ''}" if n else "unbound",
                                 "faint"),
                      alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch()


class GesturesPage(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("page")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 24)
        root.setSpacing(14)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(make_label("Gestures", "h1"))
        col.addWidget(make_label("Each icon is the actual recorded pose. "
                                 "Record your own — it works everywhere instantly.",
                                 "dim"))
        head.addLayout(col)
        head.addStretch()
        rec = QPushButton("  Record New")
        rec.setProperty("class", "primary")
        rec.setIcon(theme.sf_icon("record.circle", 14, "white"))
        rec.setMinimumHeight(36)
        rec.setCursor(Qt.CursorShape.PointingHandCursor)
        rec.clicked.connect(self._record)
        head.addWidget(rec)
        root.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.body_lay = QVBoxLayout(body)
        self.body_lay.setContentsMargins(0, 4, 0, 12)
        self.body_lay.setSpacing(12)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        engine.library_changed.connect(self._rebuild)
        engine.mappings_changed.connect(self._rebuild)
        self._rebuild()

    def _rebuild(self):
        while self.body_lay.count():
            item = self.body_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        self.body_lay.addWidget(make_label("Poses", "h2"))
        flow_host = QWidget()
        flow = FlowLayout(flow_host, spacing=12)
        for name in self.engine.gesture_names():
            card = GestureCard(self.engine, name)
            card.delete_requested.connect(self._delete)
            flow.addWidget(card)
        self.body_lay.addWidget(flow_host)

        self.body_lay.addSpacing(6)
        self.body_lay.addWidget(make_label("Pinch signals", "h2"))
        self.body_lay.addWidget(make_label(
            "Touch your thumb to a fingertip — measured continuously, so pinches "
            "click and hold. Built in; bind them like any gesture.", "faint"))
        flow_host2 = QWidget()
        flow2 = FlowLayout(flow_host2, spacing=12)
        for name in labels.PINCH_GESTURES:
            flow2.addWidget(PinchCard(self.engine, name))
        self.body_lay.addWidget(flow_host2)
        self.body_lay.addStretch()

    def _clear_layout(self, lay):
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _delete(self, name: str):
        uses = self.engine.bindings_for_gesture(name)
        if name in _CORE:
            body = (f"“{labels.gesture_label(name)}” is part of the pointer and "
                    "joystick controls themselves — deleting it will break hand "
                    "control until it’s re-recorded.")
        elif uses:
            body = (f"“{labels.gesture_label(name)}” has {len(uses)} "
                    f"binding{'s' if len(uses) != 1 else ''}. They’ll stop firing.")
        else:
            body = f"Delete “{labels.gesture_label(name)}”?"
        box = QMessageBox(self)
        box.setWindowTitle("Delete gesture")
        box.setText(body)
        box.setStandardButtons(QMessageBox.StandardButton.Cancel |
                               QMessageBox.StandardButton.Yes)
        box.button(QMessageBox.StandardButton.Yes).setText("Delete")
        if box.exec() == QMessageBox.StandardButton.Yes:
            self.engine.delete_gesture(name)

    def _record(self):
        dlg = RecordDialog(self.engine, self)
        dlg.exec()


class RecordDialog(QDialog):
    """Guided recording: hold the pose, one click, 3-2-1, five samples."""

    SAMPLES = 5

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.samples = []
        self.setWindowTitle("Record a gesture")
        self.setModal(True)
        self.setFixedSize(520, 640)
        self.setStyleSheet(theme.QSS)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)

        lay.addWidget(make_label("Record a gesture", "h2"))
        sub = make_label(
            "Hold the pose in view, then capture. Five quick samples are "
            "averaged into a template.", "dim")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        self.camera = CameraView(engine, radius=12)
        self.camera.setMinimumSize(472, 340)
        lay.addWidget(self.camera)

        self.progress = make_label("", None, 13, QFont.Weight.DemiBold,
                                   theme.ACCENT)
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setMinimumHeight(20)
        lay.addWidget(self.progress)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name this gesture — e.g. “rock on”")
        self.name_edit.textChanged.connect(self._refresh_buttons)
        lay.addWidget(self.name_edit)

        row = QHBoxLayout()
        self.capture_btn = QPushButton("  Capture")
        self.capture_btn.setProperty("class", "primary")
        self.capture_btn.setIcon(theme.sf_icon("record.circle", 14, "white"))
        self.capture_btn.setMinimumHeight(38)
        self.capture_btn.clicked.connect(self._start_countdown)
        row.addWidget(self.capture_btn, stretch=1)

        self.save_btn = QPushButton("Save")
        self.save_btn.setMinimumHeight(38)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        row.addWidget(self.save_btn)
        lay.addLayout(row)

        self._countdown = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._update_progress()

    def _start_countdown(self):
        self.samples = []
        self._countdown = 3
        self.capture_btn.setEnabled(False)
        self.progress.setText("Hold the pose…  3")
        self._timer.start(700)

    def _tick(self):
        if self._countdown > 1:
            self._countdown -= 1
            self.progress.setText(f"Hold the pose…  {self._countdown}")
            return
        if self._countdown == 1:
            self._countdown = 0
            self._timer.setInterval(280)   # sampling cadence
            return
        # sampling phase
        sample = self.engine.capture_sample()
        if sample is None:
            self.progress.setText("No hand in view — show the pose to the camera")
            return
        self.samples.append(sample)
        self._update_progress()
        if len(self.samples) >= self.SAMPLES:
            self._timer.stop()
            self._timer.setInterval(700)
            self.capture_btn.setEnabled(True)
            self.capture_btn.setText("  Re-capture")
            self.progress.setText("Captured ✓ — name it and save")
            self.progress.setStyleSheet(f"color: {theme.GREEN};")
            self._refresh_buttons()

    def _update_progress(self):
        dots = "●" * len(self.samples) + "○" * (self.SAMPLES - len(self.samples))
        if self.samples and len(self.samples) < self.SAMPLES:
            self.progress.setText(f"Capturing  {dots}")
        elif not self.samples:
            self.progress.setText(dots)

    def _refresh_buttons(self):
        name = self._clean_name()
        self.save_btn.setEnabled(bool(name) and len(self.samples) >= self.SAMPLES)
        self.save_btn.setProperty("class", "primary" if self.save_btn.isEnabled() else "")
        self.save_btn.style().unpolish(self.save_btn)
        self.save_btn.style().polish(self.save_btn)

    def _clean_name(self) -> str:
        return self.name_edit.text().strip().lower().replace(" ", "_")

    def _save(self):
        name = self._clean_name()
        if name in self.engine.gesture_names():
            box = QMessageBox(self)
            box.setWindowTitle("Replace gesture")
            box.setText(f"“{labels.gesture_label(name)}” already exists. Replace it?")
            box.setStandardButtons(QMessageBox.StandardButton.Cancel |
                                   QMessageBox.StandardButton.Yes)
            if box.exec() != QMessageBox.StandardButton.Yes:
                return
        self.engine.save_gesture(name, self.samples)
        self.accept()

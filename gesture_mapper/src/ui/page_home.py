"""
Home: live camera hero, context + permission status, Start/Stop control.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

import theme
import labels
from widgets import CameraView, Chip, make_label


class HomePage(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("page")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 24)
        root.setSpacing(16)

        # header
        head = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_col.addWidget(make_label("Neuratect", "h1"))
        self.subtitle = make_label("Your hands are the interface.", "dim")
        title_col.addWidget(self.subtitle)
        head.addLayout(title_col)
        head.addStretch()

        self.ctx_chip = Chip("Desktop", dot=theme.TEXT_DIM)
        self.input_chip = Chip("Input control", dot=theme.TEXT_FAINT)
        self.cam_chip = Chip("Camera", dot=theme.TEXT_FAINT)
        for c in (self.cam_chip, self.input_chip, self.ctx_chip):
            head.addWidget(c)
        head.setSpacing(8)
        root.addLayout(head)

        # camera hero
        self.camera = CameraView(engine)
        root.addWidget(self.camera, stretch=1)

        # control bar
        bar = QHBoxLayout()
        bar.setSpacing(14)
        self.start_btn = QPushButton("  Start Control")
        self.start_btn.setProperty("class", "primary")
        self.start_btn.setIcon(theme.sf_icon("play.fill", 13, "white"))
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(170)
        f = self.start_btn.font()
        f.setPointSize(14)
        self.start_btn.setFont(f)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._toggle_control)
        bar.addWidget(self.start_btn)

        self.status = make_label("Preview only — gestures aren’t doing anything yet.",
                                 "dim")
        bar.addWidget(self.status)
        bar.addStretch()

        self.stats = make_label("", "faint")
        bar.addWidget(self.stats)
        root.addLayout(bar)

        engine.control_changed.connect(self._on_control_changed)
        engine.context_changed.connect(self._on_context)
        engine.camera_ready.connect(self._refresh_status)
        engine.camera_failed.connect(lambda _: self._refresh_status())
        engine.library_changed.connect(self._refresh_stats)
        engine.mappings_changed.connect(self._refresh_stats)
        self._refresh_status()
        self._refresh_stats()

    # ── slots ────────────────────────────────────────────────────────────────

    def _toggle_control(self):
        if self.engine.control_running:
            self.engine.stop_control()
        else:
            if not self.engine.input_permission():
                self.engine.request_input_permission()
                self.status.setText(
                    "macOS needs Accessibility access before gestures can "
                    "control anything — grant it in the prompt, then try again.")
                self._refresh_status()
                return
            self.engine.start_control()

    def _on_control_changed(self, running: bool):
        if running:
            self.start_btn.setText("  Stop Control")
            self.start_btn.setIcon(theme.sf_icon("stop.fill", 13, "white"))
            self.start_btn.setStyleSheet(
                f"background: {theme.RED}; color: white; border: none;")
            self.status.setText("Live — your gestures are controlling this Mac. "
                                "Switch to any app and go.")
            self.status.setStyleSheet(f"color: {theme.GREEN};")
        else:
            self.start_btn.setText("  Start Control")
            self.start_btn.setIcon(theme.sf_icon("play.fill", 13, "white"))
            self.start_btn.setStyleSheet("")
            self.status.setText("Preview only — gestures aren’t doing anything yet.")
            self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")

    def _on_context(self, ctx: str):
        self.ctx_chip.set(text=labels.context_label(ctx),
                          dot=theme.ACCENT if ctx != "default" else theme.TEXT_DIM)

    def _refresh_status(self):
        cam = self.engine.camera_permission()
        cam_ok = cam is True and self.engine.detector is not None
        self.cam_chip.set(text="Camera",
                          dot=theme.GREEN if cam_ok else
                              theme.ORANGE if cam is None else
                              theme.GREEN if cam else theme.RED)
        self.input_chip.set(
            text="Input control",
            dot=theme.GREEN if self.engine.input_permission() else theme.ORANGE)

    def _refresh_stats(self):
        n_g = len(self.engine.gesture_names())
        n_b = sum(len(t) for t in self.engine.mappings().values())
        self.stats.setText(f"{n_g} gestures · {n_b} bindings · {self.engine.camera_name()}")

    def showEvent(self, e):
        self._refresh_status()
        self._refresh_stats()
        super().showEvent(e)

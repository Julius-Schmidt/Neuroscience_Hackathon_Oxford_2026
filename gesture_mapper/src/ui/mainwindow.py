"""
Main window: sidebar + pages, with the onboarding view overlaid on first run.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QStackedWidget,
                               QVBoxLayout, QWidget)

import theme
import handicon
from services import Engine
from widgets import SidebarItem, make_label
from page_home import HomePage
from page_gestures import GesturesPage
from page_mappings import MappingsPage
from page_settings import SettingsPage
from onboarding import OnboardingView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuratect")
        self.resize(1180, 780)
        self.setMinimumSize(1020, 680)

        self.engine = Engine(self)

        # root: onboarding overlays the whole app on first run
        self.root = QStackedWidget()
        self.setCentralWidget(self.root)

        self.main_ui = self._build_main_ui()
        self.root.addWidget(self.main_ui)

        self.onboarding = OnboardingView(self.engine)
        self.root.addWidget(self.onboarding)
        self.onboarding.finished.connect(self._finish_onboarding)

        if self.engine.get_settings().get("onboarded"):
            self.root.setCurrentWidget(self.main_ui)
            # slight delay so the window paints before the camera spins up
            QTimer.singleShot(250, self.engine.start_camera)
        else:
            self.root.setCurrentWidget(self.onboarding)

        self._polished = False

    def _build_main_ui(self) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        slay = QVBoxLayout(sidebar)
        slay.setContentsMargins(12, 44, 12, 16)   # top clears the traffic lights
        slay.setSpacing(4)

        logo_row = QHBoxLayout()
        logo_row.setSpacing(8)
        logo_row.setContentsMargins(10, 0, 0, 10)
        mark = QLabel()
        mark.setPixmap(handicon.gesture_pixmap("open_palm", 24))
        logo_row.addWidget(mark)
        logo_row.addWidget(make_label("Neuratect", None, 16, QFont.Weight.Bold))
        logo_row.addStretch()
        slay.addLayout(logo_row)

        self.pages = QStackedWidget()
        self._nav: list[SidebarItem] = []
        for i, (title, symbol, page) in enumerate([
            ("Home",     "house.fill",              HomePage(self.engine)),
            ("Gestures", "hand.raised.fill",        GesturesPage(self.engine)),
            ("Mappings", "arrow.left.arrow.right",  MappingsPage(self.engine)),
            ("Settings", "gearshape.fill",          SettingsPage(self.engine)),
        ]):
            item = SidebarItem(f"  {title}", symbol)
            item.clicked.connect(lambda _=False, idx=i: self._select(idx))
            slay.addWidget(item)
            self._nav.append(item)
            self.pages.addWidget(page)
            if isinstance(page, SettingsPage):
                page.replay_onboarding.connect(self._replay_onboarding)

        slay.addStretch()

        self.live_dot = make_label("○  idle", "faint", 12)
        self.live_dot.setContentsMargins(12, 0, 0, 0)
        slay.addWidget(self.live_dot)
        self.engine.control_changed.connect(self._on_control)

        lay.addWidget(sidebar)
        lay.addWidget(self.pages, stretch=1)
        self._select(0)
        return wrap

    def _select(self, idx: int):
        for i, item in enumerate(self._nav):
            item.setChecked(i == idx)
        self.pages.setCurrentIndex(idx)

    def _on_control(self, running: bool):
        if running:
            self.live_dot.setText("●  live")
            self.live_dot.setStyleSheet(
                f"color: {theme.GREEN}; font-size: 12px; margin-left: 12px;")
        else:
            self.live_dot.setText("○  idle")
            self.live_dot.setStyleSheet(
                f"color: {theme.TEXT_FAINT}; font-size: 12px; margin-left: 12px;")

    def _finish_onboarding(self):
        self.engine.update_settings(onboarded=True)
        self.root.setCurrentWidget(self.main_ui)
        self.engine.start_camera()

    def _replay_onboarding(self):
        self.root.setCurrentWidget(self.onboarding)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._polished:
            self._polished = True
            theme.polish_macos_window(self)

    def closeEvent(self, e):
        self.engine.shutdown()
        super().closeEvent(e)

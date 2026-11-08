"""
Neuratect — the app.

    python3 gui.py

Onboards on first run, then: Home (live preview + Start/Stop control),
Gestures (library + recording), Mappings (per-context editor), Settings.
`python3 run.py` remains the headless way to run control from a terminal.
"""

import sys as _sys
if _sys.platform == "win32":   # Windows consoles default to cp1252: print the log's arrows instead of crashing on them
    for _s in (_sys.stdout, _sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)   # run.py — the control loop the GUI embeds
for sub in ("src/ui", "src/cv", "src/mapping-engine", "src/os-integration"):
    sys.path.insert(0, os.path.join(ROOT, sub))


def main():
    from PySide6.QtWidgets import QApplication
    QApplication.setApplicationName("Neuratect")
    QApplication.setApplicationDisplayName("Neuratect")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # consistent base for the QSS dark theme

    import theme
    app.setStyleSheet(theme.QSS)
    app.setFont(theme.app_font(13))

    from mainwindow import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

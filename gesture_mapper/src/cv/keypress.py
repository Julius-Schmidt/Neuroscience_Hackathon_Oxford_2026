"""
Non-blocking single-key listener for headless terminal loops.

cv2.imshow doesn't work on this machine, so there's no window to catch
cv2.waitKey() on — loops that used to rely on that (or on Ctrl+C) poll
this instead.

Usage:
    with KeyListener() as keys:
        while not keys.should_quit:
            ...
"""

import sys
import threading

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


class KeyListener:
    def __init__(self, quit_key: str = "q"):
        self._quit_key = quit_key.lower()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd = None
        self._old_settings = None

    def __enter__(self) -> "KeyListener":
        if termios and tty and sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        return self

    def _listen(self) -> None:
        while not self._quit.is_set():
            ch = sys.stdin.read(1)
            if ch.lower() == self._quit_key:
                self._quit.set()
                return

    @property
    def should_quit(self) -> bool:
        return self._quit.is_set()

    def __exit__(self, *_exc) -> None:
        self._quit.set()
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

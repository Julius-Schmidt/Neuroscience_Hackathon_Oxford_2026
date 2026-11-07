"""Non-blocking single-key reading on Windows (msvcrt) and Mac/Linux (stdin line)."""

from __future__ import annotations

import queue
import sys
import threading


class Keys:
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        try:
            import msvcrt  # Windows
            self._msvcrt = msvcrt
        except ImportError:
            self._msvcrt = None
            threading.Thread(target=self._stdin_loop, daemon=True).start()

    def _stdin_loop(self):
        for line in sys.stdin:                      # press a letter then Enter
            for ch in line.strip().lower():
                self.q.put(ch)

    def get(self) -> str | None:
        if self._msvcrt is not None:
            if self._msvcrt.kbhit():
                return self._msvcrt.getwch().lower()
            return None
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

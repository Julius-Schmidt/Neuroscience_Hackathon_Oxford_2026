import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("../mapping-engine", "../os-integration", "../cv"):
    p = os.path.normpath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))   # the hackathon repo (myomouse lives here)

from rules import Event  # noqa: E402


class Source:
    name = "source"

    def __init__(self):
        self.status = "stopped"
        self._stop = threading.Event()
        self._threads = []
        self.emit = lambda ev: None

    def start(self, emit) -> None:
        self.emit = emit
        self._stop.clear()
        self.status = "starting"
        self._start()

    def _start(self):
        raise NotImplementedError

    def _thread(self, fn, *args):
        t = threading.Thread(target=self._guard, args=(fn, *args), daemon=True)
        self._threads.append(t)
        t.start()

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except SystemExit as e:   # myomouse's "fix your setup" messages
            self.status = f"error: {e}"
        except Exception as e:
            self.status = f"error: {type(e).__name__}: {e}"

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self.status = "stopped"


class EdgeTracker:
    """Turns 'what is active now' snapshots into press / release events."""

    def __init__(self, source: str, emit):
        self.source, self.emit, self.active = source, emit, {}

    def set(self, group: str, name: str | None):
        """group = e.g. 'right' (one active pose per group); name None/'rest' = nothing."""
        old = self.active.get(group)
        new = None if name in (None, "rest", "none", "unknown") else name
        if old == new:
            return
        if old:
            self.emit(Event(self.source, f"{group}.{old}", "release"))
        if new:
            self.emit(Event(self.source, f"{group}.{new}", "press"))
        self.active[group] = new

    def clear(self):
        for g in list(self.active):
            self.set(g, None)

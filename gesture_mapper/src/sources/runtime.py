"""Runs sources -> RuleEngine -> OSExecutor. Used by the GUI's Blocks page and by
`python rules_run.py` (headless). Events go through one queue so actions run in order on a
single worker thread, whatever thread the device delivered them on."""

import queue
import threading

from base import Event  # noqa: F401  (re-export for callers)
import rules as R


class Runtime:
    def __init__(self, executor=None, data=None, context_fn=None):
        if executor is None:
            from actions import OSExecutor
            executor = OSExecutor()
        self.engine = R.RuleEngine(executor, data, context_fn)
        self.sources = {}
        self.q = queue.Queue()
        self.event_listeners = []          # callables(Event) - live UI feedback
        self._worker = None
        self._quit = threading.Event()

    def emit(self, ev) -> None:
        self.q.put(ev)

    def add_source(self, src) -> None:
        if src.name in self.sources:
            self.sources[src.name].stop()
        self.sources[src.name] = src
        src.start(self.emit)

    def remove_source(self, name) -> None:
        src = self.sources.pop(name, None)
        if src:
            src.stop()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._quit.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while not self._quit.is_set():
            try:
                ev = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            for cb in list(self.event_listeners):
                try:
                    cb(ev)
                except Exception:
                    pass
            self.engine.handle(ev)

    def stop(self) -> None:
        for name in list(self.sources):
            self.remove_source(name)
        self._quit.set()
        if self._worker:
            self._worker.join(timeout=2.0)
        self.engine.release_all()

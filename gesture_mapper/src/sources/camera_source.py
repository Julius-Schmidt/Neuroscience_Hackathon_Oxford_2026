"""Camera hand poses -> events 'camera:left.fist' etc. Reads a detector's last_gestures
(the GUI's live preview already pumps the camera; headless mode pumps its own)."""

import time

from base import EdgeTracker, Source


class CameraSource(Source):
    name = "camera"

    def __init__(self, detector=None, own_pump=False, hz=30.0):
        super().__init__()
        self.detector, self.own_pump, self.hz = detector, own_pump, hz

    def _start(self):
        if self.detector is None:
            from detector import GestureDetector
            self.detector, self.own_pump = GestureDetector(), True
        self._thread(self._loop)

    def _loop(self):
        edges = EdgeTracker("camera", self.emit)
        if self.own_pump:
            self.detector.open()
        self.status = "running"
        try:
            while not self._stop.is_set():
                gs = self.detector.next_gestures() if self.own_pump else list(self.detector.last_gestures)
                seen = {}
                for g in gs:
                    if g.hand in ("left", "right"):
                        seen[g.hand] = g.name
                for hand in ("left", "right"):
                    edges.set(hand, seen.get(hand))
                if not self.own_pump:
                    time.sleep(1.0 / self.hz)
        finally:
            edges.clear()
            if self.own_pump:
                self.detector.close()

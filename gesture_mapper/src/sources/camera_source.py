"""Camera hand poses -> events 'camera:left.fist' etc., plus thumb pinches
'camera:right.pinch_index' / pinch_middle / pinch_ring / pinch_pinky (thumb-to-finger, with
the same engage / release hysteresis as the control loop). Reads a detector's last_gestures
(the GUI's live preview already pumps the camera; headless mode pumps its own)."""

PINCH_ENGAGE, PINCH_RELEASE = 0.26, 0.40   # run.py defaults (data/settings.json can override)
FINGERS = ("index", "middle", "ring", "pinky")

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

    @staticmethod
    def _pinch_limits():
        try:
            import json, os
            path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "settings.json")
            st = json.load(open(path))
            return st.get("pinch_engage", PINCH_ENGAGE), st.get("pinch_release", PINCH_RELEASE)
        except Exception:
            return PINCH_ENGAGE, PINCH_RELEASE

    def pinches(self, edges, hand, g, engage, release):
        """Hysteresis per finger: engage below `engage`, release above `release`."""
        for f in FINGERS:
            key = f"{hand}.pinch_{f}"
            was = edges.active.get(key) is not None
            ratio = getattr(g, f"pinch_{f}", 10.0) if g is not None else 10.0
            if not was and ratio < engage:
                edges.set(key, "on")
            elif was and ratio > release:
                edges.set(key, None)

    def _loop(self):
        edges = EdgeTracker("camera", self.emit)
        pinch_edges = _PinchEdges(self.emit)
        engage, release = self._pinch_limits()
        if self.own_pump:
            self.detector.open()
        self.status = "running"
        try:
            while not self._stop.is_set():
                gs = self.detector.next_gestures() if self.own_pump else list(self.detector.last_gestures)
                seen, objs = {}, {}
                for g in gs:
                    if g.hand in ("left", "right"):
                        seen[g.hand], objs[g.hand] = g.name, g
                for hand in ("left", "right"):
                    edges.set(hand, seen.get(hand))
                    self.pinches(pinch_edges, hand, objs.get(hand), engage, release)
                if not self.own_pump:
                    time.sleep(1.0 / self.hz)
        finally:
            edges.clear()
            pinch_edges.clear()
            if self.own_pump:
                self.detector.close()


class _PinchEdges(EdgeTracker):
    """EdgeTracker whose group IS the event name ('left.pinch_index'), on/off."""

    def __init__(self, emit):
        super().__init__("camera", emit)

    def set(self, key, name):
        from base import Event
        old = self.active.get(key)
        new = None if name in (None, "rest", "none") else "on"
        if old == new:
            return
        self.emit(Event("camera", key, "press" if new else "release"))
        self.active[key] = new

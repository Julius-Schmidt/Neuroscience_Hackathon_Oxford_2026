"""Lab Streaming Layer markers -> events 'lsl:<marker>'. This is the EEG / BCI door:
almost every BCI toolkit (OpenViBE, BCI2000, MNE-LSL, PsychoPy, our own decoders) can push
a marker string such as 'left', 'select' or 'target_3' when it decides something.

    pip install pylsl       (optional; without it this source just reports so)
"""

import time

from base import Event, Source


class LslSource(Source):
    name = "lsl"

    def __init__(self, stream_type="Markers", rescan_s=3.0):
        super().__init__()
        self.stream_type, self.rescan_s = stream_type, rescan_s

    def _start(self):
        try:
            import pylsl  # noqa: F401
        except ImportError:
            self.status = "needs: pip install pylsl"
            return
        self._thread(self._loop)

    def _loop(self):
        import pylsl
        inlets = {}
        self.status = f"looking for '{self.stream_type}' streams"
        last_scan = 0.0
        while not self._stop.is_set():
            if time.monotonic() - last_scan > self.rescan_s:
                last_scan = time.monotonic()
                for info in pylsl.resolve_byprop("type", self.stream_type, timeout=0.5):
                    uid = info.uid()
                    if uid not in inlets:
                        inlets[uid] = pylsl.StreamInlet(info)
                self.status = f"listening to {len(inlets)} stream(s)" if inlets else \
                    f"no '{self.stream_type}' streams yet"
            got = False
            for inlet in list(inlets.values()):
                sample, _ts = inlet.pull_sample(timeout=0.0)
                while sample is not None:
                    got = True
                    marker = str(sample[0]).strip()
                    self.emit(Event("lsl", marker, "press"))
                    self.emit(Event("lsl", marker, "release"))
                    sample, _ts = inlet.pull_sample(timeout=0.0)
            if not got:
                time.sleep(0.01)

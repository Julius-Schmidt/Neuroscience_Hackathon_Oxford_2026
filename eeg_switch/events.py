"""Shared intent-event format and its LSL stream.

Format (agreed with Dylan, do not change without telling him):
    {"source": "eeg" | "cv" | "emg", "intent": str, "confidence": 0-1, "timestamp": LSL time}

Events travel as JSON strings on one LSL marker stream (config.EVENT_STREAM), so
any producer (EEG, camera, Myo) and any consumer (gesture mapper) can join.
"""

from __future__ import annotations

import json

from . import config as C

SOURCES = ("eeg", "cv", "emg")


def make_event(source: str, intent: str, confidence: float, timestamp: float) -> dict:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return {"source": source, "intent": str(intent),
            "confidence": float(min(1.0, max(0.0, confidence))), "timestamp": float(timestamp)}


def encode(event: dict) -> str:
    return json.dumps(event, separators=(",", ":"))


def decode(payload: str) -> dict:
    ev = json.loads(payload)
    missing = {"source", "intent", "confidence", "timestamp"} - set(ev)
    if missing:
        raise ValueError(f"event missing {missing}: {payload}")
    return ev


class EventOutlet:
    """Publish intent events. One per producer process."""

    def __init__(self, source: str):
        import pylsl
        self.pylsl, self.source = pylsl, source
        s = C.EVENT_STREAM
        info = pylsl.StreamInfo(s["name"], s["type"], 1, pylsl.IRREGULAR_RATE, "string",
                                f"{s['source_id']}_{source}")
        self.outlet = pylsl.StreamOutlet(info)

    def send(self, intent: str, confidence: float, timestamp: float | None = None) -> dict:
        ts = self.pylsl.local_clock() if timestamp is None else timestamp
        ev = make_event(self.source, intent, confidence, ts)
        self.outlet.push_sample([encode(ev)], ts)
        return ev


class EventInlet:
    """Receive intent events from every producer (EEG, CV, EMG)."""

    def __init__(self, timeout: float = 5.0):
        import pylsl
        self.pylsl = pylsl
        self.timeout = timeout
        self.inlets = []
        self.refresh()

    def refresh(self, timeout: float | None = None) -> int:
        """Pick up producers that started later. Returns number of connected producers."""
        t = self.timeout if timeout is None else timeout
        found = self.pylsl.resolve_byprop("name", C.EVENT_STREAM["name"], minimum=1, timeout=t)
        known = {i.info().source_id() for i in self.inlets}
        for info in found:
            if info.source_id() not in known:
                self.inlets.append(self.pylsl.StreamInlet(info))
        return len(self.inlets)

    def poll(self) -> list[dict]:
        out = []
        for inlet in self.inlets:
            while True:
                sample, _ = inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                try:
                    out.append(decode(sample[0]))
                except ValueError as e:
                    print(f"[events] dropped malformed event: {e}")
        return sorted(out, key=lambda e: e["timestamp"])

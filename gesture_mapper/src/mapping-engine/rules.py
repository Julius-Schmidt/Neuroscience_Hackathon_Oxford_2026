"""Rules: the general, Scratch-style mapping layer. ANY input -> ANY action.

The original mapper binds (context, hand, pose) -> Command for the camera. This layer is
the general form: an input source (camera poses, Myo bands, EEG/BCI markers over LSL, a
keyboard for testing, anything that can emit an event) produces Events; rules written as
blocks decide what happens:

    WHEN  <source> <event> <is pressed | is released | is held | changes>
    ONLY IF  <context is ...>  <mode is ...>          (optional, all must hold)
    DO    <action> <action> ...                        (run top to bottom)

"mode" is a variable the rules themselves set (Scratch-style), so a single gesture can mean
different things in different layers: e.g. double tap -> "set mode to typing", and in
typing mode the same fist types a letter instead of clicking.

Saved as plain JSON (data/rules.json) - the block editor writes it, and a hand-edited
file can't crash anything: unknown blocks are skipped with a log line.

The engine is pure logic: no OS calls. Actions go to an executor object (actions.py in
os-integration), so this file is fully testable with a fake executor.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("neuratect")

RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "rules.json")


@dataclass
class Event:
    source: str            # "camera", "myo", "lsl", "keyboard", ...
    name: str              # "left.fist", "right.wave_out", "SSVEP_target_3", ...
    kind: str = "press"    # "press" | "release" | "value"
    value: object = None   # for kind == "value": a number or (dx, dy)
    t: float = field(default_factory=time.monotonic)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.name}"


# ── block catalogue: what the editor offers, with parameters and defaults ─────
TRIGGERS = {
    "press":   "is pressed",
    "release": "is released",
    "hold":    "is held (actions held until released)",
    "value":   "changes (continuous value)",
}

CONDITIONS = {
    "context": {"label": "app in front is", "param": "value", "default": "default"},
    "mode":    {"label": "mode is", "param": "value", "default": "normal"},
    "not_mode": {"label": "mode is not", "param": "value", "default": "paused"},
}

ACTIONS = {
    "key_tap":     {"label": "press key",           "params": {"key": "space"}},
    "key_hold":    {"label": "hold key",            "params": {"key": "w"}},
    "mouse_click": {"label": "click mouse",         "params": {"button": "left"}},
    "mouse_hold":  {"label": "hold mouse button",   "params": {"button": "left"}},
    "mouse_move":  {"label": "move mouse by value x", "params": {"gain": 1.0, "flip_x": False, "flip_y": False}},
    "scroll":      {"label": "scroll",              "params": {"amount": -3}},
    "type_text":   {"label": "type",                "params": {"text": "hello"}},
    "shortcut":    {"label": "press shortcut",      "params": {"keys": "ctrl+c"}},
    "command":     {"label": "run built-in command", "params": {"name": "OpenSpotlight"}},
    "command_hold": {"label": "hold built-in command", "params": {"name": "HoldBreak"}},
    "open":        {"label": "open app / file / website", "params": {"target": "https://www.bbc.co.uk"}},
    "set_mode":    {"label": "set mode to",         "params": {"value": "normal"}},
    "toggle_mode": {"label": "toggle mode between", "params": {"a": "normal", "b": "paused"}},
    "wait":        {"label": "wait (seconds)",      "params": {"seconds": 0.2}},
    "sound":       {"label": "play sound",          "params": {"name": "pop"}},
    "say":         {"label": "show message",        "params": {"text": "hi!"}},
    "claw_step":   {"label": "claw: next step (sweep, stop, click)",
                    "params": {"button": "left", "sweep_s": 3.0, "lag_ms": 80, "refine": True, "restart": False}},
    "claw_drive":  {"label": "claw: glide while held (tap = click)", "params": {"axis": "x", "button": "left"}},
    "claw_cancel": {"label": "claw: cancel", "params": {}},
}

# event names each source can produce (the editor's dropdowns); free text is allowed too
SOURCE_EVENTS = {
    "camera": [f"{h}.{g}" for h in ("left", "right") for g in
               ("fist", "open_palm", "peace", "index_pointer", "thumbs_up", "shaaa",
                "pinch_index", "pinch_middle", "pinch_ring", "pinch_pinky")],
    "myo": [f"{s}.{g}" for s in ("right", "left") for g in
            ("fist", "index", "wave_in", "wave_out", "fingers_spread", "double_tap",
             "ulnar_deviation", "wrist_extension", "power_grip", "chuck_grip", "pinch_grip")]
           + ["right.pointer", "left.pointer"],
    "lsl": ["any marker", "left", "right", "select", "target_1", "target_2"],
    "keyboard": ["f8", "f9", "f10"],
}


def new_rule(source="myo", name="right.fist", trigger="press") -> dict:
    return {"id": uuid.uuid4().hex[:8], "enabled": True, "title": "",
            "when": {"source": source, "name": name, "trigger": trigger},
            "conditions": [], "actions": []}


def new_action(kind: str) -> dict:
    spec = ACTIONS[kind]
    return {"type": kind, **dict(spec["params"])}


def new_condition(kind: str) -> dict:
    return {"type": kind, "value": CONDITIONS[kind]["default"]}


def load(path: str = RULES_PATH) -> dict:
    if not os.path.exists(path):
        return {"rules": [], "mode": "normal"}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("rules", [])
        data.setdefault("mode", "normal")
        return data
    except Exception as e:
        log.warning(f"[rules] could not read {path}: {e} - starting empty")
        return {"rules": [], "mode": "normal"}


def save(data: dict, path: str = RULES_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _matches(when: dict, ev: Event) -> bool:
    if when.get("source") != ev.source:
        return False
    name = when.get("name", "")
    if name in ("any marker", "*", "any"):
        return True
    return name == ev.name


class RuleEngine:
    """Feed it Events; it runs the matching rules' actions through `executor`.

    executor must provide: run(action: dict, event: Event, engine) -> None for one-shot
    actions, and press(action) / release(action) for holdable ones (key_hold, mouse_hold).
    """

    def __init__(self, executor, data: dict | None = None, context_fn=None):
        self.executor = executor
        self.data = data if data is not None else load()
        self.mode = self.data.get("mode", "normal")
        self.context_fn = context_fn or (lambda: "default")
        self._held: dict[str, list[dict]] = {}       # event key -> held actions to release
        self._lock = threading.RLock()
        self.listeners = []                           # callables(rule_id, event) for live UI feedback

    # ── state ──
    def set_rules(self, data: dict) -> None:
        with self._lock:
            self.release_all()
            self.data = data
            self.mode = data.get("mode", self.mode)

    def set_mode(self, value: str) -> None:
        if value != self.mode:
            log.info(f"[rules] mode {self.mode} -> {value}")
        self.mode = value

    def _conditions_ok(self, rule: dict) -> bool:
        ctx = None
        for c in rule.get("conditions", []):
            t, v = c.get("type"), c.get("value")
            if t == "context":
                ctx = ctx if ctx is not None else self.context_fn()
                if ctx != v:
                    return False
            elif t == "mode":
                if self.mode != v:
                    return False
            elif t == "not_mode":
                if self.mode == v:
                    return False
        return True

    # ── events ──
    def handle(self, ev: Event) -> list[str]:
        """Process one event. Returns the ids of the rules that fired (for UI feedback)."""
        fired = []
        with self._lock:
            if ev.kind == "release" and ev.key in self._held:
                for a in self._held.pop(ev.key):
                    self._safe(self.executor.release, a)
            for rule in list(self.data.get("rules", [])):
                if not rule.get("enabled", True):
                    continue
                when = rule.get("when", {})
                if not _matches(when, ev):
                    continue
                trig = when.get("trigger", "press")
                if trig == "value":
                    if ev.kind != "value":
                        continue
                elif trig in ("press", "hold"):
                    if ev.kind != "press":
                        continue
                elif trig == "release":
                    if ev.kind != "release":
                        continue
                if not self._conditions_ok(rule):
                    continue
                fired.append(rule.get("id", "?"))
                self._run(rule, ev, hold=(trig == "hold"))
        for rid in fired:
            for cb in list(self.listeners):
                self._safe(cb, rid, ev)
        return fired

    def _run(self, rule: dict, ev: Event, hold: bool) -> None:
        for a in rule.get("actions", []):
            t = a.get("type")
            if t not in ACTIONS:
                log.info(f"[rules] unknown action {t!r} skipped")
                continue
            if t == "set_mode":
                self.set_mode(str(a.get("value", "normal")))
            elif t == "toggle_mode":
                self.set_mode(a.get("b") if self.mode == a.get("a") else a.get("a"))
            elif t in ("key_hold", "mouse_hold", "command_hold", "claw_drive"):
                self._safe(self.executor.press, a)
                if hold:
                    self._held.setdefault(ev.key, []).append(a)
                else:   # "is pressed" trigger with a hold block: tap it
                    self._safe(self.executor.release, a)
            elif t == "wait":
                time.sleep(min(float(a.get("seconds", 0)), 5.0))
            else:
                self._safe(self.executor.run, a, ev, self)

    def release_all(self) -> None:
        with self._lock:
            for acts in self._held.values():
                for a in acts:
                    self._safe(self.executor.release, a)
            self._held.clear()

    @staticmethod
    def _safe(fn, *args):
        try:
            fn(*args)
        except Exception as e:   # one broken block must never take the runtime down
            log.warning(f"[rules] {getattr(fn, '__name__', fn)} failed: {type(e).__name__}: {e}")

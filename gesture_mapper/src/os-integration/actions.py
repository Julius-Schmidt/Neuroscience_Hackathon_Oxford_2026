"""Executes rule-engine action blocks as real input. macOS (Quartz) and Windows (Win32 via
quartz_compat) through the same calls, so a rules.json works on both.

Key names: letters a-z, digits 0-9, space, enter, tab, esc, backspace, delete, up, down,
left, right, shift, ctrl, alt, cmd (Windows key on Windows), f1-f6.
"""

import logging
import os
import subprocess
import sys

try:
    import Quartz
except ImportError:
    from quartz_compat import Quartz

log = logging.getLogger("neuratect")

# macOS virtual key codes (quartz_compat translates these on Windows)
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22, "7": 26, "8": 28, "9": 25, "0": 29,
    "space": 49, "enter": 36, "return": 36, "tab": 48, "esc": 53, "escape": 53,
    "backspace": 51, "delete": 117, "left": 123, "right": 124, "down": 125, "up": 126,
    "shift": 56, "ctrl": 59, "control": 59, "alt": 58, "option": 58, "cmd": 55, "win": 55,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
}
_FLAGS = {"shift": Quartz.kCGEventFlagMaskShift, "ctrl": Quartz.kCGEventFlagMaskControl,
          "control": Quartz.kCGEventFlagMaskControl, "alt": Quartz.kCGEventFlagMaskAlternate,
          "cmd": Quartz.kCGEventFlagMaskCommand}


def _key(name: str, down: bool, flags: int = 0) -> None:
    code = KEYCODES.get(name.lower())
    if code is None:
        log.info(f"[actions] unknown key {name!r}")
        return
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    evt = Quartz.CGEventCreateKeyboardEvent(src, code, down)
    if flags:
        Quartz.CGEventSetFlags(evt, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def _mouse(button: str, down: bool) -> None:
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
    kind = {("left", True): Quartz.kCGEventLeftMouseDown, ("left", False): Quartz.kCGEventLeftMouseUp,
            ("right", True): Quartz.kCGEventRightMouseDown, ("right", False): Quartz.kCGEventRightMouseUp}
    btn = Quartz.kCGMouseButtonRight if button == "right" else Quartz.kCGMouseButtonLeft
    evt = Quartz.CGEventCreateMouseEvent(src, kind[(button if button == "right" else "left", down)], pos, btn)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


_carry = [0.0, 0.0]   # sub-pixel movement not sent yet


def _move(dx: float, dy: float) -> None:
    # games read the whole-pixel delta fields: keep the fractions, or slow aiming never arrives
    _carry[0] += dx
    _carry[1] += dy
    ix, iy = int(round(_carry[0], 6)), int(round(_carry[1], 6))   # round: 10 x 0.3 is 2.9999...
    if not ix and not iy:
        return
    _carry[0] -= ix
    _carry[1] -= iy
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    cur = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
    evt = Quartz.CGEventCreateMouseEvent(src, Quartz.kCGEventMouseMoved,
                                         Quartz.CGPoint(cur.x + ix, cur.y + iy), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX, ix)
    Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY, iy)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def _scroll(amount: int) -> None:
    if hasattr(Quartz, "backend"):              # Windows / stub
        Quartz.backend.scroll(int(amount))
        return
    evt = Quartz.CGEventCreateScrollWheelEvent(None, 0, 1, int(amount))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def _shortcut(spec: str) -> None:
    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return
    mods, key = parts[:-1], parts[-1]
    if sys.platform == "darwin":                 # flags on the key event, the macOS way
        flags = 0
        for m in mods:
            flags |= _FLAGS.get(m, 0)
        _key(key, True, flags)
        _key(key, False, flags)
        return
    for m in mods:                               # Windows: press the modifier keys themselves
        _key("win" if m == "cmd" else m, True)
    _key(key, True)
    _key(key, False)
    for m in reversed(mods):
        _key("win" if m == "cmd" else m, False)


_SHIFTED = {"!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8", "(": "9", ")": "0"}


def _type(text: str) -> None:
    for ch in text:
        if ch == " ":
            _key("space", True), _key("space", False)
        elif ch == "\n":
            _key("enter", True), _key("enter", False)
        elif ch.isalpha() and ch.lower() in KEYCODES:
            if ch.isupper():
                _key("shift", True)
            _key(ch.lower(), True), _key(ch.lower(), False)
            if ch.isupper():
                _key("shift", False)
        elif ch.isdigit():
            _key(ch, True), _key(ch, False)
        elif ch in _SHIFTED:
            _key("shift", True), _key(_SHIFTED[ch], True), _key(_SHIFTED[ch], False), _key("shift", False)


def _open(target: str) -> None:
    if sys.platform == "win32":
        os.startfile(target)                    # app path, file, or URL
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def click_at(x: float, y: float, button: str = "left") -> None:
    """Move the real cursor to screen pixel (x, y) and click there ('double' = two left clicks)."""
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    evt = Quartz.CGEventCreateMouseEvent(src, Quartz.kCGEventMouseMoved, Quartz.CGPoint(x, y),
                                         Quartz.kCGMouseButtonLeft)
    if hasattr(Quartz, "backend"):              # Windows / stub: always an absolute move here
        Quartz.backend.move_abs(x, y)
    else:
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    for _ in range(2 if button == "double" else 1):
        b = "right" if button == "right" else "left"
        _mouse(b, True)
        _mouse(b, False)


def _sound(name: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "..", "..", "..", "data", "sounds", f"{name}.wav"),
                  os.path.join(here, "..", "..", "data", "sounds", f"{name}.wav")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        log.info(f"[actions] no sound {name!r} (put {name}.wav in data/sounds)")
        return
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif sys.platform == "darwin":
        subprocess.Popen(["afplay", path])


class OSExecutor:
    """The real executor for RuleEngine. `messages` collects 'show message' blocks for the UI."""

    def __init__(self):
        self.messages = []
        self.hooks = {}          # UI-provided actions, e.g. "claw" (needs the GUI's overlay)

    @staticmethod
    def _cmd(a):
        from commands import by_name
        import executor
        return executor, by_name(str(a.get("name", "")))

    def press(self, a: dict) -> None:
        if a["type"] == "claw_drive":
            hook = self.hooks.get("claw")
            if hook:
                hook(dict(a, down=True))
        elif a["type"] == "command_hold":
            ex, cmd = self._cmd(a)
            if cmd:
                (ex.hold if cmd.mode == "held" else ex.tap)(cmd)
        elif a["type"] == "key_hold":
            _key(a.get("key", "w"), True)
        elif a["type"] == "mouse_hold":
            _mouse(a.get("button", "left"), True)

    def release(self, a: dict) -> None:
        if a["type"] == "claw_drive":
            hook = self.hooks.get("claw")
            if hook:
                hook(dict(a, down=False))
        elif a["type"] == "command_hold":
            ex, cmd = self._cmd(a)
            if cmd and cmd.mode == "held":
                ex.release(cmd)
        elif a["type"] == "key_hold":
            _key(a.get("key", "w"), False)
        elif a["type"] == "mouse_hold":
            _mouse(a.get("button", "left"), False)

    def run(self, a: dict, ev, engine) -> None:
        t = a["type"]
        if t == "key_tap":
            _key(a.get("key", "space"), True)
            _key(a.get("key", "space"), False)
        elif t == "mouse_click":
            _mouse(a.get("button", "left"), True)
            _mouse(a.get("button", "left"), False)
        elif t == "mouse_move":
            v = ev.value if isinstance(ev.value, (tuple, list)) else (ev.value or 0, 0)
            g = float(a.get("gain", 1.0))
            sx = -1.0 if a.get("flip_x") else 1.0     # depends on which way round the band is worn
            sy = -1.0 if a.get("flip_y") else 1.0
            _move(v[0] * g * sx, (v[1] if len(v) > 1 else 0) * g * sy)
        elif t == "scroll":
            _scroll(int(a.get("amount", -3)))
        elif t == "type_text":
            _type(str(a.get("text", "")))
        elif t == "shortcut":
            _shortcut(str(a.get("keys", "")))
        elif t == "command":
            from commands import by_name
            import executor
            cmd = by_name(str(a.get("name", "")))
            if cmd:
                executor.tap(cmd)
        elif t == "open":
            _open(str(a.get("target", "")))
        elif t == "sound":
            _sound(str(a.get("name", "pop")))
        elif t in ("claw_step", "claw_cancel"):
            hook = self.hooks.get("claw")
            if hook is None:
                log.info("[actions] the claw needs the GUI (python gui.py > Blocks)")
            else:
                hook(dict(a))
        elif t == "say":
            self.messages.append(str(a.get("text", "")))
            log.info(f"[rules] say: {a.get('text', '')}")

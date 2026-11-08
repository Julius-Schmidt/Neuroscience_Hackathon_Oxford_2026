"""A stand-in for macOS's `Quartz` module on Windows (and a recording stub elsewhere).

The mapper was written against Quartz (CGEvent for synthetic input, CGWindowList for
"which app is in front"). Rather than rewrite every call site, the few functions it uses
are re-implemented here with the Win32 API (ctypes, no extra packages), so on Windows:

    try:
        import Quartz
    except ImportError:
        from quartz_compat import Quartz

gives the same calls with the same meaning. macOS keeps using the real Quartz untouched.

Semantics kept from macOS:
  keyboard   macOS virtual key codes are translated to Windows virtual keys. Cmd+Space
             (Spotlight) becomes the Windows key (Start search). Other Cmd combos -> Ctrl.
  mouse      a move event goes to its absolute position while the cursor is visible
             (desktop, menus) and is sent as a RELATIVE move of its delta fields while
             the cursor is hidden (in-game camera look) - games read relative motion.
             Drag events are plain moves on Windows (no separate drag event type).
  windows    CGWindowListCopyWindowInfo returns visible top-level windows front to back,
             with kCGWindowOwnerName = the process name (e.g. "javaw" for Minecraft),
             kCGWindowLayer 0 and kCGWindowBounds.
On other systems (Linux CI / tests) the same API records posted events in Quartz.posted
instead of touching the OS.
"""

import sys
import types

IS_WIN = sys.platform == "win32"

# ── constants (values only need to be distinct; they mirror the CG names) ────
kCGEventSourceStateHIDSystemState = 1
kCGEventSourceStateCombinedSessionState = 0
kCGHIDEventTap = 0
kCGEventNull = 0
kCGEventLeftMouseDown = 1
kCGEventLeftMouseUp = 2
kCGEventRightMouseDown = 3
kCGEventRightMouseUp = 4
kCGEventMouseMoved = 5
kCGEventLeftMouseDragged = 6
kCGEventRightMouseDragged = 7
kCGEventKeyDown = 10
kCGEventKeyUp = 11
kCGEventScrollWheel = 22
kCGMouseButtonLeft = 0
kCGMouseButtonRight = 1
kCGMouseEventDeltaX = 4
kCGMouseEventDeltaY = 5
kCGKeyboardEventKeycode = 9
kCGEventFlagMaskShift = 0x00020000
kCGEventFlagMaskControl = 0x00040000
kCGEventFlagMaskAlternate = 0x00080000
kCGEventFlagMaskCommand = 0x00100000
kCGWindowListOptionOnScreenOnly = 1
kCGWindowListExcludeDesktopElements = 16
kCGNullWindowID = 0

# macOS virtual key code -> Windows virtual key (only the keys the mapper uses + letters/digits)
MAC_TO_VK = {
    0: 0x41, 1: 0x53, 2: 0x44, 3: 0x46, 4: 0x48, 5: 0x47, 6: 0x5A, 7: 0x58, 8: 0x43, 9: 0x56,
    11: 0x42, 12: 0x51, 13: 0x57, 14: 0x45, 15: 0x52, 16: 0x59, 17: 0x54, 31: 0x4F, 32: 0x55,
    34: 0x49, 35: 0x50, 37: 0x4C, 38: 0x4A, 40: 0x4B, 45: 0x4E, 46: 0x4D,
    18: 0x31, 19: 0x32, 20: 0x33, 21: 0x34, 23: 0x35, 22: 0x36, 26: 0x37, 28: 0x38, 25: 0x39, 29: 0x30,
    49: 0x20, 48: 0x09, 51: 0x08, 117: 0x2E, 36: 0x0D, 53: 0x1B,
    123: 0x25, 124: 0x27, 125: 0x28, 126: 0x26,
    56: 0xA0, 59: 0xA2, 58: 0xA4, 55: 0x5B,           # lshift, lctrl, lalt, cmd -> lwin
    122: 0x70, 120: 0x71, 99: 0x72, 118: 0x73, 96: 0x74, 97: 0x75,  # F1..F6
}
_EXTENDED = {0x25, 0x26, 0x27, 0x28, 0x2E, 0x5B}
VK_LWIN, VK_CONTROL = 0x5B, 0xA2


class CGPoint:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)

    def __iter__(self):
        return iter((self.x, self.y))

    def __repr__(self):
        return f"CGPoint({self.x:.1f}, {self.y:.1f})"


class _Size:
    def __init__(self, w, h):
        self.width, self.height = float(w), float(h)


class _Rect:
    def __init__(self, x, y, w, h):
        self.origin, self.size = CGPoint(x, y), _Size(w, h)


class _Event:
    def __init__(self, etype, **kw):
        self.type = etype
        self.flags = 0
        self.fields = {}
        self.__dict__.update(kw)


# ── backends ────────────────────────────────────────────────────────────────
class _StubBackend:
    """Non-Windows, non-mac: remember everything, touch nothing (for tests)."""

    def __init__(self):
        self.pos = CGPoint(400, 300)
        self.cursor_visible = True
        self.posted = []
        self.windows = []          # tests can fill: [{"kCGWindowOwnerName": ..., ...}]
        self.screen = (0, 0, 1920, 1080)

    def key(self, vk, down):
        self.posted.append(("key", vk, down))

    def move_abs(self, x, y):
        self.pos = CGPoint(x, y)
        self.posted.append(("move_abs", round(x), round(y)))

    def move_rel(self, dx, dy):
        self.pos = CGPoint(self.pos.x + dx, self.pos.y + dy)
        self.posted.append(("move_rel", int(dx), int(dy)))

    def button(self, which, down):
        self.posted.append(("button", which, down))

    def scroll(self, amount):
        self.posted.append(("scroll", amount))

    def cursor_pos(self):
        return CGPoint(self.pos.x, self.pos.y)

    def is_cursor_visible(self):
        return self.cursor_visible

    def window_list(self):
        return list(self.windows)

    def screen_rect(self):
        return self.screen


class _WinBackend(_StubBackend):
    def __init__(self):
        super().__init__()
        import ctypes
        from ctypes import wintypes as wt
        self.ct, self.wt = ctypes, wt
        self.u32 = ctypes.WinDLL("user32", use_last_error=True)
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        try:  # physical pixels everywhere, so SetCursorPos / window rects / screen agree
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                self.u32.SetProcessDPIAware()
            except Exception:
                pass
        ULONG_PTR = wt.WPARAM

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                        ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                        ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wt.DWORD), ("u", _U)]

        self.INPUT, self.MOUSEINPUT, self.KEYBDINPUT = INPUT, MOUSEINPUT, KEYBDINPUT

    def _send(self, inp):
        self.u32.SendInput(1, self.ct.byref(inp), self.ct.sizeof(inp))

    def key(self, vk, down):
        inp = self.INPUT(type=1)
        scan = self.u32.MapVirtualKeyW(vk, 0)
        flags = (0 if down else 0x0002) | (0x0001 if vk in _EXTENDED else 0)
        inp.u.ki = self.KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
        self._send(inp)

    def move_abs(self, x, y):
        self.u32.SetCursorPos(int(round(x)), int(round(y)))

    def move_rel(self, dx, dy):
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(dx=int(dx), dy=int(dy), mouseData=0, dwFlags=0x0001, time=0, dwExtraInfo=0)
        self._send(inp)

    def button(self, which, down):
        flag = {("left", True): 0x0002, ("left", False): 0x0004,
                ("right", True): 0x0008, ("right", False): 0x0010}[(which, down)]
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flag, time=0, dwExtraInfo=0)
        self._send(inp)

    def scroll(self, amount):
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(dx=0, dy=0, mouseData=int(amount) * 120 & 0xFFFFFFFF,
                                   dwFlags=0x0800, time=0, dwExtraInfo=0)
        self._send(inp)

    def cursor_pos(self):
        p = self.wt.POINT()
        self.u32.GetCursorPos(self.ct.byref(p))
        return CGPoint(p.x, p.y)

    def is_cursor_visible(self):
        ct, wt = self.ct, self.wt

        class CURSORINFO(ct.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("flags", wt.DWORD), ("hCursor", wt.HANDLE),
                        ("ptScreenPos", wt.POINT)]
        ci = CURSORINFO(cbSize=ct.sizeof(CURSORINFO))
        if not self.u32.GetCursorInfo(ct.byref(ci)):
            return True
        return bool(ci.flags & 1)

    def _proc_name(self, hwnd):
        ct, wt = self.ct, self.wt
        pid = wt.DWORD()
        self.u32.GetWindowThreadProcessId(hwnd, ct.byref(pid))
        h = self.k32.OpenProcess(0x1000, False, pid.value)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = ct.create_unicode_buffer(512)
            n = wt.DWORD(512)
            if self.k32.QueryFullProcessImageNameW(h, 0, buf, ct.byref(n)):
                name = buf.value.replace("/", "\\").split("\\")[-1]
                return name[:-4] if name.lower().endswith(".exe") else name
            return ""
        finally:
            self.k32.CloseHandle(h)

    def window_list(self):
        ct, wt = self.ct, self.wt
        out = []
        fg = self.u32.GetForegroundWindow()

        @ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def cb(hwnd, _lp):
            if not self.u32.IsWindowVisible(hwnd) or self.u32.IsIconic(hwnd):
                return True
            n = self.u32.GetWindowTextLengthW(hwnd)
            if n == 0 and hwnd != fg:
                return True
            title = ct.create_unicode_buffer(n + 1)
            self.u32.GetWindowTextW(hwnd, title, n + 1)
            r = wt.RECT()
            self.u32.GetWindowRect(hwnd, ct.byref(r))
            w, h = r.right - r.left, r.bottom - r.top
            if w <= 1 or h <= 1:
                return True
            out.append({"kCGWindowOwnerName": self._proc_name(hwnd), "kCGWindowName": title.value,
                        "kCGWindowLayer": 0,
                        "kCGWindowBounds": {"X": r.left, "Y": r.top, "Width": w, "Height": h},
                        "_foreground": hwnd == fg})
            return len(out) < 60
        self.u32.EnumWindows(cb, 0)   # top-level windows in z-order, front first
        out.sort(key=lambda d: not d["_foreground"])   # the focused window first, like macOS
        return out

    def screen_rect(self):
        return (0, 0, self.u32.GetSystemMetrics(0), self.u32.GetSystemMetrics(1))


_backend = _WinBackend() if IS_WIN else _StubBackend()


# ── the Quartz-shaped API ───────────────────────────────────────────────────
def CGEventSourceCreate(_state):
    return None


def CGEventCreate(_src):
    return _Event(kCGEventNull, pos=_backend.cursor_pos())


def CGEventGetLocation(evt):
    return evt.pos if getattr(evt, "pos", None) is not None else _backend.cursor_pos()


def CGEventCreateKeyboardEvent(_src, keycode, down):
    return _Event(kCGEventKeyDown if down else kCGEventKeyUp, keycode=keycode, down=bool(down), pos=None)


def CGEventCreateMouseEvent(_src, etype, pos, button):
    return _Event(etype, pos=CGPoint(*pos) if pos is not None else None, button=button)


def CGEventSetFlags(evt, flags):
    evt.flags = flags


def CGEventSetType(evt, etype):
    evt.type = etype


def CGEventGetType(evt):
    return evt.type


def CGEventSetIntegerValueField(evt, field, value):
    evt.fields[field] = value


def CGEventGetIntegerValueField(evt, field):
    if field == kCGKeyboardEventKeycode:
        return getattr(evt, "keycode", 0)
    return evt.fields.get(field, 0)


def CGEventSourceFlagsState(_state):
    return 0   # Windows mouse events already carry the live modifier state


def _post_key(evt):
    code = evt.keycode
    if evt.flags & kCGEventFlagMaskCommand:
        if code == 49:            # Cmd+Space (Spotlight) -> Windows key opens Start search
            if evt.down:
                _backend.key(VK_LWIN, True)
                _backend.key(VK_LWIN, False)
            return
        if evt.down:              # other Cmd shortcuts -> Ctrl
            _backend.key(VK_CONTROL, True)
    vk = MAC_TO_VK.get(code)
    if vk is None:
        return
    _backend.key(vk, evt.down)
    if evt.flags & kCGEventFlagMaskCommand and not evt.down:
        _backend.key(VK_CONTROL, False)


def CGEventPost(_tap, evt):
    t = evt.type
    if t in (kCGEventKeyDown, kCGEventKeyUp):
        _post_key(evt)
    elif t in (kCGEventMouseMoved, kCGEventLeftMouseDragged, kCGEventRightMouseDragged):
        dx = evt.fields.get(kCGMouseEventDeltaX, 0)
        dy = evt.fields.get(kCGMouseEventDeltaY, 0)
        if not _backend.is_cursor_visible() and (dx or dy):
            _backend.move_rel(dx, dy)      # in-game look: games read relative motion
        elif evt.pos is not None:
            _backend.move_abs(evt.pos.x, evt.pos.y)
    elif t == kCGEventLeftMouseDown:
        _backend.button("left", True)
    elif t == kCGEventLeftMouseUp:
        _backend.button("left", False)
    elif t == kCGEventRightMouseDown:
        _backend.button("right", True)
    elif t == kCGEventRightMouseUp:
        _backend.button("right", False)


def CGCursorIsVisible():
    return _backend.is_cursor_visible()


def CGMainDisplayID():
    return 0


def CGDisplayBounds(_display):
    return _Rect(*_backend.screen_rect())


def CGWindowListCopyWindowInfo(_options, _window):
    return _backend.window_list()


def CGPreflightPostEventAccess():
    return True    # Windows needs no Accessibility grant to synthesise input


def CGRequestPostEventAccess():
    return True


# expose as a module object so `Quartz.X` works exactly like the real thing
Quartz = types.ModuleType("Quartz")
Quartz.__dict__.update({k: v for k, v in globals().items() if k.startswith(("kCG", "CG"))})
Quartz.backend = _backend
Quartz.posted = _backend.posted

"""
OS integration: executes Commands as macOS actions.
All macOS API calls live here.
Requires Accessibility permission for the running terminal/app.

tap(command)     — fire once
hold(command)    — press and hold (for "held" mode commands)
release(command) — release a held key/button
"""

import logging
import Quartz
try:
    from .commands import Command
except ImportError:
    from commands import Command

log = logging.getLogger("neuratect")

# ── Virtual key codes ────────────────────────────────────────────────────────
_KEY = {
    "w": 13, "a": 0, "s": 1, "d": 2,
    "e": 14, "q": 12,
    "space": 49, "tab": 48, "delete": 51, "return": 36, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "1": 18, "2": 19, "3": 20, "4": 21, "5": 23,
    "6": 22, "7": 26, "8": 28, "9": 25,
    "shift": 56, "ctrl": 59,   # left-hand modifiers (Minecraft sneak / sprint)
}

# Command → key string
_CMD_KEY = {
    "ArrowUp":    "up",
    "ArrowDown":  "down",
    "ArrowLeft":  "left",
    "ArrowRight": "right",
    "Enter":      "return",
    "Delete":     "delete",
    "Tab":        "tab",
    "Space":      "space",
    "Jump":       "space",
    "Inventory":  "e",
    "Escape":     "esc",
    "HoldW":      "w",
    "HoldA":      "a",
    "HoldS":      "s",
    "HoldD":      "d",
    "Crouch":     "shift",
    "Sprint":     "ctrl",
}

# Minecraft hotbar slot tracking (1-9)
_hotbar_slot = 1


def _press_slot(slot: int) -> None:
    _key_event(str(slot), True)
    _key_event(str(slot), False)


def _key_event(key: str, down: bool) -> None:
    code = _KEY[key]
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    evt = Quartz.CGEventCreateKeyboardEvent(src, code, down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def _mouse_event(event_type: int, pos=None) -> None:
    if pos is None:
        # Post to current cursor position
        pos = Quartz.CGEventGetLocation(
            Quartz.CGEventCreate(
                Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            )
        )
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    evt = Quartz.CGEventCreateMouseEvent(src, event_type, pos, Quartz.kCGMouseButtonLeft)
    # Shift-click (inventory quick-move) needs the click to carry the modifier
    # that Crouch is holding. Measured 2026-08-08: CGEventCreateMouseEvent
    # already inherits the live session modifier state on this macOS, so this
    # is belt-and-braces rather than the fix — but it is cheap, it makes the
    # dependency explicit, and it keeps working if that inheritance ever
    # changes. Picks up both our own held keys and the real keyboard's.
    flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateCombinedSessionState)
    if flags:
        Quartz.CGEventSetFlags(evt, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def tap(command: Command) -> None:
    """Fire a one-shot command."""
    if command.name == "OpenSpotlight":
        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        down = Quartz.CGEventCreateKeyboardEvent(src, _KEY["space"], True)
        up   = Quartz.CGEventCreateKeyboardEvent(src, _KEY["space"], False)
        Quartz.CGEventSetFlags(down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetFlags(up,   Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
        return

    if command.name in ("HotbarNext", "HotbarPrev"):
        global _hotbar_slot
        if command.name == "HotbarNext":
            _hotbar_slot = _hotbar_slot % 9 + 1
        else:
            _hotbar_slot = (_hotbar_slot - 2) % 9 + 1
        log.info(f"[hotbar] slot {_hotbar_slot}")
        _press_slot(_hotbar_slot)
        return

    key = _CMD_KEY.get(command.name)
    if key:
        _key_event(key, True)
        _key_event(key, False)


def hold(command: Command) -> None:
    """Press and hold — call release() when gesture stops."""
    if command.name == "HoldBreak":
        _mouse_event(Quartz.kCGEventLeftMouseDown)
        return
    if command.name == "HoldPlace":
        _mouse_event(Quartz.kCGEventRightMouseDown)
        return
    key = _CMD_KEY.get(command.name)
    if key:
        _key_event(key, True)


# While a mouse button is held, cursor motion must be posted as a *dragged*
# event, not kCGEventMouseMoved. AppKit drives drags off mouseDragged: /
# rightMouseDragged: — a mouseMoved with the button down still moves the
# hardware cursor (so it looks like it works), but no drag session ever sees
# the new position: a dragged window only jumps to the cursor on mouse-up, and
# Finder never starts a file drag at all. GLFW/Minecraft handles both, so this
# is safe in every context.
_DRAG_EVENT = {
    "left":  (Quartz.kCGEventLeftMouseDragged,  Quartz.kCGMouseButtonLeft),
    "right": (Quartz.kCGEventRightMouseDragged, Quartz.kCGMouseButtonRight),
}


def move_event_type(button: str | None) -> tuple[int, int]:
    """(CGEventType, CGMouseButton) for moving the cursor with `button` held
    ("left" / "right" / None for no button)."""
    return _DRAG_EVENT.get(
        button, (Quartz.kCGEventMouseMoved, Quartz.kCGMouseButtonLeft))


def post_cursor_move(x: float, y: float, dx: float, dy: float,
                     button: str | None = None) -> None:
    """Move the cursor to absolute (x, y), carrying (dx, dy) as the event's
    delta fields (Minecraft reads those, not the position). Posts a drag event
    instead of a move when a button is held — see _DRAG_EVENT above."""
    etype, mbutton = move_event_type(button)
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    evt = Quartz.CGEventCreateMouseEvent(src, etype, Quartz.CGPoint(x, y), mbutton)
    Quartz.CGEventSetType(evt, etype)
    Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX, int(dx))
    Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY, int(dy))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def move_mouse_relative(dx: float, dy: float) -> None:
    """
    Move mouse by (dx, dy) pixels. Sets delta fields so GLFW/Minecraft
    picks up the movement via NSEvent.deltaX/deltaY.
    """
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    current = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
    new_pos = Quartz.CGPoint(current.x + dx, current.y + dy)
    evt = Quartz.CGEventCreateMouseEvent(src, Quartz.kCGEventMouseMoved, new_pos, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventSetType(evt, Quartz.kCGEventMouseMoved)
    Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX, int(dx))
    Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY, int(dy))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def release(command: Command) -> None:
    """Release a held key/button."""
    if command.name == "HoldBreak":
        _mouse_event(Quartz.kCGEventLeftMouseUp)
        return
    if command.name == "HoldPlace":
        _mouse_event(Quartz.kCGEventRightMouseUp)
        return
    key = _CMD_KEY.get(command.name)
    if key:
        _key_event(key, False)

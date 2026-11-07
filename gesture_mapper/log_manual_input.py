"""
Baseline input logger — records real keyboard/mouse/trackpad input system-wide,
for comparing manual play against gesture-driven play (run.py).

Requires macOS Input Monitoring permission for whatever runs this (System
Settings -> Privacy & Security -> Input Monitoring) — a different permission
from the Accessibility one run.py needs, since this only OBSERVES input
rather than sending it. Grant it to Terminal (or whichever app runs this),
then re-run.

    python log_manual_input.py
    Ctrl+C to stop.

Logs to logs/baseline.log in the same timestamped format run.py uses, for
side-by-side comparison. Discrete key/click/scroll events are always logged.
Cursor position is sampled at ~10Hz (only when it has actually moved) rather
than logging every raw move event — enough to see the path/overshoot without
drowning the log the way per-event logging would.
"""

import time
import logging
import Quartz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/baseline.log", mode="w"),
    ],
)
log = logging.getLogger("baseline")

_KEY_NAMES = {
    13: "W", 0: "A", 1: "S", 2: "D", 14: "E", 12: "Q",
    49: "Space", 48: "Tab", 51: "Delete", 36: "Return", 53: "Esc",
    123: "Left", 124: "Right", 125: "Down", 126: "Up",
    18: "1", 19: "2", 20: "3", 21: "4", 23: "5",
    22: "6", 26: "7", 28: "8", 25: "9",
}

_POSITION_HZ       = 10
_POSITION_INTERVAL = 1.0 / _POSITION_HZ
_POSITION_MIN_MOVE = 3.0   # px — skip logging if it barely moved since last sample

_last_pos_log_t = 0.0
_last_pos       = None


def _key_name(event) -> str:
    code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
    return _KEY_NAMES.get(code, f"key({code})")


def _callback(proxy, etype, event, refcon):
    global _last_pos_log_t, _last_pos

    if etype == Quartz.kCGEventKeyDown:
        log.info(f"[key] {_key_name(event)} down")
    elif etype == Quartz.kCGEventKeyUp:
        log.info(f"[key] {_key_name(event)} up")
    elif etype == Quartz.kCGEventLeftMouseDown:
        pos = Quartz.CGEventGetLocation(event)
        log.info(f"[mouse] left down  ({pos.x:.0f},{pos.y:.0f})")
    elif etype == Quartz.kCGEventLeftMouseUp:
        pos = Quartz.CGEventGetLocation(event)
        log.info(f"[mouse] left up    ({pos.x:.0f},{pos.y:.0f})")
    elif etype == Quartz.kCGEventRightMouseDown:
        pos = Quartz.CGEventGetLocation(event)
        log.info(f"[mouse] right down ({pos.x:.0f},{pos.y:.0f})")
    elif etype == Quartz.kCGEventRightMouseUp:
        pos = Quartz.CGEventGetLocation(event)
        log.info(f"[mouse] right up   ({pos.x:.0f},{pos.y:.0f})")
    elif etype == Quartz.kCGEventScrollWheel:
        dy = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventDeltaAxis1)
        if dy != 0:
            log.info(f"[scroll] {dy:+d}")
    elif etype in (Quartz.kCGEventMouseMoved,
                   Quartz.kCGEventLeftMouseDragged,
                   Quartz.kCGEventRightMouseDragged):
        now = time.monotonic()
        pos = Quartz.CGEventGetLocation(event)
        if now - _last_pos_log_t >= _POSITION_INTERVAL:
            moved_enough = (
                _last_pos is None
                or abs(pos.x - _last_pos[0]) + abs(pos.y - _last_pos[1]) >= _POSITION_MIN_MOVE
            )
            if moved_enough:
                log.info(f"[pos] ({pos.x:.0f},{pos.y:.0f})")
                _last_pos_log_t = now
                _last_pos = (pos.x, pos.y)

    return event


def main():
    mask = 0
    for e in (
        Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp,
        Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp,
        Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp,
        Quartz.kCGEventScrollWheel, Quartz.kCGEventMouseMoved,
        Quartz.kCGEventLeftMouseDragged, Quartz.kCGEventRightMouseDragged,
    ):
        mask |= Quartz.CGEventMaskBit(e)

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGHIDEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,   # observe only — never modifies/blocks events
        mask,
        _callback,
        None,
    )
    if tap is None:
        raise SystemExit(
            "Failed to create event tap. Grant Input Monitoring permission to "
            "this terminal in System Settings > Privacy & Security > Input "
            "Monitoring, then re-run."
        )

    loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), loop_source, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)

    log.info("Logging real keyboard/mouse/trackpad input. Ctrl+C to stop.")
    try:
        Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        pass
    log.info("Stopped.")


if __name__ == "__main__":
    main()

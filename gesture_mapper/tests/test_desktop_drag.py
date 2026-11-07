"""
The reported bug: outside Minecraft, dragging a window only moved it on
release, and desktop files wouldn't drag at all — because cursor motion was
posted as kCGEventMouseMoved even while a button was held, so no AppKit drag
session ever saw the movement.

Drives run.main()'s real loop in "default" (desktop) context with a scripted
right hand, and asserts what the mouse thread posts: plain moves with no button
down, *drag* events once a pinch has engaged. Nothing reaches the OS — the
posting function is stubbed, so the real cursor never moves.
"""

import sys, os, time, math

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src', 'cv'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mapping-engine'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'os-integration'))

import run
from gesture import Gesture
from executor import move_event_type
import Quartz

ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok = ok and cond


# Right hand oscillating (so the pointer keeps moving), pinching partway
# through and releasing at the end. Oscillating rather than drifting one way on
# purpose: the pointer seeds from wherever the real cursor happens to be, so a
# one-way hand can pin the target against the clamp edge and stop producing
# posts at all — which made an earlier version of this test flaky.
FRAMES = 150
PINCH_FROM, PINCH_TO = 50, 110


def frame(i):
    pinching = PINCH_FROM <= i < PINCH_TO
    return Gesture(name="none", confidence=0.9, timestamp=0, hand="right",
                   wrist_x=0.5 + 0.05 * math.sin(i * 0.35), wrist_y=0.5,
                   hand_scale=0.15, aspect=4/3,
                   pinch_index=0.04 if pinching else 1.2,
                   pinch_middle=1.4, pinch_ring=1.4, pinch_pinky=1.4)


posts: list[tuple[int, str | None]] = []   # (frame index, button)
_i = [0]


class FakeDetector:
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def next_gestures(self):
        if _i[0] >= FRAMES:
            FakeKeys.quit = True
            return []
        g = frame(_i[0])
        _i[0] += 1
        time.sleep(1 / 60)
        return [g]


class FakeKeys:
    quit = False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def should_quit(self): return FakeKeys.quit


run.GestureDetector = FakeDetector
run.KeyListener = FakeKeys
run.get_active_context = lambda: "default"
run.hold = lambda cmd: None
run.release = lambda cmd: None
run.tap = lambda cmd: None
run._pointer_bounds = lambda ctx: (0.0, 0.0, 1000.0, 800.0)
run.post_cursor_move = lambda x, y, dx, dy, button=None: posts.append((_i[0], button))

run.main()

# ── Assertions ───────────────────────────────────────────────────────────────
before = [b for i, b in posts if i < PINCH_FROM]
during = [b for i, b in posts if PINCH_FROM + 12 <= i < PINCH_TO]   # past CLICK_FREEZE
after  = [b for i, b in posts if i >= PINCH_TO + 12]

print(f"  posts: {len(posts)} total — {len(before)} before pinch, "
      f"{len(during)} during, {len(after)} after release")

check(len(posts) > 10, f"the pointer actually moved ({len(posts)} posts)")
check(before and all(b is None for b in before),
      f"no button held → plain moves ({set(before)})")
check(during and all(b == "left" for b in during),
      f"pinch held → every post is a LEFT DRAG, not a move ({set(during)})")
check(after and all(b is None for b in after),
      f"after release → back to plain moves ({set(after)})")
check(run._mouse_button[0] is None, "button state cleared at exit")

# ── Event types the button names resolve to ──────────────────────────────────
for button, want, label in (
        (None,      Quartz.kCGEventMouseMoved,         "MouseMoved"),
        ("left",    Quartz.kCGEventLeftMouseDragged,   "LeftMouseDragged"),
        ("right",   Quartz.kCGEventRightMouseDragged,  "RightMouseDragged"),
        ("garbage", Quartz.kCGEventMouseMoved,         "MouseMoved (fallback)")):
    etype, _ = move_event_type(button)
    check(etype == want, f"button={button!r} → {label}")

check(move_event_type("right")[1] == Quartz.kCGMouseButtonRight,
      "a right drag carries button 1, not button 0")

# Deltas must survive on a drag event — Minecraft's camera reads those fields,
# so the drag change must not have cost the gameplay path its input.
etype, mbtn = move_event_type("left")
src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
evt = Quartz.CGEventCreateMouseEvent(src, etype, Quartz.CGPoint(400, 300), mbtn)
Quartz.CGEventSetType(evt, etype)
Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX, 7)
Quartz.CGEventSetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY, -3)
check(Quartz.CGEventGetType(evt) == Quartz.kCGEventLeftMouseDragged
      and Quartz.CGEventGetIntegerValueField(evt, Quartz.kCGMouseEventDeltaX) == 7
      and Quartz.CGEventGetIntegerValueField(evt, Quartz.kCGMouseEventDeltaY) == -3,
      "drag event keeps its delta fields (Minecraft camera still gets input)")
loc = Quartz.CGEventGetLocation(evt)
check((round(loc.x), round(loc.y)) == (400, 300), "and its absolute position")

print("ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)

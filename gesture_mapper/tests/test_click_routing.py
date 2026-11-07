"""
Covers the two things the pinch/pointer rework changed, neither of which the
GuiPointer suites touch: which cursor model each context gets, and clicks
coming from PinchClicker rather than pose mappings. No camera, no Minecraft.
"""

import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src', 'cv'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mapping-engine'))

import run
from run import (PinchClicker, _pointer_mode, _pointer_bounds, _screen_bounds,
                 _pinch_actions, _TAP_COOLDOWN, TAP_COOLDOWN,
                 SPRINT_ENGAGE, SPRINT_RELEASE)
from mapper import map_gesture
from gesture import Gesture
from joystick import HandJoystick, MAX_RANGE

ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok = ok and cond


def g(pi=1.2, pm=1.4, pr=1.4, pp=1.4, name="fist", x=0.5, y=0.5):
    return Gesture(name=name, confidence=0.2, timestamp=0, hand="right",
                   wrist_x=x, wrist_y=y, hand_scale=0.15, aspect=4/3,
                   pinch_index=pi, pinch_middle=pm,
                   pinch_ring=pr, pinch_pinky=pp)


def feed(p, ratio, t0, finger="index", frames=6, other=1.2):
    """Hold one ratio for a few frames (the EMA needs more than one)."""
    evs, t = [], t0
    for _ in range(frames):
        t += 1/30
        sample = g(pi=ratio, pm=other) if finger == "index" else g(pi=other, pm=ratio)
        evs += p.update(sample, t)
    return evs, t


# ── 1. Pointer model per context ─────────────────────────────────────────────
check(_pointer_mode("spotlight", False) == "desktop",
      "spotlight gets the absolute pointer, not the velocity joystick")
check(_pointer_mode("default", False) == "desktop", "desktop gets the pointer")
check(_pointer_mode("minecraft", True) == "mc-gui", "MC with a screen open: pointer")
check(_pointer_mode("minecraft", False) is None, "MC gameplay stays on velocity look")
check(_pointer_mode("default", False) == _pointer_mode("spotlight", False),
      "default→spotlight is not a coordinate-space switch (no pointer re-seed)")

scr = _screen_bounds()
check(_pointer_bounds("spotlight") == scr, "spotlight points in screen coordinates")
check(_pointer_bounds("default") == scr, "default points in screen coordinates")
check(scr[2] > 0 and scr[3] > 0, f"screen bounds are real: {scr}")

# ── 2. Clicks are pinches, not poses ─────────────────────────────────────────
for pose in ("index_pointer", "peace"):
    check(map_gesture(pose, "right", "minecraft") is None,
          f"right {pose} maps to nothing in gameplay (was HoldBreak/HoldPlace)")
_jump = map_gesture("thumbs_up", "right", "minecraft")
check(_jump is not None and _jump.name == "Jump", "right thumbs_up still jumps")
for pose, name in (("shaaa", "Escape"), ("thumbs_up", "Inventory")):
    cmd = map_gesture(pose, "left", "minecraft")
    check(cmd is not None and cmd.name == name, f"left {pose} → {name} untouched")
# index_pointer/peace are disconnected on BOTH hands now: forming either pose
# perturbs the pinch ratios on the same hand (the pose/pinch clash), so the
# hotbar moved to the ring/pinky pinches.
for pose in ("index_pointer", "peace"):
    check(map_gesture(pose, "left", "minecraft") is None,
          f"left {pose} disconnected (pose/pinch clash)")
check(map_gesture("pinch_ring", "left", "minecraft").name == "HotbarNext",
      "hotbar next moved to the left ring pinch")
check(map_gesture("pinch_pinky", "left", "minecraft").name == "HotbarPrev",
      "hotbar prev moved to the left pinky pinch")

# ── 3. Pinch bindings are table data, per hand and context ───────────────────
for ctx in ("minecraft", "default", "spotlight"):
    check(map_gesture("pinch_index", "right", ctx).name == "HoldBreak",
          f"{ctx}: right index pinch = left click")
    check(map_gesture("pinch_middle", "right", ctx).name == "HoldPlace",
          f"{ctx}: right middle pinch = right click")
check(map_gesture("pinch_index", "left", "minecraft").name == "Jump",
      "minecraft: left index pinch = Jump")
check(map_gesture("pinch_middle", "left", "minecraft").name == "Crouch",
      "minecraft: left middle pinch = Crouch (shift)")
for ctx in ("default", "spotlight"):
    check(map_gesture("pinch_index", "left", ctx) is None
          and map_gesture("pinch_middle", "left", ctx) is None,
          f"{ctx}: left pinches unbound (left free for the scroll modifier)")

# ── 4. GUI-screen gating: Crouch survives (shift-click), Jump doesn't ────────
names = lambda s: {c.name for c in s}
ALL4 = {"index", "middle", "ring", "pinky"}

held, taps = _pinch_actions("left", "minecraft", False, ALL4)
check(names(held) == {"Jump", "Crouch"}, f"gameplay: held left pinches ({names(held)})")
check(names(set(taps)) == {"HotbarNext", "HotbarPrev"},
      f"gameplay: hotbar pinches are taps, not holds ({names(set(taps))})")

held, taps = _pinch_actions("left", "minecraft", True, ALL4)
check(names(held) == {"Crouch"},
      "screen open: Crouch stays live for shift-click, Jump suppressed")
check(taps == [], "screen open: hotbar cycling blocked (number keys move inventory items)")

held, _ = _pinch_actions("right", "minecraft", True, {"index"})
check(names(held) == {"HoldBreak"},
      "screen open: right pinch still clicks (the click half of shift-click)")
check(_pinch_actions("left", "default", False, ALL4) == (set(), []),
      "desktop: left pinches do nothing")

# ── 5. PinchClicker mechanism ────────────────────────────────────────────────
p = PinchClicker(run.PINCH_ENGAGE, run.PINCH_RELEASE)
evs, t = feed(p, 0.50, 0.0)
check(evs == [] and p.held_fingers() == set(), "relaxed hand: no click")

evs, t = feed(p, 0.10, t)
check(len(evs) == 1 and evs[0][:2] == ("index", "down"), "index pinch → down edge")
check(p.held_fingers() == {"index"}, "stays held while pinched")

# exclusive (mouse buttons): a middle pinch mid-hold must not add a second one
evs2 = []
for _ in range(6):
    t += 1/30
    evs2 += p.update(g(pi=0.10, pm=0.10), t)
check(evs2 == [] and p.held_fingers() == {"index"}, "exclusive: one button at a time")

evs, t = feed(p, 0.55, t)
check(any(e[1] == "up" for e in evs) and p.held_fingers() == set(),
      "relaxing releases (no splay needed)")

# the left hand is NOT exclusive — jumping while crouched is ordinary play
pl = PinchClicker(run.PINCH_ENGAGE, run.PINCH_RELEASE, exclusive=False)
evs, t = feed(pl, 0.10, 0.0, finger="middle")
check(pl.held_fingers() == {"middle"}, "left: middle pinch (Crouch) held")
evs2 = []
for _ in range(6):
    t += 1/30
    evs2 += pl.update(g(pi=0.10, pm=0.10), t)
check(pl.held_fingers() == {"index", "middle"},
      "left: index pinch engages *while* middle is held (jump while crouched)")

# hand disappearing mid-click can't leave a button stuck. The timeout runs
# from the first missing frame (t_gone), not from the last seen frame.
p2 = PinchClicker(run.PINCH_ENGAGE, run.PINCH_RELEASE)
evs, t = feed(p2, 0.10, 0.0, finger="middle")
check(len(evs) == 1 and evs[0][0] == "middle", "middle pinch → its own edge")
t_gone = t + 0.5
check(p2.hand_lost(t_gone) == [], "brief dropout does not release (drag survives)")
evs = p2.hand_lost(t_gone + run.PINCH_HAND_LOST_RELEASE + 0.2)
check(len(evs) == 1 and evs[0][1] == "up" and p2.held_fingers() == set(),
      "hand gone past the timeout force-releases")

# reset (fired when gameplay↔GUI-screen switches) must not leave a phantom hold
p3 = PinchClicker(run.PINCH_ENGAGE, run.PINCH_RELEASE)
feed(p3, 0.10, 0.0)
p3.reset()
check(p3.held_fingers() == set(), "reset drops held state on mode switch")

# the left hand's clicker watches all four fingers (ring/pinky = hotbar)
p4 = PinchClicker(run.PINCH_ENGAGE, run.PINCH_RELEASE, exclusive=False,
                  fingers=("index", "middle", "ring", "pinky"))
t = 0.0
for _ in range(8):
    t += 1/30
    evs = p4.update(g(pr=0.08), t)
check(p4.held_fingers() == {"ring"}, "ring pinch tracked (hotbar next)")
for _ in range(8):
    t += 1/30
    p4.update(g(pr=0.08, pp=0.08), t)
check(p4.held_fingers() == {"ring", "pinky"},
      "ring and pinky pinch independently (non-exclusive left hand)")
# a two-finger clicker must ignore the ring/pinky ratios entirely
p5 = PinchClicker(run.PINCH_ENGAGE, run.PINCH_RELEASE)
t = 0.0
for _ in range(8):
    t += 1/30
    p5.update(g(pr=0.02, pp=0.02), t)
check(p5.held_fingers() == set(), "right-hand clicker ignores ring/pinky (mouse = 2 buttons)")

# ── 6. Sprint: stick pushed past full deflection ─────────────────────────────
check(SPRINT_ENGAGE >= MAX_RANGE, "sprint engages only past full WASD deflection")
check(SPRINT_RELEASE < SPRINT_ENGAGE, "sprint has hysteresis")

stick = HandJoystick(name="test")
# anchor: fist held past ARM_DELAY at a fixed position
stick.calibrate(0.5, 0.5)
v = stick.read(0.5, 0.5)
check(abs(v.offset_dist) < 1e-9, "at the anchor: zero offset")
# forward = -y (image y grows downward), so move the hand *up* to push forward.
# read() smooths, so feed the position until it converges.
for _ in range(12):
    v = stick.read(0.5, 0.5 - (MAX_RANGE + 0.05))
fwd = -v.offset_y
check(fwd > SPRINT_ENGAGE,
      f"pushing past full deflection clears the sprint threshold ({fwd:.3f} > {SPRINT_ENGAGE})")
check(abs(v.y) > 0.99, "WASD output is already saturated there (travel is otherwise unused)")
for _ in range(12):
    v = stick.read(0.5, 0.5 - MAX_RANGE * 0.5)
check(-v.offset_y < SPRINT_RELEASE, "normal walking deflection does not sprint")

# ── 7. Hotbar tap cadence ────────────────────────────────────────────────────
check(_TAP_COOLDOWN["HotbarNext"] == TAP_COOLDOWN / 2
      and _TAP_COOLDOWN["HotbarPrev"] == TAP_COOLDOWN / 2,
      "hotbar cycles at half the default tap cooldown (isolated taps and repeat)")
for name in ("Inventory", "Escape", "OpenSpotlight"):
    check(name not in _TAP_COOLDOWN, f"{name} keeps the slower default (it toggles a screen)")

print("ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)

"""
Drives run.main()'s real loop with a scripted hand, in Minecraft context, with
no camera and no Minecraft.

This is the only thing that covers the *wiring* — the unit suites test
PinchClicker, the mapper and the sprint threshold in isolation, but the left-
hand pinch block, the sprint block and the held-key reconciliation only ever
run inside main(), in a branch a camera-less smoke run never reaches (it sits
in "default" context).

Everything that touches the OS is stubbed: no keys are pressed, no mouse
events are posted, the real cursor does not move.
"""

import sys, os, time, threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'src', 'cv'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mapping-engine'))

import run
from gesture import Gesture

ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok = ok and cond


def hand(h, name, x=0.5, y=0.5, pi=1.2, pm=1.4, pr=1.4, pp=1.4):
    return Gesture(name=name, confidence=0.9, timestamp=0, hand=h,
                   wrist_x=x, wrist_y=y, hand_scale=0.15, aspect=4/3,
                   pinch_index=pi, pinch_middle=pm,
                   pinch_ring=pr, pinch_pinky=pp)


# ── Script: (label, gui_open, [gestures]) held for `frames` frames each ──────
FWD = 0.5 - (run.MAX_RANGE + 0.05)      # pushed past full forward deflection
WALK = 0.5 - run.MAX_RANGE * 0.5        # ordinary walking deflection

FRAMES_PER = 14   # > joystick ARM_DELAY and enough for the pinch EMA

PHASES = [
    ("anchor",        False, [hand("left", "fist")]),
    ("walk",          False, [hand("left", "fist", y=WALK)]),
    ("sprint",        False, [hand("left", "fist", y=FWD)]),
    ("sprint+jump",   False, [hand("left", "fist", y=FWD, pi=0.05)]),
    ("crouch",        False, [hand("left", "fist", pm=0.05)]),
    # screen opens, left middle still pinched, right hand pinches to click:
    # this is inventory shift-click
    ("shift-click",   True,  [hand("left", "fist", pm=0.05),
                              hand("right", "none", pi=0.05)]),
    # held ring pinch: hotbar should cycle repeatedly, at the halved cadence
    ("hotbar",        False, [hand("left", "fist", pr=0.05)], 110),
    # ...and the same pinch must do nothing once a screen is open, because the
    # number keys rearrange a container's contents
    ("hotbar-in-gui", True,  [hand("left", "fist", pr=0.05)], 60),
    # the poses that used to drive the hotbar are disconnected entirely
    ("pose-hotbar",   False, [hand("left", "index_pointer")], 40),
]

# What was held during each phase, by phase label
held_by_phase: dict[str, set[str]] = {}
_phase = ["setup"]


class FakeDetector:
    """Feeds the script, one phase at a time, then asks the loop to quit."""
    def __init__(self):
        self._queue = []
        for phase in PHASES:
            label, gui, gestures = phase[0], phase[1], phase[2]
            for _ in range(phase[3] if len(phase) > 3 else FRAMES_PER):
                self._queue.append((label, gui, gestures))
        self._i = 0

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def next_gestures(self):
        if self._i >= len(self._queue):
            FakeKeys.quit = True
            return []
        label, gui, gestures = self._queue[self._i]
        self._i += 1
        # record what the *previous* frame ended up holding for this phase
        held_by_phase.setdefault(label, set())
        held_by_phase[label] = {c.name for c in run._held_commands}
        _phase[0] = label
        run._gui_open = gui
        time.sleep(1 / 60)
        return list(gestures)


class FakeKeys:
    quit = False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def should_quit(self): return FakeKeys.quit


taps: list[tuple[str, float, str]] = []   # (command name, monotonic time, phase)
button_posts: list[str | None] = []       # button carried by each cursor move posted

run.GestureDetector = FakeDetector
run.KeyListener = FakeKeys
run._sync_gui_open = lambda: None            # the script owns _gui_open
run.hold = lambda cmd: None                  # reconciliation still tracks _held_commands
run.release = lambda cmd: None
run.tap = lambda cmd: taps.append((cmd.name, time.monotonic(), _phase[0]))
run.move_mouse_relative = lambda dx, dy: None
run.post_cursor_move = lambda x, y, dx, dy, button=None: button_posts.append(button)
run._center_cursor_in_window = lambda: None
run._ensure_cursor_in_bounds = lambda: None
run.get_active_context = lambda: "minecraft"
run._pointer_bounds = lambda ctx: (0.0, 0.0, 1000.0, 800.0)

run.main()

# ── Assertions ───────────────────────────────────────────────────────────────
print()
for phase in PHASES:
    label = phase[0]
    print(f"  {label:<13} held: {sorted(held_by_phase.get(label, set())) or '—'}")
print()

anchor = held_by_phase.get("anchor", set())
check(anchor == set(), f"anchoring alone holds nothing ({sorted(anchor)})")

walk = held_by_phase.get("walk", set())
check("HoldW" in walk, "walking deflection holds W")
check("Sprint" not in walk, "ordinary walking does not sprint")

sprint = held_by_phase.get("sprint", set())
check("HoldW" in sprint and "Sprint" in sprint,
      f"pushing past full deflection sprints ({sorted(sprint)})")

sj = held_by_phase.get("sprint+jump", set())
check("Jump" in sj, "left index pinch jumps")
check("HoldW" in sj and "Sprint" in sj,
      "jumping does not drop walking or sprinting (pinch reads as fist)")

crouch = held_by_phase.get("crouch", set())
check("Crouch" in crouch, "left middle pinch crouches")

sc = held_by_phase.get("shift-click", set())
check("Crouch" in sc, "screen open: Crouch (shift) still held — the shift half")
check("HoldBreak" in sc, "screen open: right pinch clicks — the click half")
check("Jump" not in sc, "screen open: Jump suppressed")
check(not {"HoldW", "HoldA", "HoldS", "HoldD"} & sc, "screen open: WASD released")
check("Sprint" not in sc, "screen open: Sprint released")

check(run._held_commands == set(), "loop exit releases everything")
check(run._mouse_button[0] is None, "loop exit clears the held-button state")
# The GUI-screen pointer drags too: while the right pinch held HoldBreak, every
# cursor move the mouse thread posted must have been a left-drag, not a move
# (that's what lets you drag an item across crafting slots).
check("left" in button_posts,
      f"pointer posts left-drag events while a pinch is held ({set(button_posts)})")

# Hotbar cadence: a held ring pinch should re-fire every _TAP_COOLDOWN
# seconds, and the *first* one fires immediately (isolated tap).
hb = [t for name, t, ph in taps if name == "HotbarNext" and ph == "hotbar"]
expected = run._TAP_COOLDOWN["HotbarNext"]
check(len(hb) >= 3, f"held ring pinch repeats the hotbar tap ({len(hb)} taps)")
gaps = [b - a for a, b in zip(hb, hb[1:])]
worst = max(abs(gp - expected) for gp in gaps) if gaps else 99
check(worst < 0.12,
      f"repeat gap ≈ {expected}s, not the {run.TAP_COOLDOWN}s default "
      f"(gaps {[round(gp, 2) for gp in gaps]})")

in_gui = [name for name, _, ph in taps if ph == "hotbar-in-gui"]
check(in_gui == [],
      f"screen open: the same pinch fires nothing ({in_gui}) — no silent inventory shuffling")

pose = [name for name, _, ph in taps if ph == "pose-hotbar"]
check(pose == [], f"index_pointer pose fires nothing at all now ({pose})")

check(all(name != "Inventory" for name, _, _ in taps),
      "no stray Inventory taps (left thumbs_up never scripted)")

print("\nALL OK" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)

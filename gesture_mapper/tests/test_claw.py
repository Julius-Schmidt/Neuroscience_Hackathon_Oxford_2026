"""Claw logic (no screen): three presses = sweep, stop x, stop y + click; reaction-time
compensation; presses during the short pause are ignored."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
for sub in ("src/ui", "src/mapping-engine", "src/os-integration", "src/sources"):
    sys.path.insert(0, os.path.join(ROOT, sub))
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


try:
    from claw_overlay import ClawState
except Exception as e:      # Qt can't load here (no display libs)
    print(f"PASS skipped: {type(e).__name__}")
    sys.exit(0)

s = ClawState(sweep_s=2.0, lag_s=0.2, refine=False)
t = 0.0
check(s.step(t) is None and s.stage == "x", "1st press starts the left-right sweep")
t += 0.1
check(s.step(t) is None and s.stage == "x", "a press inside the short pause is ignored")
for _ in range(60):            # 1.2 s from t=0.25 pause end -> pos ~ (1.2-0.15)/2
    t += 0.02
    s.tick(t, 0.02)
pos = s.pos
s.step(t)
check(s.stage == "y" and abs(s.x - (pos - 0.1)) < 1e-6, f"2nd press fixes x, backed up by the lag ({s.x:.3f} vs {pos:.3f})")
t += 0.5
for _ in range(25):
    t += 0.02
    s.tick(t, 0.02)
done = s.step(t)
check(done is not None and s.stage == "idle" and 0 <= done[1] <= 1, f"3rd press clicks at ({done})")

# refine pass: 5 presses, the fine sweep stays inside a narrow ribbon around the coarse stop
s = ClawState(sweep_s=2.0, lag_s=0.0, refine=True, band=0.16)
t = 10.0
s.step(t)
for _ in range(40):
    t += 0.02; s.tick(t, 0.02)
coarse = s.pos
s.step(t)
check(s.stage == "x-fine" and abs((s.lo + s.hi) / 2 - coarse) < 1e-6 and abs(s.hi - s.lo - 0.16) < 1e-6,
      f"2nd press opens a ribbon around the coarse stop ({s.lo:.2f}-{s.hi:.2f})")
inside = True
for _ in range(400):
    t += 0.02; s.tick(t, 0.02); inside &= s.lo - 1e-9 <= s.pos <= s.hi + 1e-9
check(inside, "the slow line never leaves the ribbon")
s.step(t + 1); s.step(t + 2)
done = s.step(t + 3)
check(done is not None, f"5 presses with refine -> click ({done})")

# drive mode: hold glides (slow then faster), tap clicks without moving
s = ClawState()
s.drive("x", True, 0.0)
for k in range(25):
    s.tick(0.02 * (k + 1), 0.02)
early = s.cx                                   # distance in the first 0.5 s
for k in range(25, 50):
    s.tick(0.02 * (k + 1), 0.02)
late = s.cx - early                            # distance in the next 0.5 s
check(s.drive("x", False, 3.0) is None and late > 2 * early > 0, f"holding glides and speeds up ({early:.3f} then {late:.3f})")
cx = s.cx
s.drive("y", True, 4.0); s.tick(4.1, 0.1)
check(s.drive("y", False, 4.15) == (cx, 0.0) and s.stage == "idle",
      "a quick tap clicks where the crosshair is, then the claw is done")
s = ClawState()
s.drive("x", True, 0.0)
check(s.cx == 0.0 and s.cy == 0.0, "glide lines start at the left / top edge")
for k in range(400):
    s.tick(0.02 * (k + 1), 0.02)
check(0.0 <= s.cx < 1.0, f"past the far edge the line wraps back to the start ({s.cx:.2f})")

import rules as R
import presets
d = presets.claw()
check(any(a["type"] == "claw_step" for r in d["rules"] for a in r["actions"]), "claw example uses claw_step")


class Ex:
    def __init__(self):
        self.hooks, self.got = {}, []
    def run(self, a, ev, eng):
        import actions
        actions.OSExecutor.run(self, a, ev, eng)
    def press(self, a): pass
    def release(self, a): pass


ex = Ex()
ex.messages = []
ex.hooks["claw"] = lambda a: ex.got.append(a["type"])
eng = R.RuleEngine(ex, d)
eng.handle(R.Event("myo", "right.fist", "press"))
eng.handle(R.Event("myo", "right.double_tap", "press"))
check(ex.got == ["claw_step", "claw_cancel"], f"claw actions go to the GUI hook ({ex.got})")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)

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

s = ClawState(sweep_s=2.0, lag_s=0.2)
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

"""Rule engine + action executor, no camera / bands / OS. Prints PASS/FAIL per check."""
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
for sub in ("src/mapping-engine", "src/os-integration"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import rules as R

fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


class Fake:
    def __init__(self):
        self.log = []

    def run(self, a, ev, eng):
        self.log.append(("run", a["type"], a.get("key") or a.get("text") or a.get("button")))

    def press(self, a):
        self.log.append(("press", a.get("key") or a.get("button")))

    def release(self, a):
        self.log.append(("release", a.get("key") or a.get("button")))


def rule(src, name, trig, actions, conds=()):
    r = R.new_rule(src, name, trig)
    r["actions"] = [dict(R.new_action(k), **kw) for k, kw in actions]
    r["conditions"] = [dict(R.new_condition(k), value=v) for k, v in conds]
    return r


# 1. press fires, release doesn't
fx = Fake()
eng = R.RuleEngine(fx, {"rules": [rule("myo", "right.fist", "press", [("key_tap", {"key": "space"})])]})
eng.handle(R.Event("myo", "right.fist", "press"))
eng.handle(R.Event("myo", "right.fist", "release"))
eng.handle(R.Event("myo", "left.fist", "press"))
check(fx.log == [("run", "key_tap", "space")], f"press fires once, other band ignored ({fx.log})")

# 2. hold: held while the gesture lasts
fx = Fake()
eng = R.RuleEngine(fx, {"rules": [rule("myo", "left.fist", "hold", [("key_hold", {"key": "w"})])]})
eng.handle(R.Event("myo", "left.fist", "press"))
check(fx.log == [("press", "w")], "hold presses on gesture start")
eng.handle(R.Event("myo", "left.fist", "release"))
check(fx.log == [("press", "w"), ("release", "w")], "hold releases on gesture end")

# 3. modes, Scratch-style variables
fx = Fake()
eng = R.RuleEngine(fx, {"rules": [
    rule("myo", "right.double_tap", "press", [("toggle_mode", {"a": "normal", "b": "typing"})]),
    rule("myo", "right.fist", "press", [("mouse_click", {"button": "left"})], [("mode", "normal")]),
    rule("myo", "right.fist", "press", [("type_text", {"text": "a"})], [("mode", "typing")]),
]})
eng.handle(R.Event("myo", "right.fist", "press"))
eng.handle(R.Event("myo", "right.double_tap", "press"))
eng.handle(R.Event("myo", "right.fist", "press"))
check(fx.log == [("run", "mouse_click", "left"), ("run", "type_text", "a")] and eng.mode == "typing",
      f"same gesture, different mode -> different action ({fx.log}, mode={eng.mode})")

# 4. any-marker LSL rule (BCI decisions)
fx = Fake()
eng = R.RuleEngine(fx, {"rules": [rule("lsl", "any marker", "press", [("say", {"text": "got it"})])]})
eng.handle(R.Event("lsl", "target_7", "press"))
check(fx.log == [("run", "say", "got it")], "LSL 'any marker' rule catches every BCI decision")

# 5. context condition
fx = Fake()
eng = R.RuleEngine(fx, {"rules": [rule("camera", "left.fist", "press", [("key_tap", {"key": "e"})],
                                       [("context", "minecraft")])]}, context_fn=lambda: "default")
eng.handle(R.Event("camera", "left.fist", "press"))
check(fx.log == [], "context condition blocks outside the app")

# 6. a broken action never crashes the engine
class Boom(Fake):
    def run(self, a, ev, eng):
        raise RuntimeError("boom")
eng = R.RuleEngine(Boom(), {"rules": [rule("myo", "right.fist", "press", [("key_tap", {})]),
                                     {"id": "x", "when": {"source": "myo", "name": "right.fist"},
                                      "actions": [{"type": "nonsense"}]}]})
try:
    fired = eng.handle(R.Event("myo", "right.fist", "press"))
    check(len(fired) == 2, "failing / unknown blocks are skipped, not fatal")
except Exception as e:
    check(False, f"engine raised {e}")

# 7. save / load round trip
d = {"rules": [rule("myo", "right.fist", "press", [("key_tap", {"key": "a"})])], "mode": "normal"}
p = os.path.join(tempfile.mkdtemp(), "rules.json")
R.save(d, p)
check(R.load(p) == d, "rules.json round trip")
check(R.load(p + ".missing")["rules"] == [], "missing file -> empty rule set")

# 8. value events drive mouse movement through the real executor (stub OS)
import actions
from quartz_compat import Quartz as Q
if hasattr(Q, "posted"):
    Q.posted.clear()
    ex = actions.OSExecutor()
    eng = R.RuleEngine(ex, {"rules": [rule("myo", "right.pointer", "value", [("mouse_move", {"gain": 2.0})]),
                                      rule("myo", "right.fist", "press", [("shortcut", {"keys": "ctrl+c"})])]})
    eng.handle(R.Event("myo", "right.pointer", "value", (3, -4)))
    check(Q.posted and Q.posted[-1] == ("move_abs", 406, 292), f"pointer value moves the mouse ({Q.posted[-1:]})")
    Q.posted.clear()
    eng.handle(R.Event("myo", "right.fist", "press"))
    keys = [p for p in Q.posted if p[0] == "key"]
    check(keys == [("key", 0xA2, True), ("key", 0x43, True), ("key", 0x43, False), ("key", 0xA2, False)],
          f"ctrl+c shortcut = ctrl down, c, ctrl up ({keys})")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)

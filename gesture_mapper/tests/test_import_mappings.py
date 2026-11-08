"""Camera mappings -> Blocks rules: conversion, re-import replaces, held commands hold,
and the camera source's pinch events. No camera / OS."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
for sub in ("src/mapping-engine", "src/os-integration", "src/sources", "src/cv"):
    sys.path.insert(0, os.path.join(ROOT, sub))
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


import rules as R
import import_mappings as IM
from mapper import default_tables

rules_ = IM.rules_from_tables(default_tables())
n_bindings = sum(len(t) for t in default_tables().values())
check(len(rules_) == n_bindings, f"one rule per binding ({len(rules_)} / {n_bindings})")
jump = [r for r in rules_ if r["title"].startswith("minecraft: left pinch index")][0]
check(jump["when"] == {"source": "camera", "name": "left.pinch_index", "trigger": "hold"}
      and jump["conditions"] == [{"type": "context", "value": "minecraft"}]
      and jump["actions"] == [{"type": "command_hold", "name": "Jump"}], f"held command -> 'is held' + hold block ({jump})")
spot = [r for r in rules_ if r["actions"][0].get("name") == "OpenSpotlight"][0]
check(spot["when"]["trigger"] == "press" and spot["actions"][0]["type"] == "command", "tap command -> 'starts' + run block")

data = {"rules": [R.new_rule("myo", "right.fist")], "mode": "normal"}
IM.merge(data, rules_)
IM.merge(data, rules_[:3])
check(len(data["rules"]) == 4 and data["rules"][0]["when"]["source"] == "myo",
      "re-import replaces old imported rules, keeps the user's own")


class Rec:
    def __init__(self):
        self.log = []

    def run(self, a, ev, eng):
        self.log.append(("run", a.get("name")))

    def press(self, a):
        self.log.append(("press", a.get("name")))

    def release(self, a):
        self.log.append(("release", a.get("name")))


rec = Rec()
ctx = ["minecraft"]
eng = R.RuleEngine(rec, {"rules": rules_}, context_fn=lambda: ctx[0])
eng.handle(R.Event("camera", "left.pinch_index", "press"))
eng.handle(R.Event("camera", "left.pinch_index", "release"))
ctx[0] = "default"
eng.handle(R.Event("camera", "left.pinch_index", "press"))
check(rec.log == [("press", "Jump"), ("release", "Jump")], f"Jump held in Minecraft only ({rec.log})")

# camera source pinch hysteresis
from camera_source import CameraSource, _PinchEdges


class G:
    def __init__(self, i):
        self.pinch_index, self.pinch_middle, self.pinch_ring, self.pinch_pinky = i, 10, 10, 10


got = []
pe = _PinchEdges(lambda ev: got.append(f"{ev.name} {ev.kind}"))
cs = CameraSource(detector=object())
for ratio in (0.5, 0.2, 0.3, 0.38, 0.45, 0.2):
    cs.pinches(pe, "right", G(ratio), 0.26, 0.40)
check(got == ["right.pinch_index press", "right.pinch_index release", "right.pinch_index press"],
      f"pinch engages < 0.26, holds until > 0.40 ({got})")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)

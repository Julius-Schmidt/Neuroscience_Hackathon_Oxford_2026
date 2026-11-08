"""Myo 'gestures' mode (5 trained gestures) + the Minecraft presets, no hardware:
gesture changes -> press/release, both layers of both presets, and a simulated band
through the real gesture model if one is trained."""
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
for sub in ("src/sources", "src/mapping-engine", "src/os-integration"):
    sys.path.insert(0, os.path.join(ROOT, sub))
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


import presets  # noqa: E402
import rules as R  # noqa: E402
from base import EdgeTracker  # noqa: E402
from myo_source import MyoSource  # noqa: E402


class Fake:
    def __init__(self):
        self.log = []

    def run(self, a, ev, eng):
        self.log.append(("run", a.get("name") or a.get("key") or a["type"]))

    def press(self, a):
        self.log.append(("press", a.get("name") or a.get("key") or a.get("button")))

    def release(self, a):
        self.log.append(("release", a.get("name") or a.get("key") or a.get("button")))


def engine(data, ctx="minecraft"):
    eng = R.RuleEngine(Fake(), data)
    eng.context_fn = lambda: ctx
    return eng.executor, eng


def tap(eng, name):
    eng.handle(R.Event("myo", name, "press"))
    eng.handle(R.Event("myo", name, "release"))


def main():
    # gesture changes -> one active gesture per band, neutral / rest = nothing
    got = []
    edges = EdgeTracker("myo", lambda ev: got.append(f"{ev.name} {ev.kind}"))
    for g in ["power_grip", "pinch_grip", "neutral", "wrist_extension", "rest"]:
        MyoSource.on_gesture("left", edges, g)
    check(got == ["left.power_grip press", "left.power_grip release", "left.pinch_grip press",
                  "left.pinch_grip release", "left.wrist_extension press", "left.wrist_extension release"],
          f"gestures -> press/release, neutral and rest release ({got})")

    names = set(R.SOURCE_EVENTS["myo"])
    for key in ("minecraft5", "minecraft5_one"):
        data = presets.PRESETS[key]()
        unknown = [r["when"]["name"] for r in data["rules"] if r["when"]["name"] not in names]
        bad = [a["type"] for r in data["rules"] for a in r["actions"] if a["type"] not in R.ACTIONS]
        check(not unknown and not bad and key in presets.TITLES,
              f"{key}: known event names + actions, has a title ({unknown} {bad})")

    # minecraft5: normal layer holds W while the grip is held; ulnar deviation -> alt layer
    fx, eng = engine(presets.minecraft5(), "default")
    tap(eng, "left.power_grip")
    check(fx.log == [], f"nothing fires unless Minecraft is in front ({fx.log})")
    fx, eng = engine(presets.minecraft5())
    eng.handle(R.Event("myo", "left.wrist_extension", "press"))
    held = list(fx.log)
    eng.handle(R.Event("myo", "left.wrist_extension", "release"))
    check(held == [("press", "w")] and fx.log[-1] == ("release", "w"), f"wrist extension = hold W ({fx.log})")
    fx.log.clear()
    tap(eng, "left.power_grip")
    check(fx.log == [("press", "HoldPlace"), ("release", "HoldPlace")], f"power grip = place ({fx.log})")
    fx.log.clear()
    tap(eng, "left.ulnar_deviation")
    tap(eng, "left.chuck_grip")
    eng.handle(R.Event("myo", "left.power_grip", "press"))
    check(eng.mode == "alt" and ("run", "Inventory") in fx.log and ("press", "s") in fx.log
          and ("press", "w") not in fx.log, f"alt layer: chuck = inventory, power grip = S ({fx.log})")
    eng.handle(R.Event("myo", "left.power_grip", "release"))
    tap(eng, "left.ulnar_deviation")
    fx.log.clear()
    tap(eng, "right.fist")
    tap(eng, "left.chuck_grip")
    check(eng.mode == "normal" and fx.log == [("press", "HoldBreak"), ("release", "HoldBreak"),
                                              ("press", "Jump"), ("release", "Jump")],
          f"back to normal: right fist = break, chuck = jump ({fx.log})")
    check(not any("pinch" in r["when"]["name"] for r in presets.minecraft5()["rules"]), "minecraft5 doesn't use pinch")

    # minecraft5_one: everything on the right band
    fx, eng = engine(presets.minecraft5_one())
    tap(eng, "right.wrist_extension")
    tap(eng, "right.power_grip")
    check(fx.log == [("press", "w"), ("release", "w"), ("press", "HoldBreak"), ("release", "HoldBreak")],
          f"one band: wrist extension = W, power grip = break ({fx.log})")

    # slow aiming: fractions of a pixel add up instead of being cut to 0 (stub OS, nothing posted)
    import actions

    class StubQuartz:
        deltas = []
        def __getattr__(self, name):
            return lambda *a: (a[-1] if name == "CGPointMake" else None)
        def CGEventGetLocation(self, _e):
            class P: x = y = 0.0
            return P()
        def CGPoint(self, x, y):
            return (x, y)
        def CGEventSetIntegerValueField(self, _e, field, v):
            self.deltas.append(v)
    real, actions.Quartz = actions.Quartz, StubQuartz()
    try:
        for _ in range(10):
            actions._move(0.3, 0.0)
    finally:
        actions.Quartz = real
    xs = StubQuartz.deltas[0::2]
    check(sum(xs) == 3, f"10 moves of 0.3 px arrive as 3 whole pixels ({xs})")
    for key in ("minecraft5", "minecraft5_one"):
        check(presets.PRESETS[key]().get("move") == "game", f"{key} uses the game camera look")

    # simulated band through the real gesture model (skips if none trained)
    try:
        from myomouse.gestures11 import model_path
    except Exception as e:
        print(f"PASS skipped simulated band (no myomouse: {e})")
        model_path = None
    user = next((u for u in ("julius", "dylan") if model_path and model_path(u, "right").exists()), None)
    if user:
        got = []
        src = MyoSource("gestures", user, ["right"], fake=True)
        src.start(lambda ev: got.append(ev))
        time.sleep(4)
        status = src.status
        src.stop()
        check(status.startswith("right: gestures running") and not status.startswith("error"),
              f"simulated band runs the {user} gesture model ({status}, {len(got)} events)")
        src = MyoSource("gestures", user, ["right"], fake=True,
                        gesture_sides=presets.minecraft5()["gesture_sides"])
        src.start(lambda ev: None)
        check(src.status.startswith("no gesture model"), f"minecraft5 keeps the right band off gestures ({src.status})")
        src.stop()
        src = MyoSource("gestures", "nobody_here", ["right"], fake=True)
        src.start(lambda ev: None)
        check(src.status.startswith("no gesture model"), f"missing model -> clear status ({src.status})")
    elif model_path:
        print("PASS skipped simulated band (no gesture model trained)")

    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

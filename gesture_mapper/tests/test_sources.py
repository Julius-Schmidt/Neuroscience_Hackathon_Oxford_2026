"""Sources without hardware: edge tracking, the Myo built-in classifier path (fed fake band
events), and the trained-model path on simulated bands if myomouse + a model are present."""
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


from base import EdgeTracker
from myo_source import MyoSource


def main():
    global fails
    got = []
    emit = lambda ev: got.append(f"{ev.source}:{ev.name} {ev.kind}")

    # built-in classifier: synced, fist, wave out (fist auto-released), rest, unsynced
    src = MyoSource("builtin")
    edges = EdgeTracker("myo", emit)
    needs_unlock = src.on_classifier("left", edges, "synced", "left arm")
    for kind, detail in [("pose", "fist"), ("pose", "wave out"), ("pose", "rest"),
                         ("pose", "fingers spread"), ("unsynced", "")]:
        src.on_classifier("left", edges, kind, detail)
    check(needs_unlock and src.synced.get("left") is False, "sync / unsync tracked, unlock requested on sync")
    check(got == ["myo:left.fist press", "myo:left.fist release", "myo:left.wave_out press",
                  "myo:left.wave_out release", "myo:left.fingers_spread press", "myo:left.fingers_spread release"],
          f"poses -> press/release, one at a time, released on unsync ({got})")

    # trained model path on simulated bands (skips if no model for 'dylan')
    models = os.path.join(ROOT, "..", "models", "dylan", "fist_right.joblib")
    if os.path.exists(models):
        got.clear()
        src = MyoSource("trained", "dylan", ["right"], fake=True)
        src.start(emit)
        t0 = time.time()
        while time.time() - t0 < 12 and "myo:right.fist press" not in got:
            time.sleep(0.2)
        src.stop()
        check("myo:right.fist press" in got, f"simulated band -> trained fist events ({src.status})")
    else:
        print("PASS skipped trained-model check (no models/dylan)")

    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":   # band workers are spawned processes: they must not re-run the test
    main()

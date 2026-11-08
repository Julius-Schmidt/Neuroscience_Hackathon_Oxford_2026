"""Run the block rules (data/rules.json) without the GUI: any sources -> rules -> actions.

    python rules_run.py --myo builtin                  # Myo built-in poses (sync gesture first), both bands
    python rules_run.py --myo trained --user dylan     # our trained click model
    python rules_run.py --lsl                          # EEG / BCI markers over LSL
    python rules_run.py --camera                       # webcam hand poses
    python rules_run.py --myo builtin --lsl --preset controller   # try a ready-made rule set

Every event and every rule that fires is printed, so this doubles as a monitor:
`python rules_run.py --myo builtin --dry-run` shows what the bands send without touching
the mouse or keyboard. Ctrl+C to stop.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("src/sources", "src/mapping-engine", "src/os-integration", "src/cv"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import rules as R  # noqa: E402
from runtime import Runtime  # noqa: E402


class PrintExecutor:
    def run(self, a, ev, eng):
        print(f"      -> would {a['type']} {({k: v for k, v in a.items() if k != 'type'})}")

    def press(self, a):
        print(f"      -> would press {a}")

    def release(self, a):
        print(f"      -> would release {a}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--myo", choices=["builtin", "trained"], default=None)
    ap.add_argument("--user", default=None, help="for --myo trained")
    ap.add_argument("--sides", default=None, help="e.g. right or right,left (default: bands.json)")
    ap.add_argument("--fake", action="store_true", help="simulated bands (trained mode)")
    ap.add_argument("--lsl", action="store_true")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--preset", default=None, help="load a preset rule set instead of data/rules.json")
    ap.add_argument("--dry-run", action="store_true", help="print actions instead of doing them")
    ap.add_argument("--seconds", type=float, default=0)
    args = ap.parse_args()

    if args.preset:
        import presets
        data = presets.PRESETS[args.preset]()
    else:
        data = R.load()
    print(f"{len(data['rules'])} rule(s), mode '{data.get('mode', 'normal')}'")
    try:
        from context import get_active_context
    except Exception:
        get_active_context = None
    rt = Runtime(PrintExecutor() if args.dry_run else None, data, get_active_context)
    rt.engine.listeners.append(lambda rid, ev: print(f"   rule {rid} fired"))
    rt.event_listeners.append(lambda ev: print(f"{ev.source}:{ev.name} {ev.kind}"
                                               + (f" {ev.value}" if ev.kind == "value" else ""))
                              if ev.kind != "value" else None)
    rt.start()
    sides = args.sides.split(",") if args.sides else None
    if args.myo:
        from myo_source import MyoSource
        rt.add_source(MyoSource(args.myo, args.user, sides, fake=args.fake))
    if args.lsl:
        from lsl_source import LslSource
        rt.add_source(LslSource())
    if args.camera:
        from camera_source import CameraSource
        rt.add_source(CameraSource())
    if not rt.sources:
        print("no sources: add --myo builtin / --myo trained --user <you> / --lsl / --camera")
        return
    t0 = time.monotonic()
    last = {}
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            for n, s in rt.sources.items():
                if last.get(n) != s.status:
                    print(f"[{n}] {s.status}")
                    last[n] = s.status
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop()


if __name__ == "__main__":
    main()

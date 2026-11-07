"""
Replays real pinch-ratio sequences from the 2026-08-07 live session against the
hysteresis state machine.

Sections 1-3 are pinned to the thresholds those ratios were *recorded* under
(0.26/0.40) — they test the state machine, not the current tuning, so they must
not drift when the live defaults are retuned. Section 4 checks the live
defaults, and is where the two directions of failure trade off.
"""

import sys, random
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from run import GuiPointer, PINCH_ENGAGE, PINCH_RELEASE

# Thresholds in force when the sequences below were captured.
LOGGED_ENGAGE, LOGGED_RELEASE = 0.26, 0.40
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'cv'))
from gesture import Gesture

def g(pi, pm=0.65):
    return Gesture(name="pinch", confidence=0.2, timestamp=0, hand="right",
                   wrist_x=0.5, wrist_y=0.5, hand_scale=0.15, aspect=4/3,
                   pinch_index=pi, pinch_middle=pm)

ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok = ok and cond

def feed(p, seq, t0, jitter=0.05, frames_per=8):
    """each logged value held ~8 frames with jitter, 30fps; returns events"""
    random.seed(7); evs = []; t = t0
    for v in seq:
        for _ in range(frames_per):
            t += 1/30
            evs += [(e, t) for e in p.update_pinch(g(max(0.01, v + (random.random()-0.5)*2*jitter)), t)]
    return evs, t

p = GuiPointer([1.5], LOGGED_ENGAGE, LOGGED_RELEASE)

# 1. Relaxed hand / fist (logged 0.42-0.53 + jitter) must never engage
evs, t = feed(p, [0.53, 0.44, 0.51, 0.42, 0.45, 0.47, 0.48, 0.50, 0.44, 0.42], 0.0)
check(evs == [], f"relaxed/fist hand never engages ({len(evs)} events)")

# 2. Deliberate pinch (logged drag: 0.10..0.32 w/ spikes) engages once, holds, then releases on RELAX (0.45)
seq = [0.72, 0.30, 0.10, 0.10, 0.12, 0.23, 0.22, 0.11, 0.15, 0.31, 0.32, 0.17, 0.45, 0.48, 0.47]
evs, t = feed(p, seq, t)
downs = [e for e in evs if e[0][1] == "down"]; ups = [e for e in evs if e[0][1] == "up"]
check(len(downs) == 1, f"exactly one engage during pinch+drag ({len(downs)})")
check(len(ups) == 1, f"exactly one release ({len(ups)})")
check(len(ups) == 1 and ups[0][1] > downs[0][1] + 8*3/30, "held through the 0.31/0.32 jitter spikes")
# release happened during the relax portion (after seq index 12 starts)
relax_start_t = t - 3*8/30
check(len(ups) == 1 and ups[0][1] >= relax_start_t - 0.2, "released by RELAXING the hand (no splay needed)")

# 3. quick pinch tap = clean click
evs, t = feed(p, [0.70, 0.08, 0.08, 0.50], t, frames_per=6)
kinds = [e[0][1] for e in evs]
check(kinds == ["down", "up"], f"quick tap-pinch = one clean click ({kinds})")

# ── 4. Live defaults ─────────────────────────────────────────────────────────
# The reported bug: fingers separate, click stays down. A hand that opens only
# partway (0.30 — well short of the old 0.40 release) must release.
p2 = GuiPointer([1.5], PINCH_ENGAGE, PINCH_RELEASE)
evs, t = feed(p2, [0.06, 0.06], 0.0, jitter=0.02, frames_per=8)
check([e[0][1] for e in evs] == ["down"], "live defaults: firm pinch engages")
evs, t = feed(p2, [0.30, 0.30], t, jitter=0.02, frames_per=8)
check([e[0][1] for e in evs] == ["up"],
      "live defaults: opening the fingers to 0.30 releases (the reported stuck-click)")

# The cost of that, stated rather than asserted away: the logged 0.10-0.32 drag
# no longer survives its own mid-drag spikes at these thresholds.
p3 = GuiPointer([1.5], PINCH_ENGAGE, PINCH_RELEASE)
evs, t = feed(p3, seq, 0.0)
n_down = len([e for e in evs if e[0][1] == "down"])
print(f"NOTE the 2026-08-07 drag sequence yields {n_down} separate clicks at the live "
      f"thresholds ({PINCH_ENGAGE}/{PINCH_RELEASE}) vs 1 at the logged ones "
      f"({LOGGED_ENGAGE}/{LOGGED_RELEASE}) — if long drags start dropping mid-way, "
      f"raise pinch_release in data/settings.json.")

print("ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)

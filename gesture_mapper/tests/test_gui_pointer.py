import sys, os, math, random
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run
from run import GuiPointer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'cv'))
from gesture import Gesture

def g(x, y, scale=0.15, pi=1.2, pm=1.4, name="index_pointer"):
    return Gesture(name=name, confidence=0.2, timestamp=0, hand="right",
                   wrist_x=x, wrist_y=y, hand_scale=scale, aspect=4/3,
                   pinch_index=pi, pinch_middle=pm)

ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and cond

# 1. Jitter attenuation: stationary hand + 1px noise (1/320 ~ 0.003)
p = GuiPointer([1.5], 0.40, 0.60)
random.seed(1)
t = 0.0
total = 0.0
n = 0
for i in range(300):
    t += 1/30
    noise = lambda: (random.random()-0.5)*2*0.003
    d = p.track(g(0.5+noise(), 0.5+noise()), t)
    if d and t > 1.0:  # after settle
        total += math.hypot(d[0], d[1]); n += 1
avg = total/n
# raw unfiltered delta would be ~ (0.003*1.33/0.15)*1.5 ~ 0.04 window-heights/frame
check(avg < 0.0045, f"jitter: avg |delta| {avg:.5f} window-heights/frame (raw would be ~0.04)")

# 2. Deliberate move passes through: hand sweeps 0.3 frame-widths over 0.5s
p2 = GuiPointer([1.5], 0.40, 0.60)
t = 0.0
p2.track(g(0.6, 0.5), t)
moved = 0.0
for i in range(15):
    t += 1/30
    d = p2.track(g(0.6 - 0.02*(i+1), 0.5), t)
    if d: moved += d[0]
# 0.3 frame-widths = 0.4 iso units = 2.67 hand-widths * 1.5 = 4.0 window-heights min (before accel)
check(moved > 2.0, f"responsiveness: sweep produced {moved:+.2f} window-heights (mirror: hand left -> cursor +x... )")
check(moved > 0, "mirror flip: hand moving toward its left (image -x) drives cursor +x")

# 3. Pinch hysteresis + cooldown (ratios are EMA-smoothed: feed each value
# a few frames, like a real hand — a single-frame spike must NOT click)
p3 = GuiPointer([1.5], 0.40, 0.60)
def feed3(p, pi, pm, t0, n=3):
    evs = []
    for i in range(n):
        evs += p.update_pinch(g(0.5,0.5,pi=pi,pm=pm), t0 + i*0.033)
    return evs
ev = feed3(p3, 0.8, 1.4, 1.0); check(ev==[], "no event at ratio 0.8")
ev = p3.update_pinch(g(0.5,0.5,pi=0.20), 1.10); check(ev==[], "single-frame spike does not click (smoothing)")
ev = feed3(p3, 0.20, 1.4, 1.13); check(len(ev)==1 and ev[0][1]=="down", "down on sustained 0.20")
ev = feed3(p3, 0.50, 1.4, 1.25); check(ev==[], "held through hysteresis band (0.50)")
check(p3.held_fingers() == {"index"}, "index pinch held")
ev = feed3(p3, 0.80, 1.4, 1.40); check(len(ev)==1 and ev[0][1]=="up", "up on sustained 0.80")
ev = feed3(p3, 0.20, 1.4, 1.51, n=2); check(ev==[], "re-engage suppressed within cooldown")
ev = feed3(p3, 0.20, 1.4, 1.70); check(len(ev)==1 and ev[0][1]=="down", "re-engage after cooldown")
# middle pinch blocked while index held
ev = feed3(p3, 0.20, 0.20, 1.85)
check(ev==[] and p3.held_fingers()=={"index"}, "one button at a time")

# 4. Rewind
p4 = GuiPointer([1.5], 0.40, 0.60)
for i in range(10):
    p4.record(i*0.033, 0.1+0.05*i, 0.5)
pos = p4.rewind(9*0.033 - 0.12)  # ~3.6 frames back -> index 5 (t=0.165<=0.177)
check(pos is not None and abs(pos[0]-0.35) < 1e-9, f"rewind returns past position ({pos})")
pos2 = p4.rewind(-1.0)
check(pos2[0]==0.1, "rewind before trail start returns oldest")

# 5. Hand lost: hold then force-release after timeout
p5 = GuiPointer([1.5], 0.40, 0.60)
p5.update_pinch(g(0.5,0.5,pi=0.3), 1.0)
ev = p5.hand_lost(1.5); check(ev==[], "no release right after hand lost")
ev = p5.hand_lost(2.6); check(len(ev)==1 and ev[0][1]=="up", "force release after 1s missing")

# 6. Clutch: reset_tracking means next track() is a seed frame (no delta jump)
p6 = GuiPointer([1.5], 0.40, 0.60)
p6.track(g(0.5,0.5), 1.0); p6.track(g(0.5,0.5), 1.033)
p6.reset_tracking()
d = p6.track(g(0.9,0.9), 2.0)
check(d is None, "after clutch, re-entry seeds instead of jumping")

# 7. One Euro reset clears derivative state
from filters import OneEuroFilter
f = OneEuroFilter(1.5, 3.0)
f.filter(0.5, 0.0); f.filter(0.9, 0.033)
f.reset()
check(f.filter(0.1, 5.0) == 0.1, "filter reset -> passthrough on seed")

print("ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)

import math
from myomouse.arena import Mapping


def feed(m, angs, w=1000, h=800):
    out = None
    for a in angs:
        out = m.update(a, 0.02, w, h)
    return out


def test_relative_moves_left_when_turning():
    m = Mapping("pitch")
    m.mode = "relative"
    feed(m, [(0, 0, 0)] * 30)
    out = feed(m, [(0.01 * i, 0, 0) for i in range(1, 6)])
    assert out[0] == "delta" and out[1] < 0


def test_absolute_returns_to_centre_at_origin():
    m = Mapping("roll")
    m.mode = "absolute"
    feed(m, [(0.3, 0.1, 0.2)] * 50)
    m.set_origin()
    out = feed(m, [(0.3, 0.1, 0.2)] * 5)
    assert out[0] == "abs" and abs(out[1] - 500) < 1 and abs(out[2] - 400) < 1
    out = feed(m, [(0.3, 0.1, 0.3)] * 50)   # roll drives y on a table
    assert out[2] < 400


def test_joystick_deadzone_and_rate():
    m = Mapping("pitch")
    m.mode = "joystick"
    feed(m, [(0, 0, 0)] * 10)
    assert feed(m, [(0.03, 0, 0)] * 50)[1:] == (0.0, 0.0)   # inside dead zone
    dx = feed(m, [(0.2, 0, 0)] * 50)[1]
    assert dx < 0 and abs(dx - (-(0.2 - 0.06) * 2500 * 0.02)) < 1


def test_calibration_lopsided_range_maps_to_screen_edges():
    m = Mapping("pitch")
    m.mode = "absolute"
    feed(m, [(0, 0, 0)] * 20)
    m.start_calibration(0.0)
    # comfortable yaw from -0.15 (outwards) to +0.65 (inwards): lopsided around the start pose
    sweep = [(-0.15 + 0.8 * (i % 100) / 99, 0.1 * math.sin(i / 7), 0) for i in range(600)]
    feed(m, sweep)
    assert m.finish_calibration()
    assert abs(m.origin[0] - 0.25) < 0.05            # middle of the range, not the start pose
    m.sm = None
    left = feed(m, [(0.65 * 0.95, 0, 0)] * 80)[1]     # near the inward edge
    right = feed(m, [(-0.15 * 0.95, 0, 0)] * 80)[1]   # near the outward edge
    assert left < 150 and right > 1000 - 150


def test_calibration_tilted_reach_still_reaches_corners():
    import numpy as np
    m = Mapping("roll")
    m.mode = "absolute"
    feed(m, [(0, 0, 0)] * 20)
    m.start_calibration(0.0)
    # reach is a tilted ellipse in (yaw, roll): going outwards limits roll and vice versa
    t = np.linspace(0, 12 * np.pi, 800)
    a, b, th = 0.5, 0.12, math.radians(35)
    yaw = 0.2 + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th)
    roll = a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)
    feed(m, [(y, 0, rr) for y, rr in zip(yaw, roll)])
    assert m.finish_calibration()
    xs, ys = [], []
    for k in t[:200]:
        m.sm = None
        o = feed(m, [(y, 0, rr) for y, rr in [(0.2 + a * math.cos(k) * math.cos(th) - b * math.sin(k) * math.sin(th),
                                               a * math.cos(k) * math.sin(th) + b * math.sin(k) * math.cos(th))]] * 60)
        xs.append(o[1]); ys.append(o[2])
    xs, ys = np.array(xs), np.array(ys)
    for cx, cy in [(0, 0), (1000, 0), (0, 800), (1000, 800)]:   # every corner is touched by the reach edge
        assert np.min(np.hypot((xs - cx) / 1000, (ys - cy) / 800)) < 0.12


def test_relative_slow_movement_is_not_lost():
    m = Mapping("pitch")
    m.mode = "relative"
    feed(m, [(0, 0, 0)] * 50)
    total = 0.0
    for i in range(1, 501):                       # 0.2 rad over 10 s = ~1.1 deg/s, very slow
        out = m.update((0.0004 * i, 0, 0), 0.02, 1000, 800)
        total += out[1] if out else 0.0
    assert abs(total - (-0.2 * 1800)) < 0.05 * 360


def test_claw_two_stops_click_and_compensate_lag():
    from myomouse.arena import Claw
    c = Claw(); c.on = True; c.sweep_s = 2.0; c.lag = 0.2  # explicit
    t = 0.0
    for _ in range(50):              # 1 s -> pos 0.5, moving right
        t += 0.02; c.update(t, 0.02)
    assert not c.stop(t)
    assert abs(c.x - 0.4) < 0.02     # backed up 0.2 s of travel
    assert not c.stop(t + 0.1)       # ignored: pause after the first stop
    t += 0.4
    for _ in range(25):
        t += 0.02; c.update(t, 0.02)
    assert c.stop(t) and 0.1 < c.y < 0.3


def test_band_mode_leaves_cursor_to_the_band_pointer():
    m = Mapping("roll")
    assert m.mode == "band"
    assert feed(m, [(0.01 * i, 0, 0) for i in range(20)]) is None

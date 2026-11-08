"""Arm-on-table pointing: yaw sweep -> x, forearm roll -> y (myomouse/pointer.py)."""

import math

from myomouse import config as C
from myomouse.pointer import Pointer, roll_angle


def quat(axis, deg):
    """Myo-style int16 quaternion (w, x, y, z) for a rotation of `deg` about a world axis."""
    h = math.radians(deg) / 2
    s = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(int(round(v * 16384)) for v in (math.cos(h), s[0] * math.sin(h), s[1] * math.sin(h), s[2] * math.sin(h)))


def run(ptr, qs):
    tx = ty = 0.0
    for q in qs:
        dx, dy = ptr.update(q)
        tx, ty = tx + dx, ty + dy
    for _ in range(200):           # let the smoothing settle
        dx, dy = ptr.update(qs[-1])
        tx, ty = tx + dx, ty + dy
    return tx, ty


def test_roll_angle_matches_rotation_about_forearm():
    assert abs(math.degrees(roll_angle(quat("x", 30))) - 30) < 0.5
    assert abs(roll_angle(quat("z", 40))) < 1e-3          # a sweep is not a roll


def test_roll_mode_maps_roll_to_y_and_sweep_to_x():
    p = Pointer("roll")
    tx, ty = run(p, [quat("x", a) for a in range(0, 21)])  # roll 20 deg
    assert abs(tx) < 1 and abs(ty) > 0.8 * math.radians(20) * C.GAIN_ROLL
    p = Pointer("roll")
    tx, ty = run(p, [quat("z", a) for a in range(0, 21)])  # sweep 20 deg
    assert abs(ty) < 1 and abs(tx) > 0.8 * math.radians(20) * C.GAIN_X


def test_pitch_mode_ignores_roll():
    p = Pointer("pitch")
    tx, ty = run(p, [quat("x", a) for a in range(0, 21)])
    assert abs(tx) < 1 and abs(ty) < 1

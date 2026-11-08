"""Game camera look (--move game): filtered aiming, smooth keep-turning past the edge, drift
absorbed (myomouse/pointer.py)."""

import math
import random

from myomouse import config as C
from myomouse.pointer import GamePointer, TablePointer, make_pointer
from tests.test_pointer_roll import quat


def turned(ptr, deg, settle=100, hold=50):
    """Turn to `deg` and settle there, then return the x movement over `hold` more samples."""
    ptr.update(quat("z", 0))
    for i in range(1, 21):
        ptr.update(quat("z", deg * i / 20))
    for _ in range(settle):
        ptr.update(quat("z", deg))
    return sum(ptr.update(quat("z", deg))[0] for _ in range(hold))


def test_small_turn_aims_then_stops():
    assert abs(turned(GamePointer(), 8)) < 1.0            # inside the edge: no movement once still


def test_aim_follows_the_turn():
    ptr = GamePointer()
    ptr.update(quat("z", 0))
    total = sum(ptr.update(quat("z", 10 * i / 25))[0] for i in range(1, 26))
    total += sum(ptr.update(quat("z", 10))[0] for _ in range(50))
    expect = C.GAME_AIM_X * (math.radians(10) - 25 * C.GAME_JITTER_X)   # each sample loses the jitter step
    assert abs(abs(total) - expect) < 0.05 * expect


def test_slow_turn_still_arrives():
    ptr = GamePointer()                                   # 10 deg/s, a slow aim
    ptr.update(quat("z", 0))
    total = sum(ptr.update(quat("z", 10 * i / 50))[0] for i in range(1, 51))
    assert abs(total) > 0.5 * C.GAME_AIM_X * math.radians(10)


def test_turning_speed_ramps_up_smoothly_and_is_capped():
    edge = math.degrees(C.GAME_EDGE_X)
    just_past = abs(turned(GamePointer(), edge + 3, settle=50))
    far = abs(turned(GamePointer(), 60, settle=50))
    very_far = abs(turned(GamePointer(), 80, settle=50))
    assert just_past < 0.1 * C.GAME_MAX_X                 # gentle right after the edge
    assert abs(far - C.GAME_MAX_X) < 0.1 * C.GAME_MAX_X   # 1 s at full speed
    assert abs(very_far - far) < 0.05 * C.GAME_MAX_X      # never faster than the cap
    assert turned(GamePointer(), 60) * turned(GamePointer(), -60) < 0


def test_jitter_is_filtered_when_on(monkeypatch):
    monkeypatch.setattr(C, "GAME_FILTER", True)
    rng = random.Random(0)
    ptr = GamePointer()
    moves = [abs(ptr.update(quat("z", rng.gauss(0, 0.3)))[0]) for _ in range(300)]
    unfiltered = C.GAME_AIM_X * math.radians(0.3) * 1.1   # ~ one raw sample-to-sample jump
    assert sum(moves[50:]) / 250 < 0.3 * unfiltered


def test_drift_does_not_creep_into_turning_with_recenter(monkeypatch):
    monkeypatch.setattr(C, "GAME_RECENTER_S", 8.0)
    ptr = GamePointer()                                   # band drifts 1 deg/s for 30 s
    ptr.update(quat("z", 0))
    last = [ptr.update(quat("z", i / 50))[0] for i in range(1, 1501)]
    assert abs(sum(last[-50:])) < 0.2 * C.GAME_MAX_X      # still not turning on its own


def test_same_direction_as_table():
    t = TablePointer()
    t.update(quat("z", 0))
    aim = sum(t.update(quat("z", 3 * i))[0] for i in range(1, 11))
    assert aim * turned(GamePointer(), 40) > 0


def test_raw_by_default():
    ptr = GamePointer()                                   # no filter: a step arrives in one sample
    ptr.update(quat("z", 0))
    dx, _ = ptr.update(quat("z", 5))
    expect = C.GAME_AIM_X * (math.radians(5) - C.GAME_JITTER_X)
    assert abs(abs(dx) - expect) < 0.01 * expect


def test_resting_jitter_is_dropped():
    rng = random.Random(1)
    ptr = GamePointer()                                   # +-0.03 deg noise, arm still
    moves = [ptr.update(quat("z", rng.gauss(0, 0.015)))[0] for _ in range(200)]
    assert sum(abs(m) for m in moves) < 1.0


def test_make_pointer_knows_game():
    assert isinstance(make_pointer("game"), GamePointer)

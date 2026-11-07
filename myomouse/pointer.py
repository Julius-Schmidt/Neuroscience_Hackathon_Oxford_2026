"""Turn the pointer band's orientation into cursor movement.

We use the Myo's orientation quaternion rather than raw gyro axes: the band's x-axis runs
along the forearm, so we track where the forearm *points* (yaw / pitch). That keeps the
mapping the same no matter how the band is rotated on the arm or how the wrist is rolled.
Only changes in angle are used, so slow yaw drift (no magnetometer) does not matter.
"""

import math

import numpy as np

from . import config as C

QUAT_SCALE = 16384.0  # Myo quaternion units


def quat_to_forearm_dir(q) -> np.ndarray:
    """Myo quaternion (w, x, y, z as int16) -> unit vector of the band's x-axis in world frame."""
    w, x, y, z = (v / QUAT_SCALE for v in q)
    # first column of the rotation matrix = R @ [1, 0, 0]
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])


def yaw_pitch(q):
    d = quat_to_forearm_dir(q)
    d /= np.linalg.norm(d) + 1e-9
    return math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2])))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Pointer:
    """Feed quaternions; get (dx, dy) pixel deltas back."""

    def __init__(self):
        self.prev = None
        self.vx = self.vy = 0.0

    def update(self, q):
        yaw, pitch = yaw_pitch(q)
        if self.prev is None:
            self.prev = (yaw, pitch)
            return 0.0, 0.0
        dyaw = wrap(yaw - self.prev[0])
        dpitch = pitch - self.prev[1]
        self.prev = (yaw, pitch)
        if abs(dyaw) < C.DEADZONE_RAD:
            dyaw = 0.0
        if abs(dpitch) < C.DEADZONE_RAD:
            dpitch = 0.0
        # turning the arm left (yaw +) moves the cursor left; raising it moves the cursor up
        dx = -dyaw * C.GAIN_X * (-1 if C.INVERT_X else 1)
        dy = -dpitch * C.GAIN_Y * (-1 if C.INVERT_Y else 1)
        a = C.SMOOTHING
        self.vx = a * self.vx + (1 - a) * dx
        self.vy = a * self.vy + (1 - a) * dy
        return self.vx, self.vy

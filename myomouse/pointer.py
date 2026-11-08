"""Turn the pointer band's IMU into cursor movement. Styles (--move):

air (Pointer): where the forearm points in 3D.
table (TablePointer): forearm on the table, orientation only (no drift):
    left/right = turn the forearm on the table (yaw)
    up/down    = tilt the hand onto its side (roll), or lift the hand a little (pitch): TABLE_Y
slide (SlidePointer): forearm on the table, sliding measured with the accelerometer (can drift).

We use the Myo's orientation quaternion rather than raw gyro axes: the band's x-axis runs
along the forearm, so we track where the forearm *points* (yaw / pitch). That keeps the
mapping the same no matter how the band is rotated on the arm or how the wrist is rolled.
Only changes in angle are used, so slow yaw drift (no magnetometer) does not matter.

Slide mode (experimental; integrating an accelerometer drifts):
  left/right   = turning the forearm on the table (yaw, same as air mode), plus sideways sliding
  forward/back = sliding the arm along the table, from the accelerometer: the acceleration
                 minus gravity, in the table plane, integrated to a speed. Integrating
                 accelerometers drifts, so the speed leaks away and is reset to 0 whenever
                 the arm is still for a moment (TABLE_STILL_*). Forward = where the forearm points.
"""

import math

import numpy as np

from . import config as C

QUAT_SCALE = 16384.0  # Myo quaternion units
ACC_SCALE = 2048.0    # accelerometer units per g
GYRO_SCALE = 16.0     # gyro units per deg/s
IMU_HZ = 50.0
G = 9.81


def quat_to_forearm_dir(q) -> np.ndarray:
    """Myo quaternion (w, x, y, z as int16) -> unit vector of the band's x-axis in world frame."""
    w, x, y, z = (v / QUAT_SCALE for v in q)
    # first column of the rotation matrix = R @ [1, 0, 0]
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix band -> world (columns = band axes in world frame)."""
    w, x, y, z = (v / QUAT_SCALE for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_pitch(q):
    d = quat_to_forearm_dir(q)
    d /= np.linalg.norm(d) + 1e-9
    return math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2])))


def roll_angle(q) -> float:
    """Rotation about the band's x-axis (the forearm): palm down <-> thumb up.
    Euler roll (ZYX), accurate while the forearm is roughly level, e.g. resting on a table."""
    w, x, y, z = (v / QUAT_SCALE for v in q)
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_pointer(move: str):
    kinds = {"air": Pointer, "table": TablePointer, "slide": SlidePointer, "game": GamePointer}
    if move not in kinds:
        raise ValueError(move)
    return kinds[move]()


def soft_deadzone(d: float, dz: float) -> float:
    """Shrink by dz instead of cutting at dz: no jump when a slow movement crosses the threshold."""
    return math.copysign(max(abs(d) - dz, 0.0), d)


def forearm_roll(q) -> float:
    """Rotation of the band around the forearm axis (radians, arbitrary zero: only changes are used)."""
    R = quat_to_matrix(q)
    f, b = R[:, 0], R[:, 1]
    h = np.cross([0.0, 0.0, 1.0], f)  # horizontal, across the forearm
    nh = np.linalg.norm(h)
    if nh < 1e-3:                      # forearm pointing straight up/down: undefined
        return 0.0
    h /= nh
    v = np.cross(f, h)
    return math.atan2(float(b @ v), float(b @ h))


class Pointer:
    """Air mode. Feed IMU samples; get (dx, dy) pixel deltas back."""

    def __init__(self, y_axis=None):
        self.y_axis = y_axis or C.POINTER_Y_AXIS          # "pitch" (arm in the air) or "roll" (arm on a table)
        self.prev = None
        self.vx = self.vy = 0.0

    def update(self, q, acc=None, gyro=None):
        yaw, pitch = yaw_pitch(q)
        second = roll_angle(q) if self.y_axis == "roll" else pitch
        if self.prev is None:
            self.prev = (yaw, second)
            return 0.0, 0.0
        dyaw = wrap(yaw - self.prev[0])
        dsec = wrap(second - self.prev[1])
        # dead zone = hysteresis: a small change is held back (not thrown away) until it adds
        # up past the threshold, so slow deliberate movements still move the cursor
        if abs(dyaw) < C.DEADZONE_RAD:
            dyaw = 0.0
        else:
            self.prev = (yaw, self.prev[1])
        if abs(dsec) < C.DEADZONE_RAD:
            dsec = 0.0
        else:
            self.prev = (self.prev[0], second)
        # turning the arm left (yaw +) moves the cursor left; raising it moves the cursor up
        dx = -dyaw * C.GAIN_X * (-1 if C.INVERT_X else 1)
        if self.y_axis == "roll":
            dy = -dsec * C.GAIN_ROLL * (-1 if C.INVERT_ROLL else 1)
        else:
            dy = -dsec * C.GAIN_Y * (-1 if C.INVERT_Y else 1)
        a = C.SMOOTHING
        self.vx = a * self.vx + (1 - a) * dx
        self.vy = a * self.vy + (1 - a) * dy
        return self.vx, self.vy


class TablePointer:
    """Table mode, orientation only. Feed IMU samples; get (dx, dy) pixel deltas back."""

    def __init__(self):
        self.prev = None
        self.vx = self.vy = 0.0

    def update(self, q, acc=None, gyro=None):
        yaw, pitch = yaw_pitch(q)
        second = forearm_roll(q) if C.TABLE_Y == "roll" else pitch
        if self.prev is None:
            self.prev = (yaw, second)
            return 0.0, 0.0
        dyaw, d2 = wrap(yaw - self.prev[0]), wrap(second - self.prev[1])
        self.prev = (yaw, second)
        dyaw = 0.0 if abs(dyaw) < C.DEADZONE_RAD else dyaw   # left/right: unchanged, already smooth
        d2 = soft_deadzone(d2, C.TABLE_DEADZONE_Y)
        # turn left = cursor left; roll / lift = cursor up (flip with TABLE_INVERT_*)
        dx = -dyaw * C.TABLE_GAIN_TURN * (-1 if C.TABLE_INVERT_X else 1)
        dy = -d2 * C.TABLE_GAIN_Y * (-1 if C.TABLE_INVERT_Y else 1)
        a = C.SMOOTHING
        self.vx = a * self.vx + (1 - a) * dx
        self.vy = a * self.vy + (1 - a) * dy
        return self.vx, self.vy


class OneEuro:
    """One Euro filter (Casiez et al. 2012): heavy smoothing while the signal moves slowly (no
    jitter), light while it moves fast (no lag). For one angle, in radians, at IMU_HZ."""

    def __init__(self, min_hz, beta, d_hz=1.0, rate=IMU_HZ):
        self.min_hz, self.beta, self.d_hz, self.dt = min_hz, beta, d_hz, 1.0 / rate
        self.x = self.dx = None

    def _alpha(self, hz):
        tau = 1.0 / (2 * math.pi * hz)
        return 1.0 / (1.0 + tau / self.dt)

    def __call__(self, x):
        if self.x is None:
            self.x, self.dx = x, 0.0
            return x
        a = self._alpha(self.d_hz)
        self.dx += a * ((x - self.x) / self.dt - self.dx)
        a = self._alpha(self.min_hz + self.beta * abs(self.dx))
        self.x += a * (x - self.x)
        return self.x


class GamePointer:
    """Camera look for games, forearm on the table (yaw = left/right, TABLE_Y = up/down).

    aim:  the angle change moves the camera directly (GAME_AIM_*), no smoothing; only per-sample
          changes below GAME_JITTER_* are dropped (soft: no jump). GAME_FILTER = True adds a One Euro filter
    turn: an arm only turns so far, a game camera has to go all the way round: past GAME_EDGE_*
          from the centre it keeps turning, starting gently (quadratic ramp over GAME_FULL_*) up
          to GAME_MAX_* px/s. Back inside = stop.
    drift: optional (GAME_RECENTER_S > 0): inside the edge the centre slowly follows the arm,
          so the band's slow yaw drift never moves the edge under you."""

    def __init__(self):
        self.filt = [OneEuro(C.GAME_FILTER_MIN_HZ, C.GAME_FILTER_BETA) for _ in range(2)]
        self.raw = None          # last raw angles (to unwrap)
        self.cont = None         # unwrapped angles
        self.prev = None         # last filtered angles
        self.origin = None

    def angles(self, q):
        yaw, pitch = yaw_pitch(q)
        return yaw, (forearm_roll(q) if C.TABLE_Y == "roll" else pitch)

    @staticmethod
    def turn_speed(offset, edge, full, top):
        """px/s for an offset (rad) from the centre: 0 inside the edge, then a smooth ramp up to top."""
        excess = abs(offset) - edge
        if excess <= 0:
            return 0.0
        return math.copysign(top * min(1.0, excess / full) ** 2, offset)

    def update(self, q, acc=None, gyro=None):
        raw = self.angles(q)
        if self.raw is None:
            self.raw, self.cont = raw, list(raw)
        else:
            self.cont = [c + wrap(r - p) for c, r, p in zip(self.cont, raw, self.raw)]
            self.raw = raw
        f = [flt(c) for flt, c in zip(self.filt, self.cont)] if C.GAME_FILTER else list(self.cont)
        if self.prev is None:
            self.prev, self.origin = f, list(f)
            return 0.0, 0.0
        da = soft_deadzone(f[0] - self.prev[0], C.GAME_JITTER_X)   # small jitter filter only
        db = soft_deadzone(f[1] - self.prev[1], C.GAME_JITTER_Y)
        self.prev = f
        ox, oy = f[0] - self.origin[0], f[1] - self.origin[1]
        sx = C.GAME_AIM_X * da + self.turn_speed(ox, C.GAME_EDGE_X, C.GAME_FULL_X, C.GAME_MAX_X) / IMU_HZ
        sy = C.GAME_AIM_Y * db + self.turn_speed(oy, C.GAME_EDGE_Y, C.GAME_FULL_Y, C.GAME_MAX_Y) / IMU_HZ
        if C.GAME_RECENTER_S > 0:
            k = 1.0 / (C.GAME_RECENTER_S * IMU_HZ)
            if abs(ox) < C.GAME_EDGE_X:
                self.origin[0] += k * ox
            if abs(oy) < C.GAME_EDGE_Y:
                self.origin[1] += k * oy
        # same directions as TablePointer: turn left = look left, tilt = look up
        dx = -sx * (-1 if C.TABLE_INVERT_X else 1)
        dy = -sy * (-1 if C.TABLE_INVERT_Y else 1)
        return dx, dy


class SlidePointer:
    """Slide mode (accelerometer). Feed IMU samples (quat, accel, gyro); get (dx, dy) pixel deltas back."""

    def __init__(self):
        self.prev_yaw = None
        self.grav = None                 # gravity in world frame (g), measured, so no axis convention needed
        self.vel = np.zeros(3)           # sliding speed in world frame (m/s)
        self.still = 0
        self.sx = self.sy = 0.0          # smoothed output

    def update(self, q, acc, gyro):
        R = quat_to_matrix(q)
        a = R @ (np.asarray(acc, dtype=float) / ACC_SCALE)      # world frame, in g (gravity included)
        spin = float(np.linalg.norm(gyro)) / GYRO_SCALE          # deg/s

        if self.grav is None:
            self.grav = a.copy()
        lin = a - self.grav
        still = np.linalg.norm(lin) < C.TABLE_STILL_G and spin < C.TABLE_STILL_DPS
        self.still = self.still + 1 if still else 0
        # follow gravity quickly while still, barely while moving (else it eats the slide)
        k = 0.1 if still else 0.002
        self.grav = (1 - k) * self.grav + k * a

        up = self.grav / (np.linalg.norm(self.grav) + 1e-9)
        lin_h = lin - (lin @ up) * up                             # only the part along the table
        if np.linalg.norm(lin_h) < C.TABLE_ACC_DEADZONE_G:
            lin_h = np.zeros(3)
        dt = 1.0 / IMU_HZ
        self.vel = C.TABLE_LEAK * self.vel + lin_h * G * dt
        if self.still >= C.TABLE_STILL_N:
            self.vel[:] = 0.0

        fwd = R[:, 0] - (R[:, 0] @ up) * up                       # forearm direction on the table
        nf = np.linalg.norm(fwd)
        if nf < 0.3:  # forearm pointing steeply up/down: not on the table, don't guess
            v_fwd = v_left = 0.0
        else:
            fwd /= nf
            left = np.cross(up, fwd)
            v_fwd, v_left = float(self.vel @ fwd), float(self.vel @ left)

        yaw, _ = yaw_pitch(q)
        dyaw = 0.0 if self.prev_yaw is None else wrap(yaw - self.prev_yaw)
        self.prev_yaw = yaw
        if abs(dyaw) < C.DEADZONE_RAD:
            dyaw = 0.0

        # turning / sliding left moves the cursor left; pushing forward moves it up
        dx = -dyaw * C.TABLE_GAIN_TURN - v_left * dt * C.TABLE_GAIN_SLIDE * C.TABLE_SLIDE_X
        dy = -v_fwd * dt * C.TABLE_GAIN_SLIDE
        dx *= -1 if C.TABLE_INVERT_X else 1
        dy *= -1 if C.TABLE_INVERT_Y else 1
        a_s = C.SMOOTHING
        self.sx = a_s * self.sx + (1 - a_s) * dx
        self.sy = a_s * self.sy + (1 - a_s) * dy
        return self.sx, self.sy


class AxisFilter:
    """Restrict cursor movement to one axis (config AXES / --axes): x, y, both, or dominant.

    dominant: follows the axis with the bigger (smoothed) speed, and only switches when the
    other axis is AXES_SWITCH times stronger, so it doesn't flicker on diagonal movements.
    """

    def __init__(self, axes=None):
        self.axes = C.AXES if axes is None else axes
        if self.axes not in ("both", "x", "y", "dominant"):
            raise ValueError(self.axes)
        self.ex = self.ey = 0.0
        self.current = "x"

    def __call__(self, dx, dy):
        if self.axes == "both":
            return dx, dy
        if self.axes == "x":
            return dx, 0.0
        if self.axes == "y":
            return 0.0, dy
        a = C.AXES_SMOOTH
        self.ex = a * self.ex + (1 - a) * abs(dx)
        self.ey = a * self.ey + (1 - a) * abs(dy)
        if self.current == "x" and self.ey > C.AXES_SWITCH * self.ex:
            self.current = "y"
        elif self.current == "y" and self.ex > C.AXES_SWITCH * self.ey:
            self.current = "x"
        return (dx, 0.0) if self.current == "x" else (0.0, dy)

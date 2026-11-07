"""
Joystick: converts hand position to a movement vector, modelled on how real
controller-look code works (see Controlify's InGameInputHandler/ControllerUtils):
a fixed reference point, deadzone + curve applied to the offset's magnitude
(not per-axis), recomputed fresh every frame.

The neutral point is set by closing the driving gesture (fist) itself — like
picking up a mouse and setting it down: wherever the hand is when the fist
settles becomes the new center. It then stays fixed across releases/re-grips
(closing the fist again doesn't move it) until `release_anchor()` is called —
an explicit "let go" (open palm) rather than every fist entry, since re-
anchoring on every grip is what a real stick never does and reintroduces
drift. See `update()`.

  neutral = anchored once (after a short arm delay, see update()), held until
            release_anchor() is called
  offset  = hand position - neutral, lightly smoothed to kill landmark jitter
  output  = deadzone-rescaled, curved offset, direction preserved

Curve exponent reference:
  <1  — steep initial response, plateaus early  (snappy)
  1.0 — linear
  >1  — slow near deadzone, accelerates outward
"""

import math
import time
import logging
from dataclasses import dataclass

log = logging.getLogger("neuratect")

DEADZONE  = 0.04
MAX_RANGE = 0.15
CURVE     = 0.4

# EMA alpha on the raw hand position before computing offset — only enough
# to kill single-frame landmark jitter. Higher = snappier/less lag. The old
# default (0.35) kept 65% of history, which is what produced a multi-frame
# "still moving after the hand stopped" tail.
SMOOTH = 0.5


@dataclass
class JoystickVector:
    x: float        # -1 → 1, wasd_curve applied
    y: float        # -1 → 1, wasd_curve applied
    raw_x: float    # -1 → 1, mouse_curve applied
    raw_y: float
    # Offset from the anchor in normalized frame units, before deadzone and
    # curve (x already mirrored, y positive = downward, same signs as x/y
    # above). x/y saturate at max_range, so they cannot distinguish "at full
    # deflection" from "pushed well past it" — this can, which is what the
    # sprint trigger reads. 0.0 when uncalibrated/inactive.
    offset_x: float = 0.0
    offset_y: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return self.raw_x == 0.0 and self.raw_y == 0.0

    @property
    def offset_dist(self) -> float:
        return math.hypot(self.offset_x, self.offset_y)


def _apply_radial(ox: float, oy: float, deadzone: float, max_range: float, curve: float) -> tuple[float, float]:
    """Deadzone + curve applied to the offset's magnitude, preserving its direction."""
    dist = math.hypot(ox, oy)
    if dist < deadzone:
        return 0.0, 0.0
    frac = min(1.0, (dist - deadzone) / (max_range - deadzone))
    scaled = frac ** curve
    return scaled * (ox / dist), scaled * (oy / dist)


# Fist must be held this long before its position locks in as neutral — skips
# the first few frames of a gesture transition, which are still unstable
# (both in landmark position and recognition confidence).
ARM_DELAY = 0.12


class HandJoystick:
    def __init__(
        self,
        deadzone:    float = DEADZONE,
        max_range:   float = MAX_RANGE,
        wasd_curve:  float = CURVE,
        mouse_curve: float = 1.0,
        smooth:      float = SMOOTH,
        arm_delay:   float = ARM_DELAY,
        name:        str = "",
    ):
        self._dz          = deadzone
        self._max          = max_range
        self._wasd_curve   = wasd_curve
        self._mouse_curve  = mouse_curve
        self._smooth       = smooth
        self._arm_delay    = arm_delay
        self._name         = name or "stick"

        self._neutral:    tuple[float, float] | None = None
        self._smoothed:   tuple[float, float] | None = None
        self._entry_time: float | None = None

    def calibrate(self, x: float, y: float) -> None:
        self._neutral  = (x, y)
        self._smoothed = None
        log.info(f"[joystick] {self._name} calibrated neutral: ({x:.3f}, {y:.3f})")

    def reset_smoothing(self) -> None:
        """Drop stale position smoothing (e.g. re-entering camera-look after
        GUI-pointer mode) without touching the anchor — the EMA state is from
        wherever the hand was frames-to-minutes ago and would jump."""
        self._smoothed = None

    def release_anchor(self) -> None:
        """Explicit 'let go' (e.g. open palm) — next fist entry re-anchors."""
        if self._neutral is not None:
            self._neutral = None
            log.info(f"[joystick] {self._name} anchor released")
        self._smoothed   = None
        self._entry_time = None

    @property
    def is_calibrated(self) -> bool:
        return self._neutral is not None

    def update(self, active: bool, x: float, y: float) -> JoystickVector:
        """
        Call every frame with whether the driving gesture (fist) is active.
        The anchor locks in once, after `arm_delay` seconds of a sustained
        fist, and then stays fixed across releases and re-grips — call
        `release_anchor()` to explicitly clear it (e.g. on open palm).
        """
        if not active:
            self._entry_time = None
            return JoystickVector(0.0, 0.0, 0.0, 0.0)

        if self._neutral is None:
            now = time.monotonic()
            if self._entry_time is None:
                self._entry_time = now
            if now - self._entry_time < self._arm_delay:
                return JoystickVector(0.0, 0.0, 0.0, 0.0)
            self.calibrate(x, y)

        return self.read(x, y)

    def read(self, x: float, y: float) -> JoystickVector:
        if self._neutral is None:
            return JoystickVector(0.0, 0.0, 0.0, 0.0)

        if self._smoothed is None:
            self._smoothed = (x, y)
        else:
            sx, sy = self._smoothed
            self._smoothed = (
                self._smooth * x + (1.0 - self._smooth) * sx,
                self._smooth * y + (1.0 - self._smooth) * sy,
            )

        nx, ny = self._neutral
        sx, sy = self._smoothed
        ox = -(sx - nx)   # flip x: camera is mirrored
        oy =   sy - ny

        wx, wy = _apply_radial(ox, oy, self._dz, self._max, self._wasd_curve)
        mx, my = _apply_radial(ox, oy, self._dz, self._max, self._mouse_curve)

        return JoystickVector(x=wx, y=wy, raw_x=mx, raw_y=my, offset_x=ox, offset_y=oy)

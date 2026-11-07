"""
One Euro filter (Casiez, Roussel, Vogel 2012) — adaptive low-pass for noisy
human-input signals. At low speed the cutoff drops to min_cutoff (heavy
smoothing — kills landmark jitter while hovering); as speed rises the cutoff
rises with it (beta), so deliberate movement passes through with little lag.
This is the standard fix for the smoothing-vs-lag tradeoff a fixed EMA can't
solve: any single alpha is either too jittery for precision pointing or too
laggy for fast moves.
"""

import math


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff  # Hz — smoothing floor at zero speed
        self.beta = beta              # cutoff gain per unit/sec of signal speed
        self.d_cutoff = d_cutoff      # Hz — cutoff for the speed estimate itself
        self._prev_t: float | None = None
        self._prev_x: float = 0.0
        self._prev_dx: float = 0.0

    def reset(self) -> None:
        self._prev_t = None
        self._prev_dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self._prev_t is None:
            self._prev_t = t
            self._prev_x = x
            self._prev_dx = 0.0
            return x
        dt = max(t - self._prev_t, 1e-4)
        dx = (x - self._prev_x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._prev_dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._prev_x
        self._prev_t, self._prev_x, self._prev_dx = t, x_hat, dx_hat
        return x_hat

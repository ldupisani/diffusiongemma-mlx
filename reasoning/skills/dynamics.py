"""Dynamical systems & chaos — the logistic map and the Mandelbrot set. Zero model calls.

The logistic map x ← r·x·(1−x) is the canonical route to chaos: logistic_map iterates it,
logistic_fixed_points gives its fixed points (0 and 1−1/r), and lyapunov_logistic estimates the
Lyapunov exponent (positive ⇒ chaotic). mandelbrot tests whether c stays bounded under
z ← z²+c (membership in the Mandelbrot set).
"""

from __future__ import annotations

import math

from .base import register


@register("dynamics.logistic_map", "logistic map x←r·x·(1−x) after n iterations from x0",
          [((3.2, 0.5, 2), 0.512), ((4, 0.2, 1), 0.64), ((2, 0.1, 1), 0.18)])
def logistic_map(r: float, x0: float, n: int) -> float:
    r, x = float(r), float(x0)
    for _ in range(int(n)):
        x = r * x * (1 - x)
    return round(x, 6)


@register("dynamics.logistic_fixed_points", "fixed points of the logistic map (0 and 1−1/r)",
          [((3,), "0, 0.6667"), ((2,), "0, 0.5"), ((1,), "0, 0.0")])
def logistic_fixed_points(r: float) -> str:
    return f"0, {round(1 - 1 / float(r), 4)}"


@register("dynamics.lyapunov_logistic", "Lyapunov exponent of the logistic map (>0 ⇒ chaotic)",
          [((4, 0.2, 2000), 0.6932), ((3.2, 0.4, 2000), -0.913585)])
def lyapunov_logistic(r: float, x0: float, n: int) -> float:
    r, x, total = float(r), float(x0), 0.0
    for _ in range(int(n)):
        total += math.log(abs(r * (1 - 2 * x)))
        x = r * x * (1 - x)
    return round(total / int(n), 6)


@register("dynamics.mandelbrot", "is c in the Mandelbrot set? (z←z²+c stays bounded)",
          [((0, 0, 100), "bounded (in the set)"), ((1, 0, 100), "escapes at iteration 3"),
           ((-1, 0, 100), "bounded (in the set)")])
def mandelbrot(c_real: float, c_imag: float, max_iter: int) -> str:
    c = complex(float(c_real), float(c_imag))
    z = 0j
    for k in range(1, int(max_iter) + 1):
        z = z * z + c
        if abs(z) > 2:
            return f"escapes at iteration {k}"
    return "bounded (in the set)"

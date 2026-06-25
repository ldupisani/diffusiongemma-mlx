"""Special relativity — Lorentz factor, time dilation, length contraction, and relativistic kinetic energy.

All operations assume speed is given as a fraction of the speed of light (e.g., 0.8 for 0.8c).
Gamma (Lorentz factor) is dimensionless; time dilation and length contraction preserve units;
kinetic energy is in joules.
"""

from __future__ import annotations

import math

from .base import register

c = 299792458  # speed of light in m/s


@register("special_relativity.lorentz_factor", "Lorentz factor gamma = 1/sqrt(1 - v²/c²)",
          [((0.0,), 1.0), ((0.6,), 1.25), ((0.8,), 1.666667)])
def lorentz_factor(speed_fraction: float) -> float:
    """Compute Lorentz factor for a speed given as a fraction of c (0 ≤ v < c)."""
    if not (0 <= speed_fraction < 1):
        return None
    return round(1 / math.sqrt(1 - speed_fraction**2), 6)


@register("special_relativity.time_dilation", "dilated time = proper time × gamma",
          [((1.0, 1.0), 1.0), ((1.0, 1.25), 1.25), ((2.0, 1.666667), 3.333334)])
def time_dilation(proper_time: float, gamma: float) -> float:
    """Compute dilated time observed for a proper time in a moving frame."""
    return round(proper_time * gamma, 6)


@register("special_relativity.length_contraction", "contracted length = proper length / gamma",
          [((10.0, 1.0), 10.0), ((10.0, 1.25), 8.0), ((10.0, 2.0), 5.0)])
def length_contraction(proper_length: float, gamma: float) -> float:
    """Compute contracted length observed for a proper length in a moving frame."""
    return round(proper_length / gamma, 6)


@register("special_relativity.relativistic_kinetic_energy", "KE = (gamma - 1) × m × c²",
          [((1.0, 1.0), 0.0), ((0.000001, 1.25), 22468879468.42044),
           ((0.000001, 2.0), 89875517873.68176)])
def relativistic_kinetic_energy(mass_kg: float, gamma: float) -> float:
    """Compute relativistic kinetic energy in joules."""
    ke = (gamma - 1) * mass_kg * c**2
    return round(ke, 6)

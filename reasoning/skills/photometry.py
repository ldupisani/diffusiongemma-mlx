"""Photometry and illumination engineering — lux, lumens, candela, solid angle. Zero model calls.

Covers illuminance (from flux and area, or from intensity and distance), luminous efficacy,
luminous intensity, and inverse-square illuminance ratios.
"""

from __future__ import annotations

import math

from .base import register


@register("photometry.illuminance_from_flux_area",
          "illuminance E = luminous flux F (lumens) / area A (square meters), in lux",
          [((800, 5), 160.0), ((1200, 10), 120.0), ((500, 2), 250.0)])
def illuminance_from_flux_area(flux: float, area: float) -> float:
    """Illuminance from luminous flux and area: E = F / A."""
    return round(flux / area, 6)


@register("photometry.illuminance_inverse_square",
          "illuminance E = luminous intensity I (candela) / distance d squared, in lux",
          [((100, 2), 25.0), ((200, 5), 8.0), ((400, 10), 4.0)])
def illuminance_inverse_square(intensity: float, distance: float) -> float:
    """Illuminance from intensity and distance via inverse-square law: E = I / d²."""
    return round(intensity / (distance ** 2), 6)


@register("photometry.luminous_efficacy",
          "luminous efficacy K = luminous flux F (lumens) / electrical power P (watts), in lumens per watt",
          [((800, 10), 80.0), ((1200, 20), 60.0), ((400, 5), 80.0)])
def luminous_efficacy(flux: float, power: float) -> float:
    """Luminous efficacy: K = F / P, in lumens per watt (lm/W)."""
    return round(flux / power, 6)


@register("photometry.luminous_intensity",
          "luminous intensity I = luminous flux F (lumens) / solid angle omega (steradians), in candela",
          [((800, 4), 200.0), ((1000, 5), 200.0), ((628, 2), 314.0)])
def luminous_intensity(flux: float, solid_angle: float) -> float:
    """Luminous intensity: I = F / Ω, in candela (cd)."""
    return round(flux / solid_angle, 6)


@register("photometry.illuminance_ratio",
          "illuminance ratio at new distance: E₂ = E₁ × (d₁ / d₂)²",
          [((100, 2, 5), 16.0), ((50, 3, 6), 12.5), ((200, 1, 4), 12.5)])
def illuminance_ratio(illuminance1: float, distance1: float, distance2: float) -> float:
    """Inverse-square illuminance ratio: E₂ = E₁ × (d₁ / d₂)²."""
    return round(illuminance1 * (distance1 / distance2) ** 2, 6)
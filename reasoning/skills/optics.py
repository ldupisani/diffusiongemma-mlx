"""Optics — thin-lens imaging, magnification, lens power, refraction, and total internal reflection.
All operations are deterministic; inputs/outputs in SI units (meters, degrees)."""
from __future__ import annotations

import math

from .base import register


@register("optics.image_distance",
          "thin-lens image distance: 1/f = 1/do + 1/di; solve for di",
          [((10, 30), 15.0), ((0.2, 0.5), 0.333333)])
def image_distance(focal_length: float, object_distance: float) -> float:
    """di = (do·f) / (do - f); returns infinity if do = f."""
    if object_distance == focal_length:
        return float('inf')
    di = (object_distance * focal_length) / (object_distance - focal_length)
    return round(di, 6)


@register("optics.magnification",
          "linear magnification m = -di/do",
          [((15, 30), -0.5), ((20, 10), -2.0)])
def magnification(image_distance: float, object_distance: float) -> float:
    """Negative sign indicates inverted image."""
    m = -image_distance / object_distance
    return round(m, 6)


@register("optics.lens_power",
          "lens power in diopters: P = 1/f (focal length f in meters)",
          [((0.5,), 2.0), ((0.2,), 5.0)])
def lens_power(focal_length: float) -> float:
    """Diopters = 1/meters."""
    p = 1 / focal_length
    return round(p, 6)


@register("optics.snell_refraction",
          "Snell's law refraction angle: n1·sin(θ₁) = n2·sin(θ₂); solve for θ₂ in degrees",
          [((1, 30, 1.5), 19.471221), ((1.5, 30, 1), 48.590378)])
def snell_refraction(n1: float, theta1_deg: float, n2: float) -> float:
    """Returns refraction angle in degrees. Clamps sin to [-1, 1] to avoid domain errors."""
    theta1_rad = math.radians(theta1_deg)
    sin_theta2 = (n1 / n2) * math.sin(theta1_rad)
    sin_theta2 = max(-1.0, min(1.0, sin_theta2))
    theta2_rad = math.asin(sin_theta2)
    theta2_deg = math.degrees(theta2_rad)
    return round(theta2_deg, 6)


@register("optics.critical_angle",
          "critical angle for total internal reflection: θc = arcsin(n₂/n₁) in degrees",
          [((1.5, 1), 41.810315), ((1.33, 1), 48.753467)])
def critical_angle(n1: float, n2: float) -> float:
    """n1 > n2 for TIR to exist (light going from denser to rarer medium)."""
    if n1 <= n2:
        return None
    ratio = n2 / n1
    ratio = max(-1.0, min(1.0, ratio))
    theta_c_rad = math.asin(ratio)
    theta_c_deg = math.degrees(theta_c_rad)
    return round(theta_c_deg, 6)
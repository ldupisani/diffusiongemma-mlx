"""Trigonometry / angle skills — sin/cos/tan of degrees, and degree↔radian conversion."""

from __future__ import annotations

import math

from .base import register


@register("trig.sin_deg", "sine of an angle in degrees", [((30,), 0.5), ((90,), 1.0)])
def sin_deg(deg: float) -> float:
    return math.sin(math.radians(deg))


@register("trig.cos_deg", "cosine of an angle in degrees", [((60,), 0.5), ((0,), 1.0)])
def cos_deg(deg: float) -> float:
    return math.cos(math.radians(deg))


@register("trig.tan_deg", "tangent of an angle in degrees", [((45,), 1.0), ((0,), 0.0)])
def tan_deg(deg: float) -> float:
    return math.tan(math.radians(deg))


@register("trig.deg_to_rad", "convert degrees to radians",
          [((180,), 3.14159265), ((90,), 1.57079633)])
def deg_to_rad(deg: float) -> float:
    return math.radians(deg)


@register("trig.rad_to_deg", "convert radians to degrees",
          [((3.141592653589793,), 180.0), ((1.5707963267948966,), 90.0)])
def rad_to_deg(rad: float) -> float:
    return math.degrees(rad)

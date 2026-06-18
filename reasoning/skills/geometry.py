"""Geometry / measurement skills — area, perimeter, circumference, volume, surface area.

Each is a single closed-form formula. Unlike units (one factor table), geometry formulas
have heterogeneous inputs (radius vs length+width vs base+height vs legs), so they stay
distinct skills — but one generalized orchestrator dispatches (shape, measurement) to the
right one and extracts the labeled dimensions.
"""

from __future__ import annotations

import math

from .base import register


@register("geometry.circle_area", "area of a circle from its radius (pi r^2)",
          [((5,), 78.539816), ((1,), 3.141593)])
def circle_area(radius: float) -> float:
    return math.pi * radius ** 2


@register("geometry.circle_circumference", "circumference of a circle from its radius (2 pi r)",
          [((7,), 43.982297)])
def circle_circumference(radius: float) -> float:
    return 2 * math.pi * radius


@register("geometry.rectangle_area", "area of a rectangle (length x width)",
          [((4, 6), 24), ((8, 3), 24)])
def rectangle_area(length: float, width: float) -> float:
    return length * width


@register("geometry.rectangle_perimeter", "perimeter of a rectangle (2(length + width))",
          [((8, 3), 22)])
def rectangle_perimeter(length: float, width: float) -> float:
    return 2 * (length + width)


@register("geometry.rectangle_diagonal", "diagonal of a rectangle (sqrt(length^2 + width^2))",
          [((3, 4), 5.0)])
def rectangle_diagonal(length: float, width: float) -> float:
    return math.hypot(length, width)


@register("geometry.triangle_area", "area of a triangle (half base x height)",
          [((10, 5), 25.0)])
def triangle_area(base: float, height: float) -> float:
    return 0.5 * base * height


@register("geometry.square_area", "area of a square from its side (side^2)",
          [((9,), 81)])
def square_area(side: float) -> float:
    return side ** 2


@register("geometry.cube_volume", "volume of a cube from its side (side^3)",
          [((3,), 27)])
def cube_volume(side: float) -> float:
    return side ** 3


@register("geometry.sphere_volume", "volume of a sphere from its radius (4/3 pi r^3)",
          [((4,), 268.082573)])
def sphere_volume(radius: float) -> float:
    return 4 / 3 * math.pi * radius ** 3


@register("geometry.sphere_surface_area", "surface area of a sphere from its radius (4 pi r^2)",
          [((6,), 452.389342)])
def sphere_surface_area(radius: float) -> float:
    return 4 * math.pi * radius ** 2


@register("geometry.cylinder_volume", "volume of a cylinder from radius and height (pi r^2 h)",
          [((3, 10), 282.743339)])
def cylinder_volume(radius: float, height: float) -> float:
    return math.pi * radius ** 2 * height


@register("geometry.hypotenuse", "hypotenuse of a right triangle from its legs (sqrt(a^2+b^2))",
          [((3, 4), 5.0), ((5, 12), 13.0)])
def hypotenuse(a: float, b: float) -> float:
    return math.hypot(a, b)

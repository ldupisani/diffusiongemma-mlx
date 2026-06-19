"""Coordinate-geometry skills — distance, midpoint, slope between two points."""

from __future__ import annotations

import math

from .base import register


@register("coord.distance", "straight-line distance between two points (x1,y1)-(x2,y2)",
          [((0, 0, 3, 4), 5.0), ((0, 0, 5, 12), 13.0)])
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


@register("coord.slope", "slope of the line through two points: (y2-y1)/(x2-x1)",
          [((1, 2, 3, 6), 2.0), ((2, 3, 5, 3), 0.0)])
def slope(x1: float, y1: float, x2: float, y2: float) -> float:
    return (y2 - y1) / (x2 - x1)


@register("coord.midpoint", "midpoint of two points, formatted '(x, y)'",
          [((2, 4, 6, 8), "(4, 6)"), ((0, 0, 10, 10), "(5, 5)")])
def midpoint(x1: float, y1: float, x2: float, y2: float) -> str:
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    fmt = lambda v: str(int(v)) if float(v).is_integer() else str(v)
    return f"({fmt(mx)}, {fmt(my)})"

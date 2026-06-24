"""Differential geometry — curvature and arc length of a plane curve y = f(x). Zero model
calls (SymPy differentiates/integrates).

The signed curvature is κ = |f''| / (1 + f'^2)^(3/2); the radius of curvature is 1/κ; the arc
length over [a,b] is ∫ √(1 + f'^2) dx.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                        parse_expr, standard_transformations)

from .base import register

_x = sp.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _f(expr: str):
    return parse_expr(str(expr).replace("^", "**"), local_dict={"x": _x}, transformations=_T)


def _kappa(expr: str, x0: float) -> float:
    f = _f(expr)
    fp = float(sp.diff(f, _x).subs(_x, float(x0)))
    fpp = float(sp.diff(f, _x, 2).subs(_x, float(x0)))
    return abs(fpp) / (1 + fp ** 2) ** 1.5


@register("diffgeo.curvature", "curvature κ of y=f(x) at x=x0",
          [(("x^2", 0), 2.0), (("x^2", 1), 0.178885), (("sqrt(1-x^2)", 0), 1.0)])
def curvature(expr: str, x0: float) -> float:
    return round(_kappa(expr, x0), 6)


@register("diffgeo.radius_of_curvature", "radius of curvature 1/κ of y=f(x) at x=x0",
          [(("x^2", 0), 0.5), (("sqrt(1-x^2)", 0), 1.0)])
def radius_of_curvature(expr: str, x0: float) -> float:
    return round(1 / _kappa(expr, x0), 6)


@register("diffgeo.arc_length", "arc length of y=f(x) over [a,b]",
          [(("x", 0, 1), 1.414214), (("x^2", 0, 1), 1.478943)])
def arc_length(expr: str, a: float, b: float) -> float:
    f = _f(expr)
    integrand = sp.sqrt(1 + sp.diff(f, _x) ** 2)
    val = sp.integrate(integrand, (_x, float(a), float(b)))
    return round(float(sp.N(val)), 6)

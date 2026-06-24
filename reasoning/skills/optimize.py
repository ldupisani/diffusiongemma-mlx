"""Optimization — unconstrained optima of a smooth function, and gradient descent. Zero model
calls (SymPy finds the stationary points; gradient descent iterates in plain Python).

minimize/maximize solve f'(x)=0 (or ∇f=0 in two variables) and report the extreme value and its
location; gradient_descent runs the fixed-step update x ← x − lr·f'(x).
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                        parse_expr, standard_transformations)

from .base import register

_x, _y = sp.symbols("x y")
_T = standard_transformations + (implicit_multiplication_application,)


def _f(expr: str, two=False):
    # parse_expr with implicit multiplication so "4x" means 4*x (not just "4*x")
    return parse_expr(str(expr).replace("^", "**"), local_dict={"x": _x, "y": _y},
                      transformations=_T)


def _n(v):
    f = float(sp.N(v))
    return int(f) if f == int(f) else round(f, 4)


def _opt1(expr: str, maximize: bool) -> str:
    f = _f(expr)
    crit = [c for c in sp.solve(sp.diff(f, _x), _x) if c.is_real]
    if not crit:
        raise ValueError("no interior optimum")
    pairs = [(float(sp.N(f.subs(_x, c))), c) for c in crit]
    val, xs = max(pairs) if maximize else min(pairs)
    return f"{_n(val)} at x = {_n(xs)}"


@register("optimize.minimize", "minimum value of f(x) and where it occurs",
          [(("x**2 - 4*x + 7",), "3 at x = 2"), (("x**2 + 1",), "1 at x = 0")])
def minimize(expr: str) -> str:
    return _opt1(expr, maximize=False)


@register("optimize.maximize", "maximum value of f(x) and where it occurs",
          [(("-x**2 + 4*x - 1",), "3 at x = 2"), (("-x**2",), "0 at x = 0")])
def maximize(expr: str) -> str:
    return _opt1(expr, maximize=True)


@register("optimize.minimize2", "minimum of f(x,y) (solve ∇f = 0)",
          [(("x**2 + y**2 - 2*x - 4*y",), "-5 at (x, y) = (1, 2)"),
           (("x**2 + y**2",), "0 at (x, y) = (0, 0)")])
def minimize2(expr: str) -> str:
    f = _f(expr, two=True)
    sols = sp.solve([sp.diff(f, _x), sp.diff(f, _y)], [_x, _y], dict=True)
    if not sols:
        raise ValueError("no stationary point")
    s = sols[0]
    return f"{_n(f.subs(s))} at (x, y) = ({_n(s[_x])}, {_n(s[_y])})"


@register("optimize.gradient_descent", "x after `steps` of gradient descent on f(x) from x0",
          [(("x**2 - 4*x + 7", 0, 0.1, 100), 2.0), (("x**2", 5, 0.1, 200), 0.0)])
def gradient_descent(expr: str, x0: float, lr: float, steps: int) -> float:
    f = _f(expr)
    fp = sp.diff(f, _x)
    xi = float(x0)
    for _ in range(int(steps)):
        xi -= float(lr) * float(fp.subs(_x, xi))
    return round(xi, 6)

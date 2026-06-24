"""Numerical methods — root finding, quadrature, and ODE stepping on a symbolic f. Zero model
calls (SymPy parses/evaluates the expression; the iteration is plain Python).

Newton and bisection find a root of f(x); trapezoidal and Simpson approximate ∫f dx; Euler's
method advances dy/dx = f(x,y); fixed-point iterates x ← g(x). Results are rounded to 6 places.
"""

from __future__ import annotations

import sympy as sp

from .base import register

_x, _y = sp.symbols("x y")


def _f(expr: str):
    return sp.sympify(str(expr).replace("^", "**"), locals={"x": _x, "y": _y})


@register("numerical.newton", "Newton's method root of f(x) from x0",
          [(("x**2 - 2", 1), 1.414214), (("x**3 - x - 2", 1.5), 1.521380)])
def newton(expr: str, x0: float, iterations: int = 30) -> float:
    f = _f(expr)
    fp = sp.diff(f, _x)
    xi = float(x0)
    for _ in range(int(iterations)):
        d = float(fp.subs(_x, xi))
        if d == 0:
            break
        xi -= float(f.subs(_x, xi)) / d
    return round(xi, 6)


@register("numerical.bisection", "bisection-method root of f(x) on [a,b]",
          [(("x**2 - 2", 1, 2), 1.414214), (("x**3 - x - 2", 1, 2), 1.521380)])
def bisection(expr: str, a: float, b: float, iterations: int = 60) -> float:
    f = _f(expr)
    a, b = float(a), float(b)
    fa = float(f.subs(_x, a))
    for _ in range(int(iterations)):
        m = (a + b) / 2
        fm = float(f.subs(_x, m))
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return round((a + b) / 2, 6)


@register("numerical.trapezoidal", "trapezoidal-rule estimate of ∫f dx on [a,b] with n intervals",
          [(("2*x", 0, 3, 10), 9.0), (("x**2", 0, 1, 100), 0.33335)])
def trapezoidal(expr: str, a: float, b: float, n: int = 100) -> float:
    f = _f(expr)
    a, b, n = float(a), float(b), int(n)
    h = (b - a) / n
    s = (float(f.subs(_x, a)) + float(f.subs(_x, b))) / 2
    for i in range(1, n):
        s += float(f.subs(_x, a + i * h))
    return round(s * h, 6)


@register("numerical.simpson", "Simpson's-rule estimate of ∫f dx on [a,b] (exact for cubics)",
          [(("x**2", 0, 1, 2), 0.333333), (("x**3", 0, 2, 2), 4.0)])
def simpson(expr: str, a: float, b: float, n: int = 100) -> float:
    f = _f(expr)
    a, b, n = float(a), float(b), int(n)
    if n % 2:
        n += 1
    h = (b - a) / n
    s = float(f.subs(_x, a)) + float(f.subs(_x, b))
    for i in range(1, n):
        s += (4 if i % 2 else 2) * float(f.subs(_x, a + i * h))
    return round(s * h / 3, 6)


@register("numerical.euler_ode", "Euler's method for dy/dx=f(x,y): y after `steps` of size h",
          [(("y", 0, 1, 1, 1), 2.0), (("y", 0, 1, 0.5, 2), 2.25)])
def euler_ode(expr: str, x0: float, y0: float, h: float, steps: int) -> float:
    f = _f(expr)
    xi, yi, h = float(x0), float(y0), float(h)
    for _ in range(int(steps)):
        yi += h * float(f.subs({_x: xi, _y: yi}))
        xi += h
    return round(yi, 6)


@register("numerical.fixed_point", "fixed-point iteration x ← g(x) from x0",
          [(("cos(x)", 1), 0.739085), (("(x+2/x)/2", 1), 1.414214)])
def fixed_point(expr: str, x0: float, iterations: int = 100) -> float:
    g = _f(expr)
    xi = float(x0)
    for _ in range(int(iterations)):
        xi = float(g.subs(_x, xi))
    return round(xi, 6)

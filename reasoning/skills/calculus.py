"""Calculus — symbolic calculus via SymPy (deterministic, zero model calls): derivatives,
indefinite and definite integrals, and limits. Shares the algebra parser (^ powers, implicit
multiplication, and the elementary functions sin/cos/exp/log/etc.).
"""

from __future__ import annotations

import sympy

from .algebra import parse
from .base import register


@register("calculus.derivative", "the derivative of an expression with respect to a variable",
          [(("x^3 + 3x^2", "x"), "3*x**2 + 6*x"), (("sin(x)", "x"), "cos(x)")])
def derivative(expr: str, var: str) -> str:
    return str(sympy.diff(parse(expr), sympy.Symbol(var)))


@register("calculus.integral", "the indefinite integral (antiderivative), without +C",
          [(("x^2", "x"), "x**3/3"), (("cos(x)", "x"), "sin(x)")])
def integral(expr: str, var: str) -> str:
    return str(sympy.integrate(parse(expr), sympy.Symbol(var)))


@register("calculus.definite_integral", "the definite integral from a to b",
          [(("x^2", "x", "0", "1"), "1/3"), (("x", "x", "0", "2"), "2")])
def definite_integral(expr: str, var: str, a: str, b: str) -> str:
    x = sympy.Symbol(var)
    return str(sympy.integrate(parse(expr), (x, parse(a), parse(b))))


@register("calculus.limit", "the limit of an expression as the variable approaches a point "
          "(use 'oo' for infinity)",
          [(("sin(x)/x", "x", "0"), "1"), (("1/x", "x", "oo"), "0")])
def limit(expr: str, var: str, point: str) -> str:
    return str(sympy.limit(parse(expr), sympy.Symbol(var), parse(point)))


@register("calculus.nth_derivative", "the n-th derivative of an expression",
          [(("x^4", "x", 2), "12*x**2"), (("x^3", "x", 3), "6")])
def nth_derivative(expr: str, var: str, n: int) -> str:
    return str(sympy.diff(parse(expr), sympy.Symbol(var), n))


@register("calculus.taylor_series", "the Taylor series of an expression around a point, to a "
          "given order (the +O term dropped)",
          [(("exp(x)", "x", "0", 4), "x**3/6 + x**2/2 + x + 1")])
def taylor_series(expr: str, var: str, point: str, order: int) -> str:
    return str(sympy.series(parse(expr), sympy.Symbol(var), parse(point), order).removeO())


@register("calculus.critical_points", "the critical points (where the derivative is zero)",
          [(("x^3 - 3x", "x"), "[-1, 1]"), (("x^2", "x"), "[0]")])
def critical_points(expr: str, var: str) -> str:
    x = sympy.Symbol(var)
    pts = sympy.solve(sympy.diff(parse(expr), x), x)
    pts = sorted(pts, key=lambda v: (sympy.re(v), sympy.im(v)))
    return "[" + ", ".join(str(p) for p in pts) + "]"

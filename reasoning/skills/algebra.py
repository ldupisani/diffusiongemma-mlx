"""Algebra — symbolic algebra via SymPy (a deterministic CAS, zero model calls): factor,
expand, simplify, and solve equations. Expressions use ^ for powers and implicit
multiplication ("5x"), normalized by the parser.
"""

from __future__ import annotations

import sympy
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

from .base import register

_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse(expr: str):
    """Parse a human expression ('x^2 - 5x + 6') into a SymPy expression."""
    return parse_expr(expr, transformations=_TRANS)


@register("algebra.factor", "factor a polynomial expression",
          [(("x^2 - 5x + 6",), "(x - 3)*(x - 2)")])
def factor(expr: str) -> str:
    return str(sympy.factor(parse(expr)))


@register("algebra.expand", "expand a product of expressions",
          [(("(x+1)(x+2)",), "x**2 + 3*x + 2")])
def expand(expr: str) -> str:
    return str(sympy.expand(parse(expr)))


@register("algebra.simplify", "simplify an expression",
          [(("(x^2-1)/(x-1)",), "x + 1")])
def simplify(expr: str) -> str:
    return str(sympy.simplify(parse(expr)))


@register("algebra.solve", "solve an equation for a variable (RHS defaults to 0)",
          [(("x^2 - 5x + 6", "x"), "[2, 3]"), (("2x + 6 = 0", "x"), "[-3]")])
def solve(equation: str, var: str) -> str:
    if "=" in equation:
        lhs, rhs = equation.split("=", 1)
        eq = sympy.Eq(parse(lhs), parse(rhs))
    else:
        eq = sympy.Eq(parse(equation), 0)
    sols = sympy.solve(eq, sympy.Symbol(var))
    sols = sorted(sols, key=lambda v: (sympy.re(v), sympy.im(v)))
    return "[" + ", ".join(str(s) for s in sols) + "]"


@register("algebra.evaluate", "evaluate an expression by substituting a value for a variable",
          [(("x^2 + 1", "x", "3"), "10"), (("2x + 1", "x", "5"), "11")])
def evaluate(expr: str, var: str, value: str) -> str:
    return str(parse(expr).subs(sympy.Symbol(var), parse(value)))


@register("algebra.degree", "the degree of a polynomial in a variable",
          [(("x^3 + 2x", "x"), "3"), (("x^2 + x + 1", "x"), "2")])
def degree(expr: str, var: str) -> str:
    return str(sympy.degree(parse(expr), sympy.Symbol(var)))


@register("algebra.solve_system", "solve a system of equations (';'-separated) for the variables "
          "(comma-separated)",
          [(("x + y = 5; x - y = 1", "x, y"), "{x: 3, y: 2}")])
def solve_system(equations: str, variables: str) -> str:
    eqs = []
    for part in equations.split(";"):
        lhs, rhs = part.split("=", 1)
        eqs.append(sympy.Eq(parse(lhs), parse(rhs)))
    syms = [sympy.Symbol(v.strip()) for v in variables.split(",")]
    sol = sympy.solve(eqs, syms, dict=True)
    if not sol:
        return "no solution"
    s = sol[0]
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(s.items(), key=lambda kv: str(kv[0]))) + "}"

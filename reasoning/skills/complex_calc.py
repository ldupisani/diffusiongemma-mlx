"""Complex calculus / analysis — residues, contour integrals (residue theorem), and complex
derivatives. SymPy-backed (a deterministic CAS), zero model calls. The variable is z; 'i' or
'I' is the imaginary unit; ^ is power."""

from __future__ import annotations

import sympy
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

from .base import register

_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
_Z = sympy.Symbol("z")


def _cparse(s: str):
    return parse_expr(s, transformations=_TRANS,
                      local_dict={"i": sympy.I, "I": sympy.I, "z": _Z})


@register("ccalc.residue", "residue of f(z) at a point (pole)",
          [(("1/z", "0"), "1"), (("1/(z^2+1)", "I"), "-I/2")])
def residue(expr: str, point: str) -> str:
    return str(sympy.residue(_cparse(expr), _Z, _cparse(point)))


@register("ccalc.contour_integral", "contour integral ∮ f(z) dz enclosing a single pole, via "
          "the residue theorem (2πi·Res)", [(("1/z", "0"), "2*I*pi"), (("3/z", "0"), "6*I*pi")])
def contour_integral(expr: str, point: str) -> str:
    return str(sympy.simplify(2 * sympy.pi * sympy.I * sympy.residue(_cparse(expr), _Z,
                                                                     _cparse(point))))


@register("ccalc.derivative", "complex derivative d/dz of an analytic function",
          [(("z^2", ), "2*z"), (("z^3 + 2z",), "3*z**2 + 2")])
def derivative(expr: str) -> str:
    return str(sympy.diff(_cparse(expr), _Z))

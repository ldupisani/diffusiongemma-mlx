"""Applied mathematics — linear algebra (matrices via SymPy) and vector operations.
Deterministic (SymPy is a CAS), zero model calls. Matrices are python-list strings
('[[1, 2], [3, 4]]'); vectors are tuples."""

from __future__ import annotations

import ast
import math

import sympy

from .base import register


def _matrix(s: str):
    return sympy.Matrix(ast.literal_eval(s))


@register("linalg.determinant", "determinant of a square matrix",
          [(("[[1, 2], [3, 4]]",), -2), (("[[2, 0], [0, 3]]",), 6)])
def determinant(matrix: str):
    return _matrix(matrix).det()


@register("linalg.inverse", "inverse of a square matrix (as a nested list)",
          [(("[[1, 2], [3, 4]]",), "[[-2, 1], [3/2, -1/2]]")])
def inverse(matrix: str) -> str:
    return str(_matrix(matrix).inv().tolist())


@register("linalg.eigenvalues", "eigenvalues of a square matrix (sorted)",
          [(("[[2, 0], [0, 3]]",), "[2, 3]"), (("[[2, 1], [1, 2]]",), "[1, 3]")])
def eigenvalues(matrix: str) -> str:
    ev = _matrix(matrix).eigenvals()
    return "[" + ", ".join(str(v) for v in sorted(ev, key=lambda z: (sympy.re(z), sympy.im(z)))) + "]"


@register("linalg.dot", "dot product of two vectors", [(((1, 2, 3), (4, 5, 6)), 32)])
def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


@register("linalg.cross", "cross product of two 3-vectors", [(((1, 0, 0), (0, 1, 0)), (0, 0, 1))])
def cross(u, v) -> tuple:
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


@register("linalg.magnitude", "magnitude (Euclidean norm) of a vector",
          [(((3, 4),), 5.0), (((1, 2, 2),), 3.0)])
def magnitude(v) -> float:
    return math.sqrt(sum(x * x for x in v))

"""Functional analysis — the computable fragment: vector p-norms, inner products, and the
Cauchy–Schwarz inequality. Zero model calls.

(Functional analysis is mostly about infinite-dimensional spaces and operators — only these
finite-dimensional pieces compute deterministically.)
"""

from __future__ import annotations

import re

from .base import register


def _vec(s):
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(s))]


def _norm(v, p):
    if p == "inf":
        return max(abs(x) for x in v)
    if p == 1:
        return sum(abs(x) for x in v)
    return sum(abs(x) ** p for x in v) ** (1 / p)


@register("funcanalysis.vector_norm", "L_p norm of a vector (p = 1, 2, or 'inf')",
          [(("[3, 4]", 2), 5.0), (("[3, 4]", 1), 7.0), (("[3, -4]", "inf"), 4.0),
           (("[1, 2, 2]", 2), 3.0)])
def vector_norm(vector, p) -> float:
    p = "inf" if str(p).strip().lower() in ("inf", "infinity", "∞") else int(p)
    return round(_norm(_vec(vector), p), 6)


@register("funcanalysis.inner_product", "Euclidean inner product ⟨u, v⟩ of two vectors",
          [(("[1, 2, 3]", "[4, 5, 6]"), 32.0), (("[1, 0]", "[0, 1]"), 0.0)])
def inner_product(u, v) -> float:
    return round(sum(a * b for a, b in zip(_vec(u), _vec(v))), 6)


@register("funcanalysis.cauchy_schwarz", "check the Cauchy–Schwarz inequality |⟨u,v⟩| ≤ ‖u‖‖v‖",
          [(("[1, 2]", "[3, 4]"), "holds: 11 ≤ 11.18034"),
           (("[1, 0]", "[2, 0]"), "holds (equality): 2 ≤ 2.0")])
def cauchy_schwarz(u, v) -> str:
    a, b = _vec(u), _vec(v)
    lhs = abs(sum(x * y for x, y in zip(a, b)))
    rhs = _norm(a, 2) * _norm(b, 2)
    lhs_s = int(lhs) if lhs == int(lhs) else round(lhs, 6)
    eq = " (equality)" if abs(lhs - rhs) < 1e-9 else ""
    return f"holds{eq}: {lhs_s} ≤ {round(rhs, 6)}"

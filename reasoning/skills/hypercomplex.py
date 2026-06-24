"""Hypercomplex numbers — quaternions (and octonion norm/conjugate). Zero model calls.

A quaternion a + bi + cj + dk extends the complex numbers with three imaginary units obeying
i² = j² = k² = ijk = −1 (so ij = k, jk = i, ki = j, and multiplication is NON-commutative:
ij = k but ji = −k). Supported: addition, the Hamilton product, conjugate, norm, and inverse.
Octonions (8 components) add norm and conjugate — their product is non-associative and
convention-dependent, so it is intentionally omitted.
"""

from __future__ import annotations

import math

from .base import register

_UNITS = {"i": 1, "j": 2, "k": 3}


def _parse_q(s: str):
    """Parse 'a+bi+cj+dk' (terms optional, bare/unit-only terms allowed) -> (a, b, c, d)."""
    out = [0.0, 0.0, 0.0, 0.0]
    for term in s.replace(" ", "").replace("-", "+-").split("+"):
        if not term:
            continue
        unit = term[-1] if term[-1] in _UNITS else ""
        coeff = term[:-1] if unit else term
        coeff = {"": 1.0, "+": 1.0, "-": -1.0}.get(coeff, None) if coeff in ("", "+", "-") else float(coeff)
        out[_UNITS.get(unit, 0)] += coeff
    return tuple(out)


def _num(x: float):
    return int(x) if x == int(x) else round(x, 4)


def _fmt_q(q) -> str:
    labels = ["", "i", "j", "k"]
    parts = []
    for x, lab in zip(q, labels):
        x = _num(x)
        if x == 0:
            continue
        if lab and abs(x) == 1:
            token = ("-" if x < 0 else "") + lab
        else:
            token = f"{x}{lab}"
        parts.append(token)
    if not parts:
        return "0"
    s = parts[0]
    for p in parts[1:]:
        s += (p if p.startswith("-") else "+" + p)
    return s


def _qmul(x, y):
    a1, b1, c1, d1 = x
    a2, b2, c2, d2 = y
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)


@register("hyper.q_conjugate", "quaternion conjugate (negate the i, j, k parts)",
          [(("1+2i+3j+4k",), "1-2i-3j-4k"), (("2-3j",), "2+3j")])
def q_conjugate(q: str) -> str:
    a, b, c, d = _parse_q(q)
    return _fmt_q((a, -b, -c, -d))


@register("hyper.q_norm", "quaternion norm sqrt(a²+b²+c²+d²)",
          [(("1+2i+3j+4k",), math.sqrt(30)), (("3+4i",), 5.0)])
def q_norm(q: str) -> float:
    return math.sqrt(sum(x * x for x in _parse_q(q)))


@register("hyper.q_multiply", "Hamilton product of two quaternions (non-commutative)",
          [(("1+1i", "1+1j"), "1+i+j+k"), (("1i", "1j"), "k"), (("1j", "1i"), "-k")])
def q_multiply(a: str, b: str) -> str:
    return _fmt_q(_qmul(_parse_q(a), _parse_q(b)))


@register("hyper.q_add", "sum of two quaternions",
          [(("1+2i", "3+4k"), "4+2i+4k"), (("1i", "1i"), "2i")])
def q_add(a: str, b: str) -> str:
    return _fmt_q(tuple(x + y for x, y in zip(_parse_q(a), _parse_q(b))))


@register("hyper.q_inverse", "quaternion inverse conj(q)/|q|²",
          [(("1i",), "-i"), (("2",), "0.5")])
def q_inverse(q: str) -> str:
    a, b, c, d = _parse_q(q)
    n2 = a * a + b * b + c * c + d * d
    return _fmt_q((a / n2, -b / n2, -c / n2, -d / n2))


@register("hyper.o_norm", "octonion norm sqrt(sum of 8 squared components)",
          [(([1, 2, 3, 4, 0, 0, 0, 0],), math.sqrt(30)),
           (([1, 1, 1, 1, 1, 1, 1, 1],), math.sqrt(8))])
def o_norm(coeffs: list) -> float:
    return math.sqrt(sum(x * x for x in coeffs))


@register("hyper.o_conjugate", "octonion conjugate (negate all 7 imaginary components)",
          [(([1, 2, 3, 4, 5, 6, 7, 8],), [1, -2, -3, -4, -5, -6, -7, -8])])
def o_conjugate(coeffs: list) -> list:
    return [coeffs[0]] + [-x for x in coeffs[1:]]

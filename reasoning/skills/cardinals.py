"""Cardinal numbers — cardinal arithmetic over finite cardinals, ℵ₀ (countable infinity), and
𝔠 = 2^ℵ₀ (the continuum). Zero model calls.

For infinite cardinals, sum and product collapse to the larger operand: ℵ₀ + ℵ₀ = ℵ₀·ℵ₀ = ℵ₀,
n + ℵ₀ = ℵ₀, ℵ₀ + 𝔠 = 𝔠. Exponentiation grows them: 2^ℵ₀ = ℵ₀^ℵ₀ = 𝔠, while ℵ₀^n = ℵ₀ and
𝔠^ℵ₀ = 𝔠. Cardinalities: |ℕ| = |ℤ| = |ℚ| = ℵ₀ and |ℝ| = |P(ℕ)| = 𝔠 (Cantor: ℵ₀ < 𝔠). Only the
order finite < ℵ₀ < 𝔠 is used, so no Continuum Hypothesis is assumed (higher alephs are omitted).
"""

from __future__ import annotations

from .base import register

_ALEPH, _CONT, _POW2C = "ℵ₀", "𝔠", "2^𝔠"


def _parse(s):
    t = str(s).strip().lower()
    if t.lstrip("-").isdigit():
        return ("fin", int(t))
    if "aleph" in t or t.startswith("ℵ") or "countable" in t:
        return ("aleph0",)
    if "continuum" in t or t in {"𝔠", "c", "beth1", "beth_1"} or t.replace(" ", "") in {"2^aleph0", "2^ℵ₀", "2^ℵ0"}:
        return ("cont",)
    raise ValueError(f"unrecognized cardinal {s!r}")


_RANK = {"fin": 0, "aleph0": 1, "cont": 2}


def _rank(c):
    return (_RANK[c[0]], c[1] if c[0] == "fin" else 0)


def _fmt(c) -> str:
    if c[0] == "fin":
        return str(c[1])
    return _ALEPH if c[0] == "aleph0" else _CONT


@register("card.add", "cardinal sum (infinite cardinals collapse to the larger)",
          [(("aleph_0", "aleph_0"), "ℵ₀"), (("5", "aleph_0"), "ℵ₀"), (("3", "4"), "7"),
           (("aleph_0", "continuum"), "𝔠")])
def add(a, b) -> str:
    A, B = _parse(a), _parse(b)
    if A[0] == "fin" and B[0] == "fin":
        return str(A[1] + B[1])
    return _fmt(max(A, B, key=_rank))


@register("card.multiply", "cardinal product (infinite cardinals collapse to the larger)",
          [(("aleph_0", "aleph_0"), "ℵ₀"), (("0", "aleph_0"), "0"), (("3", "4"), "12"),
           (("continuum", "continuum"), "𝔠")])
def multiply(a, b) -> str:
    A, B = _parse(a), _parse(b)
    if A == ("fin", 0) or B == ("fin", 0):
        return "0"
    if A[0] == "fin" and B[0] == "fin":
        return str(A[1] * B[1])
    return _fmt(max(A, B, key=_rank))


@register("card.power", "cardinal exponentiation (2^ℵ₀ = 𝔠, ℵ₀^n = ℵ₀, 𝔠^ℵ₀ = 𝔠)",
          [(("2", "aleph_0"), "𝔠"), (("aleph_0", "aleph_0"), "𝔠"), (("aleph_0", "3"), "ℵ₀"),
           (("continuum", "aleph_0"), "𝔠"), (("2", "3"), "8"), (("continuum", "continuum"), "2^𝔠")])
def power(base, exp) -> str:
    B, E = _parse(base), _parse(exp)
    if B[0] == "fin" and E[0] == "fin":
        return str(B[1] ** E[1])
    if B == ("fin", 0):
        return "0"
    if B == ("fin", 1):
        return "1"
    if E[0] == "fin":               # infinite base, finite exponent: ℵ₀^n = ℵ₀, 𝔠^n = 𝔠
        return _fmt(B)
    if E == ("aleph0",):            # 2^ℵ₀ = ℵ₀^ℵ₀ = 𝔠^ℵ₀ = 𝔠
        return _CONT
    return _POW2C                   # exponent 𝔠: base^𝔠 = 2^𝔠


_COUNTABLE_HINTS = ("natural", "integer", "rational")
_CONTINUUM_HINTS = ("real", "irrational", "complex", "continuum", "power set", "plane")


@register("card.cardinality", "cardinality of a standard set (ℵ₀ for ℕ/ℤ/ℚ, 𝔠 for ℝ/P(ℕ))",
          [(("the naturals",), "ℵ₀"), (("the rationals",), "ℵ₀"), (("the reals",), "𝔠"),
           (("the power set of N",), "𝔠"), (("the integers",), "ℵ₀")])
def cardinality(name) -> str:
    t = str(name).strip().lower()
    if any(h in t for h in _COUNTABLE_HINTS) and "irrational" not in t:
        return _ALEPH
    if t in {"n", "ℕ", "z", "ℤ", "q", "ℚ"}:
        return _ALEPH
    if any(h in t for h in _CONTINUUM_HINTS) or t in {"r", "ℝ", "c", "ℂ"}:
        return _CONT
    raise ValueError(f"unknown set {name!r}")


@register("card.compare", "order two cardinals (ℵ₀ < 𝔠)",
          [(("aleph_0", "continuum"), "ℵ₀ < 𝔠"), (("5", "aleph_0"), "5 < ℵ₀"),
           (("continuum", "continuum"), "𝔠 = 𝔠")])
def compare(a, b) -> str:
    A, B = _parse(a), _parse(b)
    ra, rb = _rank(A), _rank(B)
    sym = "<" if ra < rb else (">" if ra > rb else "=")
    return f"{_fmt(A)} {sym} {_fmt(B)}"

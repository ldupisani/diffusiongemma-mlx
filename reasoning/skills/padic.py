"""p-adic numbers — valuation, absolute value, expansion, and distance, for a prime p.

For a prime p, the p-adic valuation v_p(x) is the exponent of p dividing x, and the p-adic
absolute value is |x|_p = p^(-v_p(x)) — so a number is "small" p-adically when it is highly
divisible by p. The p-adic metric d_p(x, y) = |x - y|_p makes 1 and 28 close in the 3-adics
(their difference 27 = 3^3). Every integer also has a p-adic (base-p) expansion a0 + a1·p + …;
negatives and fractions give an infinite, eventually-repeating tail (−1 = …(p-1) in the p-adics).

Built on stdlib integer/Fraction arithmetic; zero model calls.
"""

from __future__ import annotations

from fractions import Fraction

from .base import register


@register("padic.valuation", "p-adic valuation v_p(n): exponent of p dividing n",
          [((24, 2), 3), ((54, 3), 3), ((100, 5), 2), ((7, 5), 0)])
def valuation(n: int, p: int) -> int:
    n = abs(int(n))
    if n == 0:
        raise ValueError("v_p(0) is infinite")
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


@register("padic.abs_value", "p-adic absolute value |n|_p = p^(-v_p(n))",
          [((24, 2), 0.125), ((100, 5), 0.04), ((7, 5), 1.0), ((0, 5), 0.0)])
def abs_value(n: int, p: int) -> float:
    if n == 0:
        return 0.0
    return float(p) ** (-valuation(n, p))


@register("padic.distance", "p-adic distance |a-b|_p between two integers",
          [((1, 28, 3), 1 / 27), ((1, 9, 2), 0.125), ((4, 4, 5), 0.0)])
def distance(a: int, b: int, p: int) -> float:
    if a == b:
        return 0.0
    return float(p) ** (-valuation(a - b, p))


def _digits(n: int, p: int, k: int):
    """First k p-adic digits (ascending powers) of integer n, plus whether more remain.
    Uses the standard digit-peel over Q_p, so negatives yield the eventually-repeating tail."""
    x = Fraction(int(n))
    digits = []
    for _ in range(k):
        inv = pow(x.denominator % p, -1, p)          # denominator is a p-adic unit here
        d = (x.numerator % p) * inv % p
        digits.append(d)
        x = (x - d) / p
    return digits, x != 0


def _fmt_expansion(digits, p: int, more: bool) -> str:
    terms = []
    for i, d in enumerate(digits):
        terms.append(str(d) if i == 0 else (f"{d}·{p}" if i == 1 else f"{d}·{p}^{i}"))
    return " + ".join(terms) + (" + …" if more else "")


@register("padic.expansion", "first k digits of the p-adic expansion of n (ascending powers)",
          [((100, 5, 3), "0 + 0·5 + 4·5^2"),
           ((-1, 5, 3), "4 + 4·5 + 4·5^2 + …"),
           ((13, 2, 4), "1 + 0·2 + 1·2^2 + 1·2^3")])
def expansion(n: int, p: int, k: int = 4) -> str:
    digits, more = _digits(n, p, k)
    return _fmt_expansion(digits, p, more)


@register("padic.add_mod", "p-adic sum truncated to k digits: (a+b) mod p^k",
          [((7, 13, 5, 2), 20), ((24, 1, 2, 3), 1)])
def add_mod(a: int, b: int, p: int, k: int) -> int:
    return (a + b) % (p ** k)


@register("padic.mul_mod", "p-adic product truncated to k digits: (a*b) mod p^k",
          [((7, 13, 5, 2), 16), ((3, 4, 2, 3), 4)])
def mul_mod(a: int, b: int, p: int, k: int) -> int:
    return (a * b) % (p ** k)

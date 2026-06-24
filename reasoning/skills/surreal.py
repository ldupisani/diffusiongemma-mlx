"""Surreal numbers — the finite-birthday fragment (the dyadic rationals). Zero model calls.

A surreal number {L | R} is the SIMPLEST number strictly between the greatest left option and
the least right option (an empty side is ±∞). So {|}=0, {0|}=1, {|0}=−1, {0|1}=1/2, {1|2}=3/2.
The numbers born by a finite day are exactly the dyadic rationals (denominator a power of 2),
and "simplest" means fewest binary-search steps from 0 — which is the number's birthday.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .base import register


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _simplest(a, b) -> Fraction:
    """Simplest dyadic strictly inside the open interval (a, b); a/b are Fraction or None=±∞."""
    if (a is None or a < 0) and (b is None or b > 0):
        return Fraction(0)
    if a is not None and a >= 0:                 # closest integer above a, if it fits
        n = math.floor(a) + 1
        if b is None or n < b:
            return Fraction(n)
    if b is not None and b <= 0:                 # closest integer below b, if it fits
        n = math.ceil(b) - 1
        if a is None or n > a:
            return Fraction(n)
    d = 2                                        # no integer fits: smallest-denominator dyadic
    while d <= 2 ** 40:
        for num in range(math.floor(a * d) + 1, math.ceil(b * d)):
            x = Fraction(num, d)
            if a < x < b:
                return x
        d *= 2
    raise ValueError("interval too narrow")


def _side(s: str):
    return [Fraction(t.strip()) for t in s.split(",") if t.strip()]


@register("surr.value", "value of a surreal {L | R}: simplest number between the sides",
          [(("{ | }",), "0"), (("{0 | }",), "1"), (("{ | 0}",), "-1"),
           (("{0 | 1}",), "1/2"), (("{1 | 2}",), "3/2")])
def value(form: str) -> str:
    left, right = form.strip().strip("{}").split("|")
    ls, rs = _side(left), _side(right)
    a = max(ls) if ls else None
    b = min(rs) if rs else None
    return _fmt(_simplest(a, b))


@register("surr.simplest_between", "simplest surreal strictly between a and b",
          [(("0", "1"), "1/2"), (("1/2", "1"), "3/4"), (("-1", "1"), "0"), (("1", "2"), "3/2")])
def simplest_between(a: str, b: str) -> str:
    return _fmt(_simplest(Fraction(a), Fraction(b)))


@register("surr.birthday", "birthday of a dyadic surreal (binary-search depth from 0)",
          [(("0",), 0), (("3",), 3), (("1/2",), 2), (("3/4",), 3), (("-2",), 2)])
def birthday(x: str) -> int:
    target = Fraction(x)
    if target.denominator & (target.denominator - 1):     # denominator not a power of 2
        raise ValueError("not a finite-birthday (dyadic) surreal")
    lo, hi, day = None, None, 0
    while True:
        s = _simplest(lo, hi)
        if s == target:
            return day
        day += 1
        if target > s:
            lo = s
        else:
            hi = s

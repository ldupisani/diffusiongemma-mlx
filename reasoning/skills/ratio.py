"""Ratio / proportion / fraction skills."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from .base import register


@register("ratio.simplify_fraction", "simplify a fraction to lowest terms, as 'a/b'",
          [((18, 24), "3/4"), ((12, 18), "2/3")])
def simplify_fraction(num: int, den: int) -> str:
    g = gcd(int(num), int(den)) or 1
    return f"{int(num) // g}/{int(den) // g}"


@register("ratio.simplify_ratio", "simplify a ratio to lowest terms, as 'a:b'",
          [((12, 18), "2:3"), ((4, 6), "2:3")])
def simplify_ratio(a: int, b: int) -> str:
    g = gcd(int(a), int(b)) or 1
    return f"{int(a) // g}:{int(b) // g}"


@register("ratio.fraction_to_decimal", "a fraction as a decimal", [((3, 4), 0.75), ((1, 2), 0.5)])
def fraction_to_decimal(num: float, den: float) -> float:
    return num / den


@register("ratio.decimal_to_fraction", "a decimal as a simplified fraction 'a/b'",
          [((0.75,), "3/4"), ((0.25,), "1/4")])
def decimal_to_fraction(x: float) -> str:
    fr = Fraction(str(x)).limit_denominator(10000)
    return f"{fr.numerator}/{fr.denominator}"


@register("ratio.fraction_to_percent", "a fraction as a percentage value",
          [((3, 5), 60.0), ((1, 4), 25.0)])
def fraction_to_percent(num: float, den: float) -> float:
    return num / den * 100


@register("ratio.solve_proportion", "solve a:b = x:d for x (= a*d/b)",
          [((2, 5, 20), 8.0), ((3, 4, 8), 6.0)])
def solve_proportion(a: float, b: float, d: float) -> float:
    return a * d / b


@register("ratio.unit_rate", "unit rate = amount / per", [((150, 3), 50.0), ((100, 4), 25.0)])
def unit_rate(amount: float, per: float) -> float:
    return amount / per


@register("ratio.scale", "scale a quantity: base * to_n / from_n",
          [((2, 4, 6), 3.0), ((2, 3, 12), 8.0)])
def scale(base: float, from_n: float, to_n: float) -> float:
    return base * to_n / from_n

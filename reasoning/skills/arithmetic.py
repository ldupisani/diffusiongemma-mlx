"""Arithmetic / percentage skills — the family that proves the model-free thesis.

Each is a single atomic operation (mirroring the teacher's "one operation per step"
contract). Money output is rounded ONLY through round_money, which uses Decimal +
ROUND_HALF_UP — the human "round half up" people expect, not Python's banker's rounding.
That single chokepoint is the one place to get money rounding right and to test.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .base import register


@register("arith.add", "add two numbers (sum, plus, total)", [((2, 3), 5), ((0.1, 0.2), 0.3)])
def add(a: float, b: float) -> float:
    return a + b


@register("arith.subtract", "subtract b from a (minus, difference, less)",
          [((5, 3), 2), ((240, 40.8), 199.2)])
def subtract(a: float, b: float) -> float:
    return a - b


@register("arith.multiply", "multiply two numbers (product, times, multiplied by)",
          [((4, 5), 20), ((240, 0.17), 40.8)])
def multiply(a: float, b: float) -> float:
    return a * b


@register("arith.divide", "divide a by b (quotient, divided by, per)",
          [((10, 2), 5.0), ((1, 4), 0.25)])
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("divide by zero")
    return a / b


@register("arith.percent_of", "find what a percentage of a value is (percent of)",
          [((240, 17), 40.8), ((50, 10), 5.0)])
def percent_of(value: float, pct: float) -> float:
    return value * pct / 100


@register("arith.apply_discount", "reduce a price by a percentage (percent off, discount)",
          [((240, 17), 199.2), ((50, 20), 40.0)])
def apply_discount(price: float, pct: float) -> float:
    return price * (1 - pct / 100)


@register("arith.apply_tax", "increase an amount by a percentage tax (tax, vat, plus tax)",
          [((199.2, 8), 215.136), ((100, 7), 107.0)])
def apply_tax(amount: float, pct: float) -> float:
    return amount * (1 + pct / 100)


@register("arith.percent_change", "percentage change from an old value to a new value (increase/decrease)",
          [((50, 75), 50.0), ((75, 50), -33.333333), ((100, 110), 10.0)])
def percent_change(old: float, new: float) -> float:
    return (new - old) / old * 100


@register("arith.round_to", "round a number to N decimal places, half up",
          [((37.777, 2), 37.78), ((2.5, 0), 3.0)])
def round_to(x: float, places: int = 2) -> float:
    quant = Decimal(1).scaleb(-places)  # e.g. places=2 -> Decimal("0.01")
    return float(Decimal(str(x)).quantize(quant, rounding=ROUND_HALF_UP))


@register("arith.round_money", "round a monetary amount to N decimal places, half up",
          [((215.136, 2), 215.14), ((2.675, 2), 2.68), ((2.5, 0), 3.0)])
def round_money(x: float, places: int = 2) -> float:
    return round_to(x, places)

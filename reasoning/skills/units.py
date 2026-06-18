"""Unit-conversion skills — each a single named conversion (one formula, one step).

Replaces the teacher's "subtract 32, multiply by 5, divide by 9" chain with the named
concept units.f_to_c. Conversions come in symmetric pairs; the bind layer maps a
(source, target) unit pair to the right skill.
"""

from __future__ import annotations

from .base import register


@register("units.f_to_c", "convert Fahrenheit to Celsius",
          [((212,), 100.0), ((32,), 0.0)])
def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


@register("units.c_to_f", "convert Celsius to Fahrenheit",
          [((100,), 212.0), ((0,), 32.0)])
def c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


@register("units.mi_to_km", "convert miles to kilometers",
          [((1,), 1.609344), ((10,), 16.09344)])
def mi_to_km(mi: float) -> float:
    return mi * 1.609344


@register("units.km_to_mi", "convert kilometers to miles",
          [((1.609344,), 1.0)])
def km_to_mi(km: float) -> float:
    return km / 1.609344


@register("units.lb_to_kg", "convert pounds to kilograms",
          [((1,), 0.45359237)])
def lb_to_kg(lb: float) -> float:
    return lb * 0.45359237


@register("units.kg_to_lb", "convert kilograms to pounds",
          [((1,), 2.2046226218487757)])
def kg_to_lb(kg: float) -> float:
    return kg / 0.45359237

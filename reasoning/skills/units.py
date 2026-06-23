"""Unit-conversion skills — each a single named conversion (one formula, one step).

Replaces the teacher's "subtract 32, multiply by 5, divide by 9" chain with the named
concept units.f_to_c. Conversions come in symmetric pairs; the bind layer maps a
(source, target) unit pair to the right skill.
"""

from __future__ import annotations

import re as _re

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


# --- general unit conversion (canonical base-unit factor table) -----------------------
# Each unit's factor converts it TO its dimension's base unit; conversion within a
# dimension is value * factor[from] / factor[to]. ONE table + one convert() replaces the
# combinatorial explosion of pairwise skills (N units → N*(N-1) pairwise functions).
# Adding a unit is one row, and it converts to/from every sibling. Temperature is affine
# (offset, not just scale), so it is handled separately.
_FACTORS = {
    # length (base: metre)
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01),
    "mm": ("length", 0.001), "mi": ("length", 1609.344), "ft": ("length", 0.3048),
    "in": ("length", 0.0254), "yd": ("length", 0.9144),
    # mass (base: gram)
    "g": ("mass", 1.0), "kg": ("mass", 1000.0), "mg": ("mass", 0.001),
    "lb": ("mass", 453.59237), "oz": ("mass", 28.349523125),
    # volume (base: litre)
    "l": ("volume", 1.0), "ml": ("volume", 0.001), "gal": ("volume", 3.785411784),
    "qt": ("volume", 0.946352946), "pt": ("volume", 0.473176473),
    # time (base: second)
    "s": ("time", 1.0), "min": ("time", 60.0), "hr": ("time", 3600.0),
    "day": ("time", 86400.0), "week": ("time", 604800.0),
    # speed (base: metre/second)
    "mps": ("speed", 1.0), "kph": ("speed", 1000 / 3600), "mph": ("speed", 1609.344 / 3600),
    # data size (base: byte; 1024-based, the common informal KB/MB/...)
    "byte": ("data", 1.0), "kb": ("data", 1024.0), "mb": ("data", 1024.0 ** 2),
    "gb": ("data", 1024.0 ** 3), "tb": ("data", 1024.0 ** 4), "pb": ("data", 1024.0 ** 5),
}
_TEMP = {"c", "f", "k"}

# spelling/alias -> canonical token (multiword phrases included; bind matches longest-first)
_CANON = {
    "meters": "m", "meter": "m", "metres": "m", "metre": "m", "m": "m",
    "kilometers": "km", "kilometer": "km", "kilometres": "km", "kilometre": "km", "km": "km",
    "centimeters": "cm", "centimeter": "cm", "centimetres": "cm", "cm": "cm",
    "millimeters": "mm", "millimetres": "mm", "mm": "mm",
    "miles": "mi", "mile": "mi", "mi": "mi", "feet": "ft", "foot": "ft", "ft": "ft",
    "inches": "in", "inch": "in", "in": "in", "yards": "yd", "yard": "yd", "yd": "yd",
    "grams": "g", "gram": "g", "g": "g",
    "kilograms": "kg", "kilogram": "kg", "kilos": "kg", "kilo": "kg", "kg": "kg",
    "milligrams": "mg", "mg": "mg",
    "pounds": "lb", "pound": "lb", "lbs": "lb", "lb": "lb",
    "ounces": "oz", "ounce": "oz", "oz": "oz",
    "liters": "l", "liter": "l", "litres": "l", "litre": "l", "l": "l",
    "milliliters": "ml", "millilitres": "ml", "ml": "ml",
    "gallons": "gal", "gallon": "gal", "gal": "gal",
    "quarts": "qt", "quart": "qt", "qt": "qt", "pints": "pt", "pint": "pt", "pt": "pt",
    "seconds": "s", "second": "s", "secs": "s", "sec": "s", "s": "s",
    "minutes": "min", "minute": "min", "mins": "min", "min": "min",
    "hours": "hr", "hour": "hr", "hrs": "hr", "hr": "hr",
    "days": "day", "day": "day", "weeks": "week", "week": "week",
    "kilometers per hour": "kph", "kilometres per hour": "kph", "km/h": "kph", "kmh": "kph",
    "kph": "kph", "miles per hour": "mph", "mph": "mph", "mi/h": "mph",
    "meters per second": "mps", "metres per second": "mps", "m/s": "mps", "mps": "mps",
    "celsius": "c", "centigrade": "c", "c": "c", "fahrenheit": "f", "f": "f",
    "kelvin": "k", "k": "k",
    "bytes": "byte", "byte": "byte",
    "kilobytes": "kb", "kilobyte": "kb", "kb": "kb",
    "megabytes": "mb", "megabyte": "mb", "mb": "mb",
    "gigabytes": "gb", "gigabyte": "gb", "gb": "gb",
    "terabytes": "tb", "terabyte": "tb", "tb": "tb",
    "petabytes": "pb", "petabyte": "pb", "pb": "pb",
}

# longest phrase first so "miles per hour" beats "miles", "centimeters" beats "meters", etc.
UNIT_PHRASES = sorted(_CANON, key=len, reverse=True)
UNIT_WORDS = {a for a in _CANON if " " not in a and "/" not in a}  # cheap membership for routing

# display suffix (with separator) per canonical unit
SUFFIX = {u: f" {u}" for u in _FACTORS}
SUFFIX.update({"l": " L", "ml": " mL", "mps": " m/s", "kph": " km/h", "mph": " mph",
               "c": "°C", "f": "°F", "k": " K", "byte": " bytes", "kb": " KB", "mb": " MB",
               "gb": " GB", "tb": " TB", "pb": " PB"})


def canonical(token: str):
    return _CANON.get(token.strip().lower())


# canonical unit → dimension, and dimension → its unit phrases (longest-first), for the
# unit-AWARE extraction the bind layer uses to canonicalize a quantity before synthesis.
DIMENSION = {u: dim for u, (dim, _) in _FACTORS.items()}
_DIM_PHRASES: dict = {}
for _ph in UNIT_PHRASES:
    _dim = DIMENSION.get(_CANON[_ph])
    if _dim:
        _DIM_PHRASES.setdefault(_dim, []).append(_ph)  # UNIT_PHRASES is already longest-first


def parse_quantity(text: str, dimension: str):
    """First '<number> <unit>' whose unit belongs to `dimension` → (value, canonical_unit), else
    None. Collision-safe: a bare 'm' inside 'm/s' or 'm/s²' is rejected by the trailing lookahead,
    so a length scan never steals a speed/acceleration token."""
    phrases = _DIM_PHRASES.get(dimension)
    if not phrases:
        return None
    alt = "|".join(_re.escape(p) for p in phrases)
    m = _re.search(rf"(-?\d+(?:\.\d+)?)\s*({alt})(?![A-Za-z0-9/²³^])", text, _re.IGNORECASE)
    return (float(m.group(1)), canonical(m.group(2))) if m else None


def _to_celsius(v, u):
    return v if u == "c" else (v - 32) * 5 / 9 if u == "f" else v - 273.15


def _from_celsius(v, u):
    return v if u == "c" else v * 9 / 5 + 32 if u == "f" else v + 273.15


@register("units.convert",
          "convert a value between two units of the same dimension "
          "(length / mass / volume / time / speed / temperature)",
          [((5, "km", "m"), 5000.0), ((2, "lb", "oz"), 32.0),
           ((300, "k", "c"), 26.85), ((60, "mph", "kph"), 96.56064)])
def convert(value: float, from_unit: str, to_unit: str) -> float:
    f = canonical(from_unit) or from_unit.strip().lower()
    t = canonical(to_unit) or to_unit.strip().lower()
    f_temp, t_temp = f in _TEMP, t in _TEMP
    if f_temp or t_temp:
        if not (f_temp and t_temp):
            raise ValueError(f"cannot convert {from_unit} to {to_unit} "
                             "(temperature and non-temperature units are incompatible)")
        return _from_celsius(_to_celsius(value, f), t)
    if f not in _FACTORS:
        raise ValueError(f"unknown unit: {from_unit!r}")
    if t not in _FACTORS:
        raise ValueError(f"unknown unit: {to_unit!r}")
    (dim_f, ff), (dim_t, ft) = _FACTORS[f], _FACTORS[t]
    if dim_f != dim_t:
        raise ValueError(f"cannot convert {from_unit} ({dim_f}) to {to_unit} ({dim_t})")
    return value * ff / ft

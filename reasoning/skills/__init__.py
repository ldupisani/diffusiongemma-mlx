"""Deterministic skill library — the code that replaces model one-shot calls.

Importing this package registers every skill family (the family modules call register()
at import time), so `from reasoning import skills; skills.SKILLS` is fully populated.
"""

from .base import SKILLS, Skill, call, get, register

# Import families for their registration side effects (must come after base import).
from . import (  # noqa: E402,F401
    algebra, applied_math, arithmetic, bases, biochem, calculus, cardinals, chemistry, clock,
    complex_calc, complex_math, coord, dates, discrete, expr, finance, genetics, geo, geometry,
    group_theory, health, hypercomplex, logic, materials, molecular, ndim, number_theory,
    ordinals, padic, pchem, physics, probability, ratio, set_theory, sets, stats, tensor, text,
    topology, trig, units,
)

__all__ = ["SKILLS", "Skill", "call", "get", "register"]

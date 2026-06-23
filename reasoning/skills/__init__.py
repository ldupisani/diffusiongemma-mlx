"""Deterministic skill library — the code that replaces model one-shot calls.

Importing this package registers every skill family (the family modules call register()
at import time), so `from reasoning import skills; skills.SKILLS` is fully populated.
"""

from .base import SKILLS, Skill, call, get, register

# Import families for their registration side effects (must come after base import).
from . import (  # noqa: E402,F401
    algebra, arithmetic, bases, calculus, chemistry, clock, coord, dates, expr, finance,
    geometry, group_theory, health, logic, number_theory, physics, probability, ratio,
    set_theory, sets, stats, text, topology, trig, units,
)

__all__ = ["SKILLS", "Skill", "call", "get", "register"]

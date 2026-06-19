"""Deterministic skill library — the code that replaces model one-shot calls.

Importing this package registers every skill family (the family modules call register()
at import time), so `from reasoning import skills; skills.SKILLS` is fully populated.
"""

from .base import SKILLS, Skill, call, get, register

# Import families for their registration side effects (must come after base import).
from . import (  # noqa: E402,F401
    arithmetic, bases, chemistry, dates, finance, geometry, number_theory, physics, stats,
    text, units,
)

__all__ = ["SKILLS", "Skill", "call", "get", "register"]

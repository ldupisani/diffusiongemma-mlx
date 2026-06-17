"""Deterministic skill library — the code that replaces model one-shot calls.

Importing this package registers every skill family (the family modules call register()
at import time), so `from reasoning import skills; skills.SKILLS` is fully populated.
"""

from .base import SKILLS, Skill, call, get, register

# Import families for their registration side effects (must come after base import).
from . import arithmetic, dates, text  # noqa: E402,F401

__all__ = ["SKILLS", "Skill", "call", "get", "register"]

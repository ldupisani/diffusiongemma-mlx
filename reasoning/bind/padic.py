"""bind layer — p-adic numbers. Gated on the unambiguous "<p>-adic" phrasing (no other domain
uses it), so it routes confidently even when fuzzy lexical/embedding scores are ~0."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_PRIME = re.compile(r"(\d+)\s*-\s*adic", re.IGNORECASE)   # "5-adic", "5 - adic"
_INT = re.compile(r"-?\d+")
_DIGITS_K = re.compile(r"(?:first\s+)?(\d+)\s+digits", re.IGNORECASE)


def bind_padic(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    pm = _PRIME.search(query)
    if not pm:
        return False, None, "code", 0
    p = int(pm[1])
    if p < 2:
        return False, None, "code", 0

    # integers in the query that are NOT the prime p (the "5" in "5-adic"). Strip the
    # "<p>-adic" span first so p is never mistaken for an operand.
    rest = _PRIME.sub(" ", query)
    operands = [int(n) for n in _INT.findall(rest)]

    if "distance" in q or "close" in q:
        m = re.search(r"between\s+(-?\d+)\s+and\s+(-?\d+)", q)
        if m:
            return ok("padic.distance", ["a", "b", "p"], [int(m[1]), int(m[2]), p],
                      round=True, places=4)
        if len(operands) >= 2:
            return ok("padic.distance", ["a", "b", "p"], [operands[0], operands[1], p],
                      round=True, places=4)
        return False, None, "code", 0

    if "expansion" in q or "digits" in q:
        km = _DIGITS_K.search(query)
        # the digit count is not the operand: drop it from operands if it appears as "k digits"
        k = int(km[1]) if km else 4
        nums = [n for n in operands if not (km and n == k)] or operands
        if nums:
            return ok("padic.expansion", ["n", "p", "k"], [nums[0], p, k], round=False)
        return False, None, "code", 0

    if "valuation" in q:
        if operands:
            return ok("padic.valuation", ["n", "p"], [operands[0], p], round=False)
        return False, None, "code", 0

    if "absolute value" in q or "norm" in q or "|" in query:
        if operands:
            return ok("padic.abs_value", ["n", "p"], [operands[0], p], round=True, places=4)
        return False, None, "code", 0

    return False, None, "code", 0

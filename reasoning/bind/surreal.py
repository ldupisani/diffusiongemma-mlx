"""bind layer — surreal numbers. The {L | R} form (a brace containing a pipe) is unambiguous
and routes on its own; birthday/simplest-between require the word 'surreal' (so 'birthday' is
not mistaken for a date, and 'simplest between' for arithmetic)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_BRACE = re.compile(r"\{[^{}]*\|[^{}]*\}")          # {L | R}
_DY = r"-?\d+(?:/\d+)?"                              # a dyadic rational literal
_BIRTHDAY = re.compile(rf"birthday of\s+(?:the\s+)?(?:surreal\s+)?({_DY})", re.IGNORECASE)
_BETWEEN = re.compile(rf"between\s+({_DY})\s+and\s+({_DY})", re.IGNORECASE)
_INTERVAL = re.compile(rf"\(\s*({_DY})\s*,\s*({_DY})\s*\)")


def bind_surreal(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    bm = _BRACE.search(query)
    if bm:
        return ok("surr.value", ["form"], [bm.group()], round=False)

    if "surreal" not in q:
        return False, None, "code", 0

    bd = _BIRTHDAY.search(query)
    if bd:
        return ok("surr.birthday", ["x"], [bd[1]], round=False)

    if "simplest" in q or "between" in q:
        m = _BETWEEN.search(query) or _INTERVAL.search(query)
        if m:
            return ok("surr.simplest_between", ["a", "b"], [m[1], m[2]], round=False)
    return False, None, "code", 0

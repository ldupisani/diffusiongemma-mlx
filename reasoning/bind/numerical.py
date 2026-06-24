"""bind layer — numerical methods. Gated on the method name (Newton / bisection / Simpson /
trapezoidal / fixed-point), which keeps it distinct from the exact symbolic calculus domain.
The expression is captured as the text before the numeric anchor, with leading filler/method
words stripped. The Euler-ODE skill is left unbound (its two-variable phrasing is too varied)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_N = r"-?\d+(?:\.\d+)?"
_FROM = r"(?:from|at|starting(?:\s+at)?|x0\s*=)"
_FILLER = re.compile(
    r"^(?:\W*\b(?:use|using|apply|find|compute|estimate|approximate|a|an|the|root|roots|of|for|"
    r"to|on|by|newton|newtons|method|bisection|simpson|simpsons|trapezoidal|trapezoid|rule|"
    r"integral|fixed|point|iteration|with)\b\W*)+", re.IGNORECASE)


def _expr(s: str) -> str:
    s = re.sub(r"['’]s\b", "", s)   # drop possessive: "Newton's" -> "Newton" (then stripped)
    return _FILLER.sub("", s).strip(" .?")


def bind_numerical(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "newton" in q:
        m = re.search(rf"(.+?)\s+{_FROM}\s*({_N})", query, re.IGNORECASE)
        if m:
            return ok("numerical.newton", ["expr", "x0"], [_expr(m[1]), float(m[2])],
                      round=True, places=6)
        return False, None, "code", 0

    if "bisection" in q:
        iv = (re.search(rf"\[\s*({_N})\s*,\s*({_N})\s*\]", query)
              or re.search(rf"(?:on|in|over)\s+\(?\s*({_N})\s*,\s*({_N})\)?", query))
        em = re.search(r"(.+?)\s*(?:by\s+bisection|bisection|on\b|in\b|over\b|\[)", query, re.IGNORECASE)
        if iv and em:
            return ok("numerical.bisection", ["expr", "a", "b"],
                      [_expr(em[1]), float(iv[1]), float(iv[2])], round=True, places=6)
        return False, None, "code", 0

    if "simpson" in q or "trapezoid" in q:
        skill = "numerical.simpson" if "simpson" in q else "numerical.trapezoidal"
        m = re.search(rf"(.+?)\s+from\s+({_N})\s+to\s+({_N})(?:\s+(?:with|using)\s+(\d+))?",
                      query, re.IGNORECASE)
        if m:
            n = int(m[4]) if m[4] else 100
            return ok(skill, ["expr", "a", "b", "n"],
                      [_expr(m[1]), float(m[2]), float(m[3]), n], round=True, places=6)
        return False, None, "code", 0

    if "fixed point" in q or "fixed-point" in q:
        m = re.search(rf"(.+?)\s+{_FROM}\s*({_N})", query, re.IGNORECASE)
        if m:
            return ok("numerical.fixed_point", ["expr", "x0"], [_expr(m[1]), float(m[2])],
                      round=True, places=6)
        return False, None, "code", 0

    return False, None, "code", 0

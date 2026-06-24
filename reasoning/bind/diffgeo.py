"""bind layer — differential geometry (plane curves). Gated on curvature / radius of curvature
/ arc length of y=f(x)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_N = r"-?\d+(?:\.\d+)?"


def _expr(s: str) -> str:
    s = re.sub(r"^\s*(?:the\s+curve\s+|y\s*=\s*|f\s*\(\s*x\s*\)\s*=\s*|f\s*=\s*)", "", s,
               flags=re.IGNORECASE)
    return s.strip(" .?")


def bind_diffgeo(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "arc length" in q:
        m = re.search(rf"arc length of\s+(.+?)\s+(?:from|over)\s*\[?\s*({_N})\s*(?:to|,)\s*({_N})",
                      query, re.IGNORECASE)
        if m:
            return ok("diffgeo.arc_length", ["expr", "a", "b"],
                      [_expr(m[1]), float(m[2]), float(m[3])], round=True, places=6)
        return False, None, "code", 0

    if "radius of curvature" in q:
        m = re.search(rf"radius of curvature of\s+(.+?)\s+at\s+(?:x\s*=\s*)?({_N})", query, re.IGNORECASE)
        if m:
            return ok("diffgeo.radius_of_curvature", ["expr", "x0"],
                      [_expr(m[1]), float(m[2])], round=True, places=6)
        return False, None, "code", 0

    if "curvature" in q:
        m = re.search(rf"curvature of\s+(.+?)\s+at\s+(?:x\s*=\s*)?({_N})", query, re.IGNORECASE)
        if m:
            return ok("diffgeo.curvature", ["expr", "x0"], [_expr(m[1]), float(m[2])],
                      round=True, places=6)
        return False, None, "code", 0

    return False, None, "code", 0

"""bind layer — optimization. Gated on minimize / maximize / gradient descent. A two-variable
objective (mentions y) routes to the 2-D minimizer."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_N = r"-?\d+(?:\.\d+)?"


def _expr(s: str) -> str:
    s = re.sub(r"^\s*(?:the\s+function\s+|f\s*\(\s*x\s*,?\s*y?\s*\)\s*=\s*|f\s*=\s*)", "", s,
               flags=re.IGNORECASE)
    return s.strip(" .?")


def bind_optimize(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "gradient descent" in q:
        em = re.search(r"gradient descent\s+(?:on|of)\s+(.+?)\s+from", query, re.IGNORECASE)
        x0 = re.search(rf"from\s+({_N})", q)
        lr = re.search(rf"(?:learning rate|lr|step size|rate)\s*(?:of|=)?\s*({_N})", q)
        st = re.search(r"(\d+)\s+steps", q)
        if em and x0 and lr and st:
            return ok("optimize.gradient_descent", ["expr", "x0", "lr", "steps"],
                      [_expr(em[1]), float(x0[1]), float(lr[1]), int(st[1])], round=True, places=6)
        return False, None, "code", 0

    m = re.search(r"(?:maximize|maximum of)\s+(.+?)\s*[?.]?$", query, re.IGNORECASE)
    if m:
        return ok("optimize.maximize", ["expr"], [_expr(m[1])], round=False)

    m = re.search(r"(?:minimize|minimum of|optimi[sz]e)\s+(.+?)\s*[?.]?$", query, re.IGNORECASE)
    if m:
        e = _expr(m[1])
        skill = "optimize.minimize2" if re.search(r"\by\b", e) else "optimize.minimize"
        return ok(skill, ["expr"], [e], round=False)

    return False, None, "code", 0

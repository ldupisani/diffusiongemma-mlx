"""bind layer — cardinal numbers. Gated on a cardinal signal (ℵ / aleph / continuum / 𝔠, or
'cardinality of' a standard infinite set). Cardinality of an explicit finite set like {1,2,3}
is left to the sets domain."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_CARDTOK = re.compile(r"ℵ[₀0]?|aleph[\s_-]?(?:null|naught|zero|0)?|continuum|𝔠|\d+", re.IGNORECASE)
_SET_HINTS = ("natural", "integer", "rational", "irrational", "real", "complex", "continuum",
              "power set", "plane", "ℕ", "ℤ", "ℚ", "ℝ", "ℂ")
_GATE = re.compile(r"ℵ|aleph|continuum|𝔠|cardinal", re.IGNORECASE)


def _canon(tok: str) -> str:
    t = tok.lower()
    if t.startswith("ℵ") or "aleph" in t:
        return "aleph_0"
    if "continuum" in t or t == "𝔠":
        return "continuum"
    return t  # a finite integer


def bind_cardinal(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0

    if "cardinality" in q or "how many elements" in q:
        m = re.search(r"(?:cardinality of|size of)\s+(.+?)\s*[?.]?$", query, re.IGNORECASE)
        if m and any(h in m[1].lower() for h in _SET_HINTS):
            return ok("card.cardinality", ["name"], [m[1].strip()], round=False)
        return False, None, "code", 0  # finite/explicit set → leave to the sets domain

    toks = [_canon(t) for t in _CARDTOK.findall(query)]
    if len(toks) < 2:
        return False, None, "code", 0
    a, b = toks[0], toks[1]

    if "^" in query or "power" in q or "to the" in q:
        return ok("card.power", ["base", "exp"], [a, b], round=False)
    if any(w in q for w in ("compare", "greater", "less", "larger", "smaller", "than", "bigger")):
        return ok("card.compare", ["a", "b"], [a, b], round=False)
    if any(w in q for w in ("times", "product", "multipl", "·", "*")):
        return ok("card.multiply", ["a", "b"], [a, b], round=False)
    if any(w in q for w in ("+", "plus", "sum", "add")):
        return ok("card.add", ["a", "b"], [a, b], round=False)
    return False, None, "code", 0

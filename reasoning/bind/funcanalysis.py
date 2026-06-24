"""bind layer — functional analysis fragment. Gated on a SPECIFIC norm type (L1/taxicab,
L∞/max, Lp), an inner product, or Cauchy–Schwarz — so a plain "magnitude/norm of a vector"
(the L2 case) is left to linalg."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_VEC = re.compile(r"\[[^\]]*\]")
_GATE = re.compile(r"inner product|cauchy|manhattan|taxicab|euclidean norm|infinity norm|"
                   r"max norm|chebyshev|\bl[_-]?\d\b|\d-norm|p-norm", re.IGNORECASE)


def _p(q: str):
    if "manhattan" in q or "taxicab" in q:
        return 1
    if "infinity" in q or "max norm" in q or "chebyshev" in q or "∞" in q:
        return "inf"
    if "euclidean" in q:
        return 2
    m = re.search(r"\bl[_-]?(\d+)\b", q) or re.search(r"\b(\d+)-norm\b", q)
    return int(m[1]) if m else None


def bind_funcanalysis(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    vecs = _VEC.findall(query)

    if "cauchy" in q and len(vecs) >= 2:
        return ok("funcanalysis.cauchy_schwarz", ["u", "v"], [vecs[0], vecs[1]], round=False)
    if "inner product" in q and len(vecs) >= 2:
        return ok("funcanalysis.inner_product", ["u", "v"], [vecs[0], vecs[1]], round=True, places=6)
    p = _p(q)
    if p is not None and vecs:
        return ok("funcanalysis.vector_norm", ["vector", "p"], [vecs[0], p], round=True, places=6)
    return False, None, "code", 0

"""bind layer — tensor maths (rank/shape/trace/norm, Kronecker & outer products)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_MATRIX = re.compile(r"\[\[.*?\]\]")
_VECTOR = re.compile(r"\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*\]")


def bind_tensor(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    mats = _MATRIX.findall(query)
    if "kronecker" in q and len(mats) >= 2:
        return ok("tensor.kronecker", ["a", "b"], [mats[0], mats[1]], round=False)
    if "outer" in q:
        vecs = [v for v in _VECTOR.findall(query) if "[[" not in v]
        if len(vecs) >= 2:
            return ok("tensor.outer", ["u", "v"], [vecs[0], vecs[1]], round=False)
    if mats:
        if "rank" in q or "number of dimensions" in q or "number of axes" in q:
            return ok("tensor.rank", ["t"], [mats[0]], round=False)
        if "shape" in q or "dimensions of" in q:
            return ok("tensor.shape", ["t"], [mats[0]], round=False)
        if "trace" in q:
            return ok("tensor.trace", ["t"], [mats[0]], round=False)
        if "frobenius" in q or "norm" in q:
            return ok("tensor.frobenius_norm", ["t"], [mats[0]], places=4)
    return False, None, "code", 0

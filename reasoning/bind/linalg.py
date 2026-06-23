"""bind layer — applied mathematics (matrices, vectors)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_MATRIX = re.compile(r"\[\[.*\]\]")
_VEC = re.compile(r"\(\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*\)")


def _vectors(query: str) -> list:
    out = []
    for body in _VEC.findall(query):
        out.append(tuple(float(x) if "." in x else int(x)
                         for x in re.split(r"\s*,\s*", body)))
    return out


def bind_linalg(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, args, kind):
        return True, {"skill": skill, "args": args, "kind": kind}, "code", 0

    mat = _MATRIX.search(query)
    if mat:
        ms = mat.group(0)
        if "determinant" in q or "det " in q:
            return ok("linalg.determinant", [ms], "raw")
        if "inverse" in q or "invert" in q:
            return ok("linalg.inverse", [ms], "raw")
        if "eigenvalue" in q or "eigen" in q:
            return ok("linalg.eigenvalues", [ms], "raw")
    vecs = _vectors(query)
    if len(vecs) >= 2:
        if "dot" in q or "scalar product" in q:
            return ok("linalg.dot", [vecs[0], vecs[1]], "raw")
        if "cross" in q:
            return ok("linalg.cross", [vecs[0], vecs[1]], "tuple")
    if vecs and ("magnitude" in q or "norm" in q or "length of the vector" in q):
        return ok("linalg.magnitude", [vecs[0]], "num")
    return False, None, "code", 0

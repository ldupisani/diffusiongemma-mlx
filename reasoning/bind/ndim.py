"""bind layer — N-dimensional shapes (n-ball / n-sphere / hypercube)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_DIM = re.compile(r"(\d+)\s*[-\s]?\s*(?:dimensional|dimensions?|d\b)"
                  r"|(\d+)-(?:cube|ball|sphere|simplex)", re.IGNORECASE)


def _labeled(q, label):
    m = re.search(rf"{label}\s+(?:of\s+)?(\d+(?:\.\d+)?)", q, re.IGNORECASE)
    return float(m[1]) if m else None


def bind_ndim(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    dm = _DIM.search(query)
    if not dm:
        return False, None, "code", 0
    n = int(dm[1] or dm[2])
    radius, side = _labeled(q, "radius"), _labeled(q, "side")
    if "ball" in q or ("sphere" in q and "surface" not in q and "area" not in q):
        if radius is not None:
            return ok("ndim.ball_volume", ["n", "r"], [n, radius], places=4)
    if "sphere" in q and ("surface" in q or "area" in q):
        if radius is not None:
            return ok("ndim.sphere_surface", ["n", "r"], [n, radius], places=4)
    if "cube" in q:
        if "diagonal" in q and side is not None:
            return ok("ndim.hypercube_diagonal", ["n", "s"], [n, side], places=4)
        if "vert" in q or "corner" in q:
            return ok("ndim.hypercube_vertices", ["n"], [n])
        if "edge" in q:
            return ok("ndim.hypercube_edges", ["n"], [n])
        if ("volume" in q or "hypervolume" in q) and side is not None:
            return ok("ndim.hypercube_volume", ["n", "s"], [n, side])
    return False, None, "code", 0

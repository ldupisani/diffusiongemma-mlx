"""bind layer — discrete mathematics (graphs from edge lists; combinatorial sequences)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind import entities as _E


def bind_discrete(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, args):
        return True, {"skill": skill, "args": args}, "code", 0

    if any(c in q for c in ("graph", "edges", "vertex", "vertices")):
        edges = tuple(tuple(int(x) for x in p) for p in _E.pairs(query))
        if edges:
            if "shortest path" in q:
                m = re.search(r"from\s+(\d+)\s+to\s+(\d+)", q)
                if m:
                    return ok("discrete.shortest_path", [edges, int(m[1]), int(m[2])])
            if "component" in q:
                return ok("discrete.graph_components", [edges])
            if "degree" in q:
                m = re.search(r"vertex\s+(\d+)|degree of\s+(\d+)", q)
                if m:
                    return ok("discrete.graph_degree", [edges, int(m[1] or m[2])])
    if "catalan" in q:
        n = _E.only_number(query)
        if n is not None:
            return ok("discrete.catalan", [int(n)])
    if "fibonacci" in q:
        n = _E.only_number(query)
        if n is not None:
            return ok("discrete.fibonacci", [int(n)])
    if "bell" in q and ("number" in q or "partition" in q):
        n = _E.only_number(query)
        if n is not None:
            return ok("discrete.bell", [int(n)])
    if "stirling" in q:
        nums = _E.exactly(query, 2)
        if nums:
            return ok("discrete.stirling2", [int(nums[0]), int(nums[1])])
    return False, None, "code", 0

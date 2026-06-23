"""Discrete mathematics — finite graphs (from an edge list) and combinatorial sequences.
Pure stdlib, zero model calls. Edges are a tuple of (u, v) pairs (undirected)."""

from __future__ import annotations

from collections import deque
from math import comb

from .base import register


def _adj(edges) -> dict:
    g: dict = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g


@register("discrete.graph_degree", "degree of a vertex (number of incident edges)",
          [((((1, 2), (2, 3), (3, 1)), 2), 2), ((((1, 2), (1, 3), (1, 4)), 1), 3)])
def graph_degree(edges, vertex) -> int:
    return sum(1 for a, b in edges for x in (a, b) if x == vertex)


@register("discrete.graph_components", "number of connected components",
          [((((1, 2), (2, 3), (4, 5)),), 2), ((((1, 2), (2, 3), (3, 1)),), 1)])
def graph_components(edges) -> int:
    g, seen, comps = _adj(edges), set(), 0
    for v in g:
        if v not in seen:
            comps += 1
            stack = [v]
            while stack:
                x = stack.pop()
                if x not in seen:
                    seen.add(x)
                    stack.extend(g[x] - seen)
    return comps


@register("discrete.shortest_path", "length (hops) of the shortest path between two vertices, "
          "or -1 if unreachable",
          [((((1, 2), (2, 3), (3, 4)), 1, 4), 3), ((((1, 2), (3, 4)), 1, 4), -1)])
def shortest_path(edges, start, end) -> int:
    g, dist, dq = _adj(edges), {start: 0}, deque([start])
    while dq:
        x = dq.popleft()
        if x == end:
            return dist[x]
        for w in g.get(x, ()):
            if w not in dist:
                dist[w] = dist[x] + 1
                dq.append(w)
    return dist.get(end, -1)


@register("discrete.catalan", "the n-th Catalan number", [((4,), 14), ((5,), 42)])
def catalan(n: int) -> int:
    return comb(2 * n, n) // (n + 1)


@register("discrete.fibonacci", "the n-th Fibonacci number (F(0)=0, F(1)=1)",
          [((10,), 55), ((7,), 13)])
def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@register("discrete.stirling2", "Stirling number of the second kind S(n, k) (partitions of n "
          "items into k non-empty subsets)", [((4, 2), 7), ((5, 2), 15)])
def stirling2(n: int, k: int) -> int:
    if k == 0:
        return 1 if n == 0 else 0
    if k > n:
        return 0
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


@register("discrete.bell", "the n-th Bell number (number of partitions of an n-set)",
          [((4,), 15), ((3,), 5)])
def bell(n: int) -> int:
    return sum(stirling2(n, k) for k in range(n + 1))

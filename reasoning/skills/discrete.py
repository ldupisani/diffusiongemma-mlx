"""Discrete mathematics — finite graphs (from an edge list) and combinatorial sequences.
Pure stdlib, zero model calls. Edges are a tuple of (u, v) pairs (undirected)."""

from __future__ import annotations

from collections import deque
from math import comb, factorial, prod

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


# --- combinatorics ---------------------------------------------------------

@register("discrete.derangement", "subfactorial !n: permutations of n items with no fixed point",
          [((4,), 9), ((3,), 2), ((0,), 1), ((5,), 44)])
def derangement(n: int) -> int:
    a, b = 1, 0          # !0, !1
    if n == 0:
        return 1
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b


@register("discrete.multinomial", "multinomial coefficient (sum k_i)! / prod(k_i!)",
          [(([2, 1, 1],), 12), (([1, 1, 1],), 6), (([2, 2],), 6)])
def multinomial(counts) -> int:
    return factorial(sum(counts)) // prod(factorial(c) for c in counts)


@register("discrete.stars_and_bars", "ways to place n identical items into k bins: C(n+k-1, k-1)",
          [((5, 3), 21), ((10, 4), 286)])
def stars_and_bars(n: int, k: int) -> int:
    return comb(n + k - 1, k - 1)


@register("discrete.integer_partitions", "number of integer partitions p(n)",
          [((4,), 5), ((5,), 7), ((10,), 42), ((0,), 1)])
def integer_partitions(n: int) -> int:
    p = [1] + [0] * n
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            p[j] += p[j - i]
    return p[n]


# --- graph theory ----------------------------------------------------------

@register("discrete.degree_sequence", "sorted (descending) degree sequence of a graph",
          [(([(1, 2), (2, 3), (3, 1)],), "[2, 2, 2]"),
           (([(1, 2), (2, 3), (2, 4)],), "[3, 1, 1, 1]")])
def degree_sequence(edges) -> str:
    g = _adj(edges)
    return str(sorted((len(v) for v in g.values()), reverse=True))


@register("discrete.is_bipartite", "is the graph 2-colourable (bipartite)?",
          [(([(1, 2), (2, 3), (3, 4)],), True), (([(1, 2), (2, 3), (3, 1)],), False)])
def is_bipartite(edges) -> bool:
    g = _adj(edges)
    color: dict = {}
    for start in g:
        if start in color:
            continue
        color[start] = 0
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for w in g[u]:
                if w not in color:
                    color[w] = color[u] ^ 1
                    dq.append(w)
                elif color[w] == color[u]:
                    return False
    return True


@register("discrete.eulerian", "does the graph have an Eulerian circuit / path / neither?",
          [(([(1, 2), (2, 3), (3, 1)],), "Eulerian circuit"),
           (([(1, 2), (2, 3), (3, 4)],), "Eulerian path"),
           (([(1, 2), (2, 3), (3, 1), (1, 4), (4, 2)],), "Eulerian path")])
def eulerian(edges) -> str:
    g = _adj(edges)
    odd = sum(len(v) % 2 for v in g.values())
    if odd == 0:
        return "Eulerian circuit"
    if odd == 2:
        return "Eulerian path"
    return "neither"


@register("discrete.mst_weight", "total weight of a minimum spanning tree (Kruskal)",
          [(([(1, 2, 1), (2, 3, 2), (1, 3, 3)],), 3),
           (([(1, 2, 4), (2, 3, 1), (3, 4, 2), (4, 1, 3)],), 6)])
def mst_weight(edges) -> int:
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            total += w
    return total

"""Topology — finite topological spaces. A space is (X, T): X a finite set, T a collection of
open subsets. Pure stdlib, zero model calls.

X is a frozenset; T is a frozenset of frozensets. Example used below — the Sierpiński space
X = {1, 2}, T = {∅, {1}, {1, 2}}.
"""

from __future__ import annotations

from .base import register

_X = frozenset({1, 2})
_SIERP = frozenset({frozenset(), frozenset({1}), _X})          # {∅, {1}, {1,2}}
_DISCRETE = frozenset({frozenset(), frozenset({1}), frozenset({2}), _X})


def _T(T):
    return {frozenset(o) for o in T}


@register("topology.is_topology", "is T a topology on X (contains ∅ and X, closed under "
          "union and intersection)?",
          [((_X, _SIERP), True), ((_X, frozenset({frozenset({1}), _X})), False)])
def is_topology(X, T) -> bool:
    X, T = frozenset(X), _T(T)
    if frozenset() not in T or X not in T:
        return False
    return all((a & b) in T and (a | b) in T for a in T for b in T)


@register("topology.is_open", "is A an open set (A ∈ T)?",
          [((_SIERP, frozenset({1})), True), ((_SIERP, frozenset({2})), False)])
def is_open(T, A) -> bool:
    return frozenset(A) in _T(T)


@register("topology.is_closed", "is A closed (its complement X \\ A is open)?",
          [((_X, _SIERP, frozenset({2})), True), ((_X, _SIERP, frozenset({1})), False)])
def is_closed(X, T, A) -> bool:
    return (frozenset(X) - frozenset(A)) in _T(T)


@register("topology.interior", "the interior of A (largest open subset of A)",
          [((_X, _SIERP, frozenset({1})), frozenset({1})),
           ((_X, _SIERP, frozenset({2})), frozenset())])
def interior(X, T, A) -> frozenset:
    A = frozenset(A)
    opens = [o for o in _T(T) if o <= A]
    return frozenset().union(*opens) if opens else frozenset()


@register("topology.closure", "the closure of A (smallest closed superset of A)",
          [((_X, _SIERP, frozenset({1})), _X),
           ((_X, _SIERP, frozenset({2})), frozenset({2}))])
def closure(X, T, A) -> frozenset:
    X, A = frozenset(X), frozenset(A)
    res = X
    for closed in (X - o for o in _T(T)):  # closed sets = complements of opens
        if A <= closed:
            res &= closed
    return res


@register("topology.boundary", "the boundary of A (closure minus interior)",
          [((_X, _SIERP, frozenset({1})), frozenset({2}))])
def boundary(X, T, A) -> frozenset:
    return closure(X, T, A) - interior(X, T, A)


@register("topology.is_connected", "is the space connected (no proper nonempty clopen set)?",
          [((_X, _SIERP), True), ((_X, _DISCRETE), False)])
def is_connected(X, T) -> bool:
    X, T = frozenset(X), _T(T)
    clopen = [o for o in T if (X - o) in T]
    return not any(c and c != X for c in clopen)


@register("topology.is_t1", "is the space T1 (every singleton is closed)?",
          [((_X, _DISCRETE), True), ((_X, _SIERP), False)])
def is_t1(X, T) -> bool:
    X, T = frozenset(X), _T(T)
    return all((X - frozenset({x})) in T for x in X)


@register("topology.is_hausdorff", "is the space Hausdorff/T2 (distinct points have disjoint "
          "open neighborhoods)?",
          [((_X, _DISCRETE), True), ((_X, _SIERP), False)])
def is_hausdorff(X, T) -> bool:
    X, T = frozenset(X), _T(T)
    pts = sorted(X)
    for i, x in enumerate(pts):
        for y in pts[i + 1:]:
            if not any(x in u and y in v and not (u & v)
                       for u in T for v in T):
                return False
    return True

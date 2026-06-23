"""Set theory — explicit set operations on finite sets. Pure stdlib, zero model calls.

Sets are frozensets (of ints); set-valued results are frozensets too, rendered to standard
notation by `format_set`. (Distinct from the `sets` family, which is inclusion-exclusion
COUNTING — this is actual set algebra on explicit elements.)
"""

from __future__ import annotations

from itertools import combinations

from .base import register

_A, _B = frozenset({1, 2, 3}), frozenset({2, 3, 4})


@register("setops.union", "union A ∪ B", [((_A, _B), frozenset({1, 2, 3, 4}))])
def union(a, b) -> frozenset:
    return frozenset(a) | frozenset(b)


@register("setops.intersection", "intersection A ∩ B", [((_A, _B), frozenset({2, 3}))])
def intersection(a, b) -> frozenset:
    return frozenset(a) & frozenset(b)


@register("setops.difference", "set difference A \\ B", [((_A, _B), frozenset({1}))])
def difference(a, b) -> frozenset:
    return frozenset(a) - frozenset(b)


@register("setops.symmetric_difference", "symmetric difference A △ B",
          [((_A, _B), frozenset({1, 4}))])
def symmetric_difference(a, b) -> frozenset:
    return frozenset(a) ^ frozenset(b)


@register("setops.complement", "complement of A in universe U (U \\ A)",
          [((frozenset({1, 2, 3, 4, 5}), _A), frozenset({4, 5}))])
def complement(universe, a) -> frozenset:
    return frozenset(universe) - frozenset(a)


@register("setops.cardinality", "cardinality |A|", [((_A,), 3), ((frozenset(),), 0)])
def cardinality(a) -> int:
    return len(frozenset(a))


@register("setops.is_subset", "is A a subset of B (A ⊆ B)",
          [((frozenset({1, 2}), _A), True), ((_A, _B), False)])
def is_subset(a, b) -> bool:
    return frozenset(a) <= frozenset(b)


@register("setops.is_disjoint", "are A and B disjoint (A ∩ B = ∅)",
          [((frozenset({1, 2}), frozenset({3, 4})), True), ((_A, _B), False)])
def is_disjoint(a, b) -> bool:
    return frozenset(a).isdisjoint(frozenset(b))


@register("setops.power_set", "the power set 𝒫(A) (all subsets)",
          [((frozenset({1, 2}),),
            frozenset({frozenset(), frozenset({1}), frozenset({2}), frozenset({1, 2})}))])
def power_set(a) -> frozenset:
    items = list(a)
    return frozenset(frozenset(c) for r in range(len(items) + 1)
                     for c in combinations(items, r))


@register("setops.cartesian_product", "the Cartesian product A × B",
          [((frozenset({1, 2}), frozenset({3})), frozenset({(1, 3), (2, 3)}))])
def cartesian_product(a, b) -> frozenset:
    return frozenset((x, y) for x in a for y in b)


def _fmt_elem(e) -> str:
    if isinstance(e, frozenset):
        return format_set(e)
    if isinstance(e, tuple):
        return "(" + ", ".join(str(x) for x in e) + ")"
    return str(e)


def format_set(s) -> str:
    """Render a frozenset in standard notation: ∅, {1, 2, 3}, nested sets, or pair tuples."""
    if not s:
        return "∅"
    # sort by (size, then sorted elements) for sets-of-sets; by value for flat int sets
    def key(e):
        if isinstance(e, frozenset):
            return (1, len(e), sorted(map(str, e)))
        if isinstance(e, tuple):
            return (2, e)
        return (0, e)

    return "{" + ", ".join(_fmt_elem(e) for e in sorted(s, key=key)) + "}"


@register("setops.is_proper_subset", "is A a proper subset of B (A ⊊ B)",
          [((frozenset({1, 2}), _A), True), ((_A, _A), False)])
def is_proper_subset(a, b) -> bool:
    return frozenset(a) < frozenset(b)


@register("setops.is_superset", "is A a superset of B (A ⊇ B)",
          [((_A, frozenset({1, 2})), True), ((frozenset({1, 2}), _A), False)])
def is_superset(a, b) -> bool:
    return frozenset(a) >= frozenset(b)


@register("setops.set_equals", "are A and B the same set",
          [((_A, _A), True), ((_A, _B), False)])
def set_equals(a, b) -> bool:
    return frozenset(a) == frozenset(b)


@register("setops.count_subsets", "the number of subsets of A (2^|A|)",
          [((_A,), 8), ((frozenset(),), 1)])
def count_subsets(a) -> int:
    return 2 ** len(frozenset(a))

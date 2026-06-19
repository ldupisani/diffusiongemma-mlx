"""Set / inclusion-exclusion skills — for two-set Venn word problems."""

from __future__ import annotations

from .base import register


@register("sets.union_count", "|A ∪ B| = |A| + |B| - |A ∩ B| (also 'at least one')",
          [((18, 15, 8), 25), ((50, 45, 30), 65)])
def union_count(a: int, b: int, both: int) -> int:
    return a + b - both


@register("sets.neither", "items in neither set = total - (|A| + |B| - both)",
          [((50, 30, 25, 10), 5), ((60, 35, 40, 20), 5)])
def neither(total: int, a: int, b: int, both: int) -> int:
    return total - (a + b - both)


@register("sets.only_first", "items in the first set only = |A| - both",
          [((25, 12), 13), ((70, 25), 45)])
def only_first(a: int, both: int) -> int:
    return a - both

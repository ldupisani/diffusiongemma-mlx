"""Probability skills — the cleanly-computable subset.

Pure-computation cases only: complement, all-heads coin runs, a single die face, a
two-dice sum, and the expected value of a fair die. Cases needing domain knowledge of a
sample space (e.g. "an ace from a deck") are left to the teacher; combinatorial counts
("8 choose 3", "4 people in a row") are handled by the number_theory family.
"""

from __future__ import annotations

from .base import register


@register("prob.complement_percent", "complement of a percentage chance: 100 - p",
          [((30,), 70), ((25,), 75)])
def complement_percent(pct: float) -> float:
    return 100 - pct


@register("prob.coin_all_heads", "probability of n heads in a row = (1/2)^n",
          [((2,), 0.25), ((3,), 0.125)])
def coin_all_heads(n: int) -> float:
    return 0.5 ** int(n)


@register("prob.die_face", "probability of a specific face on a fair die = 1/sides",
          [((6,), 0.166667), ((20,), 0.05)])
def die_face(sides: int = 6) -> float:
    return 1 / int(sides)


@register("prob.dice_sum", "probability that two fair six-sided dice sum to a target",
          [((7,), 0.166667), ((2,), 0.027778)])
def dice_sum(target: int) -> float:
    ways = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == int(target))
    return ways / 36


@register("prob.expected_die", "expected value of a fair die = (sides + 1) / 2",
          [((6,), 3.5), ((20,), 10.5)])
def expected_die(sides: int = 6) -> float:
    return (int(sides) + 1) / 2

"""Statistics skills — summarize a list of numbers in one named operation.

These exist because the teacher decomposes "the average of these numbers" into a chain
of adds + a divide — correct, but a fixed chain that only works for that exact count.
stats.mean over a variable-length list captures the *concept*, so one orchestrator
handles any-length input. The list-of-primitives chain is the thing we're replacing.
"""

from __future__ import annotations

from statistics import median as _median

from .base import register


@register("stats.mean", "the average (mean) of a list of numbers",
          [(([4, 8, 15, 16, 42],), 17.0), (([10, 20],), 15.0)])
def mean(nums: list) -> float:
    return sum(nums) / len(nums)


@register("stats.median", "the median (middle value) of a list of numbers",
          [(([1, 3, 5],), 3), (([1, 2, 3, 4],), 2.5)])
def median(nums: list) -> float:
    return _median(nums)


@register("stats.total", "the sum (total) of a list of numbers",
          [(([4, 8, 15, 16, 42],), 85), (([1, 2, 3],), 6)])
def total(nums: list) -> float:
    return sum(nums)


@register("stats.minimum", "the smallest (minimum, lowest) value in a list",
          [(([4, 8, 15],), 4), (([-2, 5],), -2)])
def minimum(nums: list):
    return min(nums)


@register("stats.maximum", "the largest (maximum, highest) value in a list",
          [(([4, 8, 15],), 15), (([-2, 5],), 5)])
def maximum(nums: list):
    return max(nums)


@register("stats.range_of", "the range (largest minus smallest) of a list",
          [(([4, 8, 15],), 11), (([10, 10],), 0)])
def range_of(nums: list) -> float:
    return max(nums) - min(nums)

"""Statistics skills — summarize a list of numbers in one named operation.

These exist because the teacher decomposes "the average of these numbers" into a chain
of adds + a divide — correct, but a fixed chain that only works for that exact count.
stats.mean over a variable-length list captures the *concept*, so one orchestrator
handles any-length input. The list-of-primitives chain is the thing we're replacing.
"""

from __future__ import annotations

from math import prod as _prod
from statistics import median as _median
from statistics import mode as _mode
from statistics import pstdev as _pstdev
from statistics import pvariance as _pvariance

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


@register("stats.mode", "the most frequent value (mode) in a list",
          [(([4, 7, 4, 9, 4, 7],), 4), (([1, 1, 2],), 1)])
def mode(nums: list):
    return _mode(nums)


@register("stats.product", "the product of all numbers in a list (multiplied together)",
          [(([2, 3, 4, 5],), 120), (([10, 0.5],), 5.0)])
def product(nums: list) -> float:
    return _prod(nums)


@register("stats.geometric_mean", "the geometric mean (nth root of the product) of a list",
          [(([2, 8, 32],), 8.0), (([4, 9],), 6.0)])
def geometric_mean(nums: list) -> float:
    return _prod(nums) ** (1 / len(nums))


@register("stats.stdev", "the population standard deviation of a list",
          [(([2, 4, 4, 4, 5, 5, 7, 9],), 2.0)])
def stdev(nums: list) -> float:
    return _pstdev(nums)


@register("stats.variance", "the population variance of a list",
          [(([2, 4, 4, 4, 5, 5, 7, 9],), 4.0)])
def variance(nums: list) -> float:
    return _pvariance(nums)


@register("stats.percentile", "the p-th percentile of a list (linear interpolation)",
          [(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90), 9.1), (([1, 2, 3, 4], 50), 2.5)])
def percentile(nums: list, p: float) -> float:
    s = sorted(nums)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@register("stats.weighted_mean", "the weighted average of values given matching weights",
          [(([80, 90, 100], [1, 2, 3]), 93.333333), (([10, 20], [1, 1]), 15.0)])
def weighted_mean(values: list, weights: list) -> float:
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

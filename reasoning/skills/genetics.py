"""Genetics — Hardy–Weinberg genotype frequencies, monohybrid (Punnett) crosses, and allele
frequency. Pure stdlib, zero model calls."""

from __future__ import annotations

from collections import Counter
from functools import reduce
from math import gcd

from .base import register


@register("genetics.hardy_weinberg", "Hardy–Weinberg genotype frequencies (AA, Aa, aa) from "
          "the dominant allele frequency p (q = 1−p)",
          [((0.7,), (0.49, 0.42, 0.09)), ((0.5,), (0.25, 0.5, 0.25))])
def hardy_weinberg(p: float) -> tuple:
    q = 1 - p
    return (round(p * p, 6), round(2 * p * q, 6), round(q * q, 6))


@register("genetics.monohybrid_cross", "offspring genotype ratio of a monohybrid cross of two "
          "genotypes (e.g. 'Aa' × 'Aa')",
          [(("Aa", "Aa"), "1 AA : 2 Aa : 1 aa"), (("Aa", "aa"), "1 Aa : 1 aa")])
def monohybrid_cross(parent1: str, parent2: str) -> str:
    def norm(g):
        return "".join(sorted(g, key=lambda c: (c.lower(), c.islower())))

    counts = Counter()
    for a in parent1:
        for b in parent2:
            counts[norm(a + b)] += 1
    g = reduce(gcd, counts.values())
    return " : ".join(f"{counts[k] // g} {k}" for k in sorted(counts))


@register("genetics.allele_frequency", "allele frequency = count of that allele / total alleles",
          [((35, 50), 0.7), ((1, 4), 0.25)])
def allele_frequency(allele_count: int, total_alleles: int) -> float:
    return allele_count / total_alleles

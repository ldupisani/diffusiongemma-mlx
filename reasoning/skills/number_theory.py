"""Number-theory / integer skills — gcd, lcm, factorial, primes, combinatorics.

Single- and dual-integer operations with mixed return types (int, bool, string). One
generalized orchestrator dispatches them; the boolean ones (is_prime, is_perfect_square)
phrase as Yes/No, the rest return their value.
"""

from __future__ import annotations

import math

from .base import register


@register("numtheory.gcd", "greatest common divisor of two integers",
          [((24, 36), 12), ((4, 6), 2)])
def gcd(a: int, b: int) -> int:
    return math.gcd(int(a), int(b))


@register("numtheory.lcm", "least common multiple of two integers",
          [((4, 6), 12), ((3, 5), 15)])
def lcm(a: int, b: int) -> int:
    return math.lcm(int(a), int(b))


@register("numtheory.factorial", "factorial of a non-negative integer (n!)",
          [((5,), 120), ((7,), 5040), ((0,), 1)])
def factorial(n: int) -> int:
    return math.factorial(int(n))


@register("numtheory.combinations", "number of combinations, n choose k (nCr)",
          [((8, 3), 56), ((10, 2), 45)])
def combinations(n: int, k: int) -> int:
    return math.comb(int(n), int(k))


@register("numtheory.permutations", "number of permutations of n taken k (nPr)",
          [((5, 2), 20), ((5, 5), 120)])
def permutations(n: int, k: int) -> int:
    return math.perm(int(n), int(k))


@register("numtheory.divisor_count", "how many positive divisors an integer has",
          [((28,), 6), ((12,), 6), ((1,), 1)])
def divisor_count(n: int) -> int:
    n = int(n)
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1 if i * i == n else 2
        i += 1
    return count


@register("numtheory.is_prime", "whether an integer is prime",
          [((17,), True), ((100,), False), ((2,), True), ((1,), False)])
def is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@register("numtheory.is_perfect_square", "whether an integer is a perfect square",
          [((100,), True), ((99,), False), ((1,), True), ((0,), True)])
def is_perfect_square(n: int) -> bool:
    n = int(n)
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


@register("numtheory.prime_factorization", "prime factorization as a formatted string",
          [((60,), "2^2 × 3 × 5"), ((12,), "2^2 × 3"), ((7,), "7")])
def prime_factorization(n: int) -> str:
    n = int(n)
    if n < 2:
        return str(n)
    factors: dict[int, int] = {}
    m, d = n, 2
    while d * d <= m:
        while m % d == 0:
            factors[d] = factors.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    return " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items()))

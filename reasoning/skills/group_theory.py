"""Group theory — finite groups: cyclic groups Z_n, permutations (symmetric group), and a
Cayley-table group-axiom check. Pure stdlib, zero model calls.

Permutations are one-line tuples over 1..n: perm[i-1] is the image of i. So (2, 3, 1) sends
1→2, 2→3, 3→1.
"""

from __future__ import annotations

from functools import reduce
from math import gcd, lcm

from .base import register


# --- cyclic group Z_n (addition mod n) ---
@register("group.zn_add", "addition in Z_n: (a + b) mod n", [((3, 4, 5), 2), ((2, 2, 4), 0)])
def zn_add(a: int, b: int, n: int) -> int:
    return (a + b) % n


@register("group.element_order", "order of element a in (Z_n, +) = n / gcd(a, n)",
          [((2, 6), 3), ((1, 5), 5), ((3, 6), 2)])
def element_order(a: int, n: int) -> int:
    a %= n
    if a == 0:
        return 1
    return n // gcd(a, n)


@register("group.zn_generators", "the generators of Z_n (elements coprime to n)",
          [((6,), (1, 5)), ((5,), (1, 2, 3, 4)), ((8,), (1, 3, 5, 7))])
def zn_generators(n: int) -> tuple:
    return tuple(a for a in range(1, n) if gcd(a, n) == 1)


# --- permutations / symmetric group ---
def _cycles(perm: tuple) -> list:
    n, seen, cycles = len(perm), set(), []
    for start in range(1, n + 1):
        if start in seen:
            continue
        cyc, x = [], start
        while x not in seen:
            seen.add(x)
            cyc.append(x)
            x = perm[x - 1]
        cycles.append(cyc)
    return cycles


@register("group.permutation_order", "order of a permutation = lcm of its cycle lengths",
          [(((2, 3, 1),), 3), (((2, 1, 4, 3),), 2), (((1, 2, 3),), 1)])
def permutation_order(perm: tuple) -> int:
    return reduce(lcm, (len(c) for c in _cycles(perm)), 1)


@register("group.permutation_parity", "parity of a permutation (even/odd by transposition count)",
          [(((2, 1, 3),), "odd"), (((2, 3, 1),), "even"), (((1, 2, 3),), "even")])
def permutation_parity(perm: tuple) -> str:
    transpositions = sum(len(c) - 1 for c in _cycles(perm))
    return "even" if transpositions % 2 == 0 else "odd"


@register("group.permutation_cycles", "a permutation in cycle notation",
          [(((2, 3, 1),), "(1 2 3)"), (((2, 1, 4, 3),), "(1 2)(3 4)"), (((1, 2, 3),), "e")])
def permutation_cycles(perm: tuple) -> str:
    parts = ["(" + " ".join(map(str, c)) + ")" for c in _cycles(perm) if len(c) > 1]
    return "".join(parts) if parts else "e"


@register("group.compose_permutations", "compose permutations (p∘q)(i) = p(q(i))",
          [(((2, 3, 1), (2, 1, 3)), (3, 2, 1))])
def compose_permutations(p: tuple, q: tuple) -> tuple:
    return tuple(p[q[i] - 1] for i in range(len(q)))


@register("group.symmetric_group_order", "the order of the symmetric group S_n = n!",
          [((4,), 24), ((3,), 6), ((5,), 120)])
def symmetric_group_order(n: int) -> int:
    return reduce(lambda acc, k: acc * k, range(1, n + 1), 1)


# --- Cayley-table group-axiom check (elements 0..n-1; table[i][j] = i·j) ---
@register("group.is_group", "does a Cayley table define a group (closure, identity, inverses, "
          "associativity)?",
          [((((0, 1), (1, 0)),), True),                      # Z_2
           ((((0, 1), (1, 1)),), False)])                    # no inverse for 1
def is_group(table: tuple) -> bool:
    n = len(table)
    rng = range(n)
    if any(len(row) != n or any(table[i][j] not in rng for j in rng) for i, row in enumerate(table)):
        return False
    ident = [e for e in rng if all(table[e][x] == x and table[x][e] == x for x in rng)]
    if not ident:
        return False
    e = ident[0]
    if any(not any(table[x][y] == e and table[y][x] == e for y in rng) for x in rng):
        return False
    return all(table[table[a][b]][c] == table[a][table[b][c]] for a in rng for b in rng for c in rng)

"""Game theory — 2×2 game analysis. Zero model calls.

A payoff matrix is given as [[a,b],[c,d]] (row player's payoffs; rows are the row player's
moves, columns the column player's). Supports the value of a 2×2 zero-sum game (pure saddle
point if one exists, otherwise the mixed-strategy value), pure-strategy Nash equilibria of a
bimatrix game, the expected payoff under mixed strategies, and dominant-strategy detection.
"""

from __future__ import annotations

import re

from .base import register


def _m4(s: str):
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(s))]
    if len(nums) < 4:
        raise ValueError("need a 2x2 matrix (four numbers)")
    return nums[:4]


@register("game.zero_sum_value", "value of a 2x2 zero-sum game (saddle point or mixed value)",
          [(("[[1,-1],[-1,1]]",), 0.0), (("[[4,3],[2,1]]",), 3.0), (("[[2,0],[-1,3]]",), 1.0)])
def zero_sum_value(matrix: str) -> float:
    a, b, c, d = _m4(matrix)
    maximin = max(min(a, b), min(c, d))     # row player guarantees at least this
    minimax = min(max(a, c), max(b, d))     # column player holds row to at most this
    if maximin == minimax:
        return float(maximin)               # pure saddle point
    return round((a * d - b * c) / (a + d - b - c), 6)


@register("game.pure_nash", "pure-strategy Nash equilibria of a 2x2 bimatrix game (row, col payoffs)",
          [(("[[3,0],[5,1]]", "[[3,5],[0,1]]"), "(2,2)"),
           (("[[1,0],[0,1]]", "[[1,0],[0,1]]"), "(1,1), (2,2)"),
           (("[[0,0],[0,0]]", "[[1,0],[0,1]]"), "(1,1), (2,2)")])
def pure_nash(row_payoffs: str, col_payoffs: str) -> str:
    a = _m4(row_payoffs)
    b = _m4(col_payoffs)
    A = [[a[0], a[1]], [a[2], a[3]]]
    B = [[b[0], b[1]], [b[2], b[3]]]
    eq = [f"({i+1},{j+1})" for i in (0, 1) for j in (0, 1)
          if A[i][j] >= A[1 - i][j] and B[i][j] >= B[i][1 - j]]
    return ", ".join(eq) if eq else "none"


@register("game.expected_payoff", "expected row payoff with row playing row1 w.p. p, col col1 w.p. q",
          [(("[[3,1],[0,2]]", 0.5, 0.5), 1.5), (("[[2,0],[0,2]]", 1, 0), 0.0)])
def expected_payoff(matrix: str, p: float, q: float) -> float:
    a, b, c, d = _m4(matrix)
    p, q = float(p), float(q)
    return round(p * q * a + p * (1 - q) * b + (1 - p) * q * c + (1 - p) * (1 - q) * d, 6)


@register("game.dominant_strategy", "the row player's strictly dominant strategy, or none",
          [(("[[3,1],[0,0]]",), "row 1"), (("[[3,1],[0,2]]",), "none"),
           (("[[0,0],[3,1]]",), "row 2")])
def dominant_strategy(matrix: str) -> str:
    a, b, c, d = _m4(matrix)
    if a > c and b > d:
        return "row 1"
    if c > a and d > b:
        return "row 2"
    return "none"

"""bind layer — game theory. Gated on game-theory cues (Nash / payoff / zero-sum / dominant
strategy / saddle), so a bare [[a,b],[c,d]] matrix without them stays with linalg."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_MAT = re.compile(r"\[\s*\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*\]")
_GATE = re.compile(r"nash|payoff|zero[ -]?sum|dominant strategy|saddle|game theory|equilibri",
                   re.IGNORECASE)


def bind_game(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    mats = _MAT.findall(query)
    if not mats:
        return False, None, "code", 0

    if "nash" in q or "equilibri" in q:
        if len(mats) >= 2:
            return ok("game.pure_nash", ["row_payoffs", "col_payoffs"], [mats[0], mats[1]],
                      round=False)
        return False, None, "code", 0
    if "dominant" in q:
        return ok("game.dominant_strategy", ["matrix"], [mats[0]], round=False)
    if "expected payoff" in q or ("payoff" in q and "p=" in q.replace(" ", "")):
        pm = re.search(r"p\s*=\s*([\d.]+)", q)
        qm = re.search(r"q\s*=\s*([\d.]+)", q)
        if pm and qm:
            return ok("game.expected_payoff", ["matrix", "p", "q"],
                      [mats[0], float(pm[1]), float(qm[1])], round=True, places=6)
        return False, None, "code", 0
    if "value" in q or "zero" in q or "saddle" in q:
        return ok("game.zero_sum_value", ["matrix"], [mats[0]], round=True, places=6)
    return False, None, "code", 0

"""bind layer — ordinal numbers. Gated on an ordinal signal (the symbol ω, the word 'omega',
or 'ordinal') — no other domain uses ω, so it routes confidently. Default op is normalize
(evaluate the ω-expression to Cantor normal form); 'compare/greater/less/than' compares two."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_CMP = ("compare", "greater", "less", "larger", "smaller", "bigger", "than", "versus", " vs ",
        "equal")


def _exprs(query: str):
    q = query.lower().replace("omega", "ω").replace("×", "*").replace("·", "*")
    q = q.replace(" multiplied by ", "*").replace(" times ", "*").replace(" plus ", "+")
    q = q.replace(" ", "")
    return [m for m in re.findall(r"[ω\d+*^]+", q) if "ω" in m]


def bind_ordinal(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "ω" not in query and "omega" not in q and "ordinal" not in q:
        return False, None, "code", 0
    exprs = _exprs(query)
    if not exprs:
        return False, None, "code", 0

    if any(w in q for w in _CMP) and len(exprs) >= 2:
        return ok("ord.compare", ["a", "b"], [exprs[0], exprs[1]], round=False)

    # otherwise normalize the (longest) ordinal expression to Cantor normal form
    return ok("ord.normalize", ["expr"], [max(exprs, key=len)], round=False)

"""bind layer — hyperreal numbers. ω alone means an ORDINAL, so this domain gates strictly on
a hyperreal-specific signal (ε / epsilon / infinitesimal / hyperreal / 'standard part'); only
then is ω read as the infinite ε^-1. Ops: standard part, classify, compare, and normalize an
ε/ω expression (the default)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_GATE = re.compile(r"ε|epsilon|infinitesimal|hyperreal|standard[ -]?part", re.IGNORECASE)


def _norm(query: str) -> str:
    return (query.lower().replace("omega", "ω").replace("epsilon", "ε")
            .replace("·", "*").replace("×", "*").replace(" ", "").replace("1/ε", "ω"))


def _toks(query: str):
    # rstrip trailing dots so a sentence-ending "." isn't swallowed into a token (the "." in the
    # class is for decimal coefficients like 0.5ε).
    return [t.rstrip(".") for t in re.findall(r"[εω\d+*^.]+", _norm(query)) if t.rstrip(".")]


def _exprs(query: str):
    """ε/ω-bearing expressions (for single-operand ops and normalize)."""
    return [t for t in _toks(query) if "ε" in t or "ω" in t]


def _operands(query: str):
    """ε/ω expressions OR plain numbers (for compare, whose 2nd operand may be a real)."""
    return [t for t in _toks(query) if any(c.isdigit() or c in "εω" for c in t)]


def bind_hyperreal(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    exprs = _exprs(query)
    if not exprs:
        return False, None, "code", 0

    if "standard part" in q or re.search(r"\bst\b", q):
        return ok("hreal.standard_part", ["x"], [exprs[-1]], round=False)
    if "classify" in q or "infinitesimal" in q or "infinite" in q or "finite" in q:
        return ok("hreal.classify", ["x"], [exprs[-1]], round=False)
    if any(w in q for w in ("compare", "greater", "less", "than", "larger", "smaller", "bigger")):
        ops = _operands(query)
        if len(ops) >= 2:
            return ok("hreal.compare", ["a", "b"], [ops[0], ops[1]], round=False)
        return False, None, "code", 0
    # default: evaluate/normalize the (longest) hyperreal expression
    return ok("hreal.normalize", ["expr"], [max(exprs, key=len)], round=False)

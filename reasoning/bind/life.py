"""bind layer — life sciences (genetics; biochemistry/molecular appended later)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind import entities as _E

# a genotype is two of the SAME letter (Aa/AA/aa), so it won't match words like "of"/"in"
_GENO = re.compile(r"\b([A-Za-z])([A-Za-z])\b")


def _genotypes(query: str) -> list:
    return [m.group(0) for m in _GENO.finditer(query)
            if m.group(1).lower() == m.group(2).lower()]


def bind_genetics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, args, kind):
        return True, {"skill": skill, "args": args, "kind": kind}, "code", 0

    if "hardy" in q and "weinberg" in q:
        p = _E.only_number(query)
        if p is not None and 0 <= p <= 1:
            return ok("genetics.hardy_weinberg", [p], "genotypes")
    gts = _genotypes(query)
    if ("cross" in q or "×" in query or "punnett" in q) and len(gts) >= 2:
        return ok("genetics.monohybrid_cross", [gts[0], gts[1]], "str")
    if "allele frequency" in q or ("allele" in q and "frequency" in q):
        m = re.search(r"(\d+)\s*(?:out of|of|/)\s*(\d+)", q)
        if m:
            return ok("genetics.allele_frequency", [int(m[1]), int(m[2])], "num")
    return False, None, "code", 0

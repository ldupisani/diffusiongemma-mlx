"""bind layer — biochemistry (DNA/RNA ops, buffer & enzyme equations)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_SEQ = re.compile(r"\b([ACGTU]{3,})\b")  # uppercase nucleotides, length ≥ 3


def _after(q, label):  # (?:…) so an alternation in `label` doesn't swallow the capture group
    m = re.search(rf"(?:{label})\D{{0,8}}?(\d+(?:\.\d+)?)", q)
    return float(m[1]) if m else None


def bind_biochem(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    seq = _SEQ.search(query)
    if seq:
        s = seq.group(1)
        if "gc content" in q or "gc-content" in q:
            return ok("biochem.gc_content", ["dna"], [s], suffix="%", places=2)
        if "reverse complement" in q:
            return ok("biochem.reverse_complement", ["dna"], [s], round=False)
        if "transcribe" in q or "mrna" in q or "to rna" in q or "messenger rna" in q:
            return ok("biochem.transcribe", ["dna"], [s], round=False)
    if "henderson" in q or ("pka" in q and ("base" in q or "acid" in q)):
        pka, base, acid = _after(q, "pka"), _after(q, "base"), _after(q, "acid")
        if pka is not None and base is not None and acid is not None:
            return ok("biochem.henderson_hasselbalch", ["pka", "base", "acid"],
                      [pka, base, acid], places=2)
    if "michaelis" in q or ("vmax" in q and "km" in q):
        vmax, km = _after(q, "vmax"), _after(q, "km")
        s = _after(q, r"substrate|\[s\]")
        if vmax is not None and km is not None and s is not None:
            return ok("biochem.michaelis_menten", ["vmax", "km", "s"], [vmax, km, s], places=2)
    return False, None, "code", 0

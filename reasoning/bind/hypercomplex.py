"""bind layer — hypercomplex (quaternion) numbers. Gated on a quaternion signal: a j/k
imaginary unit (which complex numbers lack) or the word 'quaternion'.

Only the unambiguous SINGLE-operand ops are bound — conjugate, norm, inverse. The Hamilton
product and sum exist as tested skills but stay unbound: "product of A and B" is claimed by the
decomposition (compose) tier before any bind-gated orchestrator runs — the same reason
complex_numbers leaves multiply/divide unbound.
"""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

# a j or k imaginary unit appears, or the word quaternion → quaternion territory (not complex).
_QSIGNAL = re.compile(r"[\d.\s+\-][jk]\b|\b[jk]\b|quaternion", re.IGNORECASE)
# a quaternion literal token: a signed number with optional i/j/k tail, or a bare i/j/k that is
# not part of a word (so the 'i' inside "quaternion" is not mistaken for the unit i).
_QTOK = re.compile(r"(?<![a-z])[-+]?\d[\d+\-.ijk]*|(?<![a-z])[ijk](?![a-z])", re.IGNORECASE)


def _operand(query: str):
    toks = [m.lower() for m in _QTOK.findall(query)]
    units = [t for t in toks if any(u in t for u in "ijk")]
    cand = units or toks
    return cand[-1] if cand else None


def bind_hypercomplex(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, **extra):
        lit = _operand(query)
        if not lit:
            return False, None, "code", 0
        return True, {"skill": skill, "dims": ["q"], "q": lit, **extra}, "code", 0

    if not _QSIGNAL.search(query):
        return False, None, "code", 0
    if "conjugate" in q:
        return ok("hyper.q_conjugate", round=False)
    if "norm" in q or "magnitude" in q or "absolute value" in q:
        return ok("hyper.q_norm", round=True, places=4)
    if "inverse" in q or "reciprocal" in q:
        return ok("hyper.q_inverse", round=False)
    return False, None, "code", 0

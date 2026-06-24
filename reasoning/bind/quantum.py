"""bind layer — quantum mechanics (single qubit). Gated on quantum cues (qubit / quantum /
Hadamard / Pauli / measurement / expectation value / a ket like |0>). States are read from a
ket (|0>,|1>,|+>,|->) or an amplitude pair (α, β)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_GATE = re.compile(r"qubit|quantum|hadamard|pauli|\bgate\b|measurement|measuring|born rule|"
                   r"expectation value|superposition|amplitude|\|[01+\-]\s*(?:>|⟩)", re.IGNORECASE)
_KETS = {"|0>": "1, 0", "|0⟩": "1, 0", "|1>": "0, 1", "|1⟩": "0, 1",
         "|+>": "1, 1", "|+⟩": "1, 1", "|->": "1, -1", "|-⟩": "1, -1"}


def _state(query: str):
    for k, v in _KETS.items():
        if k in query.replace(" ", ""):
            return v
    m = re.search(r"\(\s*([^)]+?,[^)]+?)\s*\)", query)        # (α, β)
    if m:
        return m[1].strip()
    m = re.search(r"state\s+([-\d./a-z()\s]+?,[-\d./a-z()\s]+?)(?:[?.]|$)", query, re.IGNORECASE)
    return m[1].strip() if m else None


def _gate(q: str):
    if "hadamard" in q:
        return "H"
    if "pauli-x" in q or "pauli x" in q or "x gate" in q:
        return "X"
    if "pauli-z" in q or "pauli z" in q or "z gate" in q:
        return "Z"
    return None


def bind_quantum(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    state = _state(query)
    if not state:
        return False, None, "code", 0

    if "expectation" in q or re.search(r"<\s*[xz]\s*>", q):
        om = re.search(r"(?:expectation value of\s+(?:the\s+)?(?:pauli[- ])?|<\s*)([xz])", q)
        if om:
            return ok("quantum.expectation", ["pauli", "state"], [om[1].upper(), state],
                      round=True, places=6)
        return False, None, "code", 0
    if "measur" in q or "born" in q or "probabilit" in q:
        return ok("quantum.measurement_probabilities", ["state"], [state], round=False)
    if "normali" in q:
        return ok("quantum.normalize", ["state"], [state], round=False)
    g = _gate(q)
    if g:
        return ok("quantum.apply_gate", ["gate", "state"], [g, state], round=False)
    return False, None, "code", 0

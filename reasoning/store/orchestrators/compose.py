"""Compose — answer a compound query by code-first decomposition (Phase 4).

Splits on conjunctions and routes each clause to a coded orchestrator; claims a query only
when EVERY clause answers model-free, so it never mis-splits a single query. The ≤1-call
model decomposition fallback (decompose.decompose_with_model) lives in the CLI's no-match
branch, not here — this orchestrator is the pure 0-model-call path.

Declared model_free=False so it stays OUT of the leaf-fidelity eval (its answer is a join of
already-verified parts, and the joined format isn't meant to match a single teacher answer);
its 0-model-call guarantee is pinned by tests/test_decompose.py instead.
"""

from reasoning.compiled import explain
from reasoning.decompose import decompose_code_first

META = {
    "id": "compose",
    "query_type": "compound",
    "description": "a compound request whose parts each map to a coded orchestrator",
    "example_queries": [
        "Convert 5 miles to km, and what is 20% of 50?",
        "Convert 100 Celsius to Fahrenheit and what is the GCD of 24 and 36?",
    ],
    "keywords": [],
    "model_free": False,   # excluded from leaf-fidelity eval (see module docstring)
    "fallback": True,      # only after specific / bind-gated / clarify tiers
    "composer": True,      # routed by the composer tier when its precondition decomposes
}


def precondition(query):
    return decompose_code_first(query) is not None  # ≥2 clauses, each answers model-free


def run(query):
    return decompose_code_first(query) or explain("could not decompose into coded sub-answers")

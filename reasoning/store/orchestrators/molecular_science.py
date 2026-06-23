"""Molecular science — Avogadro conversions (moles ↔ molecules) and degrees of unsaturation.
Coded, zero model calls. Avogadro-scale counts are shown in scientific notation."""

from reasoning import skills
from reasoning.bind import bind_molecular
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "molecular_science",
    "query_type": "molecular_science",
    "description": "number of molecules in n moles (and the reverse) via Avogadro's number, "
                   "and degrees of unsaturation of a molecular formula",
    "example_queries": [
        "How many molecules are in 2 moles?",
        "How many moles are in 1.204428152e24 molecules?",
        "What are the degrees of unsaturation of C6H6?",
    ],
    "keywords": ["molecules", "moles", "avogadro", "degrees of unsaturation"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_molecular(query)[0]


def run(query):
    ok, data, via, _ = bind_molecular(query)
    if not ok:
        return explain("could not parse the molecular inputs")
    r = skills.call(data["skill"], *data["args"])
    if data["kind"] == "sci":
        answer = f"{r:.4e}"
    elif data["kind"] == "int":
        answer = str(r)
    else:
        answer = str(round(r, 4))
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": str(r)},
                                {"step": "summary", "model_calls": 0}])

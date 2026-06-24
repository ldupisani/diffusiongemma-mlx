"""Synthesis-backed chemistry — the bridge's fourth live domain.

"How many moles are in 36 grams of H2O?" → synthesize molar_mass(formula) → moles
(grams / molar_mass). The existing chemistry orchestrator needs the molar mass GIVEN; this
computes it from the formula first — a derivation no single step does. 0 model calls.
"""

from reasoning.bind import bind_synthesis_chemistry
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import (CHEM_OPS, COERCION_OPS, CROSS_OPS, MOLECULAR_OPS, WEIGHT_OPS,
                                  synthesize_and_run)

META = {
    "id": "synthesized_chemistry",
    "query_type": "chemistry_derivation",
    "description": "moles in a mass of a compound (or the mass of N moles), computing the "
                   "molar mass from the chemical formula first",
    "example_queries": [
        "How many moles are in 36 grams of H2O?",
        "What is the mass of 2 moles of H2O?",
        "How many moles are in 100 grams of NaCl?",
        "What is the cost of 3 moles of NaCl at $0.50 per gram?",
    ],
    "keywords": ["moles", "mass", "grams", "formula", "cost"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_synthesis_chemistry(query)[0]


def run(query):
    ok, data, via, _ = bind_synthesis_chemistry(query)
    if not ok:
        return explain("could not extract a chemical formula and a mass or mole quantity")
    r = synthesize_and_run(data["knowns"], data["goal"],
                           CHEM_OPS + MOLECULAR_OPS + WEIGHT_OPS + COERCION_OPS + CROSS_OPS,
                           unique=data.get("unique", False))
    if r is None:
        return explain("no derivation path to the goal")
    # Avogadro-scale molecule counts read better in scientific notation than a 25-digit integer
    answer = f"{r.data['molecules']:.4e}" if data["goal"] == "molecules" else r.answer
    return PatternResult(answer=data["prefix"] + answer + data["suffix"],
                         trace=r.trace, data=r.data)

"""Synthesis-backed material science — physics force + a geometry shape's area → stress, and
stress + strain → Young's modulus. Two physical-science domains feed one materials goal, with
the area DERIVED from a shape (the single-formula case stays with the `materials` orchestrator).
Coded, zero model calls.
"""

from reasoning.bind import bind_synthesis_materials
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import (GEOMETRY_COST_OPS, MATERIALS_OPS, PHYSICS_OPS,
                                  synthesize_and_run)

META = {
    "id": "synthesized_materials",
    "query_type": "materials_derivation",
    "description": "mechanical stress (force over a shape's area) and Young's modulus, with the "
                   "area derived from the shape's dimensions",
    "example_queries": [
        "What is the stress on a circular plate of radius 2 m under a 100 N force?",
        "What is the stress on a 3 by 4 m plate under a 1200 N force?",
        "What is Young's modulus of a 3 by 4 m plate under 1200 N with strain 0.001?",
    ],
    "keywords": ["stress", "young", "modulus", "plate", "force"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_synthesis_materials(query)[0]


def run(query):
    ok, data, via, _ = bind_synthesis_materials(query)
    if not ok:
        return explain("could not extract a force, a shape, and a materials goal")
    r = synthesize_and_run(data["knowns"], data["goal"],
                           GEOMETRY_COST_OPS + PHYSICS_OPS + MATERIALS_OPS, unique=True)
    if r is None:
        return explain("no derivation path to the materials goal")
    return PatternResult(answer=r.answer + data["suffix"], trace=r.trace, data=r.data)

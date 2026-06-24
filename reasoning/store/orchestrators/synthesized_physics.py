"""Synthesis-backed physics — the first LIVE use of the plan synthesizer.

Answers physics questions that need a MULTI-STEP derivation the single-formula `physics`
orchestrator can't do (e.g. power from mass + acceleration + distance + time, via
force → work → power). The bind extracts unit-cued quantities as typed knowns and the asked
quantity as the goal; synthesize() finds the operation path; run_plan executes it — 0 model
calls. Mutually exclusive with `physics`: it claims only ≥2-op derivations.
"""

from reasoning.bind import bind_synthesis_physics
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import (GEOMETRY_COST_OPS, MOTION_OPS, PHYSICS_OPS,
                                  synthesize_and_run)

META = {
    "id": "synthesized_physics",
    "query_type": "physics_derivation",
    "description": "derive a physics quantity that requires chaining several formulas "
                   "(e.g. power from mass, acceleration, distance, and time)",
    "example_queries": [
        "What is the power if a 2 kg mass accelerates at 3 m/s^2 over 4 meters in 5 seconds?",
        "How much work is done accelerating a 10 kg mass at 2 m/s^2 over 5 meters?",
        "What is the power delivered moving a 4 kg object at 3 m/s^2 across 6 meters in 2 seconds?",
        "What is the pressure of a 100 N force on a circular plate of radius 2 m?",
        "What is the kinetic energy of a 1000 kg car that travels 200 m in 10 s?",
    ],
    "keywords": ["power", "work", "force", "kinetic", "momentum", "pressure"],
    "model_free": True,
    "bind_gated": True,  # the synthesizable parse is dispositive (and ≥2-op, so physics can't bind)
}


def precondition(query):
    return bind_synthesis_physics(query)[0]


def run(query):
    ok, data, via, _ = bind_synthesis_physics(query)
    if not ok:
        return explain("could not extract physics quantities and a derivable goal")
    r = synthesize_and_run(data["knowns"], data["goal"],
                           PHYSICS_OPS + MOTION_OPS + GEOMETRY_COST_OPS, unique=True)
    if r is None:
        return explain("no derivation path to the goal")
    return PatternResult(answer=r.answer + data["suffix"], trace=r.trace, data=r.data)

"""Algebra — factor, expand, simplify, solve (SymPy-backed, deterministic, zero model calls)."""

from reasoning import engine
from reasoning.bind import bind_algebra
from reasoning.compiled import CodeStep, Lit, Plan, explain, run_plan

META = {
    "id": "algebra",
    "query_type": "algebra",
    "description": "symbolic algebra: factor, expand, simplify an expression, or solve an equation",
    "example_queries": [
        # [0] is power-free ("x + 1") so eval-all isn't tripped by the teacher writing powers
        # as the superscript x² (which the comparator strips) vs the coded x**2.
        "Simplify (x^2 - 1)/(x - 1).",
        "Expand (x + 1)(x + 2).",
        "Factor x^2 - 5x + 6.",
        "Solve x^2 - 5x + 6 = 0 for x.",
    ],
    "keywords": ["factor", "expand", "simplify", "solve", "equation", "polynomial"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_algebra(query)[0]


def run(query):
    ok, data, via, _ = bind_algebra(query)
    if not ok:
        return explain("could not parse the algebraic expression")
    # run through run_plan to inherit the crash-proofing backstop (a CAS that can't
    # solve/compute degrades to "(no result: …)" instead of raising)
    plan = Plan(steps=[CodeStep(data["skill"], [Lit(a) for a in data["args"]], out="r")],
                answer_ref="r")
    return run_plan(plan, {}, shot=engine._no_model)

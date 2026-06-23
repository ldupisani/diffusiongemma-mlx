"""Algebra — factor, expand, simplify, solve (SymPy-backed, deterministic, zero model calls)."""

from reasoning import skills
from reasoning.bind import bind_algebra
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

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
    try:
        answer = skills.call(data["skill"], *data["args"])
    except Exception as e:  # SymPy can raise many things on hard/unsupported input
        return explain(f"{data['skill']} could not run ({type(e).__name__})")
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": answer},
                                {"step": "summary", "model_calls": 0}])

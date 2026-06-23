"""Synthesis-backed geometry → cost — the bridge's third live domain.

"How much to tile a 4 meter by 6 meter floor at $8 per square meter?" → synthesize
rectangle_area → ×price (area → total_cost). A new capability no single formula does: the
bind extracts the surface's dimensions and a per-area price; the plan is discovered. 0 model
calls.
"""

from reasoning.bind import bind_synthesis_geometry
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import CROSS_OPS, GEOMETRY_COST_OPS, synthesize_and_run

META = {
    "id": "synthesized_geometry",
    "query_type": "geometry_cost",
    "description": "total cost to cover a surface: its area times a price per unit area "
                   "(tiling, painting, carpeting)",
    "example_queries": [
        "How much does it cost to tile a 4 meter by 6 meter floor at $8 per square meter?",
        "What is the cost to carpet a 5 by 3 room at $12 per square meter?",
        "How much to paint a wall with a radius of 5 at $3 per square foot?",
        "What does it cost to tile a 4 by 6 meter floor at $8 per square meter, "
        "split between 3 people?",
    ],
    "keywords": ["cost", "tile", "carpet", "paint", "per square", "how much", "split"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_synthesis_geometry(query)[0]


def run(query):
    ok, data, via, _ = bind_synthesis_geometry(query)
    if not ok:
        return explain("could not extract a surface's dimensions and a per-area price")
    r = synthesize_and_run(data["knowns"], data["goal"], GEOMETRY_COST_OPS + CROSS_OPS,
                           unique=data.get("unique", False))
    if r is None:
        return explain("no derivation path to the total cost")
    return PatternResult(answer=data["prefix"] + r.answer + data["suffix"],
                         trace=r.trace, data=r.data)

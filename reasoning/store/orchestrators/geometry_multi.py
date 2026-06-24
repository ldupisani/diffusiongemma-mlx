"""Multi-output geometry — several properties of one shape in a single synthesized plan, with
shared sub-steps computed once. Coded, zero model calls."""

from reasoning.bind import bind_geometry_multi
from reasoning.compiled import explain, format_number
from reasoning.patterns import PatternResult
from reasoning.synthesize import GEOMETRY_COST_OPS, synthesize_multi_and_run

META = {
    "id": "geometry_multi",
    "query_type": "geometry_multi",
    "description": "two or more properties of one shape at once (e.g. the area and perimeter "
                   "of a rectangle), derived in a single multi-goal plan",
    "example_queries": [
        "What are the area and perimeter of a 4 by 6 rectangle?",
        "Find the perimeter and area of a 5 by 3 rectangle.",
    ],
    "keywords": ["area", "perimeter", "and"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_geometry_multi(query)[0]


def run(query):
    ok, data, via, _ = bind_geometry_multi(query)
    if not ok:
        return explain("could not extract a shape and two properties")
    res = synthesize_multi_and_run(data["knowns"], data["goals"], GEOMETRY_COST_OPS, unique=True)
    if res is None:
        return explain("no derivation path to the requested properties")
    r, values = res
    answer = ", ".join(f"{g}: {format_number(v)}" for g, v in values.items())
    return PatternResult(answer=answer, trace=r.trace, data=r.data)

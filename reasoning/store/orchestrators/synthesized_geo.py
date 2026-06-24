"""Synthesis-backed geospatial — haversine distance chained into a finance cost ($/km) or a
motion time (km/h). geo → finance / geo → motion. Coded, zero model calls."""

from reasoning.bind import bind_synthesis_geo
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import COERCION_OPS, CROSS_OPS, GEO_OPS, synthesize_and_run

META = {
    "id": "synthesized_geo",
    "query_type": "geo_derivation",
    "description": "great-circle distance between two points chained into a travel cost (per km) "
                   "or a flight time (at a km/h speed)",
    "example_queries": [
        "What is the travel cost between (40.7128, -74.0060) and (51.5074, -0.1278) at $0.10 per km?",
        "How long is a flight between (40.7128, -74.0060) and (51.5074, -0.1278) at 900 km/h?",
    ],
    "keywords": ["travel cost", "flight", "how long", "per km", "km/h", "coordinates"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_synthesis_geo(query)[0]


def run(query):
    ok, data, via, _ = bind_synthesis_geo(query)
    if not ok:
        return explain("could not extract two points and a price or speed")
    r = synthesize_and_run(data["knowns"], data["goal"], GEO_OPS + COERCION_OPS + CROSS_OPS,
                           unique=True)
    if r is None:
        return explain("no derivation path to the goal")
    return PatternResult(answer=data["prefix"] + r.answer + data["suffix"],
                         trace=r.trace, data=r.data)

"""Synthesis: travel between two NAMED places — geography (name → coordinates) → geo (haversine
distance) → motion/finance (time at a speed, or cost per distance). A three-domain chain over a
shared entity (a place); all lookups + arithmetic, 0 model calls."""
from reasoning.bind import bind_travel
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import CROSS_OPS, GEO_OPS, synthesize_and_run

META = {
    "id": "synthesized_travel",
    "query_type": "travel",
    "description": "travel between two named places (resolved via the geography reference): the "
                   "great-circle distance, the travel time at a given speed, or the cost per distance",
    "example_queries": [
        "How far is London from Casablanca?",
        "What is the distance between Tokyo and Sydney?",
        "How long would it take to travel from London to Casablanca at 60 mph?",
        "What is the travel cost from Paris to Berlin at $0.20 per km?",
    ],
    "keywords": ["how far", "distance between", "travel from", "how long", "flight from",
                 "drive from"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_travel(query)[0]


def run(query):
    ok, d, _, _ = bind_travel(query)
    if not ok:
        return explain("could not resolve two known places and a travel goal")
    r = synthesize_and_run(d["knowns"], d["goal"], GEO_OPS + CROSS_OPS, unique=d.get("unique", True))
    if r is None:
        return explain("no derivation path for the travel query")
    return PatternResult(answer=d["prefix"] + r.answer + d["suffix"], trace=r.trace, data=r.data)

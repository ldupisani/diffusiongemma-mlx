"""Priced quantities → cost — labour (hours × $/hour) and electricity (kW × hours = kWh ×
$/kWh, a physics → finance chain), then the universal split/tax funnel. Coded, 0 model calls."""

from reasoning.bind import bind_priced_cost
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import CROSS_OPS, PRICED_OPS, synthesize_and_run

META = {
    "id": "priced_cost",
    "query_type": "priced_cost",
    "description": "the cost of a priced quantity — a job (hours at an hourly rate) or "
                   "electricity (a power draw over time at a per-kWh price) — optionally split "
                   "among people or taxed",
    "example_queries": [
        "How much is a 5-hour job at $40 per hour?",
        "What does a 6-hour job at $50 per hour cost, split between 2 people?",
        "What is the cost of running a 2 kW heater for 3 hours at $0.15 per kWh?",
    ],
    "keywords": ["per hour", "per kwh", "hourly", "job", "kw", "electricity"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_priced_cost(query)[0]


def run(query):
    ok, data, via, _ = bind_priced_cost(query)
    if not ok:
        return explain("could not extract a priced quantity and a per-unit rate")
    r = synthesize_and_run(data["knowns"], data["goal"], PRICED_OPS + CROSS_OPS,
                           unique=data.get("unique", False))
    if r is None:
        return explain("no derivation path to the cost")
    return PatternResult(answer=data["prefix"] + r.answer + data["suffix"],
                         trace=r.trace, data=r.data)

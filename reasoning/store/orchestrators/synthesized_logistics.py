"""Synthesis-backed logistics → cost — a logistics quantity priced by finance.

"How much does it cost to ship 1000 units if each truck holds 150 units and costs $500 per
truck?" → synthesize trucks_needed (= ceil(1000/150) = 7) → ×$500 → $3500 (and optionally
split/taxed via the cross-domain cost funnel). A new capability no single formula does; the
plan is discovered. 0 model calls.
"""

from reasoning.bind import bind_synthesis_logistics
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import CROSS_OPS, LOGISTICS_OPS, synthesize_and_run

META = {
    "id": "synthesized_logistics",
    "query_type": "logistics_cost",
    "description": "total shipping cost: the number of trucks/containers needed to move some "
                   "units (ceil(units / capacity)) times a per-truck cost, optionally split or taxed",
    "example_queries": [
        "How much does it cost to ship 1000 units if each truck holds 150 units and costs $500 per truck?",
        "What is the total cost to deliver 600 boxes if each truck holds 200 and costs $400 per truck?",
        "How much to ship 1000 units, trucks hold 150 each at $500 per truck, split between 2 warehouses?",
    ],
    "keywords": ["cost", "ship", "trucks", "per truck", "how much", "deliver", "containers"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_synthesis_logistics(query)[0]


def run(query):
    ok, data, _, _ = bind_synthesis_logistics(query)
    if not ok:
        return explain("could not extract units, a truck capacity, and a per-truck cost")
    r = synthesize_and_run(data["knowns"], data["goal"], LOGISTICS_OPS + CROSS_OPS,
                           unique=data.get("unique", False))
    if r is None:
        return explain("no derivation path to the shipping cost")
    return PatternResult(answer=data["prefix"] + r.answer + data["suffix"],
                         trace=r.trace, data=r.data)

"""Synthesis-backed finance — the bridge generalizes to a second domain.

Answers "a $240 jacket, 17% off, 8% tax, split between 3 — cost each?" by synthesizing
discount → tax → split (3 ops) — the chain percent_discount_tax (discount + tax) can't do.
The bind reuses the existing discount/tax/people extractors to produce typed knowns; the
plan is discovered by synthesize(). 0 model calls.
"""

from reasoning.bind import bind_synthesis_finance
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.synthesize import FINANCE_OPS, synthesize_and_run

META = {
    "id": "synthesized_finance",
    "query_type": "finance_split_chain",
    "description": "final per-person cost after a discount and tax, split among several people",
    "example_queries": [
        "A $240 jacket is 17% off with 8% tax, split between 3 people — how much each?",
        "A $90 meal, 10% off, 8% tax, split between 4 friends — cost per person?",
        "$120 of groceries, 20% off, 5% tax, divided among 2 people — each?",
    ],
    "keywords": ["each", "per person", "split", "off", "tax", "between"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_synthesis_finance(query)[0]


def run(query):
    ok, data, via, _ = bind_synthesis_finance(query)
    if not ok:
        return explain("could not extract price, discount, tax, and a split count")
    r = synthesize_and_run(data["knowns"], data["goal"], FINANCE_OPS)
    if r is None:
        return explain("no derivation path to the per-person cost")
    return PatternResult(answer=data["prefix"] + r.answer + data["suffix"],
                         trace=r.trace, data=r.data)

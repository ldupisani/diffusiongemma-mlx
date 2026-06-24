"""Profit margin % = (price − cost) / price × 100 — coded, zero model calls.

Distinct from markup ((price−cost)/cost, which is arith.percent_change): margin divides by
PRICE. Keeping them separate avoids a silent wrong-denominator answer.
"""

from reasoning.bind import bind_margin, validate_margin
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "profit_margin",
    "query_type": "profit_margin",
    "description": "profit margin as a percentage, from cost and selling price",
    "example_queries": [
        "A product costs $40 to make and sells for $60. What is the profit margin as a percentage?",
        "If it costs $8 and sells for $20, what's the margin?",
    ],
    "keywords": ["profit", "margin"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.margin", [Inp("cost"), Inp("price")], out="m"),
        CodeStep("arith.round_to", [Step("m"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="{:g}%",
)


def precondition(query):
    return "margin" in query.lower()


def run(query):
    return run_guarded(bind_margin(query), PLAN, validate=validate_margin,
                       parse_message="could not find a cost and price")

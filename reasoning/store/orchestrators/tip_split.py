"""Add a tip to a bill and split it N ways — coded, zero model calls.

No new skill: composes the existing arith.apply_tax (the tip) with arith.divide (the split).
A clean example of generalization by COMPOSITION rather than a new formula.
"""

from reasoning.bind import bind_tip_split, validate_tip_split
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "tip_split",
    "query_type": "tip_split",
    "description": "add a percentage tip to a bill and split the total among people",
    "example_queries": [
        "A dinner bill is $120. Add an 18% tip and split it among 4 people. How much does each person pay?",
        "$80 bill, 20% tip, split between 2 — how much each?",
    ],
    "keywords": ["tip", "split", "bill", "each", "person", "people"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("arith.apply_tax", [Inp("bill"), Inp("tip_pct")], out="total"),
        CodeStep("arith.divide", [Step("total"), Inp("people")], out="each"),
        CodeStep("arith.round_to", [Step("each"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="${:.2f}",
)


def precondition(query):
    q = query.lower()
    return "tip" in q and any(w in q for w in ("split", "each", "per person", "people", "among"))


def run(query):
    return run_guarded(bind_tip_split(query), PLAN, validate=validate_tip_split,
                       parse_message="could not find a bill, tip %, and head count")

"""Break-even units = fixed costs / (price per unit − cost per unit) — coded, zero model."""

from reasoning.bind import bind_break_even, validate_break_even
from reasoning.compiled import CodeStep, Inp, Plan, run_guarded

META = {
    "id": "break_even",
    "query_type": "break_even",
    "description": "units needed to break even from fixed costs, unit price, and unit cost",
    "example_queries": [
        "Fixed costs are $5000, each unit sells for $25 and costs $15 to make. How many units to break even?",
        "With $12000 fixed costs, $40 price and $16 cost per unit, what's the break-even quantity?",
    ],
    "keywords": ["break", "even", "fixed", "units", "breakeven"],
    "model_free": True,
}

PLAN = Plan(
    steps=[CodeStep("finance.break_even_units", [Inp("fixed"), Inp("price"), Inp("cost")], out="u")],
    answer_ref="u",
    answer_format="{:g} units",
)


def precondition(query):
    q = query.lower()
    return ("break" in q and "even" in q) or "breakeven" in q


def run(query):
    return run_guarded(bind_break_even(query), PLAN, validate=validate_break_even,
                       parse_message="could not find fixed costs, unit price, and unit cost")

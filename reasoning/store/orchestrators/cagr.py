"""Compound annual growth rate % — coded, zero model calls.

Collapses the teacher's "divide, take the nth root, subtract 1, ×100" chain into the
finance.cagr formula (a fractional power), which the model is error-prone at by hand.
"""

from reasoning.bind import bind_cagr, validate_cagr
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "cagr",
    "query_type": "cagr",
    "description": "compound annual growth rate from a starting and ending value over a number of years",
    "example_queries": [
        "An investment grew from $10000 to $16000 over 4 years. What is the compound annual growth rate?",
        "Revenue went from $2M to $5M in 3 years — what's the CAGR?",
    ],
    "keywords": ["compound", "annual", "growth", "rate", "cagr", "grew"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.cagr", [Inp("begin"), Inp("end"), Inp("years")], out="g"),
        CodeStep("arith.round_to", [Step("g"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="{:g}%",
)


def precondition(query):
    q = query.lower()
    return "growth rate" in q or "cagr" in q


def run(query):
    return run_guarded(bind_cagr(query), PLAN, validate=validate_cagr,
                       parse_message="could not find begin/end values and a year count")

"""Present value of a future amount — coded, zero model calls.

The teacher unrolled (1.04)^5 into five literal multiplies; finance.present_value is the
single-step discount that generalizes to any term.
"""

from reasoning.bind import bind_present_value, validate_present_value
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "present_value",
    "query_type": "present_value",
    "description": "the present value of a future amount discounted at an annual rate over years",
    "example_queries": [
        "What is the present value of $10000 received in 5 years at a 4% annual discount rate?",
        "Present value of $5000 in 3 years at 6%?",
    ],
    "keywords": ["present", "value", "discount", "received", "today"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.present_value", [Inp("fv"), Inp("rate_pct"), Inp("years")], out="pv"),
        CodeStep("arith.round_to", [Step("pv"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="${:.2f}",
)


def precondition(query):
    return "present value" in query.lower()


def run(query):
    return run_guarded(bind_present_value(query), PLAN, validate=validate_present_value,
                       parse_message="could not find a future value, rate, and term")

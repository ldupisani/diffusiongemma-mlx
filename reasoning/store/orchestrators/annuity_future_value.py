"""Future value of regular deposits (annuity) — coded, zero model calls.

The teacher's 7-step chain (monthly rate, count, (1+i)^n, −1, /i, ×deposit) collapses to
finance.future_value_annuity — one of the highest-payoff replacements, since that chain is
exactly where a model slips.
"""

from reasoning.bind import bind_annuity, validate_annuity
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "annuity_future_value",
    "query_type": "annuity_future_value",
    "description": "the future value of regular deposits earning compound interest",
    "example_queries": [
        "If I save $200 per month at 5% annual interest for 10 years, how much will I have?",
        "Depositing $500 every month at 4% for 20 years — final amount?",
    ],
    "keywords": ["save", "deposit", "contribute", "per", "month", "monthly", "savings"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.future_value_annuity",
                 [Inp("pmt"), Inp("rate_pct"), Inp("years"), Inp("n")], out="fv"),
        CodeStep("arith.round_to", [Step("fv"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="${:.2f}",
)


def precondition(query):
    q = query.lower()
    contributes = any(w in q for w in ("save", "deposit", "contribute", "put away"))
    regular = "per month" in q or "per year" in q or "each month" in q or "every month" in q
    has_rate = "%" in q or "percent" in q
    return contributes and regular and has_rate


def run(query):
    return run_guarded(bind_annuity(query), PLAN, validate=validate_annuity,
                       parse_message="could not find a deposit, rate, and term")

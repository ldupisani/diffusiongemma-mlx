"""Monthly loan payment (amortization) — coded, zero model calls.

Replaces a 7-step amortization chain with finance.loan_payment.
"""

from reasoning.bind import bind_loan, validate_loan
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "loan_payment",
    "query_type": "loan_payment",
    "description": "the level payment per period to amortize a loan over a term",
    "example_queries": [
        "What is the monthly payment on a $20000 loan at 6% annual interest over 5 years?",
        "Monthly payment on a $300000 mortgage at 5% over 30 years?",
    ],
    "keywords": ["loan", "mortgage", "payment", "borrow", "monthly", "repay"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.loan_payment",
                 [Inp("principal"), Inp("rate_pct"), Inp("years"), Inp("n")], out="pmt"),
        CodeStep("arith.round_to", [Step("pmt"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="${:.2f}",
)


def precondition(query):
    q = query.lower()
    return ("loan" in q or "mortgage" in q) and "payment" in q


def run(query):
    return run_guarded(bind_loan(query), PLAN, validate=validate_loan,
                       parse_message="could not find a principal, rate, and term")

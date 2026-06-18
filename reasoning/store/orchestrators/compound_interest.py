"""Compound-interest final balance — coded, zero model calls.

Generalizes the teacher's fixed "×1.05 three times" chain: one formula handles any term
and any compounding frequency (annual/monthly/quarterly/...).
"""

from reasoning.bind import bind_compound_interest, validate_compound
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "compound_interest",
    "query_type": "compound_interest",
    "description": "the final balance of a principal earning compound interest over time",
    "example_queries": [
        "What is the final balance on $1000 invested at 5% annual interest compounded annually for 3 years?",
        "$5000 at 4% compounded monthly for 10 years — final amount?",
    ],
    "keywords": ["compound", "compounded", "interest", "invested", "principal",
                 "balance", "annually", "monthly"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.compound_interest",
                 [Inp("principal"), Inp("rate_pct"), Inp("years"), Inp("n")], out="amt"),
        CodeStep("arith.round_to", [Step("amt"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="${:.2f}",
)


def precondition(query):
    q = query.lower()
    has_pct = "%" in q or "percent" in q
    return "compound" in q and has_pct and "year" in q


def run(query):
    return run_guarded(bind_compound_interest(query), PLAN, validate=validate_compound,
                       parse_message="could not parse principal/rate/years from the query")

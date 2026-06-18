"""Difference between two dates in days / weeks / months / years — coded, zero model.

Generalized from days-only: the bind detects the unit asked for and run() picks the
matching skill (rounding weeks to 2 dp).
"""

from reasoning.bind import bind_date_difference
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "days_between_dates",
    "query_type": "date_difference",
    "description": "the difference between two calendar dates in days, weeks, months, or years",
    "example_queries": [
        "How many days are between 2024-03-03 and 2024-07-15?",
        "How many weeks are between 2024-01-01 and 2024-12-31?",
        "How many months are between 2023-02-15 and 2024-08-15?",
        "How many years are between 2000-05-20 and 2024-05-20?",
    ],
    "keywords": ["days", "weeks", "months", "years", "between", "from", "to", "until"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    q = query.lower()
    has_unit = any(u in q for u in ("day", "week", "month", "year"))
    return has_unit and ("between" in q or ("from" in q and "to" in q))


def _plan(data):
    steps = [
        CodeStep("date.parse", [Inp("d1")], out="date1"),
        CodeStep("date.parse", [Inp("d2")], out="date2"),
        CodeStep(data["skill"], [Step("date1"), Step("date2")], out="diff"),
    ]
    answer_ref = "diff"
    if data["rounded"]:  # weeks come out fractional
        steps.append(CodeStep("arith.round_to", [Step("diff"), Lit(2)], out="r"))
        answer_ref = "r"
    return Plan(steps=steps, answer_ref=answer_ref, answer_format="{:g} " + data["label"])


def run(query):
    return run_guarded(bind_date_difference(query), _plan,
                       parse_message="could not find two ISO dates")

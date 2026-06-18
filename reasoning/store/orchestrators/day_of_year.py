"""The ordinal day-of-year for a date (Jan 1 = 1) — coded, zero model calls."""

from reasoning.bind import bind_single_date
from reasoning.compiled import CodeStep, Inp, Plan, Step, run_guarded

META = {
    "id": "day_of_year",
    "query_type": "day_of_year",
    "description": "the ordinal day of the year for a calendar date",
    "example_queries": [
        "What is the day of the year for 2024-03-01?",
        "Which day number of the year is 2024-12-31?",
    ],
    "keywords": ["day", "year", "ordinal"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("date.parse", [Inp("date")], out="d"),
        CodeStep("date.day_of_year", [Step("d")], out="r"),
    ],
    answer_ref="r",
    answer_format="{:g}",
)


def precondition(query):
    q = query.lower()
    return "day of the year" in q or "day of year" in q


def run(query):
    return run_guarded(bind_single_date(query), PLAN,
                       parse_message="could not find a single ISO date")

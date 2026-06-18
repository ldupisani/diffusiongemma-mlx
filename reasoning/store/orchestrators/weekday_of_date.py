"""The day of the week for a given date — coded, zero model calls."""

from reasoning.bind import bind_single_date
from reasoning.compiled import CodeStep, Inp, Plan, Step, run_guarded

META = {
    "id": "weekday_of_date",
    "query_type": "weekday_of_date",
    "description": "the day of the week (Monday..Sunday) for a calendar date",
    "example_queries": [
        "What day of the week is 2024-07-15?",
        "Which weekday was 2000-01-01?",
    ],
    "keywords": ["day", "week", "weekday"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("date.parse", [Inp("date")], out="d"),
        CodeStep("date.weekday_name", [Step("d")], out="r"),
    ],
    answer_ref="r",
)


def precondition(query):
    q = query.lower()
    return ("day of the week" in q or "weekday" in q) and "between" not in q


def run(query):
    return run_guarded(bind_single_date(query), PLAN,
                       parse_message="could not find a single date")

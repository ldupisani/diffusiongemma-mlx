"""The date N days or weeks after a given date — coded, zero model calls."""

from reasoning.bind import bind_date_add
from reasoning.compiled import CodeStep, Inp, Plan, Step, run_guarded

META = {
    "id": "date_add",
    "query_type": "date_add",
    "description": "the calendar date a number of days or weeks after a given date",
    "example_queries": [
        "What date is 90 days after 2024-01-01?",
        "What date is 12 weeks after 2024-06-01?",
    ],
    "keywords": ["date", "days", "weeks", "after", "from"],
    "model_free": True,
}


def precondition(query):
    q = query.lower()
    return "after" in q and ("day" in q or "week" in q)


def _plan(data):
    return Plan(
        steps=[
            CodeStep("date.parse", [Inp("date")], out="d"),
            CodeStep(data["skill"], [Step("d"), Inp("n")], out="r"),
        ],
        answer_ref="r",
    )


def run(query):
    return run_guarded(bind_date_add(query), _plan,
                       parse_message="could not find 'N days/weeks after <date>'")

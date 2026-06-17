"""Days between two ISO dates — fully coded, zero model calls."""

from reasoning import engine
from reasoning.bind import bind_days_between
from reasoning.compiled import CodeStep, Inp, Plan, Step, run_plan

META = {
    "id": "days_between_dates",
    "query_type": "days_between_dates",
    "description": "the number of days between two calendar dates",
    "example_queries": [
        "How many days are between 2024-03-03 and 2024-07-15?",
        "Days from 2025-01-01 to 2025-12-31?",
    ],
    "keywords": ["days", "between", "from", "to", "date", "until"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("date.parse", [Inp("d1")], out="date1"),
        CodeStep("date.parse", [Inp("d2")], out="date2"),
        CodeStep("date.days_between", [Step("date1"), Step("date2")], out="days"),
    ],
    answer_ref="days",
)


def precondition(query):
    q = query.lower()
    return "day" in q and ("between" in q or ("from" in q and "to" in q))


def run(query):
    ok, data, via, _bind_calls = bind_days_between(query)
    if not ok:
        from reasoning.patterns import PatternResult
        return PatternResult(answer="(could not find two ISO dates in the query)",
                             trace=[{"step": "bind", "via": via, "ok": False},
                                    {"step": "summary", "model_calls": 0}])
    return run_plan(PLAN, data, shot=engine._no_model)

"""Percentage change from an old value to a new value — coded, zero model calls."""

from reasoning import engine
from reasoning.bind import bind_percent_change
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, explain, run_plan

META = {
    "id": "percentage_change",
    "query_type": "percentage_change",
    "description": "the percentage increase or decrease from one value to another",
    "example_queries": [
        "What is the percentage increase from 50 to 75?",
        "What is the percent decrease from 200 to 150?",
    ],
    "keywords": ["percentage", "percent", "increase", "decrease", "change", "from", "to"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("arith.percent_change", [Inp("old"), Inp("new")], out="pc"),
        CodeStep("arith.round_to", [Step("pc"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="{:g}%",
)


def precondition(query):
    q = query.lower()
    has_pct = "percent" in q or "%" in q
    has_change = any(w in q for w in ("increase", "decrease", "change"))
    return has_pct and has_change and "from" in q and "to" in q


def run(query):
    ok, data, via, _bind_calls = bind_percent_change(query)
    if not ok:
        return explain("could not parse 'from <old> to <new>' from the query")
    return run_plan(PLAN, data, shot=engine._no_model)

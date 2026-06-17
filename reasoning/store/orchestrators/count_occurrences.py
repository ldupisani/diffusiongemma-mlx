"""Count occurrences of a quoted needle in a quoted haystack — fully coded, zero model."""

from reasoning import engine
from reasoning.bind import bind_count_occurrences
from reasoning.compiled import CodeStep, Inp, Plan, run_plan

META = {
    "id": "count_occurrences",
    "query_type": "count_occurrences",
    "description": "count how many times a substring appears in a string",
    "example_queries": [
        "How many times does 'a' appear in 'banana'?",
        "How many times does 'ss' appear in 'mississippi'?",
    ],
    "keywords": ["many", "times", "appear", "occur", "count", "occurrences"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("text.count_occurrences", [Inp("haystack"), Inp("needle")], out="n"),
    ],
    answer_ref="n",
)


def precondition(query):
    q = query.lower()
    return ("times" in q or "occur" in q or "appear" in q) and query.count("'") >= 4


def run(query):
    ok, data, via, _bind_calls = bind_count_occurrences(query)
    if not ok:
        from reasoning.patterns import PatternResult
        return PatternResult(answer="(could not find two quoted strings in the query)",
                             trace=[{"step": "bind", "via": via, "ok": False},
                                    {"step": "summary", "model_calls": 0}])
    return run_plan(PLAN, data, shot=engine._no_model)

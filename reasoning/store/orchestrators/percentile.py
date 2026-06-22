"""The p-th percentile of a list of numbers — coded, zero model calls."""

from reasoning.bind import bind_percentile, validate_percentile
from reasoning.compiled import CodeStep, Inp, Plan, run_guarded

META = {
    "id": "percentile",
    "query_type": "percentile",
    "description": "the p-th percentile of a list of numbers",
    "example_queries": [
        # [0] is method-agnostic (a median: all percentile conventions agree) so eval-all
        # isn't flagged by the teacher flip-flopping between INC/EXC interpolation methods.
        "What is the 50th percentile of 10, 20, 30, 40, and 50?",
        "What is the 90th percentile of 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10?",
        "What is the 25th percentile of 5, 10, 15, and 20?",
    ],
    "keywords": ["percentile"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[CodeStep("stats.percentile", [Inp("nums"), Inp("p")], out="r")],
    answer_ref="r",
    answer_format="{:g}",
)


def precondition(query):
    return "percentile" in query.lower()


def run(query):
    return run_guarded(bind_percentile(query), PLAN, validate=validate_percentile,
                       parse_message="could not find a percentile and a list of numbers")

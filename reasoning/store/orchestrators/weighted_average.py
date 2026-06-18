"""Weighted average of values with matching weights — coded, zero model calls."""

from reasoning.bind import bind_weighted_average, validate_weighted_average
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "weighted_average",
    "query_type": "weighted_average",
    "description": "the weighted average of a list of values given matching weights",
    "example_queries": [
        "What is the weighted average of 80, 90, and 100 with weights 1, 2, and 3?",
        "Weighted mean of grades 70 and 95 with weights 2 and 3?",
    ],
    "keywords": ["weighted", "weights", "weight"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("stats.weighted_mean", [Inp("values"), Inp("weights")], out="w"),
        CodeStep("arith.round_to", [Step("w"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="{:g}",
)


def precondition(query):
    return "weighted" in query.lower()


def run(query):
    return run_guarded(bind_weighted_average(query), PLAN, validate=validate_weighted_average,
                       parse_message="could not find values and matching weights")

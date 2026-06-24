"""Annual straight-line depreciation = (cost − salvage) / years — coded, zero model calls."""

from reasoning.bind import bind_straight_line_dep, validate_depreciation
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "straight_line_depreciation",
    "query_type": "straight_line_depreciation",
    "description": "annual straight-line depreciation from cost, salvage value, and useful life",
    "example_queries": [
        "A machine costs $15000 with a salvage value of $3000 over 6 years. What is the annual straight-line depreciation?",
        "Asset worth $9000, salvage $1000, 4-year life — annual depreciation?",
    ],
    "keywords": ["depreciation", "salvage", "depreciate", "useful", "life"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("finance.straight_line_depreciation",
                 [Inp("cost"), Inp("salvage"), Inp("years")], out="d"),
        CodeStep("arith.round_to", [Step("d"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="${:.2f}",
)


def precondition(query):
    return "depreciation" in query.lower() or "depreciate" in query.lower()


def run(query):
    return run_guarded(bind_straight_line_dep(query), PLAN, validate=validate_depreciation,
                       parse_message="could not find cost, salvage, and life")

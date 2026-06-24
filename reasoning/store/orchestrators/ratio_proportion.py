"""Ratios / proportions / fractions — simplify, convert, solve, unit rate, scale. Zero model."""

from reasoning.bind import bind_ratio, bind_synthesis_geometry
from reasoning.compiled import dispatch

META = {
    "id": "ratio_proportion",
    "query_type": "ratio_proportion",
    "description": "simplify a fraction/ratio, convert fraction↔decimal↔percent, solve a "
                   "proportion, a unit rate, or scale a quantity",
    "example_queries": [
        "Simplify the fraction 18/24.",
        "What is 3/4 as a decimal?",
        "What is 0.75 as a fraction?",
        "What is 3/5 as a percentage?",
        "Solve the proportion 2:5 = x:20.",
        "Simplify the ratio 12:18.",
        "What is the unit rate if 150 miles takes 3 hours?",
        "If a recipe for 4 servings needs 2 cups of flour, how much for 6 servings?",
        "If 3 apples cost $2, how much do 12 apples cost?",
    ],
    "keywords": ["simplify", "fraction", "ratio", "decimal", "percentage", "proportion",
                 "unit rate", "servings", "cost"],
    "model_free": True,
}


def precondition(query):
    # defer an area×price ("$8 per square meter") query to the geometry-cost synthesis chain,
    # which its "cost + how much" item-scaling branch would otherwise mis-grab
    return bind_ratio(query)[0] and not bind_synthesis_geometry(query)[0]


def run(query):
    return dispatch(bind_ratio(query), parse_message="could not parse the ratio/fraction inputs")

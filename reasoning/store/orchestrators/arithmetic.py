"""Raw arithmetic — a single binary operation, percent-of, power, or square root.

Closes the gap where bare arithmetic ("17 percent of 240", "100 divided by 4") fell to the
numeric_consensus model fallback. Conservative: the bind claims a query only when exactly
one operation is unambiguously present, so compound expressions still defer to the model.
"""

from reasoning.bind import bind_arithmetic
from reasoning.compiled import dispatch

META = {
    "id": "arithmetic",
    "query_type": "arithmetic",
    "description": "a single arithmetic operation: add, subtract, multiply, divide, percent-of, "
                   "power, or square root",
    "example_queries": [
        "What is 17 percent of 240?",
        "What is 25 plus 17?",
        "What is 100 minus 30?",
        "What is 7 times 8?",
        "What is 100 divided by 4?",
        "What is 2 to the power of 10?",
        "What is the square root of 144?",
        "What is 15% of 200?",
    ],
    "keywords": ["plus", "minus", "times", "multiplied", "divided", "percent", "power",
                 "square", "root"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    return bind_arithmetic(query)[0]  # binds only one unambiguous operation


def run(query):
    return dispatch(bind_arithmetic(query), parse_message="could not parse a single arithmetic operation")

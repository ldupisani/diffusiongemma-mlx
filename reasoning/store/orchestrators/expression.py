"""Compound arithmetic — a full expression with ≥2 operations and order-of-operations.

Complements `arithmetic` (one operation): this claims multi-operation expressions
("15% of 200 plus 10", "(5+3)*2-4") that `arithmetic` and `bind_arithmetic` deliberately
bail on. Evaluated by skills/expr.py — pure code, 0 model calls.
"""

from reasoning.bind import bind_expression
from reasoning.compiled import dispatch

META = {
    "id": "expression",
    "query_type": "arithmetic_expression",
    "description": "evaluate a compound arithmetic expression with order of operations "
                   "(multiple operators, parentheses, percent-of, powers, square roots)",
    "example_queries": [
        "What is 15% of 200 plus 10?",
        "What is (5 + 3) * 2 - 4?",
        "What is 100 divided by 4 plus 7?",
        "Calculate 3 times 4 plus 5 times 6.",
        "What is 2 to the power of 10 minus 24?",
    ],
    "keywords": ["plus", "minus", "times", "divided", "percent", "power", "calculate"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,  # the parse is dispositive — a symbolic "(5+3)*2-4" has no keywords
}


def precondition(query):
    return bind_expression(query)[0]  # binds only compound (≥2-operation) expressions


def run(query):
    return dispatch(bind_expression(query),
                    parse_message="could not parse a compound arithmetic expression")

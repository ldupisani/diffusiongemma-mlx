"""Calculus — derivative, integral (indefinite/definite), limit (SymPy-backed, deterministic,
zero model calls).
"""

from reasoning import skills
from reasoning.bind import bind_calculus
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "calculus",
    "query_type": "calculus",
    "description": "symbolic calculus: derivative, indefinite/definite integral, or limit",
    "example_queries": [
        # [0] is a plain number ("1") — robust to the teacher's symbolic notation.
        "What is the limit of sin(x)/x as x approaches 0?",
        "What is the derivative of x^3 + 3x^2?",
        "What is the integral of x^2?",
        "What is the integral of x^2 from 0 to 1?",
    ],
    "keywords": ["derivative", "differentiate", "integral", "integrate", "limit",
                 "antiderivative"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_calculus(query)[0]


def run(query):
    ok, data, via, _ = bind_calculus(query)
    if not ok:
        return explain("could not parse the calculus expression")
    answer = skills.call(data["skill"], *data["args"])
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": answer},
                                {"step": "summary", "model_calls": 0}])

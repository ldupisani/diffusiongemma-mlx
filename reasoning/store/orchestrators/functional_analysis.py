"""Functional analysis (computable fragment) — vector p-norms, inner products,
Cauchy–Schwarz. Coded, 0 calls."""
from reasoning.bind import bind_funcanalysis
from reasoning.compiled import dispatch

META = {
    "id": "functional_analysis", "query_type": "functional_analysis",
    "description": "computable functional analysis: L_p vector norms (L1/L2/L∞), the Euclidean "
                   "inner product of two vectors, and the Cauchy–Schwarz inequality",
    "example_queries": [
        "What is the L1 norm of [3, 4]?",
        "What is the infinity norm of [3, -4]?",
        "What is the inner product of [1, 2, 3] and [4, 5, 6]?",
        "Check the Cauchy-Schwarz inequality for [1, 2] and [3, 4].",
    ],
    "keywords": ["l1 norm", "l2 norm", "infinity norm", "inner product", "cauchy-schwarz"],
    "model_free": True,
}


def precondition(query):
    return bind_funcanalysis(query)[0]


def run(query):
    return dispatch(bind_funcanalysis(query), parse_message="could not parse the functional-analysis query")

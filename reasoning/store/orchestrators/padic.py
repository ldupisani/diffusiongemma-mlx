"""p-adic numbers — valuation, absolute value, expansion, distance. Coded, 0 calls."""
from reasoning.bind import bind_padic
from reasoning.compiled import dispatch

META = {
    "id": "padic", "query_type": "padic",
    "description": "p-adic number operations for a prime p: p-adic valuation v_p, p-adic "
                   "absolute value |n|_p, p-adic expansion (base-p digits), and p-adic distance",
    "example_queries": [
        "What is the 2-adic valuation of 24?",
        "What is the 5-adic absolute value of 100?",
        "Give the 5-adic expansion of 100 to 3 digits.",
        "What is the 3-adic distance between 1 and 28?",
    ],
    "keywords": ["p-adic", "padic", "adic", "valuation"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_padic(query)[0]


def run(query):
    return dispatch(bind_padic(query), parse_message="could not parse the p-adic query")

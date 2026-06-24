"""Cardinal numbers — cardinal arithmetic (ℵ₀, 𝔠) and cardinalities of standard sets.
Coded, 0 calls."""
from reasoning.bind import bind_cardinal
from reasoning.compiled import dispatch

META = {
    "id": "cardinals", "query_type": "cardinals",
    "description": "cardinal arithmetic over finite cardinals, ℵ₀, and 𝔠 = 2^ℵ₀: sum, "
                   "product, exponentiation, comparison, and the cardinality of standard sets",
    "example_queries": [
        "What is aleph_0 + aleph_0?",
        "What is 2 to the power of aleph_0?",
        "What is the cardinality of the real numbers?",
        "Is aleph_0 less than the continuum?",
    ],
    "keywords": ["cardinal", "aleph", "continuum", "cardinality", "transfinite", "ℵ"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_cardinal(query)[0]


def run(query):
    return dispatch(bind_cardinal(query), parse_message="could not parse the cardinal query")

"""Ordinal numbers — Cantor normal form and ordinal comparison (non-commutative + and ·).
Coded, 0 calls."""
from reasoning.bind import bind_ordinal
from reasoning.compiled import dispatch

META = {
    "id": "ordinals", "query_type": "ordinals",
    "description": "ordinal arithmetic below ω^ω: Cantor normal form of an ω-expression "
                   "(non-commutative + and ·, e.g. 1+ω=ω but ω+1≠ω) and ordinal comparison",
    "example_queries": [
        "What is the Cantor normal form of 1+ω?",
        "Simplify the ordinal ω·2+ω.",
        "Is ω+1 greater than ω?",
        "Is 1+ω less than ω+1?",
    ],
    "keywords": ["ordinal", "omega", "ω", "cantor normal form", "transfinite"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_ordinal(query)[0]


def run(query):
    return dispatch(bind_ordinal(query), parse_message="could not parse the ordinal expression")

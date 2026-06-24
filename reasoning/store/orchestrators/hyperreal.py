"""Hyperreal numbers — infinitesimal/infinite arithmetic, standard part, classification,
comparison (ε infinitesimal, ω = 1/ε infinite). Coded, 0 calls."""
from reasoning.bind import bind_hyperreal
from reasoning.compiled import dispatch

META = {
    "id": "hyperreal", "query_type": "hyperreal",
    "description": "hyperreal arithmetic with an infinitesimal ε and infinite ω = 1/ε: "
                   "evaluate ε/ω expressions, the standard part st(x), classification "
                   "(infinitesimal/finite/infinite), and comparison",
    "example_queries": [
        "What is the standard part of 3+ε?",
        "Classify the hyperreal ε.",
        "What is ω·ε?",
        "Is ω greater than 5 in the hyperreals?",
    ],
    "keywords": ["hyperreal", "infinitesimal", "epsilon", "standard part", "ε"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_hyperreal(query)[0]


def run(query):
    return dispatch(bind_hyperreal(query), parse_message="could not parse the hyperreal query")

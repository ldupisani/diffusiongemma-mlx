"""Surreal numbers — value of {L | R}, simplest number between two, and birthday.
Coded, 0 calls."""
from reasoning.bind import bind_surreal
from reasoning.compiled import dispatch

META = {
    "id": "surreal", "query_type": "surreal",
    "description": "surreal numbers (dyadic fragment): the value of {L | R} (simplest number "
                   "between the sides), the simplest surreal between two numbers, and birthday",
    "example_queries": [
        "What is the value of the surreal {0 | 1}?",
        "What is the surreal {1 | 2}?",
        "What is the birthday of the surreal 3/4?",
        "What is the simplest surreal between 0 and 1?",
    ],
    "keywords": ["surreal", "birthday", "{l|r}"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_surreal(query)[0]


def run(query):
    return dispatch(bind_surreal(query), parse_message="could not parse the surreal query")

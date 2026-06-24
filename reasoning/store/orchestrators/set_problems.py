"""Two-set inclusion-exclusion word problems — union, neither, only-one. Zero model calls."""

from reasoning.bind import bind_sets
from reasoning.compiled import dispatch

META = {
    "id": "set_problems",
    "query_type": "set_problems",
    "description": "two-set Venn counts: union / at-least-one, neither, or only-one-set",
    "example_queries": [
        "In a group of 30 people, 18 like coffee and 15 like tea, and 8 like both. "
        "How many like coffee or tea?",
        "Out of 50 people, 30 own a dog and 25 own a cat, and 10 own both. How many own neither?",
        "In a class of 40, 25 play soccer and 20 play basketball, and 12 play both. "
        "How many play only soccer?",
        "If set A has 12 elements, set B has 8, and 3 are in both, how many are in A union B?",
    ],
    "keywords": ["both", "neither", "union", "either", "only", "group", "set"],
    "model_free": True,
}


def precondition(query):
    return bind_sets(query)[0]


def run(query):
    return dispatch(bind_sets(query), parse_message="could not parse the set counts")

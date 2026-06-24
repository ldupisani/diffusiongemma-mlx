"""Set theory — union, intersection, difference, complement, power set, Cartesian product,
subset/disjoint tests, cardinality on explicit {…} sets. Coded, zero model calls.
"""

from reasoning import skills
from reasoning.bind import bind_set_operations
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.skills.set_theory import format_set

META = {
    "id": "set_operations",
    "query_type": "set_theory",
    "description": "set algebra on explicit sets: union, intersection, difference, symmetric "
                   "difference, complement, power set, Cartesian product, subset/disjoint, cardinality",
    "example_queries": [
        "What is {1, 2, 3} union {2, 3, 4}?",
        "What is the intersection of {1, 2, 3} and {2, 3, 4}?",
        "What is the power set of {1, 2}?",
        "Is {1, 2} a subset of {1, 2, 3}?",
        "What is the cardinality of {1, 2, 3, 4}?",
        "What is the Cartesian product of {1, 2} and {3}?",
    ],
    "keywords": ["set", "union", "intersection", "subset", "power set", "cardinality",
                 "complement", "disjoint", "cartesian"],
    "model_free": True,
    "bind_gated": True,  # the {…} parse is dispositive (symbolic, few keywords)
}


def precondition(query):
    return bind_set_operations(query)[0]


def run(query):
    ok, data, via, _ = bind_set_operations(query)
    if not ok:
        return explain("could not parse the sets and the operation")
    result = skills.call(data["skill"], *data["args"])
    if data["kind"] == "set":
        answer = format_set(result)
    elif data["kind"] == "bool":
        answer = "Yes." if result else "No."
    else:
        answer = str(result)
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": str(result)},
                                {"step": "summary", "model_calls": 0}])

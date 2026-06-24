"""Group theory — cyclic groups Z_n (addition, element order, generators), permutations
(order, parity, cycle notation, composition), and the symmetric group order. Coded, zero
model calls.
"""

from reasoning import skills
from reasoning.bind import bind_group_theory
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "group_theory",
    "query_type": "group_theory",
    "description": "finite group theory: order of an element in Z_n, generators of Z_n, "
                   "permutation order/parity/cycles/composition, symmetric group order",
    "example_queries": [
        "What is the order of the element 2 in Z_6?",
        "What are the generators of Z_8?",
        "What is the order of the permutation (2, 3, 1)?",
        "Is the permutation (2, 1, 3) even or odd?",
        "Write the permutation (2, 3, 1) in cycle notation.",
        "What is the order of the symmetric group S_4?",
        "What is 3 + 4 in Z_5?",
    ],
    "keywords": ["permutation", "cyclic", "generators", "symmetric group", "order", "cycle"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_group_theory(query)[0]


def run(query):
    ok, data, via, _ = bind_group_theory(query)
    if not ok:
        return explain("could not parse the group, element, or permutation")
    result = skills.call(data["skill"], *data["args"])
    kind = data["kind"]
    if kind == "perm":
        answer = "(" + ", ".join(map(str, result)) + ")"
    elif kind == "genset":
        answer = "{" + ", ".join(map(str, result)) + "}"
    else:
        answer = str(result)
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": str(result)},
                                {"step": "summary", "model_calls": 0}])

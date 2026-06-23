"""Topology — finite spaces (X, T): is-a-topology, interior/closure/boundary, open/closed
tests, connectedness, and separation axioms (T1, Hausdorff). Coded, zero model calls.
"""

from reasoning import skills
from reasoning.bind import bind_topology
from reasoning.compiled import explain
from reasoning.patterns import PatternResult
from reasoning.skills.set_theory import format_set

META = {
    "id": "topology",
    "query_type": "topology",
    "description": "finite topology: whether a collection is a topology, interior/closure/"
                   "boundary of a set, open/closed tests, connectedness, separation axioms",
    "example_queries": [
        "Is {∅, {1}, {1, 2}} a topology on {1, 2}?",
        "What is the closure of {1} in the topology {∅, {1}, {1, 2}} on {1, 2}?",
        "What is the interior of {2} in the topology {∅, {1}, {1, 2}} on {1, 2}?",
        "Is the space {1, 2} with topology {∅, {1}, {1, 2}} connected?",
        "Is the topology {∅, {1}, {2}, {1, 2}} on {1, 2} Hausdorff?",
    ],
    "keywords": ["topology", "closure", "interior", "boundary", "connected", "hausdorff",
                 "open set", "closed set"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_topology(query)[0]


def run(query):
    ok, data, via, _ = bind_topology(query)
    if not ok:
        return explain("could not parse the space (X, T) and the operation")
    result = skills.call(data["skill"], *data["args"])
    answer = format_set(result) if data["kind"] == "set" else ("Yes." if result else "No.")
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": str(result)},
                                {"step": "summary", "model_calls": 0}])

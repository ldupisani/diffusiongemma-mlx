"""Discrete mathematics — graph degree/components/shortest-path, Catalan/Fibonacci/Stirling/
Bell numbers. Coded, zero model calls."""

from reasoning import skills
from reasoning.bind import bind_discrete
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "discrete_math",
    "query_type": "discrete_math",
    "description": "finite graphs (degree, connected components, shortest path from an edge "
                   "list) and combinatorial sequences (Catalan, Fibonacci, Stirling, Bell)",
    "example_queries": [
        "How many connected components are in the graph with edges (1,2), (2,3), (4,5)?",
        "What is the shortest path from 1 to 4 in the graph with edges (1,2), (2,3), (3,4)?",
        "What is the 4th Catalan number?",
        "What is the 10th Fibonacci number?",
        "What is the Stirling number of the second kind for 4 and 2?",
        "How many derangements of 4 items are there?",
        "How many ways to place 5 identical items into 3 bins?",
        "How many integer partitions of 5 are there?",
        "Is the graph with edges (1,2), (2,3), (3,4) bipartite?",
        "What is the minimum spanning tree weight of the graph with edges (1,2,4), (2,3,1), (3,4,2), (4,1,3)?",
    ],
    "keywords": ["graph", "edges", "catalan", "fibonacci", "stirling", "bell", "components",
                 "derangement", "multinomial", "stars and bars", "partition", "bipartite",
                 "eulerian", "spanning tree"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_discrete(query)[0]


def run(query):
    ok, data, via, _ = bind_discrete(query)
    if not ok:
        return explain("could not parse the discrete-math inputs")
    r = skills.call(data["skill"], *data["args"])
    return PatternResult(answer=str(r),
                         trace=[{"step": data["skill"], "via": "code", "output": str(r)},
                                {"step": "summary", "model_calls": 0}])

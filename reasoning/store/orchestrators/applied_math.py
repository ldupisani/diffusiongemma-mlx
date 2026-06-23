"""Applied mathematics — matrix determinant/inverse/eigenvalues, vector dot/cross/magnitude.
Coded (SymPy-backed), zero model calls."""

from reasoning import skills
from reasoning.bind import bind_linalg
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "applied_math",
    "query_type": "applied_math",
    "description": "linear algebra (matrix determinant, inverse, eigenvalues) and vector "
                   "operations (dot product, cross product, magnitude)",
    "example_queries": [
        "What is the determinant of [[1, 2], [3, 4]]?",
        "What is the inverse of [[1, 2], [3, 4]]?",
        "What are the eigenvalues of [[2, 0], [0, 3]]?",
        "What is the dot product of (1, 2, 3) and (4, 5, 6)?",
        "What is the cross product of (1, 0, 0) and (0, 1, 0)?",
        "What is the magnitude of the vector (3, 4)?",
    ],
    "keywords": ["determinant", "inverse", "eigenvalues", "dot product", "cross product",
                 "magnitude", "matrix", "vector"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_linalg(query)[0]


def run(query):
    ok, data, via, _ = bind_linalg(query)
    if not ok:
        return explain("could not parse the matrix or vectors")
    try:
        r = skills.call(data["skill"], *data["args"])
    except Exception as e:  # singular matrix, bad shape, …
        return explain(f"{data['skill']} could not run ({type(e).__name__})")
    if data["kind"] == "tuple":
        answer = "(" + ", ".join(str(x) for x in r) + ")"
    elif data["kind"] == "num":
        answer = str(round(r, 4))
    else:
        answer = str(r)
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": str(r)},
                                {"step": "summary", "model_calls": 0}])

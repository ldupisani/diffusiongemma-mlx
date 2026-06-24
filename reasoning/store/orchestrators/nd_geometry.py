"""N-dimensional shapes — n-ball volume, n-sphere surface, hypercube measures. Coded, 0 calls."""
from reasoning.bind import bind_ndim
from reasoning.compiled import dispatch

META = {
    "id": "nd_geometry", "query_type": "nd_geometry",
    "description": "measures of shapes in n dimensions: n-ball volume, n-sphere surface area, "
                   "and hypercube volume / diagonal / vertices / edges",
    "example_queries": [
        "What is the volume of a 4-dimensional ball of radius 2?",
        "What is the surface area of a 3-dimensional sphere of radius 1?",
        "What is the volume of a 4-dimensional cube of side 2?",
        "How many edges does a 4-dimensional hypercube have?",
    ],
    "keywords": ["dimensional", "hypercube", "n-ball", "n-sphere", "hypersphere", "dimensions"],
    "sub_orchestrators": [], "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_ndim(query)[0]


def run(query):
    return dispatch(bind_ndim(query), parse_message="could not parse the n-dimensional shape")

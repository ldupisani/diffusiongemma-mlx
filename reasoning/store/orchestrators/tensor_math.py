"""Tensor maths — rank, shape, trace, Frobenius norm, Kronecker & outer products. Coded."""
from reasoning.bind import bind_tensor
from reasoning.compiled import dispatch

META = {
    "id": "tensor_math", "query_type": "tensor_math",
    "description": "tensor operations: rank, shape, trace, Frobenius norm, Kronecker product, "
                   "and outer product",
    "example_queries": [
        "What is the rank of [[1, 2], [3, 4]]?",
        "What is the trace of [[1, 2], [3, 4]]?",
        "What is the Frobenius norm of [[3, 4]]?",
        "What is the Kronecker product of [[1, 2], [3, 4]] and [[0, 1], [1, 0]]?",
        "What is the outer product of [1, 2] and [3, 4]?",
    ],
    "keywords": ["tensor", "rank", "trace", "frobenius", "kronecker", "outer product", "shape"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_tensor(query)[0]


def run(query):
    return dispatch(bind_tensor(query), parse_message="could not parse the tensor")

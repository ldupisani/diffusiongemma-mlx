"""Hypercomplex (quaternion) numbers — conjugate, norm, inverse, Hamilton product, sum.
Coded, 0 calls."""
from reasoning.bind import bind_hypercomplex
from reasoning.compiled import dispatch

META = {
    "id": "hypercomplex", "query_type": "hypercomplex",
    "description": "quaternion operations (a+bi+cj+dk): conjugate, norm, and inverse "
                   "(the Hamilton product and sum exist as skills, computed by code)",
    "example_queries": [
        "What is the conjugate of 1+2i+3j+4k?",
        "What is the norm of 1+2i+3j+4k?",
        "What is the inverse of the quaternion i?",
        "What is the magnitude of the quaternion 2+3i+6k?",
    ],
    "keywords": ["quaternion", "hypercomplex", "octonion", "hamilton"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_hypercomplex(query)[0]


def run(query):
    return dispatch(bind_hypercomplex(query), parse_message="could not parse the quaternion query")

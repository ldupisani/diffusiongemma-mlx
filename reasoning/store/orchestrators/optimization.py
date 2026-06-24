"""Optimization — unconstrained minima/maxima of a function, and gradient descent.
Coded, 0 calls."""
from reasoning.bind import bind_optimize
from reasoning.compiled import dispatch

META = {
    "id": "optimization", "query_type": "optimization",
    "description": "unconstrained optimization: the minimum/maximum of a function of x (or the "
                   "2-D minimum of f(x,y)), and gradient descent on f(x)",
    "example_queries": [
        "Minimize x^2 - 4x + 7.",
        "Maximize -x^2 + 4x - 1.",
        "Minimize x^2 + y^2 - 2x - 4y.",
        "Run gradient descent on x^2 - 4x + 7 from 0 with learning rate 0.1 for 100 steps.",
    ],
    "keywords": ["minimize", "maximize", "optimization", "gradient descent", "minimum of"],
    "model_free": True,
}


def precondition(query):
    return bind_optimize(query)[0]


def run(query):
    return dispatch(bind_optimize(query), parse_message="could not parse the optimization query")

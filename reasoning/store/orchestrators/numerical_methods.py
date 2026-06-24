"""Numerical methods — Newton/bisection root finding, Simpson/trapezoidal quadrature,
fixed-point iteration. Coded, 0 calls."""
from reasoning.bind import bind_numerical
from reasoning.compiled import dispatch

META = {
    "id": "numerical_methods", "query_type": "numerical_methods",
    "description": "numerical methods on a function of x: Newton's method and bisection roots, "
                   "Simpson's-rule and trapezoidal integration, and fixed-point iteration",
    "example_queries": [
        "Use Newton's method on x^2 - 2 from 1.",
        "Find a root of x^2 - 2 by bisection on [1, 2].",
        "Estimate the integral of x^2 from 0 to 1 with Simpson's rule using 2 intervals.",
        "Apply fixed-point iteration to cos(x) from 1.",
    ],
    "keywords": ["newton's method", "bisection", "simpson", "trapezoidal", "fixed-point"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_numerical(query)[0]


def run(query):
    return dispatch(bind_numerical(query), parse_message="could not parse the numerical-method query")

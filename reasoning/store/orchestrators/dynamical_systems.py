"""Dynamical systems & chaos — logistic map iteration, fixed points, Lyapunov exponent,
Mandelbrot membership. Coded, 0 calls."""
from reasoning.bind import bind_dynamics
from reasoning.compiled import dispatch

META = {
    "id": "dynamical_systems", "query_type": "dynamical_systems",
    "description": "dynamical systems & chaos: iterate the logistic map, its fixed points, the "
                   "Lyapunov exponent, and Mandelbrot-set membership of a point c",
    "example_queries": [
        "Iterate the logistic map with r=3.2 and x0=0.5 for 2 iterations.",
        "What are the fixed points of the logistic map with r=3?",
        "What is the Lyapunov exponent of the logistic map at r=4 with x0=0.2 over 2000 iterations?",
        "Is c = 1 in the Mandelbrot set?",
    ],
    "keywords": ["logistic map", "lyapunov", "mandelbrot", "chaos", "dynamical system"],
    "model_free": True,
}


def precondition(query):
    return bind_dynamics(query)[0]


def run(query):
    return dispatch(bind_dynamics(query), parse_message="could not parse the dynamical-systems query")

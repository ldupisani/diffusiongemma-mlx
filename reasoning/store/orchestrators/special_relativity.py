"""Special relativity — Lorentz factor, time dilation, length contraction, relativistic kinetic energy.
All coded, 0 calls."""

from reasoning.bind import bind_special_relativity
from reasoning.compiled import dispatch

META = {
    "id": "special_relativity",
    "query_type": "special_relativity",
    "description": "special relativity: Lorentz factor (gamma), time dilation, length contraction, "
                   "and relativistic kinetic energy for speeds given as fractions of light speed",
    "example_queries": [
        "What is the Lorentz factor gamma for a speed of 0.8c?",
        "Calculate the time dilation for 1 second at 0.6c.",
        "What is the length contraction of 10 meters at 0.8c?",
        "What is the relativistic kinetic energy of 1 milligram at 0.9c?",
    ],
    "keywords": ["lorentz", "gamma", "dilation", "contraction", "relativistic", "speed of light"],
    "model_free": True,
}


def precondition(query):
    return bind_special_relativity(query)[0]


def run(query):
    return dispatch(bind_special_relativity(query), parse_message="could not parse the special relativity query")

"""Differential geometry — curvature, radius of curvature, arc length of y=f(x). Coded, 0 calls."""
from reasoning.bind import bind_diffgeo
from reasoning.compiled import dispatch

META = {
    "id": "differential_geometry", "query_type": "differential_geometry",
    "description": "differential geometry of a plane curve y=f(x): curvature κ at a point, "
                   "radius of curvature, and arc length over an interval",
    "example_queries": [
        "What is the curvature of x^2 at x=0?",
        "What is the radius of curvature of x^2 at x=0?",
        "What is the arc length of x^2 from 0 to 1?",
        "What is the curvature of sqrt(1-x^2) at x=0?",
    ],
    "keywords": ["curvature", "radius of curvature", "arc length", "differential geometry"],
    "model_free": True,
}


def precondition(query):
    return bind_diffgeo(query)[0]


def run(query):
    return dispatch(bind_diffgeo(query), parse_message="could not parse the differential-geometry query")

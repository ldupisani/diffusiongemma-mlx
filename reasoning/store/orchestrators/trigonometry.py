"""Trigonometry / angles — sin/cos/tan of degrees and degree↔radian. Coded, zero model."""

from reasoning.bind import bind_geospatial, bind_trig
from reasoning.compiled import dispatch

META = {
    "id": "trigonometry",
    "query_type": "trigonometry",
    "description": "sine/cosine/tangent of an angle in degrees, or degree↔radian conversion",
    "example_queries": [
        "What is the sine of 30 degrees?",
        "What is the cosine of 60 degrees?",
        "What is the tangent of 45 degrees?",
        "Convert 180 degrees to radians.",
        "Convert 90 degrees to radians.",
        "Convert pi radians to degrees.",
    ],
    "keywords": ["sine", "cosine", "tangent", "degrees", "radians", "angle"],
    "model_free": True,
}


def precondition(query):
    return bind_trig(query)[0] and not bind_geospatial(query)[0]


def run(query):
    return dispatch(bind_trig(query), parse_message="could not parse the angle or operation")

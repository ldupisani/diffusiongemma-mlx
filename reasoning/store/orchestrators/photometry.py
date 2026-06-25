"""Photometry and illumination engineering — lux, lumens, candela, illuminance.
Coded, 0 model calls."""

from reasoning.bind import bind_photometry
from reasoning.compiled import dispatch

META = {
    "id": "photometry",
    "query_type": "photometry",
    "description": "photometry and illumination engineering: illuminance from luminous flux and area, "
                   "illuminance from luminous intensity and distance (inverse-square law), luminous "
                   "efficacy, luminous intensity from flux and solid angle, and inverse-square "
                   "illuminance ratios",
    "example_queries": [
        "What is the illuminance from 800 lumens over 5 square meters?",
        "What is the illuminance from a 100 candela source at 2 meters away?",
        "What is the luminous efficacy of an 800 lumen lamp using 10 watts?",
        "What is the luminous intensity for 1000 lumens distributed over 5 steradians?",
        "The illuminance is 100 lux at an original distance of 2 meters. What is it at 5 meters?",
    ],
    "keywords": ["lux", "lumens", "candela", "illuminance", "luminous", "steradian", "efficacy"],
    "model_free": True,
}

def precondition(query):
    return bind_photometry(query)[0]

def run(query):
    return dispatch(bind_photometry(query), parse_message="could not parse the photometry query")
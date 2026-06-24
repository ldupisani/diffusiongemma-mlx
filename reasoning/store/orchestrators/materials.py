"""Material science — stress, Young's modulus, thermal expansion, atomic packing factor.
Coded, zero model calls."""

from reasoning.bind import bind_materials
from reasoning.compiled import dispatch

META = {
    "id": "materials",
    "query_type": "materials",
    "description": "mechanical stress, Young's modulus, thermal expansion, and the atomic "
                   "packing factor of a cubic crystal structure",
    "example_queries": [
        "What is the atomic packing factor of an FCC structure?",
        "What is the stress of a 100 N force over an area of 0.5 square meters?",
        "What is Young's modulus if the stress is 200 Pa and the strain is 0.001?",
        "What is the thermal expansion of a 10 m rod with coefficient 0.000012 heated 50 degrees?",
    ],
    "keywords": ["stress", "strain", "young", "modulus", "packing factor", "thermal expansion"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_materials(query)[0]


def run(query):
    return dispatch(bind_materials(query), parse_message="could not parse the material inputs")

"""Geometry / measurement — coded, zero model calls.

One orchestrator dispatches every shape+measurement: the bind detects the formula and the
labeled dimensions (radius / length / width / base / height / side / legs), run() applies
the matching skill. Covers circle, rectangle, triangle, square, cube, sphere, cylinder, and
the Pythagorean hypotenuse.
"""

from reasoning import engine
from reasoning.bind import (bind_geometry, bind_geometry_multi, clarify_geometry,
                           validate_geometry)
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, clarify, explain, run_guarded

META = {
    "id": "geometry",
    "query_type": "geometry",
    "description": "area, perimeter, circumference, volume, surface area, or hypotenuse of a shape",
    "example_queries": [
        "What is the area of a circle with radius 5?",
        "What is the circumference of a circle with radius 7?",
        "What is the area of a rectangle with length 4 and width 6?",
        "What is the perimeter of a rectangle with length 8 and width 3?",
        "What is the area of a triangle with base 10 and height 5?",
        "What is the volume of a cube with side 3?",
        "What is the volume of a sphere with radius 4?",
        "What is the volume of a cylinder with radius 3 and height 10?",
        "What is the hypotenuse of a right triangle with legs 3 and 4?",
        "What is the surface area of a sphere with radius 6?",
        "What is the diagonal of a rectangle with length 3 and width 4?",
    ],
    "keywords": ["area", "perimeter", "circumference", "volume", "surface", "diagonal",
                 "hypotenuse", "circle", "rectangle", "triangle", "square", "cube", "sphere",
                 "cylinder", "radius"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    # bindability, not just keywords: "square meters" in a pressure query must NOT make this
    # claim the query (it detects square_area but can't find a side → does not bind).
    # Defer a multi-property request ("area AND perimeter") to geometry_multi.
    return bind_geometry(query)[0] and not bind_geometry_multi(query)[0]


clarify_probe = clarify_geometry  # routed only when nothing can ANSWER (a lower router tier)


def run(query):
    ok, data, via, _ = bind_geometry(query)
    if not ok:
        missing = clarify_geometry(query)
        if missing:
            return clarify(missing)
        return explain("could not parse the shape and its dimensions")
    plan = Plan(
        steps=[
            CodeStep(data["skill"], [Inp(d) for d in data["dims"]], out="v"),
            CodeStep("arith.round_to", [Step("v"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        answer_format="{:g}",
    )
    return run_guarded((ok, data, via, 0), plan, validate=validate_geometry, shot=engine._no_model)

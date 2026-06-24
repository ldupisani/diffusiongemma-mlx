"""Coordinate geometry — distance, midpoint, slope between two points. Coded, zero model."""

from reasoning.bind import bind_coord, bind_geospatial
from reasoning.compiled import dispatch

META = {
    "id": "coordinate_geometry",
    "query_type": "coordinate_geometry",
    "description": "distance, midpoint, or slope between two points (x1,y1) and (x2,y2)",
    "example_queries": [
        "What is the distance between the points (0,0) and (3,4)?",
        "What is the midpoint between (2,4) and (6,8)?",
        "What is the slope of the line through (1,2) and (3,6)?",
        "What is the distance between (-2,-3) and (1,1)?",
    ],
    "keywords": ["distance", "midpoint", "slope", "points", "between", "line"],
    "model_free": True,
}


def precondition(query):
    return bind_coord(query)[0] and not bind_geospatial(query)[0]


def run(query):
    return dispatch(bind_coord(query), parse_message="could not find two points")

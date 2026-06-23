"""Geospatial — great-circle distance, bearing, DMS↔decimal. Coded, zero model calls."""

from reasoning.bind import bind_geospatial, bind_synthesis_geo
from reasoning.compiled import dispatch

META = {
    "id": "geospatial",
    "query_type": "geospatial",
    "description": "great-circle (haversine) distance or bearing between two lat/long points, "
                   "and degrees-minutes-seconds ↔ decimal-degree conversion",
    "example_queries": [
        "What is the distance in km between (40.7128, -74.0060) and (51.5074, -0.1278)?",
        "What is the bearing from (0, 0) to (0, 1) on Earth?",
        "Convert 40 degrees 30 minutes 0 seconds to decimal degrees.",
        "Convert 40.5 degrees to degrees minutes seconds.",
    ],
    "keywords": ["latitude", "longitude", "bearing", "great circle", "coordinates", "gps"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_geospatial(query)[0] and not bind_synthesis_geo(query)[0]


def run(query):
    return dispatch(bind_geospatial(query), parse_message="could not parse the geospatial inputs")

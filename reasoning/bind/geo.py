"""bind layer — geospatial (lat/long distance & bearing, DMS↔decimal)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind import entities as _E

# require a geo cue so a plain "(x, y)" pair stays with coordinate_geometry (euclidean)
_GEO_CUES = ("latitude", "longitude", "gps", "great circle", "great-circle", "haversine",
             "coordinates", "on earth", "°", " km", "kilomet", "mile", "bearing")
_DMS = re.compile(r"(-?\d+)\s*(?:degrees?|°)\s*,?\s*(\d+)\s*(?:minutes?|')\s*,?\s*"
                  r"(\d+)\s*(?:seconds?|\")", re.IGNORECASE)


def bind_geospatial(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    pts = _E.pairs(query)
    if len(pts) >= 2 and any(c in q for c in _GEO_CUES):
        a = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
        dims = ["la1", "lo1", "la2", "lo2"]
        if any(w in q for w in ("bearing", "heading", "direction", "azimuth")):
            return ok("geo.bearing", dims, a, suffix="°", places=2)
        if any(w in q for w in ("distance", "how far", "km", "kilomet")):
            return ok("geo.haversine", dims, a, suffix=" km", places=2)
    m = _DMS.search(query)
    if m and "decimal" in q:
        return ok("geo.dms_to_decimal", ["d", "m", "s"],
                  [int(m[1]), int(m[2]), int(m[3])], places=4)
    if any(c in q for c in ("dms", "degrees minutes seconds", "degrees, minutes",
                            "minutes, seconds", "degrees-minutes-seconds")):
        dm = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:degrees|°)", query, re.IGNORECASE)
        if dm:
            return ok("geo.decimal_to_dms", ["dd"], [float(dm[1])], round=False)
    return False, None, "code", 0


# geo synthesis: two lat/long points → travel cost ($/km or $/mile) or flight time (km/h or
# mph), chaining haversine into a finance / motion op. haversine yields KM; a per-mile price or
# mph speed routes through a km→mi conversion so the units always match. Requires a price/speed
# so plain "distance between" stays with the geospatial orchestrator.
_PER_KM = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:km|kilomet\w*)", re.IGNORECASE)
_PER_MI = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:miles?|mi)\b", re.IGNORECASE)
_KMH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km/?h|kph|kilomet\w*\s*per\s*hour)", re.IGNORECASE)
_MPH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mph|mi/?h|miles?\s*per\s*hour)", re.IGNORECASE)


def bind_synthesis_geo(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    pts = _E.pairs(query)
    if len(pts) < 2:
        return False, None, "code", 0
    knowns = {"la1": pts[0][0], "lo1": pts[0][1], "la2": pts[1][0], "lo2": pts[1][1]}
    price_km, price_mi = _PER_KM.search(query), _PER_MI.search(query)
    speed_km, speed_mi = _KMH.search(query), _MPH.search(query)
    flight = any(w in q for w in ("how long", "flight time", "travel time", "how many hours"))
    if "cost" in q and (price_km or price_mi):
        if price_km:
            knowns["price_per_km"] = float(price_km[1])
        else:
            knowns["price_per_mile"] = float(price_mi[1])
        goal, suffix, prefix = "travel_cost", "", "$"
    elif flight and (speed_km or speed_mi):
        if speed_km:
            knowns["speed_kmh"] = float(speed_km[1])
        else:
            knowns["speed_mph"] = float(speed_mi[1])
        goal, suffix, prefix = "flight_hours", " hours", ""
    else:
        return False, None, "code", 0
    from reasoning.synthesize import CROSS_OPS, GEO_OPS, synthesize
    seq = synthesize(set(knowns), goal, GEO_OPS + CROSS_OPS, unique=True)
    if seq is None:
        return False, None, "code", 0
    return True, {"knowns": knowns, "goal": goal, "prefix": prefix, "suffix": suffix}, "code", 0

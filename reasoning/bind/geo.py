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

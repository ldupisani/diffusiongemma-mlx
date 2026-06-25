"""bind layer — travel between two NAMED places. Resolves both names via the geography reference
(so multi-word names like 'New York' / 'Ho Chi Minh City' work), then targets great-circle
distance, travel time (at a speed), or cost (per distance). Conservative: needs a travel/distance
phrase AND two known places, so it never steals a single-place lookup or a coordinate query."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_GATE = re.compile(r"\b(?:how far|distance|travel|drive|driving|fly|flight|flying|from)\b", re.IGNORECASE)
_MI_PER_KM = 1.60934


def _places(query: str) -> list[str]:
    """Known reference places named in the query, in order of appearance (longest match wins an
    overlap, so 'New York' beats a shorter substring)."""
    from reasoning import reference
    q = query.lower()
    hits = []
    for low, canon in reference.place_names().items():
        m = re.search(r"\b" + re.escape(low) + r"\b", q)
        if m:
            hits.append((m.start(), -len(low), canon))
    hits.sort()
    seen, out = set(), []
    for _start, _neg_len, canon in hits:
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def bind_travel(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(goal, knowns, prefix="", suffix=""):
        return True, {"knowns": knowns, "goal": goal, "prefix": prefix, "suffix": suffix,
                      "unique": True}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    places = _places(query)
    if len(places) < 2:
        return False, None, "code", 0
    from reasoning import reference
    ca, cb = reference.coordinates(places[0]), reference.coordinates(places[1])
    if not ca or not cb:
        return False, None, "code", 0
    knowns = {"la1": ca[0], "lo1": ca[1], "la2": cb[0], "lo2": cb[1]}

    rate = re.search(r"\$?(\d+(?:\.\d+)?)\s*(?:per|/|a)\s*(km|kilomet(?:er|re)s?|miles?)", q)
    speed = re.search(r"(\d+(?:\.\d+)?)\s*(mph|miles? per hour|km/?h|kph|kilomet(?:er|re)s? per hour)", q)
    if rate:
        r = float(rate.group(1))
        if rate.group(2).startswith("mile"):
            r /= _MI_PER_KM                       # $/mile → $/km
        knowns["price_per_km"] = round(r, 6)
        return ok("travel_cost", knowns, prefix="$")
    if speed:
        s = float(speed.group(1))
        if speed.group(2).startswith(("mph", "mile")):
            s *= _MI_PER_KM                        # mph → km/h
        knowns["speed_kmh"] = round(s, 6)
        return ok("flight_hours", knowns, suffix=" hours")
    return ok("geo_distance", knowns, suffix=" km")   # default: the distance itself

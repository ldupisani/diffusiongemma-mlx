"""bind layer — geography reference lookups. Gated on a geography keyword AND a known place
name (matched against the reference data, so multi-word names like 'United States' / 'New Delhi'
work without fragile parsing). Plain questions with no reference place are never claimed."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_GATE = re.compile(r"population|coordinates|latitude|longitude|where is|\bcapital\b|continent|"
                   r"\barea\b", re.IGNORECASE)


def _find_place(query: str):
    from reasoning import reference
    q = query.lower()
    best = None
    for low, canon in reference.place_names().items():
        if re.search(r"\b" + re.escape(low) + r"\b", q) and (best is None or len(low) > len(best[0])):
            best = (low, canon)
    return best[1] if best else None


def bind_geography(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(op, place, **extra):
        return True, {"op": op, "place": place, **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    place = _find_place(query)
    if not place:
        return False, None, "code", 0

    if "capital" in q:
        return ok("capital", place)
    if "continent" in q:
        return ok("continent", place)
    if "area" in q:
        return ok("area", place)
    if "coordinate" in q or "latitude" in q or "longitude" in q or "where is" in q:
        return ok("coordinates", place)
    if "population" in q:
        ym = re.search(r"\b(19\d{2}|20\d{2})\b", q)
        return ok("population", place, year=int(ym[1]) if ym else None)
    return False, None, "code", 0

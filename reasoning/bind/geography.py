"""bind layer — geography reference lookups. Gated on a geography keyword AND a known place
name (matched against the reference data, so multi-word names like 'United States' / 'New Delhi'
work without fragile parsing). Plain questions with no reference place are never claimed."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_GATE = re.compile(r"population|coordinates|latitude|longitude|where is|\bcapital\b|continent|"
                   r"\barea\b|\bgrow|\bgrew\b|\bgrown\b|growth", re.IGNORECASE)

# a population CHANGE between two years — even without the word "population" ("how much did Lagos
# grow from 1990 to 2020"). Implied-population unless another metric is named (GDP, area, …).
_CHANGE_VERB = re.compile(r"\b(grow|grew|grown|growth|increas|decreas|declin|change|chang|rise|"
                          r"rose|fall|fell|gain|drop|differ)\w*", re.IGNORECASE)
_OTHER_METRIC = re.compile(r"\bgdp\b|econom|revenue|sales|income|profit|\barea\b|temperature|"
                           r"rainfall|gdp", re.IGNORECASE)


def _find_place(query: str):
    from reasoning import reference
    q = query.lower()
    best = None
    for low, canon in reference.place_names().items():
        if re.search(r"\b" + re.escape(low) + r"\b", q) and (best is None or len(low) > len(best[0])):
            best = (low, canon)
    return best[1] if best else None


_SET_NOUN = (r"cit(?:y|ies)|towns?|capitals?|countr(?:y|ies)|villages?|rivers?|mountains?|"
             r"lakes?|states?|provinces?|islands?|regions?")


def _set_intent(q: str) -> bool:
    """A request for MANY data points: 'all cities in France', 'largest cities in Germany',
    'list the countries in Africa', or 'populations of France, Germany and Italy'."""
    if re.search(r"\b(all|list|every|largest|biggest|major|top|name the|name all)\b", q) \
            and re.search(rf"\b(?:{_SET_NOUN})\b", q):
        return True
    return bool(re.search(r"\b(population|coordinate|capital|area)s?\b", q)
                and re.search(r",.*\band\b", q))           # explicit comma+and list


def _slug(query: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return s[:40].rstrip("_") or "retrieved_set"


def _all_places(query: str) -> list[str]:
    """Every known reference place named in the query, in order of appearance."""
    from reasoning import reference
    q = query.lower()
    hits = []
    for low, canon in reference.place_names().items():
        m = re.search(r"\b" + re.escape(low) + r"\b", q)
        if m:
            hits.append((m.start(), -len(low), canon))
    hits.sort()
    seen, out = set(), []
    for _s, _n, canon in hits:
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


_COMPARE = re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b|compared (?:to|with)", re.IGNORECASE)


def bind_geography(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(op, place=None, **extra):
        return True, {"op": op, "place": place, **extra}, "code", 0

    from reasoning import reference
    # batch retrieval (model-sourced dataset) — only when retrieval is enabled
    if reference.retrieval_enabled() and _set_intent(q):
        return ok("set", request=query.strip(), save_as=_slug(query))

    if not _GATE.search(query):
        return False, None, "code", 0
    place = _find_place(query)
    if not place:
        # only claim an UNKNOWN place when model retrieval is enabled (else let it fall through);
        # require a proper-noun candidate after of/in/for/is so non-geography queries are ignored.
        if reference.retrieval_enabled():
            m = re.search(r"\b(?:of|in|for|is)\s+(?:the\s+)?"
                          r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*)", query)
            place = m[1].strip() if m else None
        if not place:
            return False, None, "code", 0

    years = [int(y) for y in re.findall(r"\b(\d{4})\b", q)]
    # compare two places' population growth ("compare the growth of Lagos and Nigeria")
    if _COMPARE.search(q) and _CHANGE_VERB.search(q) and not _OTHER_METRIC.search(q):
        ps = _all_places(query)
        if len(ps) >= 2:
            return ok("compare", places=ps[:2],
                      year_from=years[0] if len(years) >= 2 else None,
                      year_to=years[1] if len(years) >= 2 else None)
    # population change between two years (with or without the word "population"), unless another
    # metric is named (so "China's GDP grew …" isn't answered as population)
    if _CHANGE_VERB.search(q) and len(years) >= 2 and not _OTHER_METRIC.search(q):
        return ok("change", place, year_from=years[0], year_to=years[1])

    if "capital" in q:
        return ok("capital", place)
    if "continent" in q:
        return ok("continent", place)
    if "area" in q:
        return ok("area", place)
    if "coordinate" in q or "latitude" in q or "longitude" in q or "where is" in q:
        return ok("coordinates", place)
    if "population" in q:
        return ok("population", place, year=years[0] if years else None)
    return False, None, "code", 0

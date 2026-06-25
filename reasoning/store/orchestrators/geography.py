"""Geography reference lookups — population (by year), coordinates, capital, continent, area —
over the bundled reference data. Coded, 0 model calls."""
from reasoning import reference
from reasoning.bind import bind_geography
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "geography", "query_type": "geography",
    "description": "look up bundled geography reference data: a country/city population (by "
                   "year), coordinates (lat/lon), a country's capital, continent, or area",
    "example_queries": [
        "What is the population of France in 2020?",
        "What is the population of Tokyo?",
        "What are the coordinates of Paris?",
        "What is the capital of Japan?",
        "What continent is Nigeria in?",
        "What is the area of Brazil?",
    ],
    "keywords": ["population of", "coordinates of", "capital of", "continent", "area of",
                 "latitude", "longitude"],
    # bind-gated: a conservative place-gated bind, dispositive across phrasings.
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_geography(query)[0]


def _lookup(op, place, year):
    """(display value, year-fidelity note) from the curated set, or (None, "")."""
    if op == "population":
        det = reference.population_detail(place, year)
        if not det:
            return None, ""
        note = (" (city population; the reference has no year-specific data for cities)"
                if year is not None and not det["year_applied"] else "")
        return str(det["value"]), note
    if op == "coordinates":
        c = reference.coordinates(place)
        return (f"{c[0]}, {c[1]}" if c else None), ""
    if op == "area":
        v = reference.country_field(place, "area_km2")
        return (str(v) if v is not None else None), ""
    return reference.country_field(place, op), ""  # capital / continent


def _run_set(d):
    """Batch retrieval: a list/open-ended request → a model-sourced dataset handle."""
    n = reference.retrieve_set(d["request"], d["save_as"])
    if not n:
        return explain("could not retrieve the requested set")
    from reasoning import datasets
    info = datasets.describe(d["save_as"])
    return PatternResult(
        answer=f"{info}\nretrieved {n} rows via Claude Sonnet → dataset '{d['save_as']}' "
               f"(model-sourced, not curated; verify before relying on it)",
        trace=[{"step": "reference.retrieve_set", "via": "model", "request": d["request"]},
               {"step": "summary", "model_calls": 1}])


def run(query):
    ok, d, _, _ = bind_geography(query)
    if not ok:
        return explain("could not parse the geography query")
    if d["op"] == "set":
        return _run_set(d)
    op, place, year = d["op"], d["place"], d.get("year")
    val, note = _lookup(op, place, year)
    model_calls, retrieved, cached = 0, False, False
    # Not in the curated set? When retrieval is enabled (opt-in via DIFG_REFERENCE_RETRIEVAL):
    # reuse a previously-retrieved value (0 calls), else retrieve via Claude Sonnet (1 call) and
    # persist it. Either way the result is validated and flagged as model-sourced, never curated.
    if val is None and reference.retrieval_enabled():
        val = reference.cached_value(op, place, year)
        if val is not None:
            retrieved, cached = True, True
        else:
            val = reference.retrieve(op, place, year)
            retrieved, model_calls = val is not None, (1 if val is not None else 0)
        note = ""   # a retrieved value is year-specific (the year is in the prompt)
    if val is None:
        return explain(f"no reference data for '{place}' ({op})")
    display = (f"{val} km^2" if op == "area" else val) + note
    if retrieved:
        src = "previously retrieved" if cached else "retrieved via Claude Sonnet"
        display += f" — {src} (not in the curated reference set; verify)"
    return PatternResult(answer=display,
                         trace=[{"step": f"reference.{op}", "place": place,
                                 "via": "code" if not retrieved else ("cache" if cached else "model")},
                                {"step": "summary", "model_calls": model_calls}])

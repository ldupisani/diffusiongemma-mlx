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
    """Curated value (display-ready) or None. Same value shapes as reference.retrieve."""
    if op == "population":
        v = reference.population(place, year)
        return str(v) if v is not None else None
    if op == "coordinates":
        c = reference.coordinates(place)
        return f"{c[0]}, {c[1]}" if c else None
    if op == "area":
        v = reference.country_field(place, "area_km2")
        return str(v) if v is not None else None
    return reference.country_field(place, op)  # capital / continent


def run(query):
    ok, d, _, _ = bind_geography(query)
    if not ok:
        return explain("could not parse the geography query")
    op, place, year = d["op"], d["place"], d.get("year")
    val, model_calls, retrieved = _lookup(op, place, year), 0, False
    # Fallback: a data point not in the curated set is RETRIEVED via Claude Sonnet (opt-in via
    # DIFG_REFERENCE_RETRIEVAL), validated for shape/range, and flagged as model-sourced.
    if val is None and reference.retrieval_enabled():
        val = reference.retrieve(op, place, year)
        retrieved, model_calls = val is not None, (1 if val is not None else 0)
    if val is None:
        return explain(f"no reference data for '{place}' ({op})")
    display = f"{val} km^2" if op == "area" else val
    if retrieved:
        display += " — retrieved via Claude Sonnet (not in the curated reference set; verify)"
    return PatternResult(answer=display,
                         trace=[{"step": f"reference.{op}", "place": place,
                                 "via": "model" if retrieved else "code"},
                                {"step": "summary", "model_calls": model_calls}])

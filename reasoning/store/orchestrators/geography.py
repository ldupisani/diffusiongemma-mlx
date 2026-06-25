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
        if det and (det["year_applied"] or year is None):
            return str(det["value"]), ""        # exact curated year, or latest when no year asked
        return None, ""                          # missing year → run() tries retrieval/projection
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
    model_calls, via = 0, "code"
    # A year/place not in the curated set, tried cheapest-meaningful-first:
    #  (1) retrieval (opt-in DIFG_REFERENCE_RETRIEVAL) — a real/cached value, flagged model-sourced;
    #  (2) for population, a deterministic TREND projection from the by-year series (0 calls),
    #      flagged an estimate; (3) the latest curated figure with a year note.
    if val is None and reference.retrieval_enabled():
        cv = reference.cached_value(op, place, year)
        if cv is not None:
            val, via, note = cv, "cache", " — previously retrieved (not in the curated reference set; verify)"
        else:
            rv = reference.retrieve(op, place, year)
            if rv is not None:
                val, via, model_calls = rv, "model", 1
                note = " — retrieved via Claude Sonnet (not in the curated reference set; verify)"
    if val is None and op == "population":
        proj = reference.project_population(place, year)
        if proj is not None:
            val, via, note = str(proj["value"]), "projection", f" ({proj['basis']}; estimate, not curated)"
        else:                                # beyond the projection horizon → nearest curated year
            near = reference.nearest_population(place, year)
            if near and near["year"] is not None:
                val, note = str(near["value"]), f" (no {year} figure; nearest available is the {near['year']} figure)"
            elif near:
                val, note = str(near["value"]), f" (no {year} figure in the reference)"
    if val is None:
        return explain(f"no reference data for '{place}' ({op})")
    display = (f"{val} km^2" if op == "area" else val) + note
    return PatternResult(answer=display,
                         trace=[{"step": f"reference.{op}", "place": place, "via": via},
                                {"step": "summary", "model_calls": model_calls}])

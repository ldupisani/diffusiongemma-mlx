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


def run(query):
    ok, d, _, _ = bind_geography(query)
    if not ok:
        return explain("could not parse the geography query")
    op, place = d["op"], d["place"]
    if op == "population":
        v = reference.population(place, d.get("year"))
        ans = str(v) if v is not None else None
    elif op == "coordinates":
        c = reference.coordinates(place)
        ans = f"{c[0]}, {c[1]}" if c else None
    elif op == "area":
        v = reference.country_field(place, "area_km2")
        ans = f"{v} km^2" if v is not None else None
    else:  # capital / continent
        ans = reference.country_field(place, op)
    if ans is None:
        return explain(f"no reference data for '{place}' ({op})")
    return PatternResult(answer=str(ans),
                         trace=[{"step": f"reference.{op}", "via": "code", "place": place},
                                {"step": "summary", "model_calls": 0}])

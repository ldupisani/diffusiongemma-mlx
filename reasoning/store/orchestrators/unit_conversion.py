"""Convert a value between units (temperature / length / weight) — coded, zero model.

The bind maps the (source, target) unit pair to the right conversion skill and the answer
symbol; run() builds a convert→round plan and formats with that symbol.
"""

from reasoning import engine
from reasoning.bind import _UNIT_ALIASES, bind_unit_convert
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_plan

META = {
    "id": "unit_conversion",
    "query_type": "unit_conversion",
    "description": "convert a value from one unit to another (temperature, length, weight)",
    "example_queries": [
        "Convert 100 degrees Fahrenheit to Celsius.",
        "How many km is 5 miles?",
        "Convert 150 pounds to kg.",
    ],
    "keywords": ["convert", "fahrenheit", "celsius", "miles", "km", "kilometers",
                 "pounds", "kg", "into"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    q = query.lower()
    tokens = set(q.replace("?", " ").replace(".", " ").split())
    has_unit = bool(tokens & set(_UNIT_ALIASES))
    has_to = any(f" {w} " in f" {q} " for w in ("to", "into", "in")) or "how many" in q
    return has_unit and has_to


def run(query):
    ok, data, via, _bind_calls = bind_unit_convert(query)
    if not ok:
        from reasoning.patterns import PatternResult
        return PatternResult(answer="(could not parse a known unit conversion from the query)",
                             trace=[{"step": "bind", "via": via, "ok": False},
                                    {"step": "summary", "model_calls": 0}])
    plan = Plan(
        steps=[
            CodeStep(data["skill"], [Inp("value")], out="conv"),
            CodeStep("arith.round_to", [Step("conv"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        answer_format="{:g}" + data["symbol"],
    )
    return run_plan(plan, {"value": data["value"]}, shot=engine._no_model)

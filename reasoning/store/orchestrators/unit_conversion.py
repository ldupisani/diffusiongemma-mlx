"""Convert a value between units of the same dimension — coded, zero model calls.

Backed by the canonical factor table in skills.units: one convert skill spans length,
mass, volume, time, speed, and temperature, so a new unit is a table row, not a new
orchestrator. The bind resolves the value and the source/target units; run() builds a
convert -> round plan and formats with the target's display symbol.
"""

from reasoning import engine
from reasoning.bind import bind_unit_convert
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded
from reasoning.skills.units import UNIT_WORDS

META = {
    "id": "unit_conversion",
    "query_type": "unit_conversion",
    "description": "convert a value from one unit to another (length, mass, volume, time, speed, temperature)",
    "example_queries": [
        "Convert 100 degrees Fahrenheit to Celsius.",
        "How many km is 5 miles?",
        "Convert 5 kilometers to meters",
        "How many feet are in 10 meters?",
        "Convert 2 gallons to liters",
        "How many seconds are in 5 minutes?",
        "Convert 60 miles per hour to kilometers per hour",
    ],
    "keywords": ["convert", "fahrenheit", "celsius", "kelvin", "miles", "km", "kilometers",
                 "meters", "feet", "inches", "centimeters", "pounds", "kg", "grams", "ounces",
                 "liters", "gallons", "milliliters", "seconds", "minutes", "hours", "into"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    q = query.lower()
    tokens = set(q.replace("?", " ").replace(".", " ").split())
    has_unit = bool(tokens & UNIT_WORDS)
    connector = any(f" {w} " in f" {q} " for w in ("to", "into", "in")) or "how many" in q
    return has_unit and connector


def run(query):
    ok, data, via, _ = bind_unit_convert(query)
    if not ok:
        from reasoning.patterns import PatternResult
        return PatternResult(answer="(no result: could not parse a known unit conversion)",
                             trace=[{"step": "bind", "via": via, "ok": False},
                                    {"step": "summary", "model_calls": 0}])
    plan = Plan(
        steps=[
            CodeStep("units.convert", [Inp("value"), Inp("from"), Inp("to")], out="c"),
            CodeStep("arith.round_to", [Step("c"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        answer_format="{:g}" + data["to_sym"],
    )
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)

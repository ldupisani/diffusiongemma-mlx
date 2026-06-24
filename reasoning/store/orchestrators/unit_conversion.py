"""Convert a value between units of the same dimension — coded, zero model calls.

Backed by the canonical factor table in skills.units: one convert skill spans length,
mass, volume, time, speed, and temperature, so a new unit is a table row, not a new
orchestrator. The bind resolves the value and the source/target units; run() builds a
convert -> round plan and formats with the target's display symbol.
"""

from reasoning import engine
from reasoning.bind import bind_unit_convert
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, explain, format_number, run_guarded

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
        "How many bytes are in 5 kilobytes?",
        "How many megabytes are in 1 terabyte?",
        "Convert 2 gigabytes to megabytes",
    ],
    "keywords": ["convert", "fahrenheit", "celsius", "kelvin", "miles", "km", "kilometers",
                 "meters", "feet", "inches", "centimeters", "pounds", "kg", "grams", "ounces",
                 "liters", "gallons", "milliliters", "seconds", "minutes", "hours", "into",
                 "bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "terabyte"],
    "model_free": True,
}


def precondition(query):
    # bindability: only claim queries we can actually resolve two compatible units for, so
    # "100 meters in 5 seconds" (a physics speed query) doesn't get mistaken for a conversion.
    return bind_unit_convert(query)[0]


def run(query):
    ok, data, via, _ = bind_unit_convert(query)
    if not ok:
        return explain("could not parse a known unit conversion")
    plan = Plan(
        steps=[
            CodeStep("units.convert", [Inp("value"), Inp("from"), Inp("to")], out="c"),
            CodeStep("arith.round_to", [Step("c"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        # clean integer rendering (data sizes get large — avoid {:g} scientific notation)
        answer_format=lambda x, s=data["to_sym"]: format_number(x) + s,
    )
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)

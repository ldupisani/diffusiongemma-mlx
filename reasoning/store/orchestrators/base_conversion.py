"""Number-base conversion (binary / octal / decimal / hex) — coded, zero model calls.

(Data-size conversion — KB/MB/GB/… — is handled by `unit_conversion`: a new "data"
dimension in the units factor table, no separate orchestrator.)
"""

from reasoning import engine
from reasoning.bind import bind_geospatial, bind_radix
from reasoning.compiled import CodeStep, Inp, Plan, run_guarded

META = {
    "id": "base_conversion",
    "query_type": "base_conversion",
    "description": "convert an integer between number bases (binary, octal, decimal, hexadecimal)",
    "example_queries": [
        "Convert 255 to binary.",
        "Convert 1010 from binary to decimal.",
        "What is 255 in hexadecimal?",
        "Convert hexadecimal FF to decimal.",
        "Convert 64 to octal.",
        "Convert binary 11111111 to decimal.",
    ],
    "keywords": ["binary", "hexadecimal", "hex", "octal", "decimal", "convert", "base"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[CodeStep("radix.convert", [Inp("value"), Inp("from_base"), Inp("to_base")], out="r")],
    answer_ref="r",
)


def precondition(query):
    return bind_radix(query)[0] and not bind_geospatial(query)[0]  # defer DMS→decimal


def run(query):
    return run_guarded(bind_radix(query), PLAN, shot=engine._no_model,
                       parse_message="could not identify the bases and value to convert")

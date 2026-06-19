"""Composed orchestrator (Phase 2 demo) — total cost when the quantity's unit differs from
the price's unit, so it must convert first.

"What is the total cost of 10 miles at $3 per km?"  →  convert 10 miles → km (delegated to
the `unit_conversion` orchestrator via a SubStep), then × $3. This is the first orchestrator
to USE `sub_orchestrators`: it composes another coded orchestrator and stays model-free iff
that one does (the SubStep folds its model-call tally into this plan's).
"""

from reasoning import engine
from reasoning.bind import bind_converted_cost
from reasoning.compiled import (CodeStep, Inp, Lit, Plan, Step, SubStep, explain,
                                format_number, run_guarded)

META = {
    "id": "converted_cost",
    "query_type": "converted_cost",
    "description": "total cost of a quantity priced per a different unit (converts, then "
                   "multiplies by the unit price)",
    "example_queries": [
        "What is the total cost of 10 miles at $3 per km?",
        "What is the cost of 5 miles at $2 per km?",
        "What is the total cost of 100 kg at $4 per pound?",
    ],
    "keywords": ["cost", "total", "per", "at"],
    "sub_orchestrators": ["unit_conversion"],  # composed via the SubStep below
    "model_free": True,
}


def precondition(query):
    return bind_converted_cost(query)[0]


def run(query):
    ok, data, via, _ = bind_converted_cost(query)
    if not ok:
        return explain("could not parse '<qty> <unit> at $<price> per <unit>'")
    plan = Plan(
        steps=[
            # delegate the unit conversion to the unit_conversion orchestrator
            SubStep("how many {to} is {qty} {frm}", out="converted",
                    args={"to": Inp("to"), "qty": Inp("qty"), "frm": Inp("frm")},
                    orchestrator="unit_conversion"),
            CodeStep("arith.multiply", [Step("converted"), Inp("price")], out="total"),
            CodeStep("arith.round_to", [Step("total"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        answer_format=lambda x: "$" + format_number(x),
    )
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)

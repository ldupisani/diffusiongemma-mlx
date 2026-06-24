"""Chemistry / stoichiometry — coded, zero model calls.

Dispatches molar mass (formula parse + atomic-weight lookup), moles, molarity, pH, and
dilution. The bind extracts the right inputs per quantity (a chemical formula for molar
mass, labelled numbers otherwise); run() applies the skill and labels the result's unit.
"""

from reasoning import engine
from reasoning.bind import bind_chemistry
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, explain, run_guarded

META = {
    "id": "chemistry",
    "query_type": "chemistry",
    "description": "molar mass of a formula, moles, molarity, pH, or dilution concentration",
    "example_queries": [
        "What is the molar mass of H2O?",
        "What is the molar mass of CO2?",
        "What is the molar mass of NaCl?",
        "What is the molar mass of C6H12O6?",
        "How many moles are in 36 grams of water given a molar mass of 18 g/mol?",
        "What is the molarity of 2 moles of solute dissolved in 4 liters of solution?",
        "What is the pH of a solution with a hydrogen ion concentration of 0.001 M?",
        "What is the final concentration when 50 mL of 2 M solution is diluted to 200 mL?",
    ],
    "keywords": ["molar", "mass", "moles", "molarity", "ph", "concentration", "diluted",
                 "solution", "moles", "grams"],
    "model_free": True,
}


def precondition(query):
    return bind_chemistry(query)[0]  # bindability (incl. valid chemical formula)


def run(query):
    ok, data, via, _ = bind_chemistry(query)
    if not ok:
        return explain("could not parse the chemistry inputs")
    plan = Plan(
        steps=[
            CodeStep(data["skill"], [Inp(d) for d in data["dims"]], out="v"),
            CodeStep("arith.round_to", [Step("v"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        answer_format="{:g}" + data["suffix"],
    )
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)

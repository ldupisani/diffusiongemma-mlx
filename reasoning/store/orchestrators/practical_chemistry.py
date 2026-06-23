"""Practical chemistry — ideal gas law, percent composition, pOH, mass percent. Coded."""

from reasoning.bind import bind_pchem
from reasoning.compiled import dispatch

META = {
    "id": "practical_chemistry",
    "query_type": "practical_chemistry",
    "description": "ideal-gas pressure (PV=nRT), percent composition of an element, pOH from "
                   "pH, and mass percent of a solution",
    "example_queries": [
        "What is the pressure of 2 moles of gas at 300 K in 10 liters?",
        "What is the percent composition of O in H2O?",
        "What is the pOH if the pH is 3?",
        "What is the mass percent of 20 g solute in 80 g of solution?",
    ],
    "keywords": ["ideal gas", "percent composition", "poh", "mass percent", "pressure", "moles"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_pchem(query)[0]


def run(query):
    return dispatch(bind_pchem(query), parse_message="could not parse the chemistry inputs")

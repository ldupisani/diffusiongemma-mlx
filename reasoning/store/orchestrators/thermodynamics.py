"""Thermodynamics — Carnot efficiency, entropy change, isothermal ideal-gas work, sensible heat
Q = mcΔT, and the Carnot COP of a refrigerator / heat pump. Coded, 0 calls.

Tier-2 domain (no `bind_gated`): the queries are keyword-rich ("Carnot", "isothermal", "specific
heat", "coefficient of performance"), so distinctive example_queries win on lexical score; the
bind's precondition still gates candidacy so a plain physics "work" or unit-conversion query is
never claimed. run() wraps the bind in guard_residue per §3d.
"""
from reasoning.bind import bind_thermo
from reasoning.bind._common import guard_residue
from reasoning.compiled import dispatch

META = {
    "id": "thermodynamics", "query_type": "thermodynamics",
    "description": "thermodynamics formulas: Carnot efficiency of a heat engine, entropy change "
                   "dS = Q/T, isothermal ideal-gas work W = nRT·ln(V2/V1), sensible heat "
                   "Q = m·c·ΔT, and the Carnot coefficient of performance (COP) of a "
                   "refrigerator or heat pump (reservoir temperatures in Kelvin)",
    "example_queries": [
        "What is the Carnot efficiency of a heat engine operating between 500 K and 300 K?",
        "What is the entropy change when 1000 J of heat is added at 250 K?",
        "What is the work done in the isothermal expansion of 1 mole of an ideal gas at 300 K "
        "from 1 L to 2 L?",
        "How much heat is needed to raise 2 kg of water with a specific heat of 4184 by 10 K?",
        "What is the COP of a refrigerator operating between 300 K and 250 K?",
        "What is the coefficient of performance of a heat pump operating between 300 K and 270 K?",
    ],
    "keywords": ["thermodynamics", "carnot", "carnot efficiency", "entropy change", "isothermal",
                 "specific heat", "coefficient of performance", "cop", "heat pump", "refrigerator",
                 "reservoir"],
    "model_free": True,
}


def precondition(query):
    return bind_thermo(query)[0]


def run(query):
    return dispatch(guard_residue(query, bind_thermo(query)),
                    parse_message="could not parse the thermodynamics query")
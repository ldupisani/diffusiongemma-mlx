"""Heat transfer — conduction, convection, radiation, thermal resistance, heat flux.
Model-free orchestrator. Zero model calls."""

from reasoning.bind import bind_heat_transfer
from reasoning.bind._common import guard_residue
from reasoning.compiled import dispatch

META = {
    "id": "heat_transfer", "query_type": "heat_transfer",
    "description": "Heat transfer by conduction (Fourier's law), convection (Newton's law), "
                   "and radiation (Stefan–Boltzmann law); thermal resistance and heat flux",
    "example_queries": [
        "What is the conduction heat rate for thermal conductivity 50 W/m·K, area 2 m², "
        "temperature difference 20 K, and thickness 0.1 m?",
        "Calculate the convection heat rate for h = 25 W/m²·K, area 2 m², and ΔT = 30 K",
        "What is the radiation heat rate for emissivity 0.9, area 1 m², and absolute temperature 300 K?",
        "Find the thermal resistance for thickness 0.1 m, conductivity 50 W/m·K, and area 2 m²",
        "What is the heat flux if the heat rate is 20000 W and area is 2 m²?",
    ],
    "keywords": ["conduction", "convection", "radiation", "thermal conductivity",
                 "heat transfer coefficient", "emissivity", "thermal resistance",
                 "heat flux", "stefan", "boltzmann"],
    "model_free": True,
}


def precondition(query):
    return bind_heat_transfer(query)[0]


def run(query):
    return dispatch(guard_residue(query, bind_heat_transfer(query)),
                    parse_message="could not parse the heat transfer query")
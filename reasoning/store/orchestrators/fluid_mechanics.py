"""Fluid mechanics — hydrostatic pressure, dynamic pressure, volumetric flow rate, Reynolds number,
continuity equation, and buoyant force. Model-free, 0 calls."""

from reasoning.bind import bind_fluid_mechanics
from reasoning.compiled import dispatch

META = {
    "id": "fluid_mechanics", "query_type": "fluid_mechanics",
    "description": "Fluid mechanics: hydrostatic pressure (P = ρ·g·h), dynamic pressure (q = 0.5·ρ·v²), "
                   "volumetric flow rate (Q = A·v), Reynolds number (Re = ρ·v·D/μ), continuity "
                   "equation outlet velocity (v₂ = A₁·v₁/A₂), and buoyant force (F = ρ·g·V)",
    "example_queries": [
        "What is the hydrostatic pressure at a depth of 10 meters in water with density 1000 kg/m³?",
        "Calculate the dynamic pressure for a fluid with density 1.225 kg/m³ and velocity 20 m/s.",
        "What is the volumetric flow rate with a cross-sectional area of 0.5 m² and velocity 2 m/s?",
        "Find the Reynolds number for a fluid with density 1000 kg/m³, velocity 2 m/s, diameter 0.05 m, and viscosity 0.001 Pa·s.",
        "What is the outlet velocity using the continuity equation for inlet area 0.2 m², inlet velocity 3 m/s, and outlet area 0.1 m²?",
        "Calculate hydrostatic pressure at 5 meters depth in seawater (density 1025 kg/m³).",
        "Find flow rate through a pipe with area 0.1 m² at velocity 5 m/s.",
        "What Reynolds number corresponds to density 1.225 kg/m³, velocity 10 m/s, diameter 0.02 m, viscosity 0.000018 Pa·s?",
        "What is the buoyant force for a fluid with density 1000 kg/m³ and displaced volume 0.5 m³?",
        "Calculate the buoyant force (Archimedes' principle) for an object with density 1025 kg/m³ and displaced volume 2 m³.",
    ],
    "keywords": ["hydrostatic", "dynamic pressure", "flow rate", "reynolds", "continuity", "viscosity",
                 "fluid", "density", "velocity", "depth", "laminar", "turbulent", "buoyant", "buoyancy",
                 "archimedes", "displaced volume", "archimedes' principle"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_fluid_mechanics(query)[0]


def run(query):
    return dispatch(bind_fluid_mechanics(query),
                    parse_message="could not parse the fluid mechanics query")
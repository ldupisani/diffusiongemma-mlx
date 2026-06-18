"""Physics formulas — coded, zero model calls.

One orchestrator dispatches mechanics/electricity/fluids formulas: the bind detects the
asked quantity (speed, force, energy, …) and extracts the inputs by their units, run()
applies the matching skill and labels the answer with the result's unit.
"""

from reasoning import engine
from reasoning.bind import bind_physics
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "physics",
    "query_type": "physics",
    "description": "a physics formula: speed, force, kinetic/potential energy, work, power, "
                   "density, Ohm's-law voltage, momentum, pressure, acceleration, or weight",
    "example_queries": [
        "What is the speed of an object that travels 100 meters in 5 seconds?",
        "What is the force on a mass of 10 kg accelerating at 3 m/s^2?",
        "What is the kinetic energy of a 2 kg object moving at 10 m/s?",
        "What is the gravitational potential energy of a 5 kg object at a height of 10 meters?",
        "How much work is done by a force of 20 N over a distance of 4 meters?",
        "What is the power if 100 joules of work is done in 5 seconds?",
        "What is the density of an object with mass 300 g and volume 100 cm^3?",
        "What is the voltage across a 5 ohm resistor with a current of 2 amps?",
        "What is the momentum of a 4 kg object moving at 8 m/s?",
        "What is the pressure of a 50 N force over an area of 10 square meters?",
        "What is the acceleration of an object that goes from 0 to 20 m/s in 4 seconds?",
        "What is the weight of a 10 kg object on Earth?",
    ],
    "keywords": ["speed", "force", "energy", "kinetic", "potential", "work", "power",
                 "density", "voltage", "momentum", "pressure", "acceleration", "weight",
                 "current", "velocity", "joules"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    return bind_physics(query)[0]  # bindability, not just keywords


def run(query):
    ok, data, via, _ = bind_physics(query)
    if not ok:
        from reasoning.patterns import PatternResult
        return PatternResult(answer="(no result: could not parse the physics quantities)",
                             trace=[{"step": "bind", "via": via, "ok": False},
                                    {"step": "summary", "model_calls": 0}])
    plan = Plan(
        steps=[
            CodeStep(data["skill"], [Inp(d) for d in data["dims"]], out="v"),
            CodeStep("arith.round_to", [Step("v"), Lit(2)], out="r"),
        ],
        answer_ref="r",
        answer_format="{:g}" + data["suffix"],
    )
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)

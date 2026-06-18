"""Physics-formula skills — mechanics, electricity, fluids.

Closed-form formulas (like geometry). The binding challenge is that inputs are cued by
UNITS (kg→mass, N→force, m/s→velocity, m/s²→acceleration) rather than explicit labels, and
several formulas share input types (kinetic energy and momentum both take mass+velocity) —
so the orchestrator disambiguates the formula by the asked quantity, the values by unit.
"""

from __future__ import annotations

from .base import register

_G = 9.8  # gravitational acceleration (m/s^2)


@register("physics.speed", "speed = distance / time (m/s)", [((100, 5), 20.0)])
def speed(distance: float, time: float) -> float:
    return distance / time


@register("physics.force", "force = mass * acceleration (N)", [((10, 3), 30)])
def force(mass: float, acceleration: float) -> float:
    return mass * acceleration


@register("physics.kinetic_energy", "kinetic energy = 1/2 m v^2 (J)", [((2, 10), 100.0)])
def kinetic_energy(mass: float, velocity: float) -> float:
    return 0.5 * mass * velocity ** 2


@register("physics.potential_energy", "gravitational PE = m g h (J)",
          [((5, 10), 490.0), ((5, 10, 9.8), 490.0)])
def potential_energy(mass: float, height: float, g: float = _G) -> float:
    return mass * g * height


@register("physics.work", "work = force * distance (J)", [((20, 4), 80)])
def work(force: float, distance: float) -> float:
    return force * distance


@register("physics.power", "power = work / time (W)", [((100, 5), 20.0)])
def power(work: float, time: float) -> float:
    return work / time


@register("physics.density", "density = mass / volume", [((300, 100), 3.0)])
def density(mass: float, volume: float) -> float:
    return mass / volume


@register("physics.ohms_law_voltage", "Ohm's law: voltage = current * resistance (V)",
          [((2, 5), 10)])
def ohms_law_voltage(current: float, resistance: float) -> float:
    return current * resistance


@register("physics.momentum", "momentum = mass * velocity (kg*m/s)", [((4, 8), 32)])
def momentum(mass: float, velocity: float) -> float:
    return mass * velocity


@register("physics.pressure", "pressure = force / area (Pa)", [((50, 10), 5.0)])
def pressure(force: float, area: float) -> float:
    return force / area


@register("physics.acceleration", "acceleration = (v_final - v_initial) / time (m/s^2)",
          [((0, 20, 4), 5.0)])
def acceleration(v_initial: float, v_final: float, time: float) -> float:
    return (v_final - v_initial) / time


@register("physics.weight", "weight = mass * g (N)", [((10,), 98.0), ((10, 9.8), 98.0)])
def weight(mass: float, g: float = _G) -> float:
    return mass * g

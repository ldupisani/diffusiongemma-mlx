"""Fluid mechanics — hydrostatic pressure, dynamic pressure, flow rate, Reynolds number, continuity equation, and buoyant force. Zero model calls.

Covers the computable pieces: hydrostatic pressure (P = ρ·g·h), dynamic pressure (q = 0.5·ρ·v²),
volumetric flow rate (Q = A·v), Reynolds number (Re = ρ·v·D/μ), continuity equation outlet
velocity (v₂ = A₁·v₁/A₂), and buoyant force (F = ρ·g·V). All inputs carry explicit units (kg/m³, m, m/s, Pa·s, m²).
"""

from __future__ import annotations

import math

from .base import register


@register("fluid_mechanics.hydrostatic_pressure", "hydrostatic pressure P = ρ·g·h (Pa); g = 9.81 m/s²",
          [((1000, 10), 98100.0), ((800, 5), 39240.0), ((1025, 20), 201105.0)])
def hydrostatic_pressure(density: float, depth: float) -> float:
    g = 9.81
    return round(density * g * depth, 6)


@register("fluid_mechanics.dynamic_pressure", "dynamic pressure q = 0.5·ρ·v² (Pa)",
          [((1.225, 20), 245.0), ((1000, 2), 2000.0), ((1.225, 10), 61.25)])
def dynamic_pressure(density: float, velocity: float) -> float:
    return round(0.5 * density * velocity ** 2, 6)


@register("fluid_mechanics.volumetric_flow_rate", "volumetric flow rate Q = A·v (m³/s)",
          [((0.5, 2), 1.0), ((0.1, 5), 0.5), ((0.2, 3), 0.6)])
def volumetric_flow_rate(area: float, velocity: float) -> float:
    return round(area * velocity, 6)


@register("fluid_mechanics.reynolds_number", "Reynolds number Re = ρ·v·D/μ (dimensionless)",
          [((1000, 2, 0.05, 0.001), 100000.0), ((1.225, 10, 0.02, 0.000018), 13611.111111),
           ((998, 1.5, 0.01, 0.001), 14970.0)])
def reynolds_number(density: float, velocity: float, diameter: float, dynamic_viscosity: float) -> float:
    return round(density * velocity * diameter / dynamic_viscosity, 6)


@register("fluid_mechanics.continuity_equation", "continuity equation outlet velocity v₂ = A₁·v₁/A₂ (m/s)",
          [((0.2, 3, 0.1), 6.0), ((0.5, 1, 0.25), 2.0), ((0.1, 4, 0.08), 5.0)])
def continuity_equation(inlet_area: float, inlet_velocity: float, outlet_area: float) -> float:
    return round(inlet_area * inlet_velocity / outlet_area, 6)


@register("fluid_mechanics.buoyant_force", "buoyant force F = ρ·g·V (N); g = 9.81 m/s²",
          [((1000, 0.5), 4905.0), ((1025, 2), 20110.5), ((800, 1.5), 11772.0)])
def buoyant_force(density: float, volume: float) -> float:
    g = 9.81
    return round(density * g * volume, 6)
"""Thermodynamics — closed-form formulas, zero model calls.

Computable fragment of classical thermodynamics: Carnot efficiency of a heat engine and the
Carnot coefficient of performance (COP) of a refrigerator / heat pump (all from reservoir
temperatures in KELVIN), entropy change dS = Q/T, isothermal ideal-gas work W = nRT·ln(V2/V1),
and sensible heat Q = m·c·ΔT.

Conventions: temperatures are absolute (Kelvin); efficiency is returned as a PERCENT; the gas
constant R = 8.314462618 J/(mol·K). Carnot quantities take (t_cold, t_hot) with t_hot > t_cold.
"""

from __future__ import annotations

import math

from .base import register

R = 8.314462618  # universal gas constant, J/(mol·K)


@register("thermo.carnot_efficiency",
          "Carnot efficiency of a heat engine = (1 - T_cold/T_hot)·100 percent (Kelvin)",
          [((300, 400), 25.0), ((300, 500), 40.0), ((300, 600), 50.0)])
def carnot_efficiency(t_cold: float, t_hot: float) -> float:
    return round((1 - t_cold / t_hot) * 100, 6)


@register("thermo.entropy_change", "entropy change dS = Q / T (J/K)",
          [((1000, 250), 4.0), ((500, 100), 5.0), ((300, 300), 1.0)])
def entropy_change(heat: float, temperature: float) -> float:
    return round(heat / temperature, 6)


@register("thermo.isothermal_work",
          "isothermal ideal-gas work W = n·R·T·ln(V2/V1) (J), rounded to 2 dp",
          [((1, 300, 1, 2), 1728.94), ((2, 300, 1, 3), 5480.62), ((1, 298, 2, 4), 1717.42)])
def isothermal_work(n: float, temperature: float, v1: float, v2: float) -> float:
    return round(n * R * temperature * math.log(v2 / v1), 2)


@register("thermo.heat", "sensible heat Q = m·c·ΔT (J)",
          [((2, 4184, 10), 83680.0), ((0.5, 900, 20), 9000.0), ((1, 4184, 5), 20920.0)])
def heat(mass: float, specific_heat: float, delta_t: float) -> float:
    return round(mass * specific_heat * delta_t, 6)


@register("thermo.cop_refrigerator",
          "Carnot COP of a refrigerator = T_cold / (T_hot - T_cold) (Kelvin)",
          [((250, 300), 5.0), ((270, 300), 9.0)])
def cop_refrigerator(t_cold: float, t_hot: float) -> float:
    return round(t_cold / (t_hot - t_cold), 6)


@register("thermo.cop_heat_pump",
          "Carnot COP of a heat pump = T_hot / (T_hot - T_cold) (Kelvin)",
          [((250, 300), 6.0), ((270, 300), 10.0)])
def cop_heat_pump(t_cold: float, t_hot: float) -> float:
    return round(t_hot / (t_hot - t_cold), 6)
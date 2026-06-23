"""Practical chemistry — ideal gas law, percent composition, pOH, mass percent. Pure stdlib +
the chemistry atomic table. Zero model calls."""

from __future__ import annotations

from .base import register
from .chemistry import ATOMIC, molar_mass, parse_formula

_R = 0.0820573  # ideal gas constant, L·atm/(mol·K)


@register("pchem.ideal_gas_pressure", "ideal-gas pressure P = nRT/V (atm; n mol, T K, V L)",
          [((1, 273.15, 22.414), 0.999997835950745), ((2, 300, 10), 4.923438)])
def ideal_gas_pressure(moles: float, temp_k: float, volume_l: float) -> float:
    return moles * _R * temp_k / volume_l


@register("pchem.percent_composition", "mass percent of an element in a compound",
          [(("H2O", "O"), 88.80932556203163), (("H2O", "H"), 11.190674437968362)])
def percent_composition(formula: str, element: str) -> float:
    counts = parse_formula(formula)
    return counts.get(element, 0) * ATOMIC[element] / molar_mass(formula) * 100


@register("pchem.poh", "pOH from pH (pH + pOH = 14)", [((3,), 11.0), ((7,), 7.0)])
def poh(ph: float) -> float:
    return 14 - ph


@register("pchem.mass_percent", "mass percent of solute in a solution",
          [((5, 100), 5.0), ((20, 80), 25.0)])
def mass_percent(solute_mass: float, solution_mass: float) -> float:
    return solute_mass / solution_mass * 100

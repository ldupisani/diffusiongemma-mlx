"""Chemistry / stoichiometry skills — molar mass, moles, molarity, pH, dilution.

The novel ingredients vs other families: a chemical-FORMULA PARSER (H2O → {H:2, O:1},
with parentheses: Ca(OH)2 → {Ca:1, O:2, H:2}) and an atomic-weight LOOKUP TABLE — the
first time a "table" holds reference data rather than conversion factors. molar_mass
combines the two; the rest are ordinary closed-form formulas.
"""

from __future__ import annotations

import math
import re

from .base import register

# standard atomic weights (g/mol) — a common subset of the periodic table
ATOMIC = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.868,
    "Sn": 118.710, "I": 126.904, "Ba": 137.327, "Pt": 195.084, "Au": 196.967,
    "Hg": 200.592, "Pb": 207.2, "U": 238.029,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\))(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """Element → count for a chemical formula, honouring nested parentheses.

    Ca(OH)2 → {Ca:1, O:2, H:2}. A stack handles groups: '(' pushes a fresh tally, ')'
    pops it and folds it into the parent scaled by the trailing multiplier.
    """
    stack: list[dict[str, int]] = [{}]
    for sym, num in _FORMULA_TOKEN.findall(formula):
        if sym == "(":
            stack.append({})
        elif sym == ")":
            group = stack.pop()
            mult = int(num) if num else 1
            for el, c in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + c * mult
        else:
            mult = int(num) if num else 1
            stack[-1][sym] = stack[-1].get(sym, 0) + mult
    return stack[0]


@register("chem.molar_mass", "molar mass (g/mol) of a chemical formula",
          [(("H2O",), 18.015), (("CO2",), 44.009), (("NaCl",), 58.44)])
def molar_mass(formula: str) -> float:
    counts = parse_formula(formula)
    if not counts:
        raise ValueError(f"could not parse chemical formula {formula!r}")
    total = 0.0
    for el, c in counts.items():
        if el not in ATOMIC:
            raise ValueError(f"unknown element {el!r} in {formula!r}")
        total += ATOMIC[el] * c
    return total


@register("chem.moles", "moles = mass / molar mass", [((36, 18), 2.0), ((44, 44), 1.0)])
def moles(mass: float, molar_mass: float) -> float:
    return mass / molar_mass


@register("chem.molarity", "molarity = moles of solute / litres of solution (mol/L)",
          [((2, 4), 0.5)])
def molarity(moles: float, volume: float) -> float:
    return moles / volume


@register("chem.ph", "pH = -log10(hydrogen-ion concentration)",
          [((0.001,), 3.0), ((1e-7,), 7.0)])
def ph(concentration: float) -> float:
    return -math.log10(concentration)


@register("chem.dilution_concentration", "final concentration from C1*V1 = C2*V2",
          [((2, 50, 200), 0.5)])
def dilution_concentration(c1: float, v1: float, v2: float) -> float:
    return c1 * v1 / v2

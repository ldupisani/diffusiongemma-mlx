"""Molecular science — Avogadro conversions and degrees of unsaturation. Stdlib + the chemistry
formula parser. Zero model calls."""

from __future__ import annotations

from .base import register
from .chemistry import parse_formula

_NA = 6.02214076e23  # Avogadro's number


@register("molec.molecules_from_moles", "number of molecules in n moles (Avogadro)",
          [((2,), 1.204428152e24), ((0.5,), 3.01107038e23)])
def molecules_from_moles(moles: float) -> float:
    return moles * _NA


@register("molec.moles_from_molecules", "moles in a number of molecules (Avogadro)",
          [((6.02214076e23,), 1.0), ((1.204428152e24,), 2.0)])
def moles_from_molecules(molecules: float) -> float:
    return molecules / _NA


@register("molec.degrees_of_unsaturation", "degrees of unsaturation of a formula "
          "(2C+2+N−H−X)/2", [(("C6H6",), 4), (("C6H12",), 1), (("C2H6",), 0)])
def degrees_of_unsaturation(formula: str) -> int:
    c = parse_formula(formula)
    carbon, hydrogen = c.get("C", 0), c.get("H", 0)
    nitrogen = c.get("N", 0)
    halogen = sum(c.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    return (2 * carbon + 2 + nitrogen - hydrogen - halogen) // 2

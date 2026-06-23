"""bind layer — practical chemistry (ideal gas, % composition, pOH, mass %) and molecular
science (Avogadro conversions, degrees of unsaturation)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind import entities as _E

_FORMULA = re.compile(r"\b((?:[A-Z][a-z]?\d*){2,})\b")  # e.g. C6H6, H2O


def bind_pchem(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "ideal gas" in q or ("pressure" in q and "mol" in q):
        n = _E.number_before(query, r"mol(?:e|es)?\b")
        t = _E.number_before(query, r"K\b|kelvin")
        v = _E.number_before(query, r"L\b|liters?|litres?")
        if n is not None and t is not None and v is not None:
            return ok("pchem.ideal_gas_pressure", ["moles", "temp_k", "volume_l"],
                      [n, t, v], suffix=" atm", places=4)
    if "poh" in q:
        m = re.search(r"ph\D{0,8}?(\d+(?:\.\d+)?)", q)
        if m:
            return ok("pchem.poh", ["ph"], [float(m[1])], places=2)
    if ("percent composition" in q or "mass percent" in q) and " in " in q:
        m = re.search(r"of\s+([A-Z][a-z]?)\s+in\s+([A-Za-z0-9]+)", query)
        if m:
            return ok("pchem.percent_composition", ["formula", "element"],
                      [m[2], m[1]], suffix="%", places=2)
    if "mass percent" in q:
        masses = _E.exactly(query, 2)
        if masses:
            return ok("pchem.mass_percent", ["solute_mass", "solution_mass"],
                      masses, suffix="%", places=2)
    return False, None, "code", 0


def bind_molecular(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, args, kind):
        return True, {"skill": skill, "args": args, "kind": kind}, "code", 0

    if "unsaturation" in q:
        m = _FORMULA.search(query)
        if m:
            return ok("molec.degrees_of_unsaturation", [m[1]], "int")
    if "molecule" in q and "mole" in q:
        if "how many molecules" in q or "number of molecules" in q:
            n = _E.number_before(query, r"mol(?:e|es)?\b") or _E.only_number(query)
            if n is not None:
                return ok("molec.molecules_from_moles", [n], "sci")
        if "how many moles" in q or "number of moles" in q:
            # molecule counts are Avogadro-scale, often in scientific notation (6.022e23) —
            # which entities' plain number regex would split, so match it here
            m = re.search(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*molecules", q)
            if m:
                return ok("molec.moles_from_molecules", [float(m[1])], "num")
    return False, None, "code", 0

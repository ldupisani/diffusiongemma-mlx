"""bind layer — material science (stress/strain/Young's modulus, thermal expansion, APF)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind import entities as _E

_STRUCTURES = ("face-centered cubic", "body-centered cubic", "simple cubic", "fcc", "bcc", "sc")


def bind_materials(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "packing factor" in q or "apf" in q:
        s = next((st for st in _STRUCTURES if st in q), None)
        if s:
            return ok("materials.packing_factor", ["s"], [s], places=4)
    if "young" in q and "modulus" in q:
        sm = re.search(r"stress\D{0,12}?(\d+(?:\.\d+)?)", q)
        em = re.search(r"strain\D{0,12}?(\d+(?:\.\d+)?)", q)
        if sm and em:
            return ok("materials.youngs_modulus", ["stress_val", "strain_val"],
                      [float(sm[1]), float(em[1])], suffix=" Pa", places=2)
    if "thermal expansion" in q or ("expansion" in q and "coefficient" in q):
        a = re.search(r"(?:coefficient|alpha|α)\D{0,12}?(\d*\.?\d+(?:e-?\d+)?)", q)
        length = _E.number_before(query, r"meters?|metres?|m\b")
        dt = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°|kelvin|k\b)", q)
        if a and length is not None and dt:
            return ok("materials.thermal_expansion", ["alpha", "length", "delta_t"],
                      [float(a[1]), length, float(dt[1])], suffix=" m", places=6)
    if "stress" in q:  # force / area
        f = _E.number_before(query, r"N\b|newtons?")
        area = _E.number_before(query, r"square meters|square metres|m\^?2|m²")
        if f is not None and area is not None:
            return ok("materials.stress", ["force", "area"], [f, area], suffix=" Pa", places=2)
    return False, None, "code", 0

"""bind layer — thermodynamics. Each branch is gated on a DISTINCTIVE phrase (Carnot, entropy,
isothermal, specific heat, coefficient of performance / refrigerator / heat pump) so it never
over-claims. Temperatures are read only when carrying a Kelvin unit; Carnot quantities take the
larger temperature as hot and the smaller as cold. The whole bind has a FIXED numeric arity, so
run() wraps it in guard_residue — a stray value (an extra temperature, a heat the formula ignores)
turns the parse into a clean decline instead of a confident-wrong answer (§3d).
"""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"(\d+(?:\.\d+)?)"
# A temperature must carry an explicit Kelvin unit (K with a boundary, or "kelvin") — this also
# keeps the engine honest: Carnot ratios are only valid on absolute temperatures, so a Celsius
# figure simply won't match and the bind declines rather than mis-using it.
_T_K = re.compile(_NUM + r"\s*(?:K\b|kelvin)", re.IGNORECASE)
_Q_J = re.compile(_NUM + r"\s*(?:J\b|joules?)", re.IGNORECASE)
_KG = re.compile(_NUM + r"\s*(?:kg|kilograms?)\b", re.IGNORECASE)
_C_SH = re.compile(r"specific heat(?:\s+capacity)?(?:\s+of\s+[a-z]+)?\s*(?:of|=|is|:|,)?\s*" + _NUM,
                   re.IGNORECASE)
_C_J = re.compile(_NUM + r"\s*J\s*/\s*\(?\s*kg", re.IGNORECASE)
_DT_BY = re.compile(r"by\s+" + _NUM + r"\s*(?:K\b|kelvin|°?\s?c\b|celsius|degrees?)", re.IGNORECASE)
_DT_LBL = re.compile(r"(?:\u0394t|delta\s*t|temperature change|change in temperature)\s*"
                     r"(?:of|=|is|:)?\s*" + _NUM, re.IGNORECASE)
_MOL = re.compile(_NUM + r"\s*mol(?:e|es)?\b", re.IGNORECASE)
_FROM_TO = re.compile(r"from\s+" + _NUM +
                      r"\s*(?:l\b|liters?|litres?|m\^?3|m\u00b3|cm\^?3)?\s*to\s+" + _NUM,
                      re.IGNORECASE)


def _temps(q: str):
    return [float(m) for m in _T_K.findall(q)]


def bind_thermo(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, suffix="", places=6):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return (True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)),
                       "round": True, "places": places, "suffix": suffix}, "code", 0)

    # 1. COP of a refrigerator / heat pump (Carnot COP from the two reservoir temperatures).
    if ("coefficient of performance" in q or re.search(r"\bcop\b", q)) and \
            ("refrigerator" in q or "fridge" in q or "heat pump" in q):
        ts = _temps(q)
        if len(ts) != 2:
            return False, None, "code", 0
        cold, hot = min(ts), max(ts)
        skill = "thermo.cop_heat_pump" if "heat pump" in q else "thermo.cop_refrigerator"
        return ok(skill, ["t_cold", "t_hot"], [cold, hot])

    # 2. Carnot efficiency of a heat engine.
    if "carnot" in q or ("efficiency" in q and ("heat engine" in q or "reservoir" in q)):
        ts = _temps(q)
        if len(ts) != 2:
            return False, None, "code", 0
        cold, hot = min(ts), max(ts)
        return ok("thermo.carnot_efficiency", ["t_cold", "t_hot"], [cold, hot], suffix="%")

    # 3. Isothermal ideal-gas work W = nRT·ln(V2/V1).
    if "isothermal" in q:
        n = _MOL.search(q)
        ts = _temps(q)
        vv = _FROM_TO.search(q)
        if not (n and ts and vv):
            return False, None, "code", 0
        return ok("thermo.isothermal_work", ["n", "temperature", "v1", "v2"],
                  [float(n.group(1)), ts[0], float(vv.group(1)), float(vv.group(2))],
                  suffix=" J", places=2)

    # 4. Entropy change dS = Q/T — gated on "entropy" PLUS a heat (J) and a temperature (K); decline
    #    Shannon/information entropy (a probability distribution), which belongs to ml_theory.
    if "entropy" in q:
        if any(w in q for w in ("shannon", "information", "bits", "probabilit", "[")):
            return False, None, "code", 0
        qm, ts = _Q_J.search(q), _temps(q)
        if not (qm and ts):
            return False, None, "code", 0
        return ok("thermo.entropy_change", ["heat", "temperature"],
                  [float(qm.group(1)), ts[0]], suffix=" J/K")

    # 5. Sensible heat Q = m·c·ΔT — gated on "specific heat".
    if "specific heat" in q:
        m = _KG.search(q)
        c = _C_SH.search(q) or _C_J.search(q)
        dt = _DT_BY.search(q) or _DT_LBL.search(q)
        if not (m and c and dt):
            return False, None, "code", 0
        return ok("thermo.heat", ["mass", "specific_heat", "delta_t"],
                  [float(m.group(1)), float(c.group(1)), float(dt.group(1))], suffix=" J")

    return False, None, "code", 0
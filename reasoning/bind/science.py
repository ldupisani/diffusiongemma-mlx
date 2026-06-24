"""bind layer — science domains (see reasoning.bind)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind.entities import _FROM_TO as _FROM_TO_VEL  # noqa: F401  (single source)
from reasoning.bind.entities import number_before as _num_before  # noqa: F401


# --- chemistry / stoichiometry family -------------------------------------------------
_GRAMS = re.compile(r"(\d+(?:\.\d+)?)\s*g(?:rams)?\b")
_GMOL = re.compile(r"(\d+(?:\.\d+)?)\s*g/mol")
_MOLES_N = re.compile(r"(\d+(?:\.\d+)?)\s*mol(?:es)?\b")
_LITERS_N = re.compile(r"(\d+(?:\.\d+)?)\s*(?:liters?|litres?|l\b)")
_MOLAR_M = re.compile(r"(\d*\.?\d+)\s*M\b")        # molar concentration, e.g. "0.001 M"
_ML_N = re.compile(r"(\d+(?:\.\d+)?)\s*mL")
_FORMULA_CAND = re.compile(r"(?:[A-Z][a-z]?\d*|\([^)]*\)\d*)+")


def _find_formula(query: str):
    """The first capitalized token that parses to KNOWN elements (so 'What' → 'Wh' is
    rejected, 'H2O' is accepted) — the validation IS the disambiguation."""
    from reasoning.skills.chemistry import ATOMIC, parse_formula
    for tok in _FORMULA_CAND.findall(query):
        counts = parse_formula(tok)
        if counts and all(el in ATOMIC for el in counts):
            return tok
    return None


def bind_chemistry(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    # order disambiguates shared words: "molarity"/"diluted" before "moles"/"molar mass"
    if "molarity" in q:
        mol, vol = _MOLES_N.search(query), _LITERS_N.search(query)
        if mol and vol:
            return True, {"skill": "chem.molarity", "dims": ["moles", "volume"], "suffix": " M",
                          "moles": float(mol[1]), "volume": float(vol[1])}, "code", 0
    elif "diluted" in q or "dilution" in q:
        c1, mls = _MOLAR_M.search(query), _ML_N.findall(query)
        if c1 and len(mls) >= 2:
            return True, {"skill": "chem.dilution_concentration", "dims": ["c1", "v1", "v2"],
                          "suffix": " M", "c1": float(c1[1]), "v1": float(mls[0]),
                          "v2": float(mls[1])}, "code", 0
    elif "mole" in q:
        mass, mm = _GRAMS.search(query), _GMOL.search(query)
        if mass and mm:
            return True, {"skill": "chem.moles", "dims": ["mass", "molar_mass"], "suffix": " mol",
                          "mass": float(mass[1]), "molar_mass": float(mm[1])}, "code", 0
    elif "molar mass" in q:
        f = _find_formula(query)
        if f:
            return True, {"skill": "chem.molar_mass", "dims": ["formula"], "suffix": " g/mol",
                          "formula": f}, "code", 0
    elif re.search(r"\bph\b", q):
        c = _MOLAR_M.search(query)
        if c:
            return True, {"skill": "chem.ph", "dims": ["concentration"], "suffix": "",
                          "concentration": float(c[1])}, "code", 0
    return False, None, "code", 0


# --- physics-formula family -----------------------------------------------------------
# Quantities are cued by their UNIT, so each input is "the number before unit X". Patterns
# are case-sensitive (kg/N/m/s have meaningful case) and ordered so m/s^2 isn't read as m/s.
_PHYSICS_UNITS = {
    "KG": r"kg", "G": r"g(?:rams)?\b", "VEL": r"m/s(?![\^²2])", "ACCEL": r"m/s(?:\^?2|²)",
    "METERS": r"meters|metres", "SECONDS": r"seconds?", "N": r"N\b|newtons?",
    "JOULES": r"joules?|J\b", "AMPS": r"amps?|amperes?", "OHM": r"ohms?",
    "AREA": r"square meters|m\^?2|m²", "CM3": r"cm\^?3|cm³|cubic centimeters?",
}
# (asked-quantity keywords, skill, [(arg, unit-key)], answer suffix). Order: compound and
# specific terms first (power before work; pressure/work before the generic "force").
_PHYSICS = [
    (("kinetic energy",), "physics.kinetic_energy", [("mass", "KG"), ("velocity", "VEL")], " J"),
    (("potential energy", "gravitational"), "physics.potential_energy",
     [("mass", "KG"), ("height", "METERS")], " J"),
    (("momentum",), "physics.momentum", [("mass", "KG"), ("velocity", "VEL")], " kg·m/s"),
    (("density",), "physics.density", [("mass", "G"), ("volume", "CM3")], " g/cm³"),
    (("voltage", "ohm"), "physics.ohms_law_voltage",
     [("current", "AMPS"), ("resistance", "OHM")], " V"),
    (("pressure",), "physics.pressure", [("force", "N"), ("area", "AREA")], " Pa"),
    (("power",), "physics.power", [("work", "JOULES"), ("time", "SECONDS")], " W"),
    (("work",), "physics.work", [("force", "N"), ("distance", "METERS")], " J"),
    (("weight",), "physics.weight", [("mass", "KG")], " N"),
    (("force",), "physics.force", [("mass", "KG"), ("acceleration", "ACCEL")], " N"),
    (("speed", "how fast"), "physics.speed", [("distance", "METERS"), ("time", "SECONDS")], " m/s"),
]
# _FROM_TO_VEL and _num_before are imported from entities (single source)


def detect_physics(query: str):
    q = query.lower()
    for kws, skill, dims, suffix in _PHYSICS:
        if any(k in q for k in kws):
            return skill, dims, suffix
    if "acceleration" in q:  # checked last so "accelerating" in a force query doesn't claim it
        return "physics.acceleration", "ACCEL", " m/s²"
    return None


def bind_physics(query: str) -> tuple[bool, dict | None, str, int]:
    det = detect_physics(query)
    if not det:
        return False, None, "code", 0
    skill, dims, suffix = det
    if dims == "ACCEL":  # a = (v_final - v_initial) / time, from "X to Y in T seconds"
        m, t = _FROM_TO_VEL.search(query), _num_before(query, _PHYSICS_UNITS["SECONDS"])
        if not m or t is None:
            return False, None, "code", 0
        return True, {"v_initial": float(m[1]), "v_final": float(m[2]), "time": t,
                      "skill": skill, "dims": ["v_initial", "v_final", "time"],
                      "suffix": suffix}, "code", 0
    data = {"skill": skill, "dims": [d for d, _ in dims], "suffix": suffix}
    for name, ukey in dims:
        v = _num_before(query, _PHYSICS_UNITS[ukey])
        if v is None:
            return False, None, "code", 0
        data[name] = v
    return True, data, "code", 0


# --- recognition bridge: NL → ({typed knowns}, goal) for the plan synthesizer -----------
# Extracts unit-cued physics quantities as typed knowns and the asked quantity as the goal,
# then asks synthesize() for a path. Claims a query ONLY when the derivation needs ≥2 ops —
# single-formula queries are left to the `physics` orchestrator (mutually exclusive: if the
# goal's direct inputs are present the path is length 1; if not, physics can't bind anyway).
_SYNTH_GOALS = [
    (("pressure",), "pressure", " Pa"),
    (("power",), "power", " W"),
    (("kinetic energy",), "kinetic_energy", " J"),
    (("momentum",), "momentum", " kg·m/s"),
    (("work",), "work", " J"),
    (("force",), "force", " N"),
    (("how fast", "speed"), "speed", " m/s"),
]
_SYNTH_ROLE_UNITS = [("mass", "KG"), ("acceleration", "ACCEL"), ("velocity", "VEL"),
                     ("distance", "METERS"), ("time", "SECONDS"), ("force", "N"),
                     ("work", "JOULES"), ("area", "AREA")]


def bind_synthesis_physics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    goal_suffix = next(((g, suf) for kws, g, suf in _SYNTH_GOALS if any(k in q for k in kws)),
                       None)
    if not goal_suffix:
        return False, None, "code", 0
    goal, suffix = goal_suffix
    from reasoning.skills import units as _units
    from reasoning.synthesize import ROLE_UNIT
    knowns = {}
    for role, ukey in _SYNTH_ROLE_UNITS:
        target = ROLE_UNIT.get(role)
        if target:  # unit-aware: accept any unit of this role's dimension, convert to canonical
            pq = _units.parse_quantity(query, _units.DIMENSION[target])
            v = _units.convert(pq[0], pq[1], target) if pq else None
        else:        # opaque physical unit (N, J, m/s²) → the precise collision-safe regex
            v = _num_before(query, _PHYSICS_UNITS[ukey])
        if v is not None:
            knowns[role] = v
    vm = _FROM_TO_VEL.search(query)  # "from <v0> to <v1> m/s" → accelerate the object
    if vm and "m/s" in q:
        knowns["v_initial"], knowns["v_final"] = float(vm[1]), float(vm[2])
    from reasoning.bind.measure import shape_dims
    knowns.update(shape_dims(query))  # a SHAPE supplies area (→ pressure) or volume
    knowns.pop(goal, None)  # we're solving FOR the goal, not reading it as a known
    from reasoning.synthesize import GEOMETRY_COST_OPS, MOTION_OPS, PHYSICS_OPS, synthesize
    # cross-domain pool: geometry shapes feed pressure; motion speed feeds KE/momentum. unique=True
    # so an ambiguous derivation declines rather than silently picking one.
    pool = PHYSICS_OPS + MOTION_OPS + GEOMETRY_COST_OPS
    seq = synthesize(set(knowns), goal, pool, unique=True)
    if seq is None or len(seq) < 2:  # single-formula → the `physics` orchestrator handles it
        return False, None, "code", 0
    return True, {"knowns": knowns, "goal": goal, "suffix": suffix}, "code", 0


# --- recognition bridge #4: chemistry (compute molar mass from a formula, then moles/mass) -
# The existing chemistry orchestrator needs molar mass GIVEN (g/mol); this computes it from
# the FORMULA (a string-valued role), then derives moles (from grams) or mass (from moles).
def bind_synthesis_chemistry(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    if _GMOL.search(query):  # molar mass given explicitly → the chemistry orchestrator handles it
        return False, None, "code", 0
    formula = _find_formula(query)
    if not formula:
        return False, None, "code", 0
    grams, moles_given = _GRAMS.search(query), _MOLES_N.search(query)
    volume = _LITERS_N.search(query)
    price_g = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/|a)\s*g(?:ram)?s?\b", query, re.IGNORECASE)
    knowns = {"formula": formula}
    prefix, unique = "", False
    if "cost" in q and price_g and moles_given:  # chem → finance: molar_mass → mass → $ per gram
        knowns["moles"], knowns["price_per_gram"] = float(moles_given[1]), float(price_g[1])
        goal, suffix, prefix, unique = "reagent_cost", "", "$", True
        # §D multi-hop tail: chem → finance → split/tax (reagent_cost funnels into cost)
        sp = (re.search(r"(?:among|between|by)\s+(\d+)|(\d+)\s+(?:people|friends|ways)", q)
              if any(w in q for w in ("split", "between", "among", "shared")) else None)
        tx = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:sales\s*)?(?:tax|vat)", q)
        if sp:
            knowns["people"], goal, suffix = float(int(sp[1] or sp[2])), "cost_per_person", " each"
        elif tx:
            knowns["tax_pct"], goal = float(tx[1]), "cost_with_tax"
    elif "molecule" in q and grams:    # chem → molecular: mass → moles → molecules (Avogadro)
        knowns["mass_g"] = float(grams[1])
        goal, suffix, unique = "molecules", "", True
    elif "weight" in q and moles_given:  # chem → coerce(g→kg) → physics: mass → weight (§F)
        knowns["moles"] = float(moles_given[1])
        goal, suffix, unique = "weight", " N", True
    elif "molarity" in q and grams and volume:  # molar_mass → moles → molarity
        knowns["mass_g"], knowns["volume"] = float(grams[1]), float(volume[1])
        goal, suffix = "molarity", " M"
    elif "mole" in q and grams:        # moles from grams + formula
        knowns["mass_g"] = float(grams[1])
        goal, suffix = "moles", " mol"
    elif "mass" in q and moles_given:  # mass of N moles of formula
        knowns["moles"] = float(moles_given[1])
        goal, suffix = "mass_g", " g"
    else:
        return False, None, "code", 0
    from reasoning.synthesize import (CHEM_OPS, COERCION_OPS, CROSS_OPS, MOLECULAR_OPS,
                                      WEIGHT_OPS, synthesize)
    seq = synthesize(set(knowns), goal,
                     CHEM_OPS + MOLECULAR_OPS + WEIGHT_OPS + COERCION_OPS + CROSS_OPS,
                     unique=unique)
    if seq is None or len(seq) < 2:
        return False, None, "code", 0
    return True, {"knowns": knowns, "goal": goal, "prefix": prefix, "suffix": suffix,
                  "unique": unique}, "code", 0


# --- recognition bridge #5: physics force + geometry area → materials stress / Young's modulus
def bind_synthesis_materials(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    if "stress" not in q and "modulus" not in q and "young" not in q:
        return False, None, "code", 0
    from reasoning.bind.measure import shape_dims
    from reasoning.synthesize import (GEOMETRY_COST_OPS, MATERIALS_OPS, PHYSICS_OPS, synthesize)
    knowns = shape_dims(query)                       # a SHAPE supplies the area
    force = _num_before(query, _PHYSICS_UNITS["N"])
    if force is not None:
        knowns["force"] = force
    area = _num_before(query, _PHYSICS_UNITS["AREA"])  # or an explicit area
    if area is not None:
        knowns["area"] = area
    sm = re.search(r"strain\D{0,10}?(\d+(?:\.\d+)?)", q)
    if sm:
        knowns["strain"] = float(sm[1])
    goal = "youngs" if ("young" in q or "modulus" in q) else "stress"
    pool = GEOMETRY_COST_OPS + PHYSICS_OPS + MATERIALS_OPS
    seq = synthesize(set(knowns), goal, pool, unique=True)
    if seq is None or len(seq) < 2:  # single-formula (explicit area) → the `materials` orchestrator
        return False, None, "code", 0
    return True, {"knowns": knowns, "goal": goal, "suffix": " Pa", "prefix": "",
                  "unique": True}, "code", 0

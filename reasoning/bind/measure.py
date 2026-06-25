"""bind layer — measure domains (see reasoning.bind)."""
from __future__ import annotations

import math
import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind import entities as _E
from reasoning.bind.entities import _POINT  # noqa: F401  (single source)


# --- geometry / measurement family ----------------------------------------------------
# (measurement keyword, shape keyword or None, skill, ordered dimension labels). Checked in
# order, so "surface area" beats "area" and shape-free formulas (circumference/hypotenuse)
# come first. dims=None marks the hypotenuse's special "legs A and B" extraction.
_GEOMETRY = [
    ("circumference", None, "geometry.circle_circumference", ["radius"]),
    ("hypotenuse", None, "geometry.hypotenuse", None),
    ("surface area", "sphere", "geometry.sphere_surface_area", ["radius"]),
    ("diagonal", "rectangle", "geometry.rectangle_diagonal", ["length", "width"]),
    ("volume", "cube", "geometry.cube_volume", ["side"]),
    ("volume", "sphere", "geometry.sphere_volume", ["radius"]),
    ("volume", "cylinder", "geometry.cylinder_volume", ["radius", "height"]),
    ("perimeter", "rectangle", "geometry.rectangle_perimeter", ["length", "width"]),
    ("area", "circle", "geometry.circle_area", ["radius"]),
    ("area", "rectangle", "geometry.rectangle_area", ["length", "width"]),
    ("area", "triangle", "geometry.triangle_area", ["base", "height"]),
    ("area", "square", "geometry.square_area", ["side"]),
]
_LEGS = re.compile(r"legs?\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def detect_geometry(query: str):
    """(skill, dim_labels) for the matching shape+measurement, else None."""
    q = query.lower()
    for measure, shape, skill, dims in _GEOMETRY:
        if measure in q and (shape is None or shape in q):
            return skill, dims
    return None


def bind_geometry(query: str) -> tuple[bool, dict | None, str, int]:
    """Detect the formula, then extract each labeled dimension ('radius 5', 'length 4')."""
    det = detect_geometry(query)
    if not det:
        return False, None, "code", 0
    skill, dims = det
    if dims is None:  # hypotenuse: "legs A and B"
        m = _LEGS.search(query)
        if not m:
            return False, None, "code", 0
        return True, {"a": float(m[1]), "b": float(m[2]), "skill": skill,
                      "dims": ["a", "b"]}, "code", 0
    data = {"skill": skill, "dims": dims}
    for label in dims:
        m = re.search(rf"{label}\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", query, re.IGNORECASE)
        if not m:
            return False, None, "code", 0
        data[label] = float(m[1])
    return True, data, "code", 0


def validate_geometry(data: dict) -> tuple[bool, str]:
    for label in data["dims"]:
        if data[label] <= 0:
            return False, f"the {label} must be positive"
    return True, ""


# --- unit-conversion family -----------------------------------------------------------
# Unit knowledge lives in skills.units (the canonical factor table); the bind only locates
# the value and the source/target unit phrases and normalizes them to canonical tokens.
from reasoning.skills import units as _units  # pure skill module, safe to import here

_HOW_MANY = re.compile(
    r"how many\s+([a-z/ ]+?)\s+(?:are|is|in|=).*?(-?\d+(?:\.\d+)?)\s*([a-z/ ]+?)\b",
    re.IGNORECASE | re.DOTALL)


def _find_unit(text: str):
    """The canonical unit for the longest alias phrase present (whole-token), else None."""
    low = text.lower()
    for phrase in _units.UNIT_PHRASES:
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            return _units.canonical(phrase)
    return None


def bind_unit_convert(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind → {value, from, to, to_sym}. Two phrasings:
      - "<number> <source> to|into <target>"        (source before target)
      - "how many <target> ... <number> <source>"   (target before source)
    Conservative: needs a number and a KNOWN unit on each side, else it bails.
    """
    num = _NUMBER.search(query)
    if not num:
        return False, None, "code", 0
    value, source, target = float(num.group()), None, None

    # phrasing 2 first: "how many <target> ... <n> <source>" (so 'in'/inches is a unit,
    # not mistaken for a separator).
    m = _HOW_MANY.match(query.strip())
    if m:
        tgt, src = _find_unit(m.group(1)), _find_unit(m.group(3))
        if tgt and src:
            value, target, source = float(m.group(2)), tgt, src

    # phrasing 1: "<source> to|into <target>"
    if source is None:
        parts = re.split(r"\bto\b|\binto\b", query.lower(), maxsplit=1)
        if len(parts) == 2:
            s, t = _find_unit(parts[0]), _find_unit(parts[1])
            if s and t:
                source, target = s, t

    if source and target and source != target and _units_compatible(source, target):
        return True, {"value": value, "from": source, "to": target,
                      "to_sym": _units.SUFFIX.get(target, f" {target}")}, "code", 0
    # source == target is an identity "conversion" (no target requested) — e.g. a bare
    # "10 kilograms of butter to make cookies" splits on "to" with a mass unit on each side.
    # Echoing "10 kg" is a confident non-answer; decline so a unit word in prose isn't claimed.
    return False, None, "code", 0


def _units_compatible(a: str, b: str) -> bool:
    """Same dimension? A real conversion needs it — so a stray 'X to Y' across dimensions
    (kg→seconds in a physics query) is NOT claimed as a conversion (it would only decline)."""
    if a in _units._TEMP or b in _units._TEMP:
        return a in _units._TEMP and b in _units._TEMP
    fa, fb = _units._FACTORS.get(a), _units._FACTORS.get(b)
    return bool(fa and fb and fa[0] == fb[0])


# --- trigonometry / angle family ------------------------------------------------------
def _angle_value(query: str):
    m = re.search(r"-?\d+(?:\.\d+)?", query)
    if m:
        return float(m.group())
    if re.search(r"\bpi\b|π", query, re.IGNORECASE):
        return math.pi
    return None


def bind_trig(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    angle = _angle_value(query)
    if angle is None:
        return False, None, "code", 0
    if "cosine" in q or re.search(r"\bcos\b", q):  # check cosine before sine ("cosine" ⊃ "sine")
        skill, suffix, places = "trig.cos_deg", "", 2
    elif "sine" in q or re.search(r"\bsin\b", q):
        skill, suffix, places = "trig.sin_deg", "", 2
    elif "tangent" in q or re.search(r"\btan\b", q):
        skill, suffix, places = "trig.tan_deg", "", 2
    elif re.search(r"(?:to|in)\s+radians", q):
        skill, suffix, places = "trig.deg_to_rad", " rad", 4
    elif re.search(r"(?:to|in)\s+degrees", q):
        skill, suffix, places = "trig.rad_to_deg", "°", 2
    else:
        return False, None, "code", 0
    return True, {"skill": skill, "dims": ["angle"], "angle": angle, "suffix": suffix,
                  "places": places}, "code", 0


# --- coordinate-geometry family (_POINT imported from entities) -----------------------
def bind_coord(query: str) -> tuple[bool, dict | None, str, int]:
    pts = _POINT.findall(query)
    if len(pts) < 2:
        return False, None, "code", 0
    q = query.lower()
    if "distance" in q:
        skill, rnd = "coord.distance", True
    elif "midpoint" in q:
        skill, rnd = "coord.midpoint", False  # returns a string
    elif "slope" in q:
        skill, rnd = "coord.slope", True
    else:
        return False, None, "code", 0
    (x1, y1), (x2, y2) = pts[0], pts[1]
    return True, {"skill": skill, "dims": ["x1", "y1", "x2", "y2"], "round": rnd, "suffix": "",
                  "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}, "code", 0


# --- clock / time-of-day family -------------------------------------------------------
_CLOCK_TIME = re.compile(r"\d{1,2}:\d{2}(?:\s*(?:AM|PM))?", re.IGNORECASE)


def bind_clock(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    times = _CLOCK_TIME.findall(query)
    if "12-hour" in q or "12 hour" in q:
        if times:
            return True, {"skill": "clock.to_12h", "dims": ["time"], "round": False,
                          "time": times[0]}, "code", 0
    elif "24-hour" in q or "24 hour" in q:
        if times:
            return True, {"skill": "clock.to_24h", "dims": ["time"], "round": False,
                          "time": times[0]}, "code", 0
    elif "minutes between" in q or "minutes are between" in q:
        if len(times) >= 2:
            return True, {"skill": "clock.minutes_between", "dims": ["t1", "t2"], "round": False,
                          "suffix": " minutes", "t1": times[0], "t2": times[1]}, "code", 0
    elif "how long" in q:
        if len(times) >= 2:
            return True, {"skill": "clock.duration", "dims": ["t1", "t2"], "round": False,
                          "t1": times[0], "t2": times[1]}, "code", 0
    elif "after" in q or "before" in q:
        if times:
            mh = re.search(r"(\d+)\s*hours?", q)
            mm = re.search(r"(\d+)\s*minutes?", q)
            offset = (int(mh[1]) * 60 if mh else 0) + (int(mm[1]) if mm else 0)
            if "before" in q:
                offset = -offset
            return True, {"skill": "clock.add_minutes", "dims": ["time", "minutes"],
                          "round": False, "time": times[0], "minutes": offset}, "code", 0
    return False, None, "code", 0


# --- recognition bridge #3: geometry → cost ---------------------------------------------
# Per-AREA price → area → total_cost (tile/paint); per-LENGTH → perimeter → fence_cost
# (fencing); per-VOLUME → volume → fill_cost (concrete/mulch). The price KIND picks the goal.
_BY_DIMS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:meters?|metres?|m|feet|ft|cm|inches?|in)?\s*(?:by|x|×)\s*"
    r"(\d+(?:\.\d+)?)", re.IGNORECASE)
_PER_VOLUME = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/|a)\s*(?:cu(?:bic)?\.?\s*\w+|\w+\s*(?:³|\^3))", re.IGNORECASE)
_PER_AREA = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/|a)\s*(?:sq(?:uare|\.)?\s*\w+|\w+\s*(?:²|\^2))", re.IGNORECASE)
_PER_LENGTH = re.compile(
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:per|/|a)\s*(?:linear\s+)?"
    r"(?:meters?|metres?|m|feet|ft|yards?|yd|cm|inch(?:es)?|in)\b", re.IGNORECASE)
# a GOAL rate ("cost/price/how much PER <unit>") asks for a per-unit cost this orchestrator
# does not compute (it returns a TOTAL); decline rather than silently answer the total. The
# price's "per" is preceded by a $amount, so this adjacency only catches the asked rate.
_RATE_GOAL = re.compile(r"(?:cost|price|charge|much|rate)\s+per\b", re.IGNORECASE)


def shape_dims(query: str) -> dict:
    """Geometry dimensions in the query → {role: value}: length/width from 'A by B', plus any
    labeled radius/side/base/height. Shared by the geometry-cost, physics-pressure, and
    materials-stress binds so a SHAPE can supply a derivable `area`/`volume`."""
    dims = {}
    m = _BY_DIMS.search(query)
    if m:
        dims["length"], dims["width"] = float(m.group(1)), float(m.group(2))
    for label in ("radius", "side", "length", "width", "base", "height"):
        lm = re.search(rf"{label}\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", query, re.IGNORECASE)
        if lm:
            dims[label] = float(lm.group(1))
    return dims


def bind_synthesis_geometry(query: str) -> tuple[bool, dict | None, str, int]:
    if _RATE_GOAL.search(query):  # asks for cost-per-unit, not a total → not ours
        return False, None, "code", 0
    vol, area, length = (_PER_VOLUME.search(query), _PER_AREA.search(query),
                         _PER_LENGTH.search(query))
    if vol:      # most specific first ("per cubic meter" before "per meter")
        knowns, goal = {"volume_price": float(vol.group(1))}, "fill_cost"
    elif area:
        knowns, goal = {"unit_price": float(area.group(1))}, "total_cost"
    elif length:
        knowns, goal = {"length_price": float(length.group(1))}, "fence_cost"
    else:
        return False, None, "code", 0
    knowns.update(shape_dims(query))  # "4 meter by 6 meter" → rectangle; labeled radius/side/…
    # cross-domain tail: a geometry cost (total/fence/fill) that's then SPLIT among N people,
    # taxed, or discounted chains geometry → finance through the `cost` funnel. unique=True is
    # required so an ambiguous shape (two area derivations) declines rather than guesses.
    unique, suffix = False, ""
    q = query.lower()
    if goal in ("total_cost", "fence_cost", "fill_cost"):
        people = (_E.count_of(query, "people", "friends", "ways", "guests", "neighbors", "of them")
                  if any(w in q for w in ("split", "shared", "between", "among")) else None)
        tax = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:sales\s*)?(?:tax|vat)", q)
        disc = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*off", q)
        if people and people > 0:
            knowns["people"], goal, unique, suffix = float(people), "cost_per_person", True, " each"
        elif tax:
            knowns["tax_pct"], goal, unique = float(tax.group(1)), "cost_with_tax", True
        elif disc:
            knowns["discount_pct"], goal, unique = float(disc.group(1)), "cost_after_discount", True
    from reasoning.synthesize import CROSS_OPS, GEOMETRY_COST_OPS, synthesize
    pool = GEOMETRY_COST_OPS + CROSS_OPS
    seq = synthesize(set(knowns), goal, pool, unique=unique)
    if seq is None or len(seq) < 2:
        return False, None, "code", 0
    return True, {"knowns": knowns, "goal": goal, "prefix": "$", "suffix": suffix,
                  "unique": unique}, "code", 0


# multi-output synthesis: a query asking for ≥2 properties of one rectangle ("the area AND
# perimeter of a 4 by 6 rectangle") → a single plan producing both, shared sub-steps once.
_MULTI_PROPS = [("area", "area"), ("perimeter", "perimeter")]


def bind_geometry_multi(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    goals = [role for kw, role in _MULTI_PROPS if kw in q]
    if len(goals) < 2:  # single property → the `geometry` orchestrator handles it
        return False, None, "code", 0
    m = _BY_DIMS.search(query)
    if not m:
        return False, None, "code", 0
    knowns = {"length": float(m.group(1)), "width": float(m.group(2))}
    from reasoning.synthesize import GEOMETRY_COST_OPS, synthesize_multi
    if synthesize_multi(set(knowns), goals, GEOMETRY_COST_OPS, unique=True) is None:
        return False, None, "code", 0
    return True, {"knowns": knowns, "goals": goals}, "code", 0


def clarify_geometry(query: str) -> dict | None:
    det = detect_geometry(query)
    if not det or det[1] is None:  # unknown shape, or hypotenuse (legs phrasing) → no probe
        return None
    _, dims = det
    missing = {}
    for label in dims:
        if not re.search(rf"{label}\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", query, re.IGNORECASE):
            missing[label] = (f"What is the {label}?", f"{label} {{}}")
    return missing or None

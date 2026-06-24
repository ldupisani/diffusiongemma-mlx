"""Goal-directed plan synthesis — search the skill graph for the shortest sequence of
operations that derives a goal from the inputs you have, updating knowns as it goes.

Forward-chaining state-space search (BFS = fewest operations). Each `Op` pairs a skill with
a SEMANTIC ROLE signature (the roles it consumes, in arg order, and the role it produces).
Given typed knowns + a goal the planner discovers the operation path that orchestrators
otherwise hand-write — at ZERO model calls.

Generalization (vs the single-domain Phase-5 prototype): roles live on OPS, not skills — a
generic skill (`apply_tax`) plays different roles in different contexts (`final_price` vs
`taxed_price`), so one skill appears in several ops. Naming roles consistently *within* a
domain lets ops chain with no ontology/aliasing; different domains use disjoint role
vocabularies (price ≠ speed ≠ principal), so they never interfere even when searched as one
pool (`ALL_OPS`). Authoring an op library is manual (like authoring orchestrators); the win
is that the planner then COMPOSES multi-step plans automatically from atomic typed ops.

Still side-by-side: a planning substrate, not yet wired into the live router (that needs an
NL → {typed knowns, goal} bind — the recognition problem again).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import sympy

from .compiled import CodeStep, Inp, Lit, Plan, Step, format_number, run_plan
from .engine import _no_model


@dataclass(frozen=True)
class Op:
    """A skill typed by the semantic roles it consumes (in arg order) and produces."""
    skill: str
    needs: tuple
    produces: str


# Unit-aware roles: a role that carries a physical dimension declares its CANONICAL unit (the
# unit the skills compute in). A bind converts an extracted quantity to this unit before adding
# it to the knowns, so "5 km" / "20 minutes" feed the planner as 5000 m / 1200 s — the magnitude
# is correct regardless of how the query phrased it. Roles absent here are unitless/opaque.
ROLE_UNIT = {
    "distance": "m", "height": "m", "time": "s", "mass": "kg",
    "velocity": "mps", "speed": "mps", "v_initial": "mps", "v_final": "mps",
}


# --- domain op libraries --------------------------------------------------------------
# percentage finance: discounted_price (produced) chains into the next op's input role.
FINANCE_OPS = [
    Op("arith.apply_discount", ("price", "discount_pct"), "discounted_price"),
    Op("arith.apply_tax", ("discounted_price", "tax_pct"), "final_price"),
    Op("arith.divide", ("final_price", "people"), "per_person"),  # discount→tax→split
    Op("arith.multiply", ("unit_price", "quantity"), "price"),    # N items at $X each → subtotal
    Op("arith.apply_tax", ("price", "tax_pct"), "taxed_price"),
    Op("arith.percent_of", ("price", "discount_pct"), "discount_amount"),
]

# simple interest returns the maturity amount directly (P·(1+rt))
INTEREST_OPS = [
    Op("finance.simple_interest", ("principal", "rate_pct", "years"), "final_balance"),
]

# distance–speed–time triangle: any one from the other two (goal-directed op selection). The
# speed→velocity rename bridges MOTION → PHYSICS so a computed speed can feed kinetic energy /
# momentum (which consume the `velocity` role).
MOTION_OPS = [
    Op("physics.speed", ("distance", "time"), "speed"),
    Op("arith.multiply", ("speed", "time"), "distance"),
    Op("arith.divide", ("distance", "speed"), "time"),
    Op("arith.identity", ("speed",), "velocity"),
]

# mechanics chain: force → work → power (a 3-op derivation, no hand-written orchestrator)
PHYSICS_OPS = [
    Op("physics.force", ("mass", "acceleration"), "force"),
    Op("physics.work", ("force", "distance"), "work"),
    Op("physics.power", ("work", "time"), "power"),
    Op("physics.kinetic_energy", ("mass", "velocity"), "kinetic_energy"),
    Op("physics.momentum", ("mass", "velocity"), "momentum"),
    Op("physics.acceleration", ("v_initial", "v_final", "time"), "acceleration"),  # from Δv
    Op("physics.pressure", ("force", "area"), "pressure"),                          # force → pressure
]

# geometry → cost. Distinct price roles (unit_price / length_price / volume_price) and goals
# (total_cost / fence_cost / fill_cost) keep the goal unambiguous even when a rectangle's
# area AND perimeter are both derivable.
GEOMETRY_COST_OPS = [
    Op("geometry.rectangle_area", ("length", "width"), "area"),
    Op("geometry.circle_area", ("radius",), "area"),
    Op("geometry.square_area", ("side",), "area"),
    Op("geometry.triangle_area", ("base", "height"), "area"),
    Op("arith.multiply", ("area", "unit_price"), "total_cost"),     # tile / paint / carpet
    Op("geometry.rectangle_perimeter", ("length", "width"), "perimeter"),
    Op("arith.multiply", ("perimeter", "length_price"), "fence_cost"),  # fencing / edging
    Op("geometry.cylinder_volume", ("radius", "height"), "volume"),
    Op("geometry.cube_volume", ("side",), "volume"),
    Op("arith.multiply", ("volume", "volume_price"), "fill_cost"),  # fill / concrete / mulch
]

# chemistry: compute molar mass from a FORMULA (string role), then moles or mass
CHEM_OPS = [
    Op("chem.molar_mass", ("formula",), "molar_mass"),
    Op("chem.moles", ("mass", "molar_mass"), "moles"),          # moles = grams / molar_mass
    Op("arith.multiply", ("moles", "molar_mass"), "mass"),       # mass  = moles × molar_mass
    Op("chem.molarity", ("moles", "volume"), "molarity"),        # molarity = moles / liters
]

# molecular science: Avogadro (chemistry → molecular — moles feeds molecules)
MOLECULAR_OPS = [
    Op("molec.molecules_from_moles", ("moles",), "molecules"),
]

# material science: stress = force/area (physics force + geometry area → materials), then
# Young's modulus = stress/strain. Two physical-science domains feed one materials goal.
MATERIALS_OPS = [
    Op("materials.stress", ("force", "area"), "stress"),
    Op("materials.youngs_modulus", ("stress", "strain"), "youngs"),
]

# practical chemistry: ideal-gas pressure from moles (chemistry) + temperature + volume
PCHEM_OPS = [
    Op("pchem.ideal_gas_pressure", ("moles", "temp_k", "volume_l"), "gas_pressure"),
]

# priced quantities → cost (the universal "quantity × per-unit price" producers). Roles carry
# their unit by name (hours, power_kw, …) so the price's unit and the quantity's unit always
# match — no canonicalization needed. Each cost funnels into `cost` for split/tax/discount.
PRICED_OPS = [
    Op("arith.multiply", ("hours", "hourly_rate"), "labor_cost"),          # a job: time × $/hr
    Op("arith.multiply", ("power_kw", "hours"), "energy_kwh"),             # physics: kW × h = kWh
    Op("arith.multiply", ("energy_kwh", "price_per_kwh"), "energy_cost"),  # … → $ (per kWh)
    Op("arith.identity", ("labor_cost",), "cost"),
    Op("arith.identity", ("energy_cost",), "cost"),
]

# §F unit coercion as a planner-inserted op: chemistry's mass is in GRAMS, physics.weight wants
# KILOGRAMS — the g→kg op bridges the same-dimension roles so "the weight of N moles of <formula>"
# chains chem (mass) → units (mass_kg) → physics (weight) without silent magnitude corruption.
COERCION_OPS = [
    Op("units.g_to_kg", ("mass",), "mass_kg"),
    Op("physics.weight", ("mass_kg",), "weight"),
]

# geospatial → finance / motion. haversine yields a distance in KM; the downstream roles are
# km-native (price_per_km, speed_kmh) so units stay consistent without canonicalizing — the
# `geo_distance` role is distinct from motion's metres `distance`, so they never clash.
GEO_OPS = [
    Op("geo.haversine", ("la1", "lo1", "la2", "lo2"), "geo_distance"),  # km between two points
    Op("units.km_to_mi", ("geo_distance",), "geo_distance_mi"),         # … or in miles
    Op("arith.multiply", ("geo_distance", "price_per_km"), "travel_cost"),     # geo → $ (per km)
    Op("arith.multiply", ("geo_distance_mi", "price_per_mile"), "travel_cost"),  # geo → $ (per mi)
    Op("arith.divide", ("geo_distance", "speed_kmh"), "flight_hours"),   # geo → motion (km/h)
    Op("arith.divide", ("geo_distance_mi", "speed_mph"), "flight_hours"),  # geo → motion (mph)
    Op("arith.identity", ("travel_cost",), "cost"),
]

# cross-domain connectors. A cost produced by ANY domain funnels into a generic `cost` role,
# and the universal finance terminals (split / tax / discount) apply to `cost` ONCE. This is a
# deliberate puncture of the disjoint-role isolation at a chosen seam: a geometry/chemistry
# derivation can finish with a finance operation — e.g. "tile a floor … then split it 3 ways" =
# (area→total_cost→cost)→(/people), or "cost of 3 mol of NaCl at $0.5/g" = (formula→molar_mass
# →mass→reagent_cost→cost). The funnel scales: a new domain's cost is ONE identity row and it
# inherits split/tax/discount for free. Binds that use this MUST pass unique=True — an ambiguous
# goal (two equal-length derivations) then declines instead of silently picking one.
CROSS_OPS = [
    Op("arith.identity", ("total_cost",), "cost"),        # geometry: tile / paint / carpet
    Op("arith.identity", ("fence_cost",), "cost"),        # geometry: fencing / edging
    Op("arith.identity", ("fill_cost",), "cost"),         # geometry: concrete / mulch
    Op("arith.multiply", ("mass", "price_per_gram"), "reagent_cost"),  # chemistry → $ (per gram)
    Op("arith.identity", ("reagent_cost",), "cost"),
    Op("arith.divide", ("cost", "people"), "cost_per_person"),         # split
    Op("arith.apply_tax", ("cost", "tax_pct"), "cost_with_tax"),       # + tax
    Op("arith.apply_discount", ("cost", "discount_pct"), "cost_after_discount"),  # − discount
]

ALL_OPS = (FINANCE_OPS + INTEREST_OPS + MOTION_OPS + PHYSICS_OPS + GEOMETRY_COST_OPS
           + CHEM_OPS + MOLECULAR_OPS + MATERIALS_OPS + PCHEM_OPS + PRICED_OPS + GEO_OPS
           + COERCION_OPS + CROSS_OPS)

_MEMO: dict = {}
_searches = 0  # BFS runs actually executed (cache misses) — for tests/telemetry


def _applicable(op: Op, known: frozenset) -> bool:
    return op.produces not in known and all(n in known for n in op.needs)


def synthesize(knowns, goal: str, ops=None, *, max_depth: int = 8, memo: dict | None = _MEMO,
               unique: bool = False):
    """Shortest op sequence deriving `goal` from `knowns` (a set of role names), or None.
    Memoized by (frozenset(known roles), goal, unique). With `unique=True`, return None when
    the goal has more than one distinct shortest derivation (under-specified)."""
    ops = ALL_OPS if ops is None else ops
    key = (frozenset(knowns), goal, unique, id(ops))
    if memo is not None and key in memo:
        return memo[key]
    if unique:
        paths = shortest_paths(knowns, goal, ops, max_depth=max_depth)
        result = paths[0] if len(paths) == 1 else None
    else:
        result = _bfs(frozenset(knowns), goal, ops, max_depth)
    if memo is not None:
        memo[key] = result
    return result


def _bfs(start: frozenset, goal: str, ops, max_depth: int):
    global _searches
    _searches += 1
    if goal in start:
        return []
    seen = {start}
    queue = deque([(start, [])])
    while queue:
        known, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for op in ops:
            if not _applicable(op, known):
                continue
            npath = path + [op]
            if op.produces == goal:
                return npath           # BFS ⇒ this is a shortest path (fewest ops)
            nxt = known | {op.produces}
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, npath))
    return None


def shortest_paths(knowns, goal: str, ops=None, *, max_depth: int = 6) -> list:
    """ALL minimal-length op sequences deriving `goal` (for ambiguity analysis). Two distinct
    paths ⇒ the goal is under-specified given the knowns. Enumerated (tiny op graphs)."""
    ops = ALL_OPS if ops is None else ops
    if goal in set(knowns):
        return [[]]
    out: list = []

    def walk(known: frozenset, path: list, depth: int):
        if depth == 0:
            return
        for op in ops:
            if not _applicable(op, known):
                continue
            if op.produces == goal:
                out.append(path + [op])
            else:
                walk(known | {op.produces}, path + [op], depth - 1)

    walk(frozenset(knowns), [], max_depth)
    if not out:
        return []
    m = min(len(p) for p in out)
    return [p for p in out if len(p) == m]


def to_plan(ops, goal: str, *, places: int | None = 2) -> Plan:
    """Compile a synthesized op sequence into an executable Plan. A role is an Inp if it is an
    original known, else a Step (produced by an earlier op)."""
    produced: set = set()
    steps = []
    for op in ops:
        args = [Step(r) if r in produced else Inp(r) for r in op.needs]
        steps.append(CodeStep(op.skill, args, out=op.produces))
        produced.add(op.produces)
    ref = goal
    if places is not None:
        steps.append(CodeStep("arith.round_to", [Step(goal), Lit(places)], out="__rounded"))
        ref = "__rounded"
    return Plan(steps=steps, answer_ref=ref, answer_format=format_number)


def synthesize_and_run(knowns_values: dict, goal: str, ops=None, *, places: int | None = 2,
                       unique: bool = False):
    """Synthesize a plan from typed knowns → goal and execute it (0 model calls), or None."""
    seq = synthesize(set(knowns_values), goal, ops, unique=unique)
    if seq is None:
        return None
    return run_plan(to_plan(seq, goal, places=places), knowns_values, shot=_no_model)


# Symbolic mirror of a skill's formula (args in the op's `needs` order), used ONLY for inverse
# synthesis: solving for an input given the output. A skill without a mirror (e.g. a table
# lookup like chem.molar_mass) can't be inverted, so a chain through it declines. The mirrors
# must match the skills (verified by a test) — they re-express the same arithmetic in SymPy.
SYMBOLIC = {
    "arith.identity": lambda x: x,
    "arith.multiply": lambda a, b: a * b,
    "arith.divide": lambda a, b: a / b,
    "arith.percent_of": lambda v, pct: v * pct / 100,
    "arith.apply_discount": lambda price, pct: price * (1 - pct / 100),
    "arith.apply_tax": lambda amt, pct: amt * (1 + pct / 100),
    "physics.force": lambda m, a: m * a,
    "physics.work": lambda f, d: f * d,
    "physics.power": lambda w, t: w / t,
    "physics.kinetic_energy": lambda m, v: sympy.Rational(1, 2) * m * v ** 2,
    "physics.momentum": lambda m, v: m * v,
    "geometry.rectangle_area": lambda length, width: length * width,
    "geometry.rectangle_perimeter": lambda length, width: 2 * (length + width),
    "geometry.square_area": lambda side: side ** 2,
    "geometry.circle_area": lambda radius: sympy.pi * radius ** 2,
    "geometry.triangle_area": lambda base, height: sympy.Rational(1, 2) * base * height,
}


def inverse_synthesize(knowns_values: dict, unknown: str, goal: str, target, ops=None, *,
                       unique: bool = True):
    """Solve for the value of input role `unknown` so the forward chain yields goal == target.

    The inverse of forward synthesis: you know the OUTPUT and all inputs but one. We forward-
    synthesize `goal` treating `unknown` as known (to discover the op chain), mirror that chain
    symbolically (the unknown a free symbol, the other knowns their values), then
    sympy.solve(chain == target). Returns the solution (a non-negative real preferred, since
    these roles are magnitudes/percentages), or None if there's no path, a non-symbolic op
    blocks it, or there's no real solution. 0 model calls."""
    ops = ALL_OPS if ops is None else ops
    seq = synthesize(set(knowns_values) | {unknown}, goal, ops, unique=unique)
    if seq is None:
        return None
    x = sympy.Symbol(unknown, real=True)
    # numeric knowns → exact rationals; non-numeric (e.g. a formula string) stays as-is and will
    # hit a mirror-less op (table lookup) → decline, before it's used in any arithmetic
    env = {r: (sympy.Rational(str(v)) if isinstance(v, (int, float)) else v)
           for r, v in knowns_values.items()}
    env[unknown] = x
    for op in seq:
        mirror = SYMBOLIC.get(op.skill)
        if mirror is None:
            return None  # an op in the path has no symbolic form → can't invert
        env[op.produces] = mirror(*[env[r] for r in op.needs])
    try:
        sols = sympy.solve(sympy.Eq(env[goal], sympy.Rational(str(target))), x)
    except (NotImplementedError, RecursionError):
        return None
    reals = [s for s in sols if s.is_real]
    nonneg = [s for s in reals if s.is_nonnegative]
    pick = (nonneg or reals)
    return float(pick[0]) if pick else None


def synthesize_multi(knowns, goals, ops=None, *, max_depth: int = 8, unique: bool = False):
    """One op sequence producing SEVERAL goals from shared knowns, or None if any is unreachable.
    Goals are derived in order; each reuses roles already produced for an earlier goal, so a
    shared sub-step (e.g. `area`, used by both `perimeter`-free and `total_cost`) is computed
    ONCE. Returns the merged ops in dependency order (deduped by produced role)."""
    ops = ALL_OPS if ops is None else ops
    produced = set(knowns)
    merged, seen = [], set(knowns)
    for goal in goals:
        seq = synthesize(produced, goal, ops, max_depth=max_depth, unique=unique)
        if seq is None:
            return None
        for op in seq:
            if op.produces not in seen:
                merged.append(op)
                seen.add(op.produces)
                produced.add(op.produces)
        produced.add(goal)  # an earlier goal is a known for the next
    return merged


def synthesize_multi_and_run(knowns_values: dict, goals, ops=None, *, places: int = 2,
                             unique: bool = False):
    """Synthesize and execute a multi-goal plan (0 model calls). Returns (result, {goal: value})
    with each goal rounded to `places`, or None if any goal is unreachable."""
    seq = synthesize_multi(set(knowns_values), goals, ops, unique=unique)
    if seq is None:
        return None
    produced: set = set()
    steps = []
    for op in seq:
        args = [Step(r) if r in produced else Inp(r) for r in op.needs]
        steps.append(CodeStep(op.skill, args, out=op.produces))
        produced.add(op.produces)
    # answer_ref is arbitrary (we read every goal from the env); first goal keeps .answer sane
    r = run_plan(Plan(steps=steps, answer_ref=goals[0], answer_format=format_number),
                 knowns_values, shot=_no_model)
    values = {g: round(r.data[g], places) for g in goals}
    return r, values

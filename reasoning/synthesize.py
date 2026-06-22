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

from .compiled import CodeStep, Inp, Lit, Plan, Step, format_number, run_plan
from .engine import _no_model


@dataclass(frozen=True)
class Op:
    """A skill typed by the semantic roles it consumes (in arg order) and produces."""
    skill: str
    needs: tuple
    produces: str


# --- domain op libraries --------------------------------------------------------------
# percentage finance: discounted_price (produced) chains into the next op's input role.
FINANCE_OPS = [
    Op("arith.apply_discount", ("price", "discount_pct"), "discounted_price"),
    Op("arith.apply_tax", ("discounted_price", "tax_pct"), "final_price"),
    Op("arith.divide", ("final_price", "people"), "per_person"),  # discount→tax→split
    Op("arith.apply_tax", ("price", "tax_pct"), "taxed_price"),
    Op("arith.percent_of", ("price", "discount_pct"), "discount_amount"),
]

# simple interest returns the maturity amount directly (P·(1+rt))
INTEREST_OPS = [
    Op("finance.simple_interest", ("principal", "rate_pct", "years"), "final_balance"),
]

# distance–speed–time triangle: any one from the other two (goal-directed op selection)
MOTION_OPS = [
    Op("physics.speed", ("distance", "time"), "speed"),
    Op("arith.multiply", ("speed", "time"), "distance"),
    Op("arith.divide", ("distance", "speed"), "time"),
]

# mechanics chain: force → work → power (a 3-op derivation, no hand-written orchestrator)
PHYSICS_OPS = [
    Op("physics.force", ("mass", "acceleration"), "force"),
    Op("physics.work", ("force", "distance"), "work"),
    Op("physics.power", ("work", "time"), "power"),
    Op("physics.kinetic_energy", ("mass", "velocity"), "kinetic_energy"),
    Op("physics.momentum", ("mass", "velocity"), "momentum"),
]

# geometry → cost: a surface's area × a unit price (tile / paint / carpet)
GEOMETRY_COST_OPS = [
    Op("geometry.rectangle_area", ("length", "width"), "area"),
    Op("geometry.circle_area", ("radius",), "area"),
    Op("geometry.square_area", ("side",), "area"),
    Op("geometry.triangle_area", ("base", "height"), "area"),
    Op("arith.multiply", ("area", "unit_price"), "total_cost"),
]

ALL_OPS = FINANCE_OPS + INTEREST_OPS + MOTION_OPS + PHYSICS_OPS + GEOMETRY_COST_OPS

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

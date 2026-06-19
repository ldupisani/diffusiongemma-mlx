"""Goal-directed plan synthesis — search the skill graph for the shortest sequence of
operations that derives a goal from the inputs you have, updating knowns as it goes.

This is forward-chaining state-space search (BFS = fewest operations). Each `Op` is a skill
typed by the semantic ROLES it consumes and produces; the planner unifies a produced role
with a later op's needed role. Given typed knowns + a goal it discovers the operation path
that today's orchestrators hand-write — e.g. for {price, discount_pct, tax_pct} → final_price
it re-derives `percent_discount_tax`'s discount→tax sequence with ZERO model calls.

Status: a working prototype proven on the finance/percent domain. Generalizing to the whole
library needs every skill annotated with roles (the crux/risk noted in the plan); kept here
side-by-side, NOT wired into the live router, until eval proves parity.
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


# Proof domain: percentage finance. `discounted_price` produced by the first op unifies with
# the `amount` the second op needs — that role-matching is what makes the search directed.
FINANCE_OPS = [
    Op("arith.apply_discount", ("price", "discount_pct"), "discounted_price"),
    Op("arith.apply_tax", ("discounted_price", "tax_pct"), "final_price"),
    Op("arith.apply_tax", ("price", "tax_pct"), "taxed_price"),
    Op("arith.percent_of", ("price", "discount_pct"), "discount_amount"),
]

_MEMO: dict = {}
_searches = 0  # how many BFS runs actually executed (cache misses) — for tests/telemetry


def synthesize(knowns, goal: str, ops, *, max_depth: int = 8, memo: dict | None = _MEMO):
    """Shortest op sequence deriving `goal` from `knowns` (a set of role names), or None.
    Memoized by (frozenset(known roles), goal) — a query type is searched once, then reused."""
    key = (frozenset(knowns), goal)
    if memo is not None and key in memo:
        return memo[key]
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
            if op.produces in known or not all(n in known for n in op.needs):
                continue
            npath = path + [op]
            if op.produces == goal:
                return npath           # BFS ⇒ this is a shortest path (fewest ops)
            nxt = known | {op.produces}
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, npath))
    return None


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


def synthesize_and_run(knowns_values: dict, goal: str, ops, *, places: int | None = 2):
    """Synthesize a plan from typed knowns → goal and execute it (0 model calls), or None."""
    seq = synthesize(set(knowns_values), goal, ops)
    if seq is None:
        return None
    return run_plan(to_plan(seq, goal, places=places), knowns_values, shot=_no_model)

"""bind layer — supply chain optimization. Gated on distinctive supply-chain phrases
(total annual cost, EPQ, newsvendor, capacity utilization, takt time); arguments are
pulled by keyword-anchored number extraction. Conservative gates to avoid stealing from
logistics (which has EOQ)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"(-?\d+(?:\.\d+)?)"


def _after(q: str, *keywords) -> float | None:
    """First number appearing after any of the keyword patterns."""
    for kw in keywords:
        m = re.search(kw + r"[^0-9-]*" + _NUM, q, re.IGNORECASE)
        if m:
            return float(m.group(m.lastindex))
    return None


def _before(q: str, *units) -> float | None:
    """A number appearing just before a unit word, e.g. '480 minutes' / '60 customer units'."""
    for u in units:
        m = re.search(_NUM + r"\s*" + u, q, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def bind_supply_chain(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # Total annual inventory cost: distinctive phrase "total annual" + "cost"
    if ("total annual" in q or "total inventory cost" in q) and "cost" in q:
        return ok("supply_chain.total_annual_cost",
                  ["demand", "order_quantity", "order_cost", "holding_cost"],
                  [_after(q, "demand", "annual demand"),
                   _after(q, "order quantity", "order qty", "q"),
                   _after(q, "order cost", "setup cost", "ordering cost"),
                   _after(q, "holding cost", "carrying cost", "h")],
                  round=True, places=6)

    # EPQ: distinctive "EPQ" or "economic production quantity" (not just "economic")
    if ("epq" in q or "economic production quantity" in q) and ("production rate" in q or "demand rate" in q):
        return ok("supply_chain.epq",
                  ["demand", "order_cost", "holding_cost", "production_rate", "demand_rate"],
                  [_after(q, "demand", "annual demand"),
                   _after(q, "order cost", "setup cost", "ordering cost"),
                   _after(q, "holding cost", "carrying cost"),
                   _after(q, "production rate", "production capacity", "p"),
                   _after(q, "demand rate", "d")],
                  round=True, places=6)

    # Newsvendor critical ratio: "newsvendor" + ratio language
    if ("newsvendor" in q or "critical ratio" in q) and ("shortage cost" in q or "excess cost" in q or "underage cost" in q or "overage cost" in q):
        return ok("supply_chain.newsvendor_critical_ratio",
                  ["underage_cost", "overage_cost"],
                  [_after(q, "shortage cost", "underage cost", "stockout cost", "cu"),
                   _after(q, "excess cost", "overage cost", "holding cost", "co")],
                  round=True, places=6)

    # Capacity utilization: "capacity utilization" or "capacity" + "utilization"
    if ("capacity utilization" in q or ("capacity" in q and "utilization" in q)) and "demand" in q:
        # anchor capacity on an occurrence NOT followed by "utiliz…" (else "capacity utilization"
        # swallows the demand value and the real capacity is left as residue).
        cap = re.search(r"capacity(?!\s*utiliz)[^0-9-]*" + _NUM, q, re.IGNORECASE)
        return ok("supply_chain.capacity_utilization",
                  ["demand", "capacity"],
                  [_after(q, "demand", "expected demand", "units"),
                   float(cap.group(1)) if cap else _after(q, "max capacity", "maximum")],
                  round=True, places=6, suffix="%")

    # Takt time: "takt time" or "takt" (very specific to manufacturing). The available time and the
    # demand are usually written as "<n> minutes" / "<n> customer units" — number BEFORE the anchor —
    # so try _before first, then fall back to label-anchored _after ("available time 480").
    if "takt time" in q or ("takt" in q and ("time" in q or "minute" in q or "hour" in q or "second" in q)):
        return ok("supply_chain.takt_time",
                  ["available_time", "customer_demand"],
                  [_before(q, "minute", "hour", "second", "min", "hr")
                   or _after(q, "available time", "available", "total time"),
                   _before(q, "customer", "unit")
                   or _after(q, "customer demand", "demand")],
                  round=True, places=6, suffix=" min/unit")

    return False, None, "code", 0
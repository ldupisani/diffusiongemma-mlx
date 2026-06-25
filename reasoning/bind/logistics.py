"""bind layer — logistics / operations research. Gated on a distinctive logistics phrase
(EOQ, reorder point, safety stock, inventory turnover, days of supply, Little's Law, trucks
needed, fuel cost); arguments are pulled by keyword-anchored number extraction."""
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


def _before_unit(q: str, units: str) -> float | None:
    m = re.search(_NUM + r"\s*(?:" + units + r")\b", q, re.IGNORECASE)
    return float(m[1]) if m else None


def bind_logistics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "eoq" in q or "economic order quantity" in q:
        return ok("logistics.eoq", ["demand", "order_cost", "holding_cost"],
                  [_after(q, "demand"), _after(q, "order(?:ing)? cost", "setup cost"),
                   _after(q, "holding cost", "carrying cost")], round=True, places=6)

    if "reorder point" in q or "re-order point" in q:
        return ok("logistics.reorder_point", ["demand_rate", "lead_time", "safety_stock"],
                  [_after(q, "demand", "usage"), _after(q, "lead time"),
                   _after(q, "safety stock") or 0], round=True, places=6)

    if "safety stock" in q:
        return ok("logistics.safety_stock", ["z", "demand_std", "lead_time"],
                  [_after(q, r"\bz\b", "service factor"),
                   _after(q, "standard deviation", "std(?:ev)?", "sigma", "σ"),
                   _after(q, "lead time")], round=True, places=6)

    if "inventory turnover" in q or "turnover ratio" in q:
        return ok("logistics.inventory_turnover", ["cogs", "avg_inventory"],
                  [_after(q, "cogs", "cost of goods sold"),
                   _after(q, "average inventory", "avg inventory", "mean inventory")],
                  round=True, places=6)

    if "days of supply" in q or "days of inventory" in q:
        return ok("logistics.days_of_supply", ["inventory", "daily_demand"],
                  [_after(q, "inventory", "on hand", "stock"),
                   _after(q, "daily demand", "demand", "usage")], round=True, places=6)

    if "little" in q and "law" in q:
        return ok("logistics.littles_law", ["throughput", "flow_time"],
                  [_after(q, "throughput", "arrival rate"),
                   _after(q, "flow time", "cycle time", "time in system")], round=True, places=6)

    if ("truck" in q or "container" in q or "vehicle" in q) and ("how many" in q or "need" in q):
        return ok("logistics.trucks_needed", ["units", "capacity"],
                  [_after(q, "ship", "deliver", "move", "transport", "units", "items", "pallets"),
                   _after(q, "capacity", "holds", "each", "per truck", "per container", "per vehicle")],
                  round=False)

    if "fuel cost" in q or "fuel for" in q:
        return ok("logistics.fuel_cost", ["distance", "fuel_efficiency", "fuel_price"],
                  [_before_unit(q, "km|kilometers|miles|mi"),
                   _before_unit(q, r"km/l|mpg|miles per gallon|kilometers per liter|km per liter"),
                   _after(q, r"\$", "price", "per liter", "per gallon", "per gal")], round=True, places=6)

    return False, None, "code", 0

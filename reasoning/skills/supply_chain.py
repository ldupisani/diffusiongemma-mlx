"""Supply chain optimization — complementary to logistics (EOQ already in logistics).
Implements: total annual inventory cost, Economic Production Quantity, newsvendor critical ratio,
capacity utilization, and takt time. Zero model calls."""

from __future__ import annotations

import math

from .base import register


@register("supply_chain.total_annual_cost",
          "total annual inventory cost TC = (D/Q)·S + (Q/2)·H",
          [((1000, 100, 10, 2), 200.0),
           ((5000, 200, 50, 5), 1750.0)])
def total_annual_cost(demand: float, order_quantity: float, order_cost: float,
                      holding_cost: float) -> float:
    """Total annual inventory cost: ordering + holding."""
    return round((demand / order_quantity) * order_cost + (order_quantity / 2) * holding_cost, 6)


@register("supply_chain.epq",
          "Economic Production Quantity EPQ = sqrt(2·D·S/H) · sqrt(p/(p−d))",
          [((5000, 50, 5, 1000, 200), 353.553391),
           ((10000, 100, 4, 2000, 500), 816.496581)])
def epq(demand: float, order_cost: float, holding_cost: float,
        production_rate: float, demand_rate: float) -> float:
    """Economic Production Quantity: balances setup cost and holding cost under constrained
    production rate p and demand rate d (p > d required)."""
    if production_rate <= demand_rate:
        raise ValueError("production_rate must exceed demand_rate")
    return round(math.sqrt(2 * demand * order_cost / holding_cost) *
                 math.sqrt(production_rate / (production_rate - demand_rate)), 6)


@register("supply_chain.newsvendor_critical_ratio",
          "newsvendor critical ratio CR = Cu / (Cu + Co)",
          [((20, 10), 0.666667),
           ((50, 50), 0.5)])
def newsvendor_critical_ratio(underage_cost: float, overage_cost: float) -> float:
    """Critical ratio for the newsvendor problem: optimal order-up-to level as a percentile
    of the demand distribution."""
    return round(underage_cost / (underage_cost + overage_cost), 6)


@register("supply_chain.capacity_utilization",
          "capacity utilization = (demand / capacity) × 100%",
          [((800, 1000), 80.0),
           ((2000, 2500), 80.0)])
def capacity_utilization(demand: float, capacity: float) -> float:
    """Capacity utilization as a percentage (0–100)."""
    return round((demand / capacity) * 100, 6)


@register("supply_chain.takt_time",
          "takt time = available production time / customer demand (time per unit)",
          [((480, 60), 8.0),
           ((600, 50), 12.0)])
def takt_time(available_time: float, customer_demand: float) -> float:
    """Takt time: the time available divided by units demanded. Synchronizes production
    pace to customer demand."""
    return round(available_time / customer_demand, 6)
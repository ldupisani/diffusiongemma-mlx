"""Logistics — inventory, operations-research, and transportation formulas. Zero model calls.

Covers the classic computable pieces: Economic Order Quantity, reorder point, safety stock,
inventory turnover and days of supply, Little's Law, trucks/containers needed (ceiling pack),
and trip fuel cost.
"""

from __future__ import annotations

import math

from .base import register


@register("logistics.eoq", "Economic Order Quantity sqrt(2·D·S / H)",
          [((1000, 10, 2), 100.0), ((2400, 50, 1), 489.897949)])
def eoq(demand: float, order_cost: float, holding_cost: float) -> float:
    return round(math.sqrt(2 * demand * order_cost / holding_cost), 6)


@register("logistics.reorder_point", "reorder point = demand rate · lead time + safety stock",
          [((50, 5, 0), 250.0), ((50, 5, 30), 280.0)])
def reorder_point(demand_rate: float, lead_time: float, safety_stock: float = 0) -> float:
    return round(demand_rate * lead_time + safety_stock, 6)


@register("logistics.safety_stock", "safety stock = z · σ · sqrt(lead time)",
          [((1.65, 20, 4), 66.0), ((2.33, 10, 9), 69.9)])
def safety_stock(z: float, demand_std: float, lead_time: float) -> float:
    return round(z * demand_std * math.sqrt(lead_time), 6)


@register("logistics.inventory_turnover", "inventory turnover = COGS / average inventory",
          [((500000, 100000), 5.0), ((1200000, 150000), 8.0)])
def inventory_turnover(cogs: float, avg_inventory: float) -> float:
    return round(cogs / avg_inventory, 6)


@register("logistics.days_of_supply", "days of supply = inventory on hand / daily demand",
          [((500, 50), 10.0), ((1200, 80), 15.0)])
def days_of_supply(inventory: float, daily_demand: float) -> float:
    return round(inventory / daily_demand, 6)


@register("logistics.littles_law", "Little's Law inventory = throughput · flow time",
          [((10, 5), 50.0), ((2.5, 4), 10.0)])
def littles_law(throughput: float, flow_time: float) -> float:
    return round(throughput * flow_time, 6)


@register("logistics.trucks_needed", "vehicles needed = ceil(units / capacity)",
          [((1000, 150), 7), ((600, 200), 3), ((300, 100), 3)])
def trucks_needed(units: int, capacity: int) -> int:
    return math.ceil(units / capacity)


@register("logistics.fuel_cost", "trip fuel cost = distance / fuel efficiency · fuel price",
          [((600, 12, 1.5), 75.0), ((300, 30, 4.0), 40.0)])
def fuel_cost(distance: float, fuel_efficiency: float, fuel_price: float) -> float:
    return round(distance / fuel_efficiency * fuel_price, 6)

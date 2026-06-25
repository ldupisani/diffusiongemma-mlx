"""Heat transfer — conduction, convection, radiation, thermal resistance, and heat flux.
Computes deterministic heat-transfer rates and thermal properties. Zero model calls."""

from __future__ import annotations

from .base import register


@register("heat_transfer.conduction_rate",
          "conduction heat rate Q = k·A·ΔT/L (W), k in W/m·K, A in m², ΔT in K, L in m",
          [((50, 2, 20, 0.1), 20000.0),
           ((0.5, 1, 10, 0.05), 100.0)])
def conduction_rate(k: float, area: float, delta_t: float, thickness: float) -> float:
    """Fourier's law: heat transfer by conduction through a material."""
    return round(k * area * delta_t / thickness, 6)


@register("heat_transfer.convection_rate",
          "convection heat rate Q = h·A·ΔT (W), h in W/m²·K, A in m², ΔT in K",
          [((25, 2, 30), 1500.0),
           ((100, 0.5, 40), 2000.0)])
def convection_rate(h: float, area: float, delta_t: float) -> float:
    """Newton's law of cooling: heat transfer by convection from a surface."""
    return round(h * area * delta_t, 6)


@register("heat_transfer.radiation_rate",
          "radiation heat rate Q = ε·σ·A·T⁴ (W), σ = 5.67e-8 W/m²·K⁴, T in Kelvin",
          [((0.9, 1, 300), 413.343),
           ((0.5, 2, 400), 1451.52)])
def radiation_rate(emissivity: float, area: float, temperature: float) -> float:
    """Stefan–Boltzmann law: heat transfer by thermal radiation. Temperature in Kelvin."""
    sigma = 5.67e-8
    return round(emissivity * sigma * area * (temperature ** 4), 6)


@register("heat_transfer.thermal_resistance",
          "thermal resistance R = L/(k·A) (K/W), L in m, k in W/m·K, A in m²",
          [((0.1, 50, 2), 0.001),
           ((0.05, 0.5, 1), 0.1)])
def thermal_resistance(thickness: float, k: float, area: float) -> float:
    """Thermal resistance for conductive heat transfer through a material layer."""
    return round(thickness / (k * area), 6)


@register("heat_transfer.heat_flux",
          "heat flux q = Q/A (W/m²), heat transfer rate per unit area",
          [((20000, 2), 10000.0),
           ((1500, 0.5), 3000.0)])
def heat_flux(heat_rate: float, area: float) -> float:
    """Heat transfer rate per unit area."""
    return round(heat_rate / area, 6)
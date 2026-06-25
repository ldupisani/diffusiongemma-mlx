"""Bind layer for heat transfer. Gate on heat-transfer context; extract using num_near."""

from __future__ import annotations

from reasoning.bind._common import *  # noqa: F401,F403


def bind_heat_transfer(query: str) -> tuple[bool, dict | None, str, int]:
    """Route heat transfer queries: conduction, convection, radiation, thermal resistance, heat flux.
    Decline thermodynamics sensible heat (specific heat + mass) and entropy."""
    q = query.lower()

    # Decline thermodynamics: sensible heat Q = m·c·ΔT
    if ("specific heat" in q or "heat capacity" in q) and "mass" in q:
        return False, None, "code", 0
    # Decline entropy queries (thermodynamic state)
    if "entropy" in q and ("thermodynamic" in q or "change" in q):
        return False, None, "code", 0

    # Top-level context gate: at least one heat-transfer keyword.
    # Every op's distinctive keyword must be here (§3f gate calibration).
    heat_context = {
        "conduction", "convection", "radiation",
        "thermal conductivity", "thermal resistance",
        "heat transfer coefficient", "emissivity",
        "heat flux", "stefan", "boltzmann"
    }
    if not any(kw in q for kw in heat_context):
        return False, None, "code", 0

    def ok(skill, dims, vals, **extra):
        """Package extracted arguments; decline if any value is missing."""
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # Op 1: Conduction — gate on thermal conductivity + thickness
    if ("conduction" in q or "thermal conductivity" in q) and any(kw in q for kw in ["thickness", "depth", "length"]):
        k = num_near(q, "thermal conductivity", "conductivity", "k")
        area = num_near(q, "area")
        delta_t = num_near(q, "temperature difference", "temp difference", "delta", "temperature", "ΔT", "Δ")
        thickness = num_near(q, "thickness", "depth", "length", "l")
        return ok("heat_transfer.conduction_rate", ["k", "area", "delta_t", "thickness"],
                  [k, area, delta_t, thickness], round=True, places=6)

    # Op 2: Convection — gate on convection/heat transfer coefficient
    if "convection" in q or "heat transfer coefficient" in q:
        h = num_near(q, "heat transfer coefficient", "convection coefficient", "h")
        area = num_near(q, "area")
        delta_t = num_near(q, "temperature difference", "temp difference", "delta", "temperature", "ΔT", "Δ")
        return ok("heat_transfer.convection_rate", ["h", "area", "delta_t"],
                  [h, area, delta_t], round=True, places=6)

    # Op 3: Radiation — gate on radiation/emissivity/stefan/boltzmann
    if any(kw in q for kw in ["radiation", "emissivity", "stefan", "boltzmann"]):
        emissivity = num_near(q, "emissivity", "epsilon")
        area = num_near(q, "area")
        temperature = num_near(q, "absolute temperature", "temperature", "temp")
        return ok("heat_transfer.radiation_rate", ["emissivity", "area", "temperature"],
                  [emissivity, area, temperature], round=True, places=6)

    # Op 4: Thermal resistance — gate on thermal resistance (decline electrical resistance)
    if "thermal resistance" in q or ("resistance" in q and "thermal" in q):
        if "electrical" in q or "ohm" in q or "circuit" in q:
            return False, None, "code", 0
        thickness = num_near(q, "thickness", "length", "depth", "l")
        k = num_near(q, "thermal conductivity", "conductivity", "k")
        area = num_near(q, "area")
        return ok("heat_transfer.thermal_resistance", ["thickness", "k", "area"],
                  [thickness, k, area], round=True, places=6)

    # Op 5: Heat flux — gate on heat flux/q
    if "heat flux" in q or ("flux" in q and "heat" in q):
        heat_rate = num_near(q, "heat rate", "heat transfer", "thermal power", "q")
        area = num_near(q, "area")
        return ok("heat_transfer.heat_flux", ["heat_rate", "area"],
                  [heat_rate, area], round=True, places=6)

    return False, None, "code", 0
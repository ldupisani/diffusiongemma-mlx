"""Bind layer — special relativity. Conservative gate on relativity-specific keywords and speed.
Recognizes Lorentz factor, time dilation, length contraction, and relativistic kinetic energy.
"""

from __future__ import annotations

import re
import math

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"(-?\d+(?:\.\d+)?)"


def bind_special_relativity(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    # Gate: require a relativity keyword
    if not any(kw in q for kw in ["lorentz", "gamma", "dilation", "contraction", "relativistic", "speed of light"]):
        return False, None, "code", 0

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        data = {"skill": skill, "dims": dims, **dict(zip(dims, vals)), "round": True, "places": 6}
        data.update(extra)
        return True, data, "code", 0

    # Extract speed fraction (e.g., 0.8c, c/2)
    speed = _extract_speed(q)

    # Lorentz factor: recognize "lorentz" or "gamma"
    if any(kw in q for kw in ["lorentz factor", "lorentz", "gamma"]):
        if speed is None:
            return False, None, "code", 0
        return ok("special_relativity.lorentz_factor", ["speed_fraction"], [speed])

    # Compute gamma for other operations
    gamma = None
    if speed is not None:
        try:
            gamma = 1 / math.sqrt(1 - speed**2)
        except (ValueError, ZeroDivisionError):
            return False, None, "code", 0

    # Time dilation: "dilation" + time context
    if "dilation" in q and any(kw in q for kw in ["time", "second", "minute", "hour", "clock"]):
        time_val = _extract_number_after(q, ["time", "second", "minute", "hour"])
        if gamma is None or time_val is None:
            return False, None, "code", 0
        return ok("special_relativity.time_dilation", ["proper_time", "gamma"], [time_val, gamma])

    # Length contraction: "contraction" + length context
    if "contraction" in q and any(kw in q for kw in ["length", "distance", "meter", "m", "km", "mile"]):
        length_val = _extract_number_after(q, ["length", "distance", "meter"])
        if gamma is None or length_val is None:
            return False, None, "code", 0
        return ok("special_relativity.length_contraction", ["proper_length", "gamma"], [length_val, gamma])

    # Relativistic kinetic energy: "energy" + mass context. The mass appears as "<n> kg"
    # (number BEFORE the unit) or "mass of <n>"; _extract_number_after would wrongly grab the
    # speed that follows the unit ("0.000001 kg at 0.6c"), so match the number before kg first.
    if any(kw in q for kw in ["kinetic energy", "rest energy", "relativistic energy"]):
        m_mass = (re.search(_NUM + r"\s*(?:kg|kilograms?)\b", q)
                  or re.search(r"mass[^0-9-]*" + _NUM, q))
        mass_val = float(m_mass.group(1)) if m_mass else None
        if gamma is None or mass_val is None:
            return False, None, "code", 0
        return ok("special_relativity.relativistic_kinetic_energy", ["mass_kg", "gamma"], [mass_val, gamma])

    return False, None, "code", 0


def _extract_speed(q: str) -> float | None:
    """Extract speed as a fraction of c (e.g., 0.8c, c/2)."""
    # Pattern: 0.8c, 0.6c, etc.
    m = re.search(r'(\d+(?:\.\d+)?)\s*c\b', q)
    if m:
        val = float(m.group(1))
        if 0 <= val < 1:
            return val

    # Pattern: c/2, c/3, etc.
    m = re.search(r'c\s*/\s*(\d+)', q)
    if m:
        divisor = float(m.group(1))
        if divisor > 0:
            return 1.0 / divisor

    return None


def _extract_number_after(q: str, keywords: list[str]) -> float | None:
    """Extract the first number appearing after any of the keywords."""
    for kw in keywords:
        m = re.search(kw + r"[^0-9-]*" + _NUM, q, re.IGNORECASE)
        if m:
            return float(m.group(m.lastindex))
    return None

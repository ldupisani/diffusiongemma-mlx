"""Photometry and illumination engineering. Gated strictly on photometric keywords;
distinguishes illuminance ops (flux+area vs intensity+distance), luminous efficacy,
intensity, and inverse-square ratio."""

from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403


def bind_photometry(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    
    photometric_keywords = r'\b(?:lux|lumen|lumens|candela|illuminance|luminous|steradian|lm|cd|illumination)\b'
    if not re.search(photometric_keywords, q):
        return False, None, "code", 0
    
    if any(w in q for w in ["kinetic energy", "velocity", "speed", "acceleration", "force", "work", "joules"]):
        if not any(w in q for w in ["lumens per watt", "luminous efficacy"]):
            return False, None, "code", 0
    
    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0
    
    if ("illuminance" in q or "lux" in q) and ("flux" in q or "lumens" in q):
        if any(w in q for w in ["square meter", "m²", "m2", "sq. m", "sq m", "area"]):
            flux = num_near(q, "lumen", "lm", "flux")
            area = num_near(q, "square meter", "m2", "m²", "area", "surface")
            if flux is not None and area is not None:
                return ok("photometry.illuminance_from_flux_area", ["flux", "area"],
                         [flux, area], round=True, places=6)
    
    if ("illuminance" in q or "lux" in q) and ("intensity" in q or "candela" in q or "inverse" in q):
        if any(w in q for w in ["meter", "m ", " m", "distance", "away", "from"]):
            intensity = num_near(q, "candela", "cd", "intensity", "source")
            distance = num_near(q, "meter", "m", "distance", "away", "from")
            if intensity is not None and distance is not None:
                return ok("photometry.illuminance_inverse_square", ["intensity", "distance"],
                         [intensity, distance], round=True, places=6)
    
    if any(w in q for w in ["efficacy", "lumens per watt", "lm/w", "lm/W", "lumens/watt"]):
        flux = num_near(q, "lumen", "lm", "flux")
        power = num_near(q, "watt", "w", "watts", "power")
        if flux is not None and power is not None:
            return ok("photometry.luminous_efficacy", ["flux", "power"],
                     [flux, power], round=True, places=6, suffix=" lm/W")
    
    if ("intensity" in q or "candela" in q) and ("steradian" in q or "solid angle" in q or "sr" in q):
        flux = num_near(q, "lumen", "lm", "flux")
        solid_angle = num_near(q, "steradian", "sr", "steradians", "solid angle")
        if flux is not None and solid_angle is not None:
            return ok("photometry.luminous_intensity", ["flux", "solid_angle"],
                     [flux, solid_angle], round=True, places=6, suffix=" cd")
    
    if "illuminance" in q and ("original distance" in q or ("distance" in q and ("moved" in q or "new" in q))):
        illum = num_near(q, "illuminance", "lux")
        d1 = num_near(q, "original distance")
        d2 = num_near(q, "new distance", "moved to", r"\bat\b", "becomes", "to")
        if illum is not None and d1 is not None and d2 is not None and d1 != d2:
            return ok("photometry.illuminance_ratio", ["illuminance1", "distance1", "distance2"],
                     [illum, d1, d2], round=True, places=6, suffix=" lux")
    
    return False, None, "code", 0
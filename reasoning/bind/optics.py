"""Bind layer — optics. Gated on optics-specific keywords (lens, focal, magnification,
Snell, critical angle, refraction, refractive index, diopter, f-number, focal ratio, aperture).
Extracts numeric arguments."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"(-?\d+(?:\.\d+)?)"


def bind_optics(query: str) -> tuple[bool, dict | None, str, int]:
    """Recognize and parse optics queries. Conservative gate on optics keywords."""
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # image_distance: "focal length", "object distance", "image distance", "thin lens"
    # Require "focal" — a magnification query also names "image distance"/"object distance" (its
    # inputs), so gating on those would swallow it here and then decline for lack of a focal length.
    if ("image distance" in q or "thin lens" in q or "1/f" in q) and "focal" in q:
        m_f = re.search(r"focal\s+length[^\d]*" + _NUM, q, re.IGNORECASE)
        m_do = re.search(r"object\s+distance[^\d]*" + _NUM, q, re.IGNORECASE)
        f = float(m_f.group(m_f.lastindex)) if m_f else None
        do = float(m_do.group(m_do.lastindex)) if m_do else None
        if f is None or do is None:
            return False, None, "code", 0
        return ok("optics.image_distance", ["focal_length", "object_distance"], [f, do], round=True, places=6)

    # magnification: "magnification", "mag", "m="
    if "magnification" in q or re.search(r"(mag|m)\s*=", q, re.IGNORECASE):
        m_di = re.search(r"image\s+distance[^\d]*" + _NUM, q, re.IGNORECASE)
        m_do = re.search(r"object\s+distance[^\d]*" + _NUM, q, re.IGNORECASE)
        di = float(m_di.group(m_di.lastindex)) if m_di else None
        do = float(m_do.group(m_do.lastindex)) if m_do else None
        if di is None or do is None:
            return False, None, "code", 0
        return ok("optics.magnification", ["image_distance", "object_distance"], [di, do], round=True, places=6)

    # lens_power: "diopter", "lens power", "power", "D" (unit)
    if ("diopter" in q or "lens power" in q or "power" in q) and ("focal" in q or re.search(r"\bcm\b|\bm\b", q)):
        m_f = re.search(r"focal\s+length[^\d]*" + _NUM, q, re.IGNORECASE)
        if not m_f:
            m_f = re.search(r"(?:focal|length)[^\d]*" + _NUM, q, re.IGNORECASE)
        f = float(m_f.group(m_f.lastindex)) if m_f else None
        if f is None or f == 0:
            return False, None, "code", 0
        return ok("optics.lens_power", ["focal_length"], [f], round=True, places=6)

    # snell_refraction: "Snell" or "refraction" (extraction still requires n1=/angle/n2=, so it
    # declines cleanly when those aren't present — no need to also demand the word "refractive").
    if "snell" in q or "refraction" in q:
        # Pattern: n1=..., theta1=...°, n2=...
        m_n1 = re.search(r"n1\s*=\s*" + _NUM, q, re.IGNORECASE)
        m_theta1 = re.search(r"(?:theta|θ|angle)[^\d]*" + _NUM, q, re.IGNORECASE)
        m_n2 = re.search(r"n2\s*=\s*" + _NUM, q, re.IGNORECASE)
        n1 = float(m_n1.group(m_n1.lastindex)) if m_n1 else None
        theta1 = float(m_theta1.group(m_theta1.lastindex)) if m_theta1 else None
        n2 = float(m_n2.group(m_n2.lastindex)) if m_n2 else None
        if n1 is None or theta1 is None or n2 is None:
            return False, None, "code", 0
        return ok("optics.snell_refraction", ["n1", "theta1_deg", "n2"], [n1, theta1, n2], round=True, places=6)

    # critical_angle: "critical angle", "total internal reflection", "TIR"
    if ("critical angle" in q or "total internal reflection" in q or "tir" in q):
        m_n1 = re.search(r"n1\s*=\s*" + _NUM, q, re.IGNORECASE)
        m_n2 = re.search(r"n2\s*=\s*" + _NUM, q, re.IGNORECASE)
        n1 = float(m_n1.group(m_n1.lastindex)) if m_n1 else None
        n2 = float(m_n2.group(m_n2.lastindex)) if m_n2 else None
        if n1 is None or n2 is None:
            return False, None, "code", 0
        return ok("optics.critical_angle", ["n1", "n2"], [n1, n2], round=True, places=6)

    # f_number: "f-number", "focal ratio", or "f/" notation — gate distinctive to this op.
    # Do NOT trigger on "focal length" alone (that's image_distance).
    if "f-number" in q or "focal ratio" in q or "f/" in q:
        f = num_near(q, "focal length")
        d = num_near(q, "aperture diameter", "aperture", "diameter")
        if f is None or d is None:
            return False, None, "code", 0
        return ok("optics.f_number", ["focal_length", "aperture_diameter"], [f, d], round=True, places=6)

    return False, None, "code", 0
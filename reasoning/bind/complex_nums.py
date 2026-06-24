"""bind layer — complex numbers (modulus/argument/conjugate/arithmetic/power) and complex
calculus (residue, contour integral, complex derivative)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

# a complex literal: a±bi or bi. Requires a DIGIT (so it won't match the bare "i" in words like
# "is" or "radii") and a word boundary after i.
_CNUM = re.compile(r"[-+]?\d+(?:\.\d+)?\s*[-+]\s*\d*(?:\.\d+)?\s*i\b"   # a ± bi
                   r"|[-+]?\d+(?:\.\d+)?\s*i\b", re.IGNORECASE)         # bi


def _complexes(query):
    return [re.sub(r"\s+", "", m) for m in _CNUM.findall(query)]


def bind_complex(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # A j/k imaginary unit or the word "quaternion" means this is a hypercomplex query, not a
    # complex one — decline so bind_hypercomplex claims it (complex numbers have only i).
    if re.search(r"[\d.\s+\-][jk]\b|\b[jk]\b|quaternion", query, re.IGNORECASE):
        return False, None, "code", 0
    nums = _complexes(query)
    if not nums:
        return False, None, "code", 0
    if "conjugate" in q:
        return ok("cplx.conjugate", ["z"], [nums[0]], round=False)
    if "modulus" in q or "magnitude" in q or "absolute value" in q or "|" in query:
        return ok("cplx.modulus", ["z"], [nums[0]], places=4)
    if "argument" in q or "phase" in q or "angle" in q:
        return ok("cplx.argument", ["z"], [nums[0]], places=4)
    if "power" in q or "raised" in q or "^" in query or "to the" in q:
        m = re.search(r"\^\s*(\d+)|power\s+(?:of\s+)?(\d+)|to the (\d+)", q)
        if m:
            return ok("cplx.power", ["z", "n"], [nums[0], int(m[1] or m[2] or m[3])], round=False)
    # multiply/divide of two complex numbers exist as skills but aren't bound here: "product of
    # A and B" collides with the decomposition (compose) tier. Single-number ops are unambiguous.
    return False, None, "code", 0


# complex calculus: residue / contour integral / complex derivative of a z-expression. \b on
# "at" so it doesn't fire inside "Wh-at".
_AT = re.compile(r"\b(?:at|around|about|enclosing)\s+([^\s,?]+)", re.IGNORECASE)
_ZEXPR = re.compile(r"of\s+(.+?)(?:\s+(?:at|around|about|enclosing|with respect)|[?.]|$)",
                    re.IGNORECASE)


def bind_ccalc(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    em = _ZEXPR.search(query)
    if not em or "z" not in em[1].lower():
        return False, None, "code", 0
    expr = em[1].strip()
    if "residue" in q:
        pm = _AT.search(query)
        if pm:
            return ok("ccalc.residue", ["expr", "point"], [expr, pm[1]], round=False)
    if "contour" in q or "∮" in query:
        pm = _AT.search(query)
        if pm:
            return ok("ccalc.contour_integral", ["expr", "point"], [expr, pm[1]], round=False)
    if "complex derivative" in q or ("derivative" in q and "d/dz" in q):
        return ok("ccalc.derivative", ["expr"], [expr], round=False)
    return False, None, "code", 0

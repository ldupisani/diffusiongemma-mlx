"""bind layer — dynamical systems & chaos. Gated on logistic-map / Lyapunov / Mandelbrot /
chaos cues."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"-?\d+(?:\.\d+)?"
_GATE = re.compile(r"logistic|lyapunov|mandelbrot|chaos|dynamical system", re.IGNORECASE)


def _c(s: str):
    s = s.strip().replace(" ", "").replace("i", "j")
    s = re.sub(r"(?<![\d.])j", "1j", s)            # bare j -> 1j
    z = complex(s)
    return z.real, z.imag


def bind_dynamics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0
    # "generate a logistic trajectory/dataset …" wants the whole series as a dataset (datagen),
    # not a single iterate — let the dataset subsystem claim it.
    if "generate" in q or "trajectory" in q or "dataset" in q:
        return False, None, "code", 0

    if "mandelbrot" in q:
        m = re.search(r"c\s*=\s*([-\d.+\s]*i?[-\d.+\s]*?)(?:\s+in|\?|$)", query, re.IGNORECASE)
        if m and m[1].strip():
            try:
                cr, ci = _c(m[1])
            except ValueError:
                return False, None, "code", 0
            it = re.search(r"(\d+)\s+iterations", q)
            return ok("dynamics.mandelbrot", ["c_real", "c_imag", "max_iter"],
                      [cr, ci, int(it[1]) if it else 100], round=False)
        return False, None, "code", 0

    r = re.search(rf"r\s*=\s*({_NUM})", q)
    x0 = re.search(rf"x_?0\s*=\s*({_NUM})", q) or re.search(rf"(?:from|starting at)\s+({_NUM})", q)
    n = re.search(r"(\d+)\s+(?:iterations|steps)", q)

    if "lyapunov" in q and r:
        return ok("dynamics.lyapunov_logistic", ["r", "x0", "n"],
                  [float(r[1]), float(x0[1]) if x0 else 0.2, int(n[1]) if n else 2000],
                  round=True, places=6)
    if "fixed point" in q and r:
        return ok("dynamics.logistic_fixed_points", ["r"], [float(r[1])], round=False)
    if "logistic" in q and r and x0 and n:
        return ok("dynamics.logistic_map", ["r", "x0", "n"],
                  [float(r[1]), float(x0[1]), int(n[1])], round=True, places=6)
    return False, None, "code", 0

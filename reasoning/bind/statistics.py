"""bind layer — mathematical statistics (bivariate / inferential): correlation, least-squares
regression, and z-scores. Distinct from the descriptive single-list list_statistic domain."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_N = r"-?\d+(?:\.\d+)?"
_LIST = re.compile(r"\[[^\]]*\]")


def bind_statistics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if "z-score" in q or "z score" in q or "standard score" in q:
        x = re.search(rf"(?:z[- ]?score|standard score)\s+of\s+({_N})", q)
        mean = re.search(rf"mean\s+(?:of\s+|is\s+|=\s*)?({_N})", q)
        sd = re.search(rf"(?:standard deviation|stdev|std|sd)\s+(?:of\s+|is\s+|=\s*)?({_N})", q)
        if x and mean and sd:
            return ok("stats.zscore", ["x", "mean", "sd"],
                      [float(x[1]), float(mean[1]), float(sd[1])], round=True, places=6)
        return False, None, "code", 0

    ls = _LIST.findall(query)
    if "correlation" in q and len(ls) >= 2:
        return ok("stats.correlation", ["xs", "ys"], [ls[0], ls[1]], round=True, places=6)
    if ("regression" in q or "least squares" in q or "line of best fit" in q) and len(ls) >= 2:
        return ok("stats.linear_regression", ["xs", "ys"], [ls[0], ls[1]], round=False)
    return False, None, "code", 0

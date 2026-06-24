"""bind layer — machine learning theory. Each op has a distinctive keyword (sigmoid / relu /
softmax / cross-entropy / MSE / precision / recall / F1 / accuracy / entropy / Gini / gradient
step), so the gate is just "is one of these named?" and the rest is argument extraction."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"-?\d+(?:\.\d+)?"
_LISTS = re.compile(r"\[[^\]]*\]")
_GATE = re.compile(r"sigmoid|relu|softmax|cross[ -]?entropy|mean squared error|\bmse\b|"
                   r"precision|recall|f1|accuracy|entropy|gini|gradient (?:step|descent)",
                   re.IGNORECASE)


def _one(query):
    m = re.search(rf"of\s+({_NUM})", query) or re.search(rf"({_NUM})", query)
    return float(m[1]) if m else None


def _list(query):
    m = _LISTS.search(query)
    if m:
        return m.group()
    m = re.search(rf"of\s+([-\d.,\s]+)", query)   # bare comma list after "of"
    return m[1].strip() if m else None


def bind_ml(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0

    if "sigmoid" in q:
        x = _one(query)
        return ok("ml.sigmoid", ["x"], [x], round=True, places=6) if x is not None else (False, None, "code", 0)
    if "relu" in q:
        x = _one(query)
        return ok("ml.relu", ["x"], [x], round=True, places=6) if x is not None else (False, None, "code", 0)
    if "softmax" in q:
        ls = _list(query)
        return ok("ml.softmax", ["logits"], [ls], round=False) if ls else (False, None, "code", 0)

    if "cross" in q and "entropy" in q:
        ls = _LISTS.findall(query)
        if len(ls) >= 2:
            return ok("ml.cross_entropy", ["y_true", "y_pred"], [ls[0], ls[1]], round=True, places=6)
        return False, None, "code", 0
    if "mean squared error" in q or re.search(r"\bmse\b", q):
        ls = _LISTS.findall(query)
        if len(ls) >= 2:
            return ok("ml.mse", ["y_true", "y_pred"], [ls[0], ls[1]], round=True, places=6)
        return False, None, "code", 0

    cm = dict(re.findall(r"\b(tp|tn|fp|fn)\s*=\s*(\d+)", q))
    if "f1" in q and all(k in cm for k in ("tp", "fp", "fn")):
        return ok("ml.f1", ["tp", "fp", "fn"], [int(cm["tp"]), int(cm["fp"]), int(cm["fn"])],
                  round=True, places=6)
    if "precision" in q and all(k in cm for k in ("tp", "fp")):
        return ok("ml.precision", ["tp", "fp"], [int(cm["tp"]), int(cm["fp"])], round=True, places=6)
    if "recall" in q and all(k in cm for k in ("tp", "fn")):
        return ok("ml.recall", ["tp", "fn"], [int(cm["tp"]), int(cm["fn"])], round=True, places=6)
    if "accuracy" in q and all(k in cm for k in ("tp", "tn", "fp", "fn")):
        return ok("ml.accuracy", ["tp", "tn", "fp", "fn"],
                  [int(cm["tp"]), int(cm["tn"]), int(cm["fp"]), int(cm["fn"])], round=True, places=6)

    if "gini" in q:
        ls = _list(query)
        return ok("ml.gini", ["probs"], [ls], round=True, places=6) if ls else (False, None, "code", 0)
    if "entropy" in q:
        ls = _list(query)
        return ok("ml.entropy", ["probs"], [ls], round=True, places=6) if ls else (False, None, "code", 0)

    if "gradient" in q:
        th = re.search(rf"(?:theta|θ|parameter|weight)\s*=\s*({_NUM})", q)
        gr = re.search(rf"grad(?:ient)?\s*=\s*({_NUM})", q)
        lr = re.search(rf"(?:lr|learning rate|step size)\s*=\s*({_NUM})", q)
        if th and gr and lr:
            return ok("ml.gradient_step", ["theta", "grad", "lr"],
                      [float(th[1]), float(gr[1]), float(lr[1])], round=True, places=6)
        return False, None, "code", 0

    return False, None, "code", 0

"""Hyperreal numbers — arithmetic with an infinitesimal ε and the infinite ω = 1/ε. Zero
model calls.

A hyperreal is carried as a finite Laurent polynomial in ε: a sum of c·ε^k where k=0 is the
real (appreciable) part, k>0 are infinitesimal parts, and k<0 are infinite parts (ε^-1 = ω).
So 3+ε is "3 plus an infinitesimal", ω = ε^-1 is infinite, and ω·ε = 1. The standard part st(x)
is the unique real infinitely close to a FINITE x (st(3+ε)=3); an infinite x has none. classify
labels x infinitesimal / finite / infinite; compare orders by the most-infinite differing term.
"""

from __future__ import annotations

import re

from .base import register


def _num(x: float):
    return int(x) if float(x) == int(x) else round(x, 4)


def _norm(s: str) -> str:
    s = s.lower().replace("omega", "ω").replace("epsilon", "ε").replace("·", "*").replace("×", "*")
    s = s.replace(" ", "")
    return s.replace("1/ε", "ω").replace("1/ω", "ε")


def _atom(tok: str) -> dict:
    """A single factor (no + or *): a real, or c·ε^k / c·ω^k."""
    m = re.fullmatch(r"([+-]?\d*\.?\d*)(ε|ω)(?:\^(-?\d+))?", tok)
    if m:
        c = {"": 1.0, "+": 1.0, "-": -1.0}.get(m[1], None)
        c = float(m[1]) if c is None else c
        k = int(m[3]) if m[3] else 1
        return {(k if m[2] == "ε" else -k): c}
    v = float(tok)
    return {0: v} if v else {}


def _mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for pa, ca in a.items():
        for pb, cb in b.items():
            out[pa + pb] = out.get(pa + pb, 0.0) + ca * cb
    return out


def _add_into(acc: dict, d: dict) -> dict:
    for p, c in d.items():
        acc[p] = acc.get(p, 0.0) + c
    return acc


def _eval(expr: str) -> dict:
    total: dict = {}
    for part in _norm(expr).replace("-", "+-").split("+"):
        if not part:
            continue
        prod = None
        for f in part.split("*"):
            a = _atom(f)
            prod = a if prod is None else _mul(prod, a)
        _add_into(total, prod or {})
    return total


def _fmt(d: dict) -> str:
    items = sorted(((p, c) for p, c in d.items() if c != 0), key=lambda x: x[0])
    if not items:
        return "0"
    parts = []
    for p, c in items:
        c = _num(c)
        if p == 0:
            parts.append(str(c))
            continue
        unit = ("ω" if p == -1 else f"ω^{-p}") if p < 0 else ("ε" if p == 1 else f"ε^{p}")
        parts.append(unit if c == 1 else (f"-{unit}" if c == -1 else f"{c}{unit}"))
    s = parts[0]
    for t in parts[1:]:
        s += t if t.startswith("-") else "+" + t
    return s


@register("hreal.normalize", "evaluate a hyperreal expression in ε/ω (+, ·, ^) to a sum of terms",
          [(("ω+ε",), "ω+ε"), (("ω*ε",), "1"), (("2ε+3ε",), "5ε"), (("ω*ω",), "ω^2")])
def normalize(expr: str) -> str:
    return _fmt(_eval(expr))


@register("hreal.add", "sum of two hyperreals",
          [(("ω", "3"), "ω+3"), (("3", "ε"), "3+ε"), (("ω", "ε"), "ω+ε")])
def add(a: str, b: str) -> str:
    return _fmt(_add_into(_eval(a), _eval(b)))


@register("hreal.multiply", "product of two hyperreals (ω·ε = 1)",
          [(("ω", "ε"), "1"), (("2", "ε"), "2ε"), (("ω", "ω"), "ω^2")])
def multiply(a: str, b: str) -> str:
    return _fmt(_mul(_eval(a), _eval(b)))


@register("hreal.standard_part", "standard part st(x): the real infinitely close to a finite x",
          [(("3+ε",), "3"), (("5",), "5"), (("ε",), "0"), (("ω",), "undefined (infinite)")])
def standard_part(x: str) -> str:
    d = _eval(x)
    if any(p < 0 and c != 0 for p, c in d.items()):
        return "undefined (infinite)"
    return str(_num(d.get(0, 0.0)))


@register("hreal.classify", "classify a hyperreal: infinitesimal / finite / infinite",
          [(("ε",), "infinitesimal"), (("3+ε",), "finite"), (("ω",), "infinite"),
           (("5",), "finite"), (("0",), "zero")])
def classify(x: str) -> str:
    d = {p: c for p, c in _eval(x).items() if c != 0}
    if not d:
        return "zero"
    if any(p < 0 for p in d):
        return "infinite"
    if all(p > 0 for p in d):
        return "infinitesimal"
    return "finite"


@register("hreal.compare", "order two hyperreals (most-infinite differing term decides)",
          [(("ω", "5"), "ω > 5"), (("3+ε", "3"), "3+ε > 3"), (("ε", "0"), "ε > 0")])
def compare(a: str, b: str) -> str:
    A, B = _eval(a), _eval(b)
    diff = dict(A)
    for p, c in B.items():
        diff[p] = diff.get(p, 0.0) - c
    nz = sorted(p for p, c in diff.items() if c != 0)  # lowest power dominates
    sym = "=" if not nz else (">" if diff[nz[0]] > 0 else "<")
    return f"{_fmt(A)} {sym} {_fmt(B)}"

"""Ordinal numbers — Cantor-normal-form arithmetic below ω^ω. Zero model calls.

An ordinal here is written in Cantor normal form as ω^k·c + … with natural-number exponents
(descending) and positive integer coefficients — e.g. ω^2·3 + ω·2 + 5. Ordinal addition and
multiplication are NON-commutative and absorb lower terms on the left: 1 + ω = ω but ω + 1 ≠ ω;
2·ω = ω but ω·2 ≠ ω. normalize() evaluates a +, ·, ^ expression to CNF (so it demonstrates these
identities); compare() orders two ordinals.

CNF is represented as a list of (exponent, coefficient) pairs in strictly descending exponent.
"""

from __future__ import annotations

import re

from .base import register


def _atom(tok: str):
    if "ω" in tok:
        m = re.fullmatch(r"ω(?:\^(\d+))?", tok)
        if not m:
            raise ValueError(f"bad ordinal atom {tok!r}")
        return [(int(m[1]) if m[1] else 1, 1)]
    v = int(tok)
    return [(0, v)] if v else []


def _add(a, b):
    if not a:
        return list(b)
    if not b:
        return list(a)
    b1 = b[0][0]
    res = [(e, c) for (e, c) in a if e > b1]
    same = [(e, c) for (e, c) in a if e == b1]
    if same:
        res.append((b1, same[0][1] + b[0][1]))
        res.extend(b[1:])
    else:
        res.extend(b)
    return res


def _mul(a, b):
    if not a or not b:
        return []
    a1 = a[0][0]
    res = []
    for (e, n) in b:
        tp = [(a1 + e, n)] if e > 0 else [(a1, a[0][1] * n)] + list(a[1:])
        res = _add(res, tp)
    return res


def _norm_input(s: str) -> str:
    s = s.lower().replace("omega", "ω").replace("×", "*").replace("·", "*")
    s = s.replace(" multiplied by ", "*").replace(" times ", "*").replace(" plus ", "+")
    return s.replace(" ", "")


def _eval(expr: str):
    total = []
    for part in _norm_input(expr).split("+"):
        if not part:
            continue
        prod = None
        for f in part.split("*"):
            a = _atom(f)
            prod = a if prod is None else _mul(prod, a)
        total = _add(total, prod or [])
    return total


def _fmt(cnf) -> str:
    if not cnf:
        return "0"
    out = []
    for (e, c) in cnf:
        if e == 0:
            out.append(str(c))
        elif e == 1:
            out.append("ω" if c == 1 else f"ω·{c}")
        else:
            out.append(f"ω^{e}" if c == 1 else f"ω^{e}·{c}")
    return "+".join(out)


def _cmp(a, b) -> int:
    for (ea, ca), (eb, cb) in zip(a, b):
        if ea != eb:
            return 1 if ea > eb else -1
        if ca != cb:
            return 1 if ca > cb else -1
    if len(a) != len(b):
        return 1 if len(a) > len(b) else -1
    return 0


@register("ord.normalize", "Cantor normal form of an ordinal expression (+, ·, ^)",
          [(("1+ω",), "ω"), (("ω+1",), "ω+1"), (("2*ω",), "ω"), (("ω*2",), "ω·2"),
           (("ω*ω",), "ω^2"), (("ω^2+ω*3+5",), "ω^2+ω·3+5")])
def normalize(expr: str) -> str:
    return _fmt(_eval(expr))


@register("ord.add", "ordinal sum a + b (non-commutative)",
          [(("ω", "1"), "ω+1"), (("1", "ω"), "ω"), (("ω+1", "ω"), "ω·2")])
def add(a: str, b: str) -> str:
    return _fmt(_add(_eval(a), _eval(b)))


@register("ord.multiply", "ordinal product a · b (non-commutative)",
          [(("ω", "2"), "ω·2"), (("2", "ω"), "ω"), (("ω", "ω"), "ω^2"),
           (("ω+1", "2"), "ω·2+1")])
def multiply(a: str, b: str) -> str:
    return _fmt(_mul(_eval(a), _eval(b)))


@register("ord.compare", "order two ordinals (returns 'a < b' / 'a = b' / 'a > b')",
          [(("1+ω", "ω+1"), "ω < ω+1"), (("ω*2", "ω+5"), "ω·2 > ω+5"), (("ω", "ω"), "ω = ω")])
def compare(a: str, b: str) -> str:
    A, B = _eval(a), _eval(b)
    sym = {-1: "<", 0: "=", 1: ">"}[_cmp(A, B)]
    return f"{_fmt(A)} {sym} {_fmt(B)}"

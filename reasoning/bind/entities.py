"""Coded NL entity extraction — the conservative, shared layer the binds compose.

Every confident-wrong bind bug we've fixed (radix "A", "times", gcd(1,7), sum(100,3),
"for z-1", a stray finance $) was the same failure: a recognizer too lenient about which
numbers it grabbed, patched locally each time. This module is where the "decline on
ambiguity" discipline lives ONCE: extractors return every match (with spans, for
position-sensitive binds), and the conservative SELECTORS (only_number, exactly, …) encode
"exactly N operands or None" so a bind can't re-introduce the greedy pattern.

Regex bodies are copied verbatim from the binds they replace — migration is behavior-
preserving. Pure module: depends only on `re`.
"""

from __future__ import annotations

import re
from collections import namedtuple

Num = namedtuple("Num", "value start end")

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_INT = re.compile(r"-?\d+")
# A bare comma directly before exactly 3 digits is a thousands separator ("20,700"), NOT a list
# delimiter — used by numeric_list to avoid splitting a grouped number into [20, 700].
_THOUSANDS_GAP = re.compile(r"^,$")
_THREE_DIGITS = re.compile(r"^\d{3}(?:\D|$)")
_LIST_GAP = re.compile(r"^[\s,$]*(?:and|or|&)?[\s,$]*$", re.IGNORECASE)
_PRICE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_POINT = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_FROM_TO = re.compile(r"from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _val(s: str):
    return float(s) if "." in s else int(s)


# --- numbers ---------------------------------------------------------------------------
def numbers(text: str) -> list:
    """Every signed int/float, with spans."""
    return [Num(_val(m.group()), m.start(), m.end()) for m in _NUMBER.finditer(text)]


def values(text: str) -> list:
    """Just the numeric values (int where whole, float otherwise)."""
    return [n.value for n in numbers(text)]


def ints(text: str) -> list:
    """Every integer (sign preserved, so e.g. factorial(-3) declines rather than reading 3)."""
    return [int(x) for x in _INT.findall(text)]


def only_number(text: str):
    """The value iff EXACTLY one number is present, else None — the conservative selector."""
    ns = numbers(text)
    return ns[0].value if len(ns) == 1 else None


def exactly(text: str, n: int):
    """The n values iff exactly n numbers are present, else None (decline on extras)."""
    ns = numbers(text)
    return [x.value for x in ns] if len(ns) == n else None


def numeric_list(text: str) -> list:
    """The longest run of numbers separated only by list delimiters (commas / and / or / $),
    so scattered numbers in prose are NOT aggregated."""
    best: list = []
    run: list = []
    for m in _NUMBER.finditer(text):
        val = _val(m.group())
        gap = text[run[-1][0]:m.start()] if run else ""
        # "20,700" tokenizes as 20 then 700 with a bare-comma gap — a thousands separator, not a
        # list. Treat it as NOT a list (break the run) so list_statistic declines instead of
        # averaging/summing the fragments ([20, 700]) into a confident-wrong answer.
        if run and _THOUSANDS_GAP.match(gap) and _THREE_DIGITS.match(m.group()):
            if len(run) > len(best):
                best = run
            run = []
            continue
        if run and _LIST_GAP.match(gap):
            run.append((m.end(), val))
        else:
            if len(run) > len(best):
                best = run
            run = [(m.end(), val)]
    if len(run) > len(best):
        best = run
    return [v for _, v in best]


def number_before(text: str, unit_re: str):
    """The number immediately before a unit token (case-sensitive — kg/N/m/s differ)."""
    m = re.search(rf"(-?\d+(?:\.\d+)?)\s*(?:{unit_re})", text)
    return float(m[1]) if m else None


def labeled_number(text: str, label: str):
    """A labelled dimension, e.g. 'radius 5' / 'length of 4'."""
    m = re.search(rf"{label}\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(m[1]) if m else None


# --- money / percent / quantity / range / pairs ----------------------------------------
def dollars(text: str) -> list:
    """All $-prefixed amounts, in order."""
    return [float(m) for m in _PRICE.findall(text)]


def only_dollar(text: str):
    d = dollars(text)
    return d[0] if len(d) == 1 else None


def percents(text: str) -> list:
    """Bare 'N%' / 'N percent' values (contextual '% off' / '% tax' stay domain-local)."""
    return [float(m) for m in _PERCENT.findall(text)]


def count_of(text: str, *nouns: str):
    """'N <noun>' for one of the given nouns, e.g. count_of(q, 'people', 'friends'), else None."""
    pat = "|".join(re.escape(n) for n in nouns)
    m = re.search(rf"(\d+)\s+(?:{pat})", text, re.IGNORECASE)
    return int(m[1]) if m else None


def from_to(text: str):
    """'from X to Y' → (X, Y), else None."""
    m = _FROM_TO.search(text)
    return (_val(m[1]), _val(m[2])) if m else None


def pairs(text: str) -> list:
    """All '(x, y)' coordinate pairs."""
    return [(float(a), float(b)) for a, b in _POINT.findall(text)]

"""Shared parsing primitives + cross-cutting regexes for the bind layer."""
from __future__ import annotations

import json as _json  # noqa: F401  (used by _try_json)
import re

from reasoning.bind.entities import _NUMBER  # noqa: F401  (single source; re-exported)
from reasoning.engine import one_shot  # noqa: F401  (re-exported: finance default arg)

__all__ = ["_parse_list", "_parse_int", "_try_json", "_NUMBER", "one_shot",
           "query_numbers", "has_unconsumed_numbers", "guard_residue"]

# --- red-flag guard: a query value the skill didn't consume ----------------------------
# Partial matches are the top source of confident-wrong answers: a bind extracts what it knows
# and silently ignores the rest (e.g. answering "population in 2010" for a query that also said
# "2020"). A bind with a FIXED set of numeric inputs can call has_unconsumed_numbers(query, used)
# and DECLINE when the query carries a standalone number it isn't accounting for. Conservative by
# design: it ignores numbers glued to a date/ratio/decimal/word (2024-03-03, 3/4, 12.5, R2D2) and
# only sees free-standing magnitudes, so it flags genuine extra values without tripping on units.
# A free-standing magnitude: optional thousands commas, optional decimal; NOT glued to a letter,
# colon, slash, or hyphen — so it skips dates (2024-03-03), times (9:00), ratios (3/4), and
# identifiers (R2D2), while reading "$1,000" / "5%" / "12.5" as single values.
_STANDALONE_NUM = re.compile(r"(?<![\w:/-])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w:/-])")


def query_numbers(query: str) -> set:
    """Free-standing numeric magnitudes in a query (not part of a date, time, ratio, or word)."""
    return {float(m.replace(",", "")) for m in _STANDALONE_NUM.findall(query)}


def has_unconsumed_numbers(query: str, used) -> bool:
    """True if the query has a standalone number the bind didn't consume — a red flag that the
    query means more than the skill handles, so the bind should decline rather than guess."""
    used_set = {float(u) for u in used if u is not None}
    return bool(query_numbers(query) - used_set)


def guard_residue(query: str, result):
    """Turn a bind result into a DECLINE when the query carries a standalone number the bound
    inputs don't account for — the general anti-silent-wrong guard (§3d). `used` is every numeric
    value the bind put in its data dict (a superset of the real inputs → conservative, few false
    declines). Apply to fixed-arity numeric binds (NOT list/variable-arity ones, whose elements
    are legitimately many numbers). Wrap the bind call: guard_residue(query, bind(query))."""
    ok, data, via, calls = result
    if ok and isinstance(data, dict):
        used = [n for v in data.values() if (n := _numlike(v)) is not None]
        if has_unconsumed_numbers(query, used):
            return False, None, via, calls
    return result


def _numlike(v):
    """v as a float if it's a number or a plain numeric string (so values stored as strings —
    a radix value '42', a duration '3' — still count as consumed); else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return None

# --- shared deterministic parsing primitives (moved from patterns.py) ----------------
def _parse_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(1))
    return items


def _parse_int(text: str, default: int) -> int:
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else default


def _try_json(text: str):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return _json.loads(t), None
    except Exception as e:
        m = re.search(r"(\{.*\}|\[.*\])", t, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(1)), None
            except Exception as e2:
                return None, str(e2)
        return None, str(e)

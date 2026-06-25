"""Shared parsing primitives + cross-cutting regexes for the bind layer."""
from __future__ import annotations

import json as _json  # noqa: F401  (used by _try_json)
import re

from reasoning.bind.entities import _NUMBER  # noqa: F401  (single source; re-exported)
from reasoning.engine import one_shot  # noqa: F401  (re-exported: finance default arg)

__all__ = ["_parse_list", "_parse_int", "_try_json", "_NUMBER", "one_shot",
           "query_numbers", "has_unconsumed_numbers"]

# --- red-flag guard: a query value the skill didn't consume ----------------------------
# Partial matches are the top source of confident-wrong answers: a bind extracts what it knows
# and silently ignores the rest (e.g. answering "population in 2010" for a query that also said
# "2020"). A bind with a FIXED set of numeric inputs can call has_unconsumed_numbers(query, used)
# and DECLINE when the query carries a standalone number it isn't accounting for. Conservative by
# design: it ignores numbers glued to a date/ratio/decimal/word (2024-03-03, 3/4, 12.5, R2D2) and
# only sees free-standing magnitudes, so it flags genuine extra values without tripping on units.
_STANDALONE_NUM = re.compile(r"(?<![\w./-])(\d+(?:\.\d+)?)(?![\w./-])")


def query_numbers(query: str) -> set:
    """Free-standing numeric magnitudes in a query (not part of a date, ratio, or word)."""
    return {float(m) for m in _STANDALONE_NUM.findall(query)}


def has_unconsumed_numbers(query: str, used) -> bool:
    """True if the query has a standalone number the bind didn't consume — a red flag that the
    query means more than the skill handles, so the bind should decline rather than guess."""
    used_set = {float(u) for u in used if u is not None}
    return bool(query_numbers(query) - used_set)

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

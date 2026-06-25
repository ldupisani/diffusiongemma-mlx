"""Shared parsing primitives + cross-cutting regexes for the bind layer."""
from __future__ import annotations

import json as _json  # noqa: F401  (used by _try_json)
import re

from reasoning.bind.entities import _NUMBER  # noqa: F401  (single source; re-exported)
from reasoning.engine import one_shot  # noqa: F401  (re-exported: finance default arg)

__all__ = ["_parse_list", "_parse_int", "_try_json", "_NUMBER", "one_shot",
           "query_numbers", "has_unconsumed_numbers", "guard_residue", "num_near"]

# --- bidirectional keyword-anchored number extraction --------------------------------
# Authored binds repeatedly hand-rolled FORWARD-only extraction ("the number after the keyword")
# and then mis-read natural phrasings where the value comes BEFORE the anchor: "480 minutes",
# "60 customer units", a mass in "0.000001 kg" (forward grabbed the speed that followed). num_near
# checks both sides, nearest-first, without jumping across another number.
_NEAR_NUM = r"(-?\d+(?:\.\d+)?)"


def _stem_anchor(a: str) -> str:
    """Drop a trailing plural 's' from a plain single-word anchor so it matches both forms — the
    stem substring-matches the plural ('lumen' is inside 'lumens'), which the bare plural can't do
    in reverse. Only touches pure-letter words longer than 3 chars and not ending in 'ss' (so
    'lumens'/'watts'/'meters' → 'lumen'/'watt'/'meter', but 'mass'/'loss' and regex/multi-word
    anchors like r'\\bat\\b' or 'order cost' pass through untouched)."""
    if a.isalpha() and len(a) > 3 and a.endswith("s") and not a.endswith("ss"):
        return a[:-1]
    return a


def num_near(text: str, *anchors: str, window: int = 24):
    """The number NEAREST to any of the given anchor words/phrases, looking BOTH directions — after
    the anchor ('demand 800', 'capacity is 1000', 'n1=1.5') and before it ('480 minutes',
    '60 customer units', a mass in '0.000001 kg'). For each anchor (tried in order), returns the
    closest number on either side whose gap to the anchor carries no other digit and is <= `window`
    chars — so it never skips across an intervening value, and a unit that follows its number
    ('480 minutes') reads correctly without the caller choosing a direction. Returns a float or None.

    Anchors are regex fragments (plain words are fine). For synonyms of one value pass them together
    (`num_near(q, "order cost", "setup cost", "ordering cost")`); the first anchor that finds a
    number wins. When a bare word also appears inside a longer phrase, nearest-wins usually still
    picks correctly ('capacity is 1000' beats the distant demand value), but listing the specific
    form first (`"capacity is", "capacity"`) is the robust choice."""
    nums = [(m.start(), m.end(), float(m.group(1))) for m in re.finditer(_NEAR_NUM, text)]
    if not nums:
        return None
    for a in anchors:
        best = None
        for am in re.finditer(_stem_anchor(a), text, re.IGNORECASE):
            for ns, ne, val in nums:
                if ns >= am.end():                       # number after the anchor
                    gap, dist = text[am.end():ns], ns - am.end()
                elif ne <= am.start():                   # number before the anchor
                    gap, dist = text[ne:am.start()], am.start() - ne
                else:
                    continue                             # number overlaps the anchor
                if dist <= window and not any(c.isdigit() for c in gap):
                    if best is None or dist < best[0]:
                        best = (dist, val)
        if best is not None:
            return best[1]
    return None

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

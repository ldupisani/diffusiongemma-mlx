"""The bind layer: natural language → typed arguments.

This is the seam that decides whether a whole query runs model-free. For a known query
type we try a CODED parser (regex/grammar) first; only if that fails do we fall back to
the model (`extract_validate_repair`). A bind returns (ok, data, via) where `via` is
"code" (parser succeeded → path stays model-free) or "model" (fell back).

Design rule — be CONSERVATIVE: a coded parser returns failure rather than guess on an
ambiguous query. A wrong coded parse is a silent confident-but-wrong answer with zero
model calls; a fallback is graceful (slower, loads the sidecar). Failure is the safer bet.

The deterministic parsing primitives (_parse_int/_parse_list/_try_json) live here so the
skill/compiled layers can share them without importing patterns; patterns.py re-exports
them for backward compatibility.
"""

from __future__ import annotations

import json as _json
import re

from .engine import one_shot


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


# --- discount+tax family --------------------------------------------------------------
_PRICE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_DISCOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*off", re.IGNORECASE)
_TAX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:sales\s*)?(?:tax|vat)", re.IGNORECASE)

_DISCOUNT_TAX_SCHEMA = "{'price': number, 'discount_pct': number, 'tax_pct': number}"


def parse_discount_tax(query: str) -> dict | None:
    """Coded parse of a 'price, %-off, %-tax' query. Returns None on any doubt."""
    price, disc, tax = _PRICE.search(query), _DISCOUNT.search(query), _TAX.search(query)
    if not (price and disc and tax):
        return None  # missing a field → let the caller fall back to the model
    return {
        "price": float(price.group(1)),
        "discount_pct": float(disc.group(1)),
        "tax_pct": float(tax.group(1)),
    }


def validate_discount_tax(obj) -> tuple[bool, list[str]]:
    """Coded predicate — same (ok, errors) contract extract_validate_repair expects."""
    errors = []
    if not isinstance(obj, dict):
        return False, ["not an object"]
    for k in ("price", "discount_pct", "tax_pct"):
        v = obj.get(k)
        if not isinstance(v, (int, float)):
            errors.append(f"{k} missing or not a number")
    if not errors:
        if obj["price"] <= 0:
            errors.append("price must be positive")
        for k in ("discount_pct", "tax_pct"):
            if not (0 <= obj[k] <= 100):
                errors.append(f"{k} out of range 0-100")
    return (not errors), errors


def _evr_model_calls(result) -> int:
    """Count the model calls an extract_validate_repair run made (extract + any repairs)."""
    return sum(1 for t in result.trace if t["step"].startswith(("extract", "repair")))


def bind_discount_tax(query: str, *, shot=one_shot) -> tuple[bool, dict | None, str, int]:
    """Return (ok, data, via, model_calls). Coded parser first; model fallback on failure."""
    coded = parse_discount_tax(query)
    if coded is not None:
        ok, _ = validate_discount_tax(coded)
        if ok:
            return True, coded, "code", 0
    # fall back to the model extractor (loads the sidecar) — deferred import avoids the
    # patterns<->bind import cycle.
    from .patterns import extract_validate_repair
    r = extract_validate_repair(query, schema=_DISCOUNT_TAX_SCHEMA,
                                validate=validate_discount_tax, shot=shot)
    return (r.data is not None), r.data, "model", _evr_model_calls(r)


# --- days-between family (ISO dates) --------------------------------------------------
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def bind_days_between(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: two ISO dates → {d1, d2}. No model fallback (yet)."""
    dates = _ISO_DATE.findall(query)
    if len(dates) == 2:
        return True, {"d1": dates[0], "d2": dates[1]}, "code", 0
    return False, None, "code", 0


# --- count-occurrences family (two quoted strings) ------------------------------------
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def bind_count_occurrences(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: 'needle' ... 'haystack' (in that order) → {needle, haystack}."""
    quoted = [a or b for a, b in _QUOTED.findall(query)]
    if len(quoted) == 2:
        return True, {"needle": quoted[0], "haystack": quoted[1]}, "code", 0
    return False, None, "code", 0


# --- list-statistic family (a list of numbers + which summary) ------------------------
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# keyword → stats skill. First match wins, so order from most to least specific.
_STAT_OPS = [
    (("median",), "stats.median"),
    (("average", "mean"), "stats.mean"),
    (("sum", "total", "altogether"), "stats.total"),
    (("highest", "largest", "maximum", "max", "biggest"), "stats.maximum"),
    (("lowest", "smallest", "minimum", "min"), "stats.minimum"),
    (("range",), "stats.range_of"),
]


def detect_stat_op(query: str) -> str | None:
    q = query.lower()
    for cues, skill in _STAT_OPS:
        if any(c in q for c in cues):
            return skill
    return None


def bind_list_statistic(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: a stat keyword + ≥2 numbers → {nums, skill}."""
    skill = detect_stat_op(query)
    nums = [float(n) if "." in n else int(n) for n in _NUMBER.findall(query)]
    if skill and len(nums) >= 2:
        return True, {"nums": nums, "skill": skill}, "code", 0
    return False, None, "code", 0


# --- percentage-change family ("from X to Y") -----------------------------------------
_FROM_TO = re.compile(r"from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def bind_percent_change(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: 'from <old> to <new>' → {old, new}."""
    m = _FROM_TO.search(query)
    if m:
        return True, {"old": float(m.group(1)), "new": float(m.group(2))}, "code", 0
    return False, None, "code", 0


# --- unit-conversion family ("<n> <unit> to <unit>") ----------------------------------
# normalize many spellings to a canonical unit token
_UNIT_ALIASES = {
    "fahrenheit": "f", "f": "f", "celsius": "c", "centigrade": "c", "c": "c",
    "miles": "mi", "mile": "mi", "mi": "mi",
    "km": "km", "kilometers": "km", "kilometres": "km", "kilometer": "km", "kilometre": "km",
    "pounds": "lb", "pound": "lb", "lbs": "lb", "lb": "lb",
    "kg": "kg", "kilograms": "kg", "kilogram": "kg", "kilos": "kg", "kilo": "kg",
}
_CONVERSIONS = {
    ("f", "c"): ("units.f_to_c", "°C"), ("c", "f"): ("units.c_to_f", "°F"),
    ("mi", "km"): ("units.mi_to_km", " km"), ("km", "mi"): ("units.km_to_mi", " mi"),
    ("lb", "kg"): ("units.lb_to_kg", " kg"), ("kg", "lb"): ("units.kg_to_lb", " lb"),
}
_UNIT_WORD = re.compile(r"[a-z]+", re.IGNORECASE)


_HOW_MANY = re.compile(
    r"how many\s+([a-z]+).*?(-?\d+(?:\.\d+)?)\s*([a-z]+)", re.IGNORECASE | re.DOTALL)


def _units_in(text: str) -> list[str]:
    return [_UNIT_ALIASES[w] for w in _UNIT_WORD.findall(text.lower()) if w in _UNIT_ALIASES]


def bind_unit_convert(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind → {value, skill, symbol}. Two supported phrasings:
      - "<number> <source> to|into|in <target>"   (source before target)
      - "how many <target> ... <number> <source>"  (target before source)
    Conservative: needs a number and a KNOWN unit on each side, else it bails.
    """
    num = _NUMBER.search(query)
    if not num:
        return False, None, "code", 0
    value, source, target = float(num.group()), None, None

    # phrasing 1: explicit "<source> to <target>"
    split = re.split(r"\b(?:to|into|in)\b", query.lower(), maxsplit=1)
    if len(split) == 2:
        left, right = _units_in(split[0]), _units_in(split[1])
        if left and right:
            source, target = left[-1], right[0]

    # phrasing 2: "how many <target> is <number> <source>"
    if source is None:
        m = _HOW_MANY.match(query.strip())
        if m and m.group(1).lower() in _UNIT_ALIASES and m.group(3).lower() in _UNIT_ALIASES:
            value = float(m.group(2))
            target, source = _UNIT_ALIASES[m.group(1).lower()], _UNIT_ALIASES[m.group(3).lower()]

    pair = (source, target)
    if pair not in _CONVERSIONS:
        return False, None, "code", 0
    skill, symbol = _CONVERSIONS[pair]
    return True, {"value": value, "skill": skill, "symbol": symbol}, "code", 0

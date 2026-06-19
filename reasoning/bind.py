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


# --- date families ---------------------------------------------------------------------
# date discovery: year-first numeric (-, /, .) and named-month (either order, with year).
# All unambiguous; locale-ambiguous numeric month-first (MM/DD/YYYY) is intentionally absent.
_MONTH_WORD = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_ISO_DATE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
_DATE_PATTERNS = [
    _ISO_DATE,
    re.compile(rf"\b\d{{1,2}}\s+{_MONTH_WORD}\.?,?\s+\d{{4}}\b", re.IGNORECASE),   # D Month YYYY
    re.compile(rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),   # Month D, YYYY
]


def _find_dates(query: str) -> list[str]:
    """Every date expression in the query, in order, de-overlapped (numeric or named-month)."""
    spans = sorted((m.start(), m.end(), m.group())
                   for pat in _DATE_PATTERNS for m in pat.finditer(query))
    out, last_end = [], -1
    for start, end, text in spans:
        if start >= last_end:  # skip overlapping matches from a second pattern
            out.append(text)
            last_end = end
    return out


def bind_days_between(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: two dates → {d1, d2}. No model fallback (yet)."""
    dates = _find_dates(query)
    if len(dates) == 2:
        return True, {"d1": dates[0], "d2": dates[1]}, "code", 0
    return False, None, "code", 0


# difference unit → (skill, whether to round, unit label)
_DIFF_UNITS = [
    ("week", "date.weeks_between", True, "weeks"),
    ("month", "date.months_between", False, "months"),
    ("year", "date.years_between", False, "years"),
    ("day", "date.days_between", False, "days"),
]


def bind_date_difference(query: str) -> tuple[bool, dict | None, str, int]:
    """Two dates + the unit asked for (days/weeks/months/years)."""
    dates = _find_dates(query)
    if len(dates) != 2:
        return False, None, "code", 0
    q = query.lower()
    skill, rounded, label = next(((s, r, lbl) for w, s, r, lbl in _DIFF_UNITS if w in q),
                                 ("date.days_between", False, "days"))
    return True, {"d1": dates[0], "d2": dates[1], "skill": skill,
                  "rounded": rounded, "label": label}, "code", 0


_AFTER = re.compile(r"(\d+)\s*(day|week)s?\s+after", re.IGNORECASE)


def bind_date_add(query: str) -> tuple[bool, dict | None, str, int]:
    """'<n> days/weeks after <date>' → {date, n, skill}."""
    m, dates = _AFTER.search(query), _find_dates(query)
    if m and dates:
        skill = "date.add_weeks" if m.group(2).lower() == "week" else "date.add_days"
        return True, {"date": dates[0], "n": int(m.group(1)), "skill": skill}, "code", 0
    return False, None, "code", 0


def bind_single_date(query: str) -> tuple[bool, dict | None, str, int]:
    """A single date → {date}. For weekday / day-of-year queries."""
    dates = _find_dates(query)
    if len(dates) == 1:
        return True, {"date": dates[0]}, "code", 0
    return False, None, "code", 0


_YEAR = re.compile(r"\b(\d{4})\b")


def bind_year(query: str) -> tuple[bool, dict | None, str, int]:
    """A 4-digit year → {year}. For leap-year queries."""
    m = _YEAR.search(query)
    if m:
        return True, {"year": int(m.group(1))}, "code", 0
    return False, None, "code", 0


# --- count-occurrences family (two quoted strings) ------------------------------------
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def bind_count_occurrences(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: 'needle' ... 'haystack' (in that order) → {needle, haystack}."""
    quoted = [a or b for a, b in _QUOTED.findall(query)]
    if len(quoted) == 2:
        return True, {"needle": quoted[0], "haystack": quoted[1]}, "code", 0
    return False, None, "code", 0


# --- single-string operation family (one quoted operand + which operation) ------------
def _quoted_strings(query: str) -> list[str]:
    return [a or b for a, b in _QUOTED.findall(query)]


# keyword → single-arg text skill. Most specific first ("longest word" before "word").
_TEXT_OPS = [
    (("reverse", "backwards"), "text.reverse"),
    (("uppercase", "upper case", "capital letters", "all caps"), "text.upper"),
    (("lowercase", "lower case"), "text.lower"),
    (("title case", "title-case", "capitalize each", "capitalize every",
      "capitalize the words", "capitalise each"), "text.title_case"),
    (("vowel",), "text.count_vowels"),
    (("consonant",), "text.count_consonants"),
    (("longest word",), "text.longest_word"),
    (("acronym", "initials", "first letter of each"), "text.acronym"),
    (("character",), "text.char_count"),
    (("word",), "text.word_count"),
]


def detect_text_op(query: str) -> str | None:
    q = query.lower()
    for cues, skill in _TEXT_OPS:
        if any(c in q for c in cues):
            return skill
    return None


def bind_string_op(query: str) -> tuple[bool, dict | None, str, int]:
    """One quoted operand + a detected single-arg operation → {text, skill}."""
    skill = detect_text_op(query)
    quoted = _quoted_strings(query)
    if skill and len(quoted) == 1:
        return True, {"text": quoted[0], "skill": skill}, "code", 0
    return False, None, "code", 0


def bind_replace(query: str) -> tuple[bool, dict | None, str, int]:
    """'replace OLD with NEW in TEXT' → three quoted strings {old, new, text}."""
    quoted = _quoted_strings(query)
    if len(quoted) == 3:
        return True, {"old": quoted[0], "new": quoted[1], "text": quoted[2]}, "code", 0
    return False, None, "code", 0


# --- geometry / measurement family ----------------------------------------------------
# (measurement keyword, shape keyword or None, skill, ordered dimension labels). Checked in
# order, so "surface area" beats "area" and shape-free formulas (circumference/hypotenuse)
# come first. dims=None marks the hypotenuse's special "legs A and B" extraction.
_GEOMETRY = [
    ("circumference", None, "geometry.circle_circumference", ["radius"]),
    ("hypotenuse", None, "geometry.hypotenuse", None),
    ("surface area", "sphere", "geometry.sphere_surface_area", ["radius"]),
    ("diagonal", "rectangle", "geometry.rectangle_diagonal", ["length", "width"]),
    ("volume", "cube", "geometry.cube_volume", ["side"]),
    ("volume", "sphere", "geometry.sphere_volume", ["radius"]),
    ("volume", "cylinder", "geometry.cylinder_volume", ["radius", "height"]),
    ("perimeter", "rectangle", "geometry.rectangle_perimeter", ["length", "width"]),
    ("area", "circle", "geometry.circle_area", ["radius"]),
    ("area", "rectangle", "geometry.rectangle_area", ["length", "width"]),
    ("area", "triangle", "geometry.triangle_area", ["base", "height"]),
    ("area", "square", "geometry.square_area", ["side"]),
]
_LEGS = re.compile(r"legs?\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def detect_geometry(query: str):
    """(skill, dim_labels) for the matching shape+measurement, else None."""
    q = query.lower()
    for measure, shape, skill, dims in _GEOMETRY:
        if measure in q and (shape is None or shape in q):
            return skill, dims
    return None


def bind_geometry(query: str) -> tuple[bool, dict | None, str, int]:
    """Detect the formula, then extract each labeled dimension ('radius 5', 'length 4')."""
    det = detect_geometry(query)
    if not det:
        return False, None, "code", 0
    skill, dims = det
    if dims is None:  # hypotenuse: "legs A and B"
        m = _LEGS.search(query)
        if not m:
            return False, None, "code", 0
        return True, {"a": float(m[1]), "b": float(m[2]), "skill": skill,
                      "dims": ["a", "b"]}, "code", 0
    data = {"skill": skill, "dims": dims}
    for label in dims:
        m = re.search(rf"{label}\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", query, re.IGNORECASE)
        if not m:
            return False, None, "code", 0
        data[label] = float(m[1])
    return True, data, "code", 0


def validate_geometry(data: dict) -> tuple[bool, str]:
    for label in data["dims"]:
        if data[label] <= 0:
            return False, f"the {label} must be positive"
    return True, ""


# --- number-theory family -------------------------------------------------------------
# (keywords, skill, arity, return-kind). Ordered so "prime factorization" beats "prime",
# "divisor" is checked before "factorial" (which contains "factor"), gcd/lcm before divisor.
_NUMBER_THEORY = [
    (("prime factorization", "prime factors"), "numtheory.prime_factorization", 1, "string"),
    (("perfect square",), "numtheory.is_perfect_square", 1, "bool"),
    (("gcd", "greatest common"), "numtheory.gcd", 2, "int"),
    (("lcm", "least common"), "numtheory.lcm", 2, "int"),
    (("choose", "combination"), "numtheory.combinations", 2, "int"),
    (("permutation",), "numtheory.permutations", 2, "int"),
    (("divisor",), "numtheory.divisor_count", 1, "int"),
    (("factorial",), "numtheory.factorial", 1, "int"),
    (("arrange",), "numtheory.factorial", 1, "int"),
    (("prime",), "numtheory.is_prime", 1, "bool"),
]
_INT = re.compile(r"-?\d+")  # keep the sign so e.g. factorial(-3) declines, not silently 3!


def detect_number_theory(query: str):
    q = query.lower()
    for kws, skill, arity, kind in _NUMBER_THEORY:
        if any(k in q for k in kws):
            return skill, arity, kind
    return None


def bind_number_theory(query: str) -> tuple[bool, dict | None, str, int]:
    det = detect_number_theory(query)
    if not det:
        return False, None, "code", 0
    skill, arity, kind = det
    nums = [int(x) for x in _INT.findall(query)]
    if len(nums) < arity:
        return False, None, "code", 0
    labels = ["n", "k"][:arity]
    data = {"skill": skill, "dims": labels, "kind": kind}
    for i, label in enumerate(labels):
        data[label] = nums[i]
    return True, data, "code", 0


# --- chemistry / stoichiometry family -------------------------------------------------
_GRAMS = re.compile(r"(\d+(?:\.\d+)?)\s*g(?:rams)?\b")
_GMOL = re.compile(r"(\d+(?:\.\d+)?)\s*g/mol")
_MOLES_N = re.compile(r"(\d+(?:\.\d+)?)\s*mol(?:es)?\b")
_LITERS_N = re.compile(r"(\d+(?:\.\d+)?)\s*(?:liters?|litres?|l\b)")
_MOLAR_M = re.compile(r"(\d*\.?\d+)\s*M\b")        # molar concentration, e.g. "0.001 M"
_ML_N = re.compile(r"(\d+(?:\.\d+)?)\s*mL")
_FORMULA_CAND = re.compile(r"(?:[A-Z][a-z]?\d*|\([^)]*\)\d*)+")


def _find_formula(query: str):
    """The first capitalized token that parses to KNOWN elements (so 'What' → 'Wh' is
    rejected, 'H2O' is accepted) — the validation IS the disambiguation."""
    from .skills.chemistry import ATOMIC, parse_formula
    for tok in _FORMULA_CAND.findall(query):
        counts = parse_formula(tok)
        if counts and all(el in ATOMIC for el in counts):
            return tok
    return None


def bind_chemistry(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    # order disambiguates shared words: "molarity"/"diluted" before "moles"/"molar mass"
    if "molarity" in q:
        mol, vol = _MOLES_N.search(query), _LITERS_N.search(query)
        if mol and vol:
            return True, {"skill": "chem.molarity", "dims": ["moles", "volume"], "suffix": " M",
                          "moles": float(mol[1]), "volume": float(vol[1])}, "code", 0
    elif "diluted" in q or "dilution" in q:
        c1, mls = _MOLAR_M.search(query), _ML_N.findall(query)
        if c1 and len(mls) >= 2:
            return True, {"skill": "chem.dilution_concentration", "dims": ["c1", "v1", "v2"],
                          "suffix": " M", "c1": float(c1[1]), "v1": float(mls[0]),
                          "v2": float(mls[1])}, "code", 0
    elif "mole" in q:
        mass, mm = _GRAMS.search(query), _GMOL.search(query)
        if mass and mm:
            return True, {"skill": "chem.moles", "dims": ["mass", "molar_mass"], "suffix": " mol",
                          "mass": float(mass[1]), "molar_mass": float(mm[1])}, "code", 0
    elif "molar mass" in q:
        f = _find_formula(query)
        if f:
            return True, {"skill": "chem.molar_mass", "dims": ["formula"], "suffix": " g/mol",
                          "formula": f}, "code", 0
    elif re.search(r"\bph\b", q):
        c = _MOLAR_M.search(query)
        if c:
            return True, {"skill": "chem.ph", "dims": ["concentration"], "suffix": "",
                          "concentration": float(c[1])}, "code", 0
    return False, None, "code", 0


# --- physics-formula family -----------------------------------------------------------
# Quantities are cued by their UNIT, so each input is "the number before unit X". Patterns
# are case-sensitive (kg/N/m/s have meaningful case) and ordered so m/s^2 isn't read as m/s.
_PHYSICS_UNITS = {
    "KG": r"kg", "G": r"g(?:rams)?\b", "VEL": r"m/s(?![\^²2])", "ACCEL": r"m/s(?:\^?2|²)",
    "METERS": r"meters|metres", "SECONDS": r"seconds?", "N": r"N\b|newtons?",
    "JOULES": r"joules?|J\b", "AMPS": r"amps?|amperes?", "OHM": r"ohms?",
    "AREA": r"square meters|m\^?2|m²", "CM3": r"cm\^?3|cm³|cubic centimeters?",
}
# (asked-quantity keywords, skill, [(arg, unit-key)], answer suffix). Order: compound and
# specific terms first (power before work; pressure/work before the generic "force").
_PHYSICS = [
    (("kinetic energy",), "physics.kinetic_energy", [("mass", "KG"), ("velocity", "VEL")], " J"),
    (("potential energy", "gravitational"), "physics.potential_energy",
     [("mass", "KG"), ("height", "METERS")], " J"),
    (("momentum",), "physics.momentum", [("mass", "KG"), ("velocity", "VEL")], " kg·m/s"),
    (("density",), "physics.density", [("mass", "G"), ("volume", "CM3")], " g/cm³"),
    (("voltage", "ohm"), "physics.ohms_law_voltage",
     [("current", "AMPS"), ("resistance", "OHM")], " V"),
    (("pressure",), "physics.pressure", [("force", "N"), ("area", "AREA")], " Pa"),
    (("power",), "physics.power", [("work", "JOULES"), ("time", "SECONDS")], " W"),
    (("work",), "physics.work", [("force", "N"), ("distance", "METERS")], " J"),
    (("weight",), "physics.weight", [("mass", "KG")], " N"),
    (("force",), "physics.force", [("mass", "KG"), ("acceleration", "ACCEL")], " N"),
    (("speed", "how fast"), "physics.speed", [("distance", "METERS"), ("time", "SECONDS")], " m/s"),
]
_FROM_TO_VEL = re.compile(r"from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _num_before(query: str, unit_re: str):
    """The number immediately before a unit token (case-sensitive — kg/N/m/s differ)."""
    m = re.search(rf"(-?\d+(?:\.\d+)?)\s*(?:{unit_re})", query)
    return float(m[1]) if m else None


def detect_physics(query: str):
    q = query.lower()
    for kws, skill, dims, suffix in _PHYSICS:
        if any(k in q for k in kws):
            return skill, dims, suffix
    if "acceleration" in q:  # checked last so "accelerating" in a force query doesn't claim it
        return "physics.acceleration", "ACCEL", " m/s²"
    return None


def bind_physics(query: str) -> tuple[bool, dict | None, str, int]:
    det = detect_physics(query)
    if not det:
        return False, None, "code", 0
    skill, dims, suffix = det
    if dims == "ACCEL":  # a = (v_final - v_initial) / time, from "X to Y in T seconds"
        m, t = _FROM_TO_VEL.search(query), _num_before(query, _PHYSICS_UNITS["SECONDS"])
        if not m or t is None:
            return False, None, "code", 0
        return True, {"v_initial": float(m[1]), "v_final": float(m[2]), "time": t,
                      "skill": skill, "dims": ["v_initial", "v_final", "time"],
                      "suffix": suffix}, "code", 0
    data = {"skill": skill, "dims": [d for d, _ in dims], "suffix": suffix}
    for name, ukey in dims:
        v = _num_before(query, _PHYSICS_UNITS[ukey])
        if v is None:
            return False, None, "code", 0
        data[name] = v
    return True, data, "code", 0


# --- list-statistic family (a list of numbers + which summary) ------------------------
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# keyword → single-list stats skill. First match wins, so order most → least specific
# (e.g. "geometric mean" must beat "mean", "standard deviation" must beat plain words).
_STAT_OPS = [
    (("geometric",), "stats.geometric_mean"),
    (("standard deviation", "std dev", "stdev"), "stats.stdev"),
    (("variance",), "stats.variance"),
    (("mode",), "stats.mode"),
    (("product",), "stats.product"),
    (("median",), "stats.median"),
    (("average", "mean"), "stats.mean"),
    (("sum", "total", "altogether"), "stats.total"),
    (("highest", "largest", "maximum", "max", "biggest"), "stats.maximum"),
    (("lowest", "smallest", "minimum", "min"), "stats.minimum"),
    (("range",), "stats.range_of"),
]


def detect_stat_op(query: str) -> str | None:
    q = query.lower()
    if "weighted" in q or "percentile" in q:
        return None  # these need extra inputs → dedicated orchestrators handle them
    for cues, skill in _STAT_OPS:
        if any(c in q for c in cues):
            return skill
    return None


def _nums(text: str) -> list:
    return [float(n) if "." in n else int(n) for n in _NUMBER.findall(text)]


def bind_list_statistic(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: a stat keyword + ≥2 numbers → {nums, skill}."""
    skill = detect_stat_op(query)
    nums = _nums(query)
    if skill and len(nums) >= 2:
        return True, {"nums": nums, "skill": skill}, "code", 0
    return False, None, "code", 0


_PERCENTILE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", re.IGNORECASE)


def bind_percentile(query: str) -> tuple[bool, dict | None, str, int]:
    """Percentile: the 'Nth percentile' value, and the list of numbers after 'of'."""
    m = _PERCENTILE.search(query)
    after = query.split(" of ", 1)[1] if " of " in query else query
    nums = _nums(after)
    if m and len(nums) >= 2:
        return True, {"nums": nums, "p": float(m.group(1))}, "code", 0
    return False, None, "code", 0


def validate_percentile(data: dict) -> tuple[bool, str]:
    if not (0 <= data["p"] <= 100):
        return False, "the percentile must be between 0 and 100"
    return True, ""


def bind_weighted_average(query: str) -> tuple[bool, dict | None, str, int]:
    """Weighted average: values (before 'weights'), weights (after). Splits on the plural
    'weights' so it never trips on 'weighted'."""
    parts = re.split(r"\bweights\b", query, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False, None, "code", 0
    left = parts[0].split(" of ", 1)[1] if " of " in parts[0] else parts[0]
    values, weights = _nums(left), _nums(parts[1])
    if values and weights:
        return True, {"values": values, "weights": weights}, "code", 0
    return False, None, "code", 0


def validate_weighted_average(data: dict) -> tuple[bool, str]:
    if len(data["values"]) != len(data["weights"]):
        return False, "the number of values and weights must match"
    if sum(data["weights"]) == 0:
        return False, "the weights must not sum to zero"
    return True, ""


# --- percentage-change family ("from X to Y") -----------------------------------------
_FROM_TO = re.compile(r"from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def bind_percent_change(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: 'from <old> to <new>' → {old, new}."""
    m = _FROM_TO.search(query)
    if m:
        return True, {"old": float(m.group(1)), "new": float(m.group(2))}, "code", 0
    return False, None, "code", 0


# --- unit-conversion family -----------------------------------------------------------
# Unit knowledge lives in skills.units (the canonical factor table); the bind only locates
# the value and the source/target unit phrases and normalizes them to canonical tokens.
from .skills import units as _units  # pure skill module, safe to import here

_HOW_MANY = re.compile(
    r"how many\s+([a-z/ ]+?)\s+(?:are|is|in|=).*?(-?\d+(?:\.\d+)?)\s*([a-z/ ]+?)\b",
    re.IGNORECASE | re.DOTALL)


def _find_unit(text: str):
    """The canonical unit for the longest alias phrase present (whole-token), else None."""
    low = text.lower()
    for phrase in _units.UNIT_PHRASES:
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            return _units.canonical(phrase)
    return None


def bind_unit_convert(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind → {value, from, to, to_sym}. Two phrasings:
      - "<number> <source> to|into <target>"        (source before target)
      - "how many <target> ... <number> <source>"   (target before source)
    Conservative: needs a number and a KNOWN unit on each side, else it bails.
    """
    num = _NUMBER.search(query)
    if not num:
        return False, None, "code", 0
    value, source, target = float(num.group()), None, None

    # phrasing 2 first: "how many <target> ... <n> <source>" (so 'in'/inches is a unit,
    # not mistaken for a separator).
    m = _HOW_MANY.match(query.strip())
    if m:
        tgt, src = _find_unit(m.group(1)), _find_unit(m.group(3))
        if tgt and src:
            value, target, source = float(m.group(2)), tgt, src

    # phrasing 1: "<source> to|into <target>"
    if source is None:
        parts = re.split(r"\bto\b|\binto\b", query.lower(), maxsplit=1)
        if len(parts) == 2:
            s, t = _find_unit(parts[0]), _find_unit(parts[1])
            if s and t:
                source, target = s, t

    if source and target:
        return True, {"value": value, "from": source, "to": target,
                      "to_sym": _units.SUFFIX.get(target, f" {target}")}, "code", 0
    return False, None, "code", 0


# --- compound-interest family ---------------------------------------------------------
_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*year", re.IGNORECASE)
_PERIODS = {  # compounding frequency word → periods per year
    "annually": 1, "annual": 1, "yearly": 1, "semiannually": 2, "semi-annually": 2,
    "quarterly": 4, "monthly": 12, "weekly": 52, "daily": 365,
}


def bind_compound_interest(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: principal ($), annual rate (%), years, and compounding frequency."""
    price, rate, years = _PRICE.search(query), _RATE.search(query), _YEARS.search(query)
    if not (price and rate and years):
        return False, None, "code", 0
    q = query.lower()
    n = next((p for word, p in _PERIODS.items() if word in q), 1)  # default annual
    data = {
        "principal": float(price.group(1)),
        "rate_pct": float(rate.group(1)),
        "years": float(years.group(1)),
        "n": n,
    }
    return True, data, "code", 0


# --- shared finance parsing -----------------------------------------------------------
def _dollars(query: str) -> list[float]:
    """All $-prefixed amounts in order — the positional inputs most finance queries use."""
    return [float(m) for m in _PRICE.findall(query)]


# payment/deposit cadence — keyed on explicit phrases so bare "annual" (as in "5% annual
# interest", which describes the RATE) never sets the frequency. Most specific first.
_CADENCE = [
    (("per month", "monthly", "a month", "each month", "every month", "/month"), 12),
    (("per quarter", "quarterly"), 4),
    (("per week", "weekly"), 52),
    (("semi-annually", "semiannually", "twice a year"), 2),
    (("per year", "annually", "yearly", "per annum"), 1),
]


def _periods_from(query: str, default: int) -> int:
    q = query.lower()
    for phrases, p in _CADENCE:
        if any(ph in q for ph in phrases):
            return p
    return default


def bind_roi(query: str) -> tuple[bool, dict | None, str, int]:
    """Return on investment: first $ = cost (old), second $ = sale (new)."""
    d = _dollars(query)
    if len(d) >= 2:
        return True, {"old": d[0], "new": d[1]}, "code", 0
    return False, None, "code", 0


def bind_margin(query: str) -> tuple[bool, dict | None, str, int]:
    """Profit margin: first $ = cost, second $ = price."""
    d = _dollars(query)
    if len(d) >= 2:
        return True, {"cost": d[0], "price": d[1]}, "code", 0
    return False, None, "code", 0


def bind_cagr(query: str) -> tuple[bool, dict | None, str, int]:
    """CAGR: first $ = begin, second $ = end, plus a year count."""
    d, years = _dollars(query), _YEARS.search(query)
    if len(d) >= 2 and years:
        return True, {"begin": d[0], "end": d[1], "years": float(years.group(1))}, "code", 0
    return False, None, "code", 0


def bind_present_value(query: str) -> tuple[bool, dict | None, str, int]:
    """Present value: first $ = future value, plus a rate and a year count."""
    d, rate, years = _dollars(query), _RATE.search(query), _YEARS.search(query)
    if d and rate and years:
        return True, {"fv": d[0], "rate_pct": float(rate.group(1)),
                      "years": float(years.group(1))}, "code", 0
    return False, None, "code", 0


def bind_annuity(query: str) -> tuple[bool, dict | None, str, int]:
    """Future value of an annuity: first $ = deposit, plus rate, years, frequency."""
    d, rate, years = _dollars(query), _RATE.search(query), _YEARS.search(query)
    if d and rate and years:
        return True, {"pmt": d[0], "rate_pct": float(rate.group(1)),
                      "years": float(years.group(1)), "n": _periods_from(query, 12)}, "code", 0
    return False, None, "code", 0


def bind_loan(query: str) -> tuple[bool, dict | None, str, int]:
    """Loan payment: first $ = principal, plus rate, years; payments default monthly."""
    d, rate, years = _dollars(query), _RATE.search(query), _YEARS.search(query)
    if d and rate and years:
        return True, {"principal": d[0], "rate_pct": float(rate.group(1)),
                      "years": float(years.group(1)), "n": _periods_from(query, 12)}, "code", 0
    return False, None, "code", 0


def bind_straight_line_dep(query: str) -> tuple[bool, dict | None, str, int]:
    """Depreciation: first $ = cost, second $ = salvage, plus a year count."""
    d, years = _dollars(query), _YEARS.search(query)
    if len(d) >= 2 and years:
        return True, {"cost": d[0], "salvage": d[1], "years": float(years.group(1))}, "code", 0
    return False, None, "code", 0


def bind_break_even(query: str) -> tuple[bool, dict | None, str, int]:
    """Break-even: $1 = fixed costs, $2 = price per unit, $3 = cost per unit."""
    d = _dollars(query)
    if len(d) >= 3:
        return True, {"fixed": d[0], "price": d[1], "cost": d[2]}, "code", 0
    return False, None, "code", 0


_PEOPLE = re.compile(r"(?:among|between|by)\s+(\d+)|(\d+)\s+(?:people|persons|friends|ways|guests)",
                     re.IGNORECASE)


def bind_tip_split(query: str) -> tuple[bool, dict | None, str, int]:
    """Tip and split: first $ = bill, a % tip, and a head count to divide by."""
    d, tip, ppl = _dollars(query), _RATE.search(query), _PEOPLE.search(query)
    if d and tip and ppl:
        people = int(ppl.group(1) or ppl.group(2))
        return True, {"bill": d[0], "tip_pct": float(tip.group(1)), "people": people}, "code", 0
    return False, None, "code", 0


# --- domain validators: (data) -> (ok, message) ---------------------------------------
# The value-space analog of the conservative bind: refuse to emit a number when the inputs
# are degenerate, rather than returning nonsense (e.g. negative break-even) or crashing
# (e.g. divide-by-zero). Each returns a plain-language reason on failure.
def _terms_ok(data: dict, amount_key: str, amount_label: str) -> tuple[bool, str]:
    if data[amount_key] <= 0:
        return False, f"the {amount_label} must be positive"
    if data.get("years", 1) <= 0:
        return False, "the number of years must be positive"
    if data.get("rate_pct", 0) <= -100:
        return False, "the interest rate is out of range"
    return True, ""


def validate_break_even(data: dict) -> tuple[bool, str]:
    if data["price"] <= data["cost"]:
        return False, (f"each unit sells for ${data['price']:g} but costs ${data['cost']:g} to "
                       "make — there is no break-even point when the selling price does not "
                       "exceed the unit cost")
    return True, ""


def validate_cagr(data: dict) -> tuple[bool, str]:
    if data["begin"] <= 0:
        return False, "the starting value must be positive"
    if data["years"] <= 0:
        return False, "the number of years must be positive"
    return True, ""


def validate_margin(data: dict) -> tuple[bool, str]:
    if data["price"] <= 0:
        return False, "the selling price must be positive"
    return True, ""


def validate_depreciation(data: dict) -> tuple[bool, str]:
    if data["years"] <= 0:
        return False, "the useful life (years) must be positive"
    if data["salvage"] > data["cost"]:
        return False, "salvage value cannot exceed the original cost for straight-line depreciation"
    return True, ""


def validate_compound(data: dict) -> tuple[bool, str]:
    return _terms_ok(data, "principal", "principal")


def validate_present_value(data: dict) -> tuple[bool, str]:
    return _terms_ok(data, "fv", "future value")


def validate_annuity(data: dict) -> tuple[bool, str]:
    return _terms_ok(data, "pmt", "deposit")


def validate_loan(data: dict) -> tuple[bool, str]:
    return _terms_ok(data, "principal", "loan principal")


def validate_tip_split(data: dict) -> tuple[bool, str]:
    if data["people"] <= 0:
        return False, "the number of people must be positive"
    return True, ""

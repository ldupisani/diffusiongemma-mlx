"""bind layer — math domains (see reasoning.bind)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
# shared entity extractors (single source); keep local names so call sites are unchanged
from reasoning.bind import entities as _E  # noqa: F401
from reasoning.bind.entities import _FROM_TO, _INT  # noqa: F401
from reasoning.bind.entities import numeric_list as _numeric_list  # noqa: F401
from reasoning.bind.entities import values as _nums  # noqa: F401


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
    (("arrange", "seated in a row", "in a row"), "numtheory.factorial", 1, "int"),
    (("prime",), "numtheory.is_prime", 1, "bool"),
]


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
    # Require EXACTLY the operation's operands. Extra integers (list markers "1.", numbers
    # from another clause) mean we can't tell which are the operands → decline rather than
    # grab the first N and emit a confident wrong answer (e.g. gcd(1,7) for a "1. …" list).
    if len(nums) != arity:
        return False, None, "code", 0
    labels = ["n", "k"][:arity]
    data = {"skill": skill, "dims": labels, "kind": kind}
    for i, label in enumerate(labels):
        data[label] = nums[i]
    return True, data, "code", 0


# --- number-base (radix) conversion family --------------------------------------------
_BASES = {"binary": 2, "hexadecimal": 16, "hex": 16, "octal": 8, "decimal": 10}
_BASE_WORD = re.compile(r"binary|hexadecimal|hex|octal|decimal", re.IGNORECASE)
_TO_BASE = re.compile(r"\b(?:to|into|in)\s+(binary|hexadecimal|hex|octal|decimal)\b",
                      re.IGNORECASE)
# strip base names + structural words so the value token is what remains. NOTE: the
# articles "a"/"an" are deliberately NOT stripped — "A" is also the hex digit ten; instead
# value selection (below) prefers a non-article token, treating a lone "a"/"A" as the value
# only when nothing else qualifies.
_RADIX_NOISE = re.compile(
    r"\b(?:binary|hexadecimal|hex|octal|decimal|convert|number|the|what|is|from|to|into|"
    r"in|of|value)\b", re.IGNORECASE)


def bind_radix(query: str) -> tuple[bool, dict | None, str, int]:
    """Detect target base (after to/into/in), source base (any other base word, else
    decimal), and the value token that parses in the source base."""
    m_to = _TO_BASE.search(query)
    if not m_to:
        return False, None, "code", 0
    to_base = _BASES[m_to.group(1).lower()]
    from_base = 10
    for m in _BASE_WORD.finditer(query):  # first base word that isn't the target → source
        if m.start() != m_to.start(1):
            from_base = _BASES[m.group().lower()]
            break
    cleaned = _RADIX_NOISE.sub(" ", query)
    valid = []
    for tok in re.findall(r"[0-9A-Fa-f]+", cleaned):
        try:
            int(tok, from_base)  # the value must be valid in its source base
            valid.append(tok)
        except ValueError:
            continue
    # prefer a real value over a stray article; a lone "a"/"A" is the value only if alone
    value = next((t for t in valid if t.lower() not in ("a", "an")), valid[0] if valid else None)
    if value is None:
        return False, None, "code", 0
    return True, {"skill": "radix.convert", "dims": ["value", "from_base", "to_base"],
                  "value": value, "from_base": from_base, "to_base": to_base}, "code", 0


# --- list-statistic family (a list of numbers + which summary) ------------------------
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


def bind_list_statistic(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: a stat keyword + a list of ≥2 numbers → {nums, skill}. The numbers
    must form an actual list (delimiters only) — scattered numbers in prose ("total cost of
    100 yards at $3") would otherwise be summed into a confident wrong answer."""
    skill = detect_stat_op(query)
    nums = _numeric_list(query)
    if skill and len(nums) >= 2:
        return True, {"nums": nums, "skill": skill}, "code", 0
    return False, None, "code", 0


_PERCENTILE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", re.IGNORECASE)


def bind_percentile(query: str) -> tuple[bool, dict | None, str, int]:
    """Percentile: the 'Nth percentile' value, and the list of numbers after 'of'."""
    m = _PERCENTILE.search(query)
    after = query.split(" of ", 1)[1] if " of " in query else query
    nums = _numeric_list(after)  # a contiguous list, not scattered numbers
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


# --- percentage-change family ("from X to Y"; _FROM_TO imported from entities) --------
def bind_percent_change(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: 'from <old> to <new>' → {old, new}."""
    m = _FROM_TO.search(query)
    if m:
        return True, {"old": float(m.group(1)), "new": float(m.group(2))}, "code", 0
    return False, None, "code", 0


# --- set / inclusion-exclusion family -------------------------------------------------
def bind_sets(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    ints = [(m.start(), int(m.group())) for m in re.finditer(r"\d+", q)]
    if len(ints) < 3:
        return False, None, "code", 0
    both = both_pos = None
    mb = q.find("both")
    if mb != -1:
        before = [(pos, v) for pos, v in ints if pos < mb]
        if before:
            both_pos, both = before[-1]
    total = total_pos = None
    mt = re.search(r"(?:group of|out of|class of)\s+(\d+)|(\d+)\s+(?:people|students|surveyed)", q)
    if mt:
        total = int(mt.group(1) or mt.group(2))
        total_pos = mt.start(1) if mt.group(1) else mt.start(2)
    rest = [v for pos, v in ints if pos != both_pos and pos != total_pos]
    if both is None or len(rest) < 2:
        return False, None, "code", 0
    a, b = rest[0], rest[1]
    if "neither" in q:
        if total is None:
            return False, None, "code", 0
        return True, {"skill": "sets.neither", "dims": ["total", "a", "b", "both"],
                      "total": total, "a": a, "b": b, "both": both}, "code", 0
    if "only" in q:
        return True, {"skill": "sets.only_first", "dims": ["a", "both"],
                      "a": a, "both": both}, "code", 0
    return True, {"skill": "sets.union_count", "dims": ["a", "b", "both"],
                  "a": a, "b": b, "both": both}, "code", 0


# --- probability family (computable subset) -------------------------------------------
def bind_probability(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    if "does not" in q or "not rain" in q or "complement" in q:
        m = re.search(r"(\d+)\s*percent|(\d+)\s*%", q)
        if m:
            return True, {"skill": "prob.complement_percent", "dims": ["pct"], "suffix": "%",
                          "places": 2, "pct": float(m.group(1) or m.group(2))}, "code", 0
    elif "sum of" in q and ("dice" in q or "die" in q):
        m = re.search(r"sum of (\d+)", q)
        if m:
            return True, {"skill": "prob.dice_sum", "dims": ["target"], "places": 4,
                          "target": int(m[1])}, "code", 0
    elif "expected value" in q and ("die" in q or "dice" in q):
        return True, {"skill": "prob.expected_die", "dims": ["sides"], "places": 2,
                      "sides": 6}, "code", 0
    elif "heads" in q or ("coin" in q and "flip" in q):
        if "twice" in q:
            n = 2
        elif "thrice" in q:
            n = 3
        else:
            m = re.search(r"(\d+)\s*times", q)
            n = int(m[1]) if m else None
        if n:
            return True, {"skill": "prob.coin_all_heads", "dims": ["n"], "places": 4,
                          "n": n}, "code", 0
    elif ("roll" in q or "rolling" in q) and ("die" in q or "dice" in q):
        return True, {"skill": "prob.die_face", "dims": ["sides"], "places": 4,
                      "sides": 6}, "code", 0
    return False, None, "code", 0


# --- ratio / proportion / fraction family ---------------------------------------------
_FRAC = re.compile(r"(\d+)\s*/\s*(\d+)")
_RATIO = re.compile(r"(\d+)\s*:\s*(\d+)")
_DEC = re.compile(r"\d+\.\d+")


def bind_ratio(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    frac, rat = _FRAC.search(query), _RATIO.search(query)
    if "proportion" in q:
        m = re.search(r"(\d+)\s*:\s*(\d+)\s*=\s*x\s*:\s*(\d+)", q)
        if m:
            return True, {"skill": "ratio.solve_proportion", "dims": ["a", "b", "d"],
                          "places": 2, "a": float(m[1]), "b": float(m[2]),
                          "d": float(m[3])}, "code", 0
    elif "simplify" in q and rat and ":" in query:
        return True, {"skill": "ratio.simplify_ratio", "dims": ["a", "b"], "round": False,
                      "a": int(rat[1]), "b": int(rat[2])}, "code", 0
    elif "simplify" in q and frac:
        return True, {"skill": "ratio.simplify_fraction", "dims": ["num", "den"], "round": False,
                      "num": int(frac[1]), "den": int(frac[2])}, "code", 0
    elif "as a decimal" in q and frac:
        return True, {"skill": "ratio.fraction_to_decimal", "dims": ["num", "den"], "places": 4,
                      "num": float(frac[1]), "den": float(frac[2])}, "code", 0
    elif "as a fraction" in q:
        d = _DEC.search(query)
        if d:
            return True, {"skill": "ratio.decimal_to_fraction", "dims": ["x"], "round": False,
                          "x": float(d.group())}, "code", 0
    elif ("percent" in q or "percentage" in q) and frac:
        return True, {"skill": "ratio.fraction_to_percent", "dims": ["num", "den"], "suffix": "%",
                      "places": 2, "num": float(frac[1]), "den": float(frac[2])}, "code", 0
    elif "unit rate" in q:
        nums = _E.exactly(query, 2)  # exactly amount + per; a 3rd number ⇒ ambiguous ⇒ decline
        if nums:
            return True, {"skill": "ratio.unit_rate", "dims": ["amount", "per"], "places": 2,
                          "amount": float(nums[0]), "per": float(nums[1])}, "code", 0
    elif "servings" in q:
        cups = re.search(r"(\d+(?:\.\d+)?)\s*cups?", q)
        servs = re.findall(r"(\d+(?:\.\d+)?)\s*servings?", q)
        if cups and len(servs) >= 2:
            return True, {"skill": "ratio.scale", "dims": ["base", "from_n", "to_n"],
                          "places": 2, "suffix": " cups", "base": float(cups[1]),
                          "from_n": float(servs[0]), "to_n": float(servs[1])}, "code", 0
    elif "cost" in q and "how much" in q and "per" not in q:
        # item-scaling ("3 apples cost $2, how much for 12 apples?") — never has "per". A
        # "per <unit>" makes it a rate query this branch would mis-scale ⇒ decline.
        amt = re.search(r"\$\s*(\d+(?:\.\d+)?)", query)
        items = re.findall(r"(\d+)\s+[a-z]+", q)  # "3 apples", "12 apples"
        if amt and len(items) >= 2:
            return True, {"skill": "ratio.scale", "dims": ["base", "from_n", "to_n"],
                          "places": 2, "base": float(amt[1]), "from_n": float(items[0]),
                          "to_n": float(items[1])}, "code", 0
    return False, None, "code", 0


# --- health / body-metrics family -----------------------------------------------------
def bind_health(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    age = re.search(r"(\d+)\s*year", q)
    if "bmi" in q:
        w = re.search(r"(\d+(?:\.\d+)?)\s*kg", q)
        h = re.search(r"(\d+(?:\.\d+)?)\s*met(?:er|re)", q)
        if w and h:
            return True, {"skill": "health.bmi", "dims": ["weight_kg", "height_m"], "places": 2,
                          "weight_kg": float(w[1]), "height_m": float(h[1])}, "code", 0
    elif "target heart rate" in q and age:
        pct = re.search(r"(\d+)\s*percent|(\d+)\s*%", q)
        if pct:
            return True, {"skill": "health.target_heart_rate", "dims": ["age", "intensity_pct"],
                          "suffix": " bpm", "places": 0, "age": float(age[1]),
                          "intensity_pct": float(pct.group(1) or pct.group(2))}, "code", 0
    elif "heart rate" in q and age:
        return True, {"skill": "health.max_heart_rate", "dims": ["age"], "suffix": " bpm",
                      "places": 0, "age": float(age[1])}, "code", 0
    elif "pace" in q:
        dist = re.search(r"(\d+(?:\.\d+)?)\s*(?:miles?|mi|km|kilometer)", q)
        mins = re.search(r"(\d+(?:\.\d+)?)\s*min", q)
        unit = " minutes per kilometer" if ("km" in q or "kilomet" in q) else " minutes per mile"
        if dist and mins:
            return True, {"skill": "health.pace", "dims": ["minutes", "distance"],
                          "suffix": unit, "places": 2, "minutes": float(mins[1]),
                          "distance": float(dist[1])}, "code", 0
    elif "bmr" in q and age:
        w = re.search(r"(\d+(?:\.\d+)?)\s*kg", q)
        h = re.search(r"(\d+(?:\.\d+)?)\s*cm", q)
        if w and h:
            skill = "health.bmr_male" if ("man" in q or "male" in q) else "health.bmr_female"
            return True, {"skill": skill, "dims": ["weight_kg", "height_cm", "age"],
                          "suffix": " kcal/day", "places": 2, "weight_kg": float(w[1]),
                          "height_cm": float(h[1]), "age": float(age[1])}, "code", 0
    elif "ideal body weight" in q:
        h = re.search(r"(\d+(?:\.\d+)?)\s*cm", q)
        if h:
            return True, {"skill": "health.ideal_body_weight", "dims": ["height_cm"],
                          "suffix": " kg", "places": 0, "height_cm": float(h[1])}, "code", 0
    return False, None, "code", 0


# --- formal-logic family --------------------------------------------------------------
_PARENS = re.compile(r"\(([^()]+)\)")
_ASSIGN = re.compile(r"\b([A-Za-z])\s+is\s+(true|false)\b", re.IGNORECASE)
_CONDITIONAL = re.compile(r"\bif\s+(.+?)\s+then\s+(.+?)['\"?.\s]*$", re.IGNORECASE)


def _after(query: str, pattern: str):
    m = re.search(pattern, query, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _valid_expr(e) -> bool:
    if not e:
        return False
    from reasoning.skills.logic import compile_expr
    try:
        compile_expr(e)
        return True
    except Exception:
        return False


def bind_logic(query: str) -> tuple[bool, dict | None, str, int]:
    from reasoning.skills.logic import compile_expr, negatable_clause, _vars
    ql = query.lower()
    parens = _PARENS.findall(query)
    e0 = parens[0] if parens else None

    if "equivalent" in ql:
        if len(parens) >= 2 and _valid_expr(parens[0]) and _valid_expr(parens[1]):
            return True, {"skill": "logic.equivalent", "dims": ["e1", "e2"], "kind": "equivalent",
                          "e1": parens[0], "e2": parens[1]}, "code", 0
        return False, None, "code", 0

    for key, kind, skill in [("tautolog", "tautology", "logic.tautology"),
                             ("contradiction", "contradiction", "logic.contradiction"),
                             ("satisfiab", "satisfiable", "logic.satisfiable")]:
        if key in ql:
            e = e0 or _after(query, r"is\s+(.+?)\s+a\b")
            if _valid_expr(e):
                return True, {"skill": skill, "dims": ["expr"], "kind": kind, "expr": e}, "code", 0
            return False, None, "code", 0

    if "truth table" in ql:
        e = e0 or _after(query, r"truth table (?:for|of)\s+(.+?)[.?]*$")
        if _valid_expr(e):
            return True, {"skill": "logic.truth_table", "dims": ["expr"], "kind": "transform",
                          "expr": e}, "code", 0
        return False, None, "code", 0

    if "negation" in ql or "de morgan" in ql:
        e = e0 or _after(query, r"negation of\s+(.+?)[.?]*$")
        if _valid_expr(e):
            return True, {"skill": "logic.de_morgan", "dims": ["expr"], "kind": "transform",
                          "expr": e}, "code", 0
        return False, None, "code", 0

    if "converse" in ql or "inverse" in ql or "contrapositive" in ql:
        m = _CONDITIONAL.search(query)
        if not m:
            return False, None, "code", 0
        hyp, con = m.group(1).strip(" '\""), m.group(2).strip(" '\".?")
        if "converse" in ql:  # swap only — always clean
            return True, {"skill": "logic.converse", "dims": ["hypothesis", "conclusion"],
                          "kind": "transform", "hypothesis": hyp, "conclusion": con}, "code", 0
        if negatable_clause(hyp) and negatable_clause(con):  # else bail to the teacher
            skill = "logic.inverse" if "inverse" in ql else "logic.contrapositive"
            return True, {"skill": skill, "dims": ["hypothesis", "conclusion"],
                          "kind": "transform", "hypothesis": hyp, "conclusion": con}, "code", 0
        return False, None, "code", 0

    # default: evaluate an expression under an assignment (or all-constant)
    e = e0 or _after(query, r"evaluate\s+(.+?)\s+when") or _after(query, r"what is\s+(.+?)[.?]*$")
    if _valid_expr(e):
        assign = {m.group(1): m.group(2).lower() == "true" for m in _ASSIGN.finditer(query)}
        if set(_vars(compile_expr(e))) <= set(assign):
            return True, {"skill": "logic.evaluate", "dims": ["expr", "values"], "kind": "evaluate",
                          "expr": e, "values": assign}, "code", 0
    return False, None, "code", 0


# --- raw arithmetic family (single binary op / percent-of / power / sqrt) -------------
# Conservative: collect candidate operations and proceed ONLY if exactly one matches, so a
# compound expression ("15% of 200 plus 10") bails to the model rather than being mis-parsed.
# operator must sit BETWEEN two numbers ("7 times 8"), so the word "times" in "how many
# times …" is not mistaken for multiplication.
_N = r"(-?\d+(?:\.\d+)?)"
_ARITH_BINARY = [
    (re.compile(rf"{_N}\s*(?:\+|plus|added to)\s*{_N}", re.IGNORECASE), "arith.add"),
    (re.compile(rf"{_N}\s+minus\s+{_N}", re.IGNORECASE), "arith.subtract"),
    (re.compile(rf"{_N}\s+(?:times|multiplied by)\s+{_N}", re.IGNORECASE), "arith.multiply"),
    (re.compile(rf"{_N}\s+divided by\s+{_N}", re.IGNORECASE), "arith.divide"),
]
_ARITH_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_ARITH_POWER = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\^|to the power of|raised to(?: the power of)?)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE)
_ARITH_SQRT = re.compile(r"square root of\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_ARITH_NTH = re.compile(r"(-?\d+(?:\.\d+)?)\s+(squared|cubed)", re.IGNORECASE)


def bind_arithmetic(query: str) -> tuple[bool, dict | None, str, int]:
    cands = []
    m = _ARITH_SQRT.search(query)
    if m:
        cands.append({"skill": "arith.sqrt", "dims": ["x"], "x": float(m[1])})
    m = _ARITH_NTH.search(query)
    if m:
        cands.append({"skill": "arith.power", "dims": ["base", "exp"],
                      "base": float(m[1]), "exp": 2.0 if m[2].lower() == "squared" else 3.0})
    m = _ARITH_POWER.search(query)
    if m:
        cands.append({"skill": "arith.power", "dims": ["base", "exp"],
                      "base": float(m[1]), "exp": float(m[2])})
    m = _ARITH_PCT.search(query)
    if m:  # "X percent of Y" -> percent_of(value=Y, pct=X)
        cands.append({"skill": "arith.percent_of", "dims": ["value", "pct"],
                      "value": float(m[2]), "pct": float(m[1])})
    for pat, skill in _ARITH_BINARY:
        m = pat.search(query)
        if m:
            cands.append({"skill": skill, "dims": ["a", "b"],
                          "a": float(m[1]), "b": float(m[2])})
    if len(cands) != 1:  # zero (not arithmetic) or several (compound) → bail to the model
        return False, None, "code", 0
    if bind_expression(query)[0]:  # a compound expression (e.g. "(5+3)*2-4") → `expression`
        return False, None, "code", 0  # owns it; arithmetic only claims a lone operation
    return True, cands[0], "code", 0


# --- compound arithmetic expression family --------------------------------------------
# `arithmetic` handles ONE operation; `expression` handles ≥2 (full order-of-operations) by
# normalizing NL operators into a symbolic string and evaluating with skills/expr.py.
from reasoning.skills import expr as _expr  # noqa: E402

_EXPR_LEADIN = re.compile(r"(?i)\b(what(?:'s| is)|whats|calculate|evaluate|compute|"
                          r"the value of|how much is)\b")
_EXPR_SQRT_NUM = re.compile(r"(?i)square root of\s+(\d+\.?\d*)")
_EXPR_NL = [
    (re.compile(r"(?i)\bto the power of\b|\braised to(?: the power of)?\b"), "^"),
    (re.compile(r"(?i)\bplus\b|\badded to\b"), "+"),
    (re.compile(r"(?i)\bminus\b"), "-"),
    (re.compile(r"(?i)\btimes\b|\bmultiplied by\b"), "*"),
    (re.compile(r"(?i)\bdivided by\b"), "/"),
]
_EXPR_PCT = re.compile(r"(?i)(\d+\.?\d*)\s*(?:percent|%)\s+of\s+")


def _normalize_expression(query: str) -> tuple[str, bool]:
    """NL → symbolic. Returns (symbolic, used_nl): used_nl is True if any natural-language
    operator word was rewritten — an unambiguous signal the query is arithmetic (vs a bare
    "2024-01-01" that only *looks* like subtraction)."""
    s = _EXPR_LEADIN.sub(" ", query).strip().rstrip("?. ").replace(",", "")
    used_nl = False
    for pat, sym in [(_EXPR_SQRT_NUM, r"sqrt(\1)")] + _EXPR_NL:
        s2 = pat.sub(sym, s)
        if s2 != s:
            used_nl, s = True, s2
    s2 = _EXPR_PCT.sub(lambda m: f"({float(m[1]) / 100})*", s)  # "15% of " → "(0.15)*"
    if s2 != s:
        used_nl, s = True, s2
    return s, used_nl


def bind_expression(query: str) -> tuple[bool, dict | None, str, int]:
    """Bind a COMPOUND arithmetic expression (≥2 ops). Conservative guard so it never hijacks
    dates/IDs/ranges that merely look like subtraction."""
    symbolic, used_nl = _normalize_expression(query)
    # guard 1: stray words (other than sqrt) ⇒ a different domain ("days between … and …")
    if re.search(r"[A-Za-z]", re.sub(r"(?i)sqrt", "", symbolic)):
        return False, None, "code", 0
    # guard 2: a purely symbolic string with only digits/dots/dashes is a date/version/range,
    # not an expression — require a strong operator unless an NL operator word was used.
    if not used_nl and not re.search(r"[+*/^()%]", symbolic):
        return False, None, "code", 0
    try:
        ast = _expr.compile_expr(symbolic)
    except Exception:
        return False, None, "code", 0
    if _expr.op_count(ast) < 2:  # single op → leave it to `arithmetic`
        return False, None, "code", 0
    return True, {"skill": "expr.evaluate", "dims": ["expression"],
                  "expression": symbolic}, "code", 0


# --- clarify probes: query TYPE recognized, but a required field is missing -------------
# Each returns {field: (question, append_template)} for the missing fields, or None. The
# append_template formats the user's answer into text the bind above already parses, so the
# CLI re-runs with `query + appended text` — no new parser (see compiled.clarify).
def clarify_health(query: str) -> dict | None:
    q = query.lower()
    if "bmi" in q:
        missing = {}
        if not re.search(r"(\d+(?:\.\d+)?)\s*kg", q):
            missing["weight_kg"] = ("What is your weight in kilograms?", "{} kg")
        if not re.search(r"(\d+(?:\.\d+)?)\s*met(?:er|re)", q):
            missing["height_m"] = ("What is your height in meters?", "{} meters")
        return missing or None
    return None

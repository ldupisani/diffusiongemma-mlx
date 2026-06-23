"""bind layer — finance domains (see reasoning.bind)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403
from reasoning.bind.entities import _PRICE  # noqa: F401  (single source)
from reasoning.bind.entities import dollars as _dollars  # noqa: F401


# --- discount+tax family --------------------------------------------------------------
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
    from reasoning.patterns import extract_validate_repair
    r = extract_validate_repair(query, schema=_DISCOUNT_TAX_SCHEMA,
                                validate=validate_discount_tax, shot=shot)
    return (r.data is not None), r.data, "model", _evr_model_calls(r)


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
    if not (price and rate and years) or len(_dollars(query)) != 1:
        return False, None, "code", 0  # exactly one principal; a stray $ ⇒ ambiguous ⇒ decline
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
# _dollars = entities.dollars (imported above). Finance binds assign these BY POSITION, so
# they require an EXACT count (==2, ==3, …): a stray/leading $ ⇒ ambiguous operands ⇒ decline.


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
    if len(d) == 2:
        return True, {"old": d[0], "new": d[1]}, "code", 0
    return False, None, "code", 0


def bind_margin(query: str) -> tuple[bool, dict | None, str, int]:
    """Profit margin: first $ = cost, second $ = price."""
    d = _dollars(query)
    if len(d) == 2:
        return True, {"cost": d[0], "price": d[1]}, "code", 0
    return False, None, "code", 0


def bind_cagr(query: str) -> tuple[bool, dict | None, str, int]:
    """CAGR: first $ = begin, second $ = end, plus a year count."""
    d, years = _dollars(query), _YEARS.search(query)
    if len(d) == 2 and years:
        return True, {"begin": d[0], "end": d[1], "years": float(years.group(1))}, "code", 0
    return False, None, "code", 0


def bind_present_value(query: str) -> tuple[bool, dict | None, str, int]:
    """Present value: first $ = future value, plus a rate and a year count."""
    d, rate, years = _dollars(query), _RATE.search(query), _YEARS.search(query)
    if len(d) == 1 and rate and years:
        return True, {"fv": d[0], "rate_pct": float(rate.group(1)),
                      "years": float(years.group(1))}, "code", 0
    return False, None, "code", 0


def bind_annuity(query: str) -> tuple[bool, dict | None, str, int]:
    """Future value of an annuity: first $ = deposit, plus rate, years, frequency."""
    d, rate, years = _dollars(query), _RATE.search(query), _YEARS.search(query)
    if len(d) == 1 and rate and years:
        return True, {"pmt": d[0], "rate_pct": float(rate.group(1)),
                      "years": float(years.group(1)), "n": _periods_from(query, 12)}, "code", 0
    return False, None, "code", 0


def bind_loan(query: str) -> tuple[bool, dict | None, str, int]:
    """Loan payment: first $ = principal, plus rate, years; payments default monthly."""
    d, rate, years = _dollars(query), _RATE.search(query), _YEARS.search(query)
    if len(d) == 1 and rate and years:
        return True, {"principal": d[0], "rate_pct": float(rate.group(1)),
                      "years": float(years.group(1)), "n": _periods_from(query, 12)}, "code", 0
    return False, None, "code", 0


def bind_straight_line_dep(query: str) -> tuple[bool, dict | None, str, int]:
    """Depreciation: first $ = cost, second $ = salvage, plus a year count."""
    d, years = _dollars(query), _YEARS.search(query)
    if len(d) == 2 and years:
        return True, {"cost": d[0], "salvage": d[1], "years": float(years.group(1))}, "code", 0
    return False, None, "code", 0


def bind_break_even(query: str) -> tuple[bool, dict | None, str, int]:
    """Break-even: $1 = fixed costs, $2 = price per unit, $3 = cost per unit."""
    d = _dollars(query)
    if len(d) == 3:
        return True, {"fixed": d[0], "price": d[1], "cost": d[2]}, "code", 0
    return False, None, "code", 0


_PEOPLE = re.compile(r"(?:among|between|by)\s+(\d+)|(\d+)\s+(?:people|persons|friends|ways|guests)",
                     re.IGNORECASE)


def bind_tip_split(query: str) -> tuple[bool, dict | None, str, int]:
    """Tip and split: first $ = bill, a % tip, and a head count to divide by."""
    d, tip, ppl = _dollars(query), _RATE.search(query), _PEOPLE.search(query)
    if len(d) == 1 and tip and ppl:
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


# --- composed: total cost where quantity unit ≠ price unit (delegates a conversion) ----
_UNIT_WORD = (r"miles?|kilometers?|kilometres?|km|yards?|yd|meters?|metres?|ft|feet|"
              r"pounds?|lbs?|lb|kilograms?|kg|grams?|g")
_CONVERTED_COST = re.compile(
    rf"(?i)cost of\s+(\d+(?:\.\d+)?)\s*({_UNIT_WORD})\s+at\s+\$?\s*(\d+(?:\.\d+)?)\s*"
    rf"(?:per|/|a)\s+({_UNIT_WORD})")


def bind_converted_cost(query: str) -> tuple[bool, dict | None, str, int]:
    """'(total) cost of <qty> <unitA> at $<price> per <unitB>' → {qty, frm, price, to}.
    The price is per <unitB> but the quantity is in <unitA>, so the total needs a unit
    conversion first — which the orchestrator delegates to `unit_conversion` via a SubStep."""
    m = _CONVERTED_COST.search(query)
    if not m:
        return False, None, "code", 0
    return True, {"qty": float(m[1]), "frm": m[2], "price": float(m[3]),
                  "to": m[4]}, "code", 0


# --- recognition bridge #2: finance discount → tax chains -------------------------------
# Reuses the discount/tax/people extractors and adds "N items at $X each". Claims a chain
# percent_discount_tax (plain discount+tax) can't do: a per-person SPLIT and/or a multi-item
# QUANTITY. Additive, not duplicative.
_QTY_AT = re.compile(r"(\d+)\s+\w+\s+at\s+\$\s*(\d+(?:\.\d+)?)\s*(?:each|apiece)", re.IGNORECASE)


def bind_synthesis_finance(query: str) -> tuple[bool, dict | None, str, int]:
    disc, tax, ppl, qty = (_DISCOUNT.search(query), _TAX.search(query),
                           _PEOPLE.search(query), _QTY_AT.search(query))
    if not (disc and tax) or not (ppl or qty):  # plain discount+tax → percent_discount_tax
        return False, None, "code", 0
    knowns = {"discount_pct": float(disc.group(1)), "tax_pct": float(tax.group(1))}
    if qty:  # "5 shirts at $20 each" → subtotal from unit price × quantity
        knowns["quantity"], knowns["unit_price"] = float(qty.group(1)), float(qty.group(2))
    else:
        price = _dollars(query)
        if len(price) != 1:
            return False, None, "code", 0
        knowns["price"] = price[0]
    suffix = ""
    if ppl:
        people = int(ppl.group(1) or ppl.group(2))
        if people <= 0:
            return False, None, "code", 0
        knowns["people"] = float(people)
    goal, suffix = ("per_person", " each") if ppl else ("final_price", "")
    from reasoning.synthesize import FINANCE_OPS, synthesize
    seq = synthesize(set(knowns), goal, FINANCE_OPS)
    if seq is None or len(seq) < 2:
        return False, None, "code", 0
    return True, {"knowns": knowns, "goal": goal, "prefix": "$", "suffix": suffix}, "code", 0

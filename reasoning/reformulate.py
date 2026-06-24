"""Model-assisted reformulation: when a single question doesn't route as-written, spend ONE
model call to rewrite it into a phrasing the coded engine recognizes, run that orchestrator
(0 further model calls), then check — deterministically, no model call — that the rewrite
didn't drift before trusting the coded answer. Returns None (→ the caller's LLM) if the
rewrite doesn't route, doesn't answer, or introduces a number absent from the original.

This is the single-question sibling of decompose.decompose_with_model (which splits a COMPOUND
query). Both turn NL the router can't parse into exact, code-computed answers at ~1 model call
instead of a full LLM answer.

The drift guard is deterministic number-preservation rather than a model YES/NO: a small local
model can't reliably self-verify (it rejects correct rewrites), whereas requiring every number
in the rewrite to appear in the original reliably catches the dominant failure mode — a changed
or invented quantity ("× 7" → "× 6") that would otherwise yield a confidently-wrong coded answer.
"""

from __future__ import annotations

import re

from .compiled import model_calls
from .decompose import _solve_clause
from .patterns import PatternResult

_REWRITE_PROMPT = (
    "Rewrite the question below as ONE standard, self-contained computation or factual "
    "lookup, phrased plainly. Preserve EVERY number, unit, and quantity exactly. Do NOT "
    "solve it and do NOT explain. Output ONLY the rewrite wrapped in <q></q> tags, e.g. "
    "<q>area of a circle with radius 5</q> or <q>convert 100 F to Celsius</q>. If it is not "
    "a single standard computation or lookup, output <q>NONE</q>.\n\n"
    "Question: {q}\n\n<q>"
)

# Map number words to digits so "five" in the original matches "5" in the rewrite. Composite
# words ("hundred", "thousand") map standalone — "one hundred" -> {1, 100}, which still lets a
# rewritten "100" pass the subset check below.
_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "dozen": 12,
}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Cheap pre-gate: skip the reformulation model call on purely conversational turns. A query
# "looks computational" if it has a digit, a number word, or a domain keyword. This is only an
# optimization — a false negative just defers to the caller's LLM (safe), so the bar is "could
# this plausibly be a computation/lookup?", kept deliberately inclusive.
_COMPUTE_WORDS = frozenset(_WORD_NUM) | frozenset({
    "percent", "percentage", "plus", "minus", "times", "multiply", "multiplied", "divide",
    "divided", "sum", "total", "average", "mean", "median", "product", "difference", "square",
    "squared", "sqrt", "root", "power", "factorial", "modulo", "remainder", "ratio",
    "proportion", "fraction", "decimal", "gcd", "lcm", "prime", "factor", "probability",
    "area", "perimeter", "volume", "radius", "diameter", "circumference", "circle",
    "rectangle", "triangle", "sphere", "square", "convert", "conversion", "celsius",
    "fahrenheit", "kelvin", "km", "kilometer", "kilometers", "mile", "miles", "meter",
    "meters", "metre", "metres", "foot", "feet", "inch", "inches", "kg", "gram", "grams",
    "pound", "pounds", "liter", "liters", "gallon", "gallons", "bmi", "interest", "discount",
    "tax", "tip", "profit", "margin", "speed", "velocity", "distance", "rate", "cost", "price",
    "mole", "moles", "molar", "mass", "density", "pressure", "force", "energy", "work", "leap",
    "weekday", "palindrome", "reverse", "count", "depreciation", "annuity", "loan", "payment",
})
_WORD_RE = re.compile(r"[a-z]+")


def _looks_computational(query: str) -> bool:
    if _NUM_RE.search(query):
        return True
    return bool(set(_WORD_RE.findall(query.lower())) & _COMPUTE_WORDS)


def _number_set(text: str) -> set[float]:
    """All numeric values in `text`, with number words normalized to digits."""
    normalized = re.sub(r"[a-z]+",
                        lambda m: str(_WORD_NUM.get(m.group(), m.group())),
                        text.lower())
    return {float(n) for n in _NUM_RE.findall(normalized)}


def _numbers_preserved(original: str, rewrite: str) -> bool:
    """True iff the rewrite introduces no number absent from the original (guards numeric drift
    such as 7 -> 6). A rewrite with no numbers — e.g. a factual lookup — trivially passes."""
    return _number_set(rewrite) <= _number_set(original)


def _extract_rewrite(text: str) -> str:
    """Pull the rewrite out of the model's output. The prompt primes "<q>", so the output
    begins with the rewrite and ends at "</q>" — take everything up to the FIRST closing tag
    (robust to trailing explanation or stray later tags), then drop an echoed opening tag if
    present. Fall back to the first non-empty line when the model emits no tag at all."""
    if "</q>" in text:
        body = text.split("</q>", 1)[0]
        body = body.split("<q>")[-1]   # keep text after an echoed opening tag, if any
    else:
        body = text.splitlines()[0] if text.strip() else ""
    return body.strip().strip('"').strip()


def reformulate_with_model(query: str, *, shot) -> PatternResult | None:
    """1-model-call reformulation: rewrite → coded answer → deterministic drift check. None if
    the rewrite bails, doesn't route to a coded orchestrator, or introduces a new number."""
    if not _looks_computational(query):
        return None  # pre-gate: don't spend a model call on a purely conversational turn
    rewrite = _extract_rewrite(shot(_REWRITE_PROMPT.format(q=query),
                                    max_tokens=64, quality="fast").text)
    if not rewrite or rewrite.upper().startswith("NONE"):
        return None
    solved = _solve_clause(rewrite)  # routes only to a coded orchestrator that genuinely answers
    if solved is None:
        return None
    _, r = solved
    if not _numbers_preserved(query, rewrite):  # guard against a drifted/invented quantity
        return None
    coded = model_calls(r) or 0
    trace = [
        {"step": "reformulate", "rewrite": rewrite, "model_calls": 1},
        {"step": "solve", "query": rewrite, "answer": r.answer, "model_calls": coded},
        {"step": "verify", "check": "numbers_preserved", "model_calls": 0},
        {"step": "summary", "model_calls": 1 + coded},
    ]
    return PatternResult(answer=r.answer, trace=trace,
                         data={"rewrite": rewrite, "source": "reformulate"})

"""Model-assisted reformulation: when a single question doesn't route as-written, spend ONE
model call to rewrite it into a phrasing the coded engine recognizes, run that orchestrator
(0 further model calls), then spend ONE more call to VERIFY the rewrite faithfully matches the
original before trusting the coded answer. Returns None (→ the full teacher) if the rewrite
doesn't route, doesn't answer, or fails verification.

This is the single-question sibling of decompose.decompose_with_model (which splits a COMPOUND
query). Both turn NL the router can't parse into exact, code-computed answers at ~1-2 model
calls instead of a full LLM answer. The verify step is the guard against the one real risk:
a drifted rewrite ("× 7" → "× 6") would otherwise yield a confidently-wrong *coded* answer.
"""

from __future__ import annotations

from .compiled import model_calls
from .decompose import _solve_clause
from .patterns import PatternResult

_REWRITE_PROMPT = (
    "Rewrite the question below as ONE standard, self-contained computation or factual "
    "lookup, phrased plainly (e.g. 'area of a circle with radius 5', 'convert 100 F to "
    "Celsius', '15% of 240'). Preserve EVERY number, unit, and quantity exactly. Do NOT "
    "solve it. If it is not a single standard computation or lookup, output exactly NONE.\n\n"
    "Question: {q}\n\nRewritten:"
)

_VERIFY_PROMPT = (
    "Question A: {original}\nQuestion B: {rewrite}\n\n"
    "Do A and B ask for the same result, using the same numbers and units? "
    "Answer with only YES or NO."
)


def _first_line(text: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip().strip('"').strip()
        if ln:
            return ln
    return ""


def reformulate_with_model(query: str, *, shot) -> PatternResult | None:
    """≤2-model-call reformulation: rewrite → coded answer → verify. None if any step fails."""
    rewrite = _first_line(shot(_REWRITE_PROMPT.format(q=query),
                               max_tokens=64, quality="fast").text)
    if not rewrite or rewrite.upper().startswith("NONE"):
        return None
    solved = _solve_clause(rewrite)  # routes only to a coded orchestrator that genuinely answers
    if solved is None:
        return None
    _, r = solved
    # Guard: confirm the rewrite didn't drift from the original before trusting a coded answer.
    verdict = shot(_VERIFY_PROMPT.format(original=query, rewrite=rewrite),
                   max_tokens=8, quality="fast").text
    if not verdict.strip().lower().startswith("y"):
        return None
    coded = model_calls(r) or 0
    trace = [
        {"step": "reformulate", "rewrite": rewrite, "model_calls": 1},
        {"step": "solve", "query": rewrite, "answer": r.answer, "model_calls": coded},
        {"step": "verify", "verdict": verdict.strip(), "model_calls": 1},
        {"step": "summary", "model_calls": 2 + coded},
    ]
    return PatternResult(answer=r.answer, trace=trace,
                         data={"rewrite": rewrite, "source": "reformulate"})

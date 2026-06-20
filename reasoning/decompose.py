"""Cross-domain decomposition: answer a compound query by splitting it into sub-questions
and routing EACH to a coded orchestrator — so the parts are model-free even when the whole
wasn't a single recognized type.

Two strategies, cheapest first:
  1. code-first split on conjunctions — 0 model calls. Conservative: only accepts a split
     when EVERY clause routes to an orchestrator that actually answers (not a clarify/decline
     and not the generic fallback), so a single query like "a rectangle with length 4 and
     width 6" is NOT mis-split.
  2. ≤1-call model fallback — the ONLY model call decomposes the query into sub-questions;
     each is then solved by a coded orchestrator (not the model). This replaces the old
     decompose_solve_combine's N+2 calls with ~1.
"""

from __future__ import annotations

import re

from . import router
from .compiled import model_calls, needs_input
from .patterns import PatternResult, _parse_list

# numbered/bulleted list items ("1. … 2. …"), with ANY separator between them (real newline,
# the literal "\n" people paste from shells, or none). Decimals like "3.5" are safe: the
# marker requires whitespace AFTER the dot.
_LIST_ITEM = re.compile(r"(?:\\n|\n|^|\s)\s*\d+[.)]\s+")
# most-specific separators first ("a, and b" before "a and b")
_SEPARATORS = ["\n", "\\n", "; ", ", and ", " and then ", " then ", " and "]


def _solve_clause(clause: str):
    """Route a clause to a coded orchestrator and run it. Returns (clause, result) only if it
    genuinely ANSWERED — not a clarify, not a decline, not the generic fallback."""
    clause = clause.strip().strip(".?").strip()
    if not clause:
        return None
    m = router.recognize(clause)
    # only coded (model_free) orchestrators — a clause must NOT trigger a model call, even
    # during compose.precondition (which runs on every recognize). Excludes fallback/clarify.
    if not m or not m.orchestrator.model_free or m.via == "clarify":
        return None
    r = m.orchestrator.run(clause)
    if needs_input(r) or r.answer.startswith("("):  # clarify / "(no result: …)"
        return None
    return clause, r


def _split(query: str) -> list[str]:
    items = [p.strip() for p in _LIST_ITEM.split(query) if p.strip()]
    if len(items) >= 2:  # a numbered/bulleted list — split regardless of inter-item separator
        return items
    for sep in _SEPARATORS:
        parts = [p for p in re.split(re.escape(sep), query, flags=re.IGNORECASE) if p.strip()]
        if len(parts) >= 2:
            return parts
    return [query]


def _combine(solved: list, *, decompose_calls: int = 0) -> PatternResult:
    total = decompose_calls + sum(model_calls(r) or 0 for _, r in solved)
    trace = [{"step": "decompose", "clauses": [c for c, _ in solved],
              "model_calls": decompose_calls}]
    for c, r in solved:
        trace.append({"step": "solve", "clause": c, "answer": r.answer,
                      "model_calls": model_calls(r)})
    trace.append({"step": "summary", "model_calls": total})
    return PatternResult(answer="; ".join(r.answer for _, r in solved), trace=trace,
                         data={"parts": [(c, r.answer) for c, r in solved]})


def decompose_code_first(query: str) -> PatternResult | None:
    """0-model-call decomposition, or None if the query doesn't split into answerable clauses."""
    parts = _split(query)
    if len(parts) < 2:
        return None
    solved = [_solve_clause(p) for p in parts]
    if not all(solved):
        return None
    return _combine(solved)


def decompose_with_model(query: str, *, shot) -> PatternResult | None:
    """≤1-model-call decomposition: the single model call only SPLITS the query; each
    sub-question is solved by a coded orchestrator. None if the sub-questions don't all route."""
    code = decompose_code_first(query)
    if code is not None:
        return code  # free path won — no model call at all
    out = shot("Break this request into independent sub-questions, one per line, numbered. "
               "Each must be answerable on its own.\n\n" + query, max_tokens=256, quality="fast")
    subqs = _parse_list(out.text)
    if len(subqs) < 2:
        return None
    solved = [_solve_clause(sq) for sq in subqs]
    if not all(solved):
        return None  # caller falls back to the full teacher
    return _combine(solved, decompose_calls=1)

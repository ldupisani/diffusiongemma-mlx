"""Fidelity: does a coded orchestrator reproduce the teacher's conclusion?

This is the gate that lets the graph grow without silently rotting. We run the query
through both the orchestrator (fast, thinking-off) and the teacher (slow, thinking-on),
then have a one-shot judge score their agreement on the substantive conclusion.
"""

from __future__ import annotations

import re

from .compiled import model_calls
from .engine import one_shot
from .registry import Orchestrator
from .teachers import get_teacher

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _as_number(text: str):
    """Extract a single number from an answer (strips $/, etc.), else None."""
    m = _NUM.search(text.replace(",", ""))
    return float(m.group()) if m else None


def fidelity(query: str, orchestrator: Orchestrator, *, teacher_seed: int = 0,
             teacher_backend: str = "diffusion") -> dict:
    t = get_teacher(teacher_backend).trace(query, seed=teacher_seed)
    teacher_ans = t.answer
    o = orchestrator.run(query)

    judge = one_shot(
        "Two answers to the same question are below. Output ONLY an integer 0-100 for how "
        "well they AGREE on the substantive conclusion (100 = same conclusion, 0 = unrelated "
        "or contradictory).\n\n"
        f"QUESTION: {query}\n\nANSWER A (reference): {teacher_ans}\n\nANSWER B: {o.answer}",
        max_tokens=8, quality="fast", temperature=0.0,
    )
    m = re.search(r"\d+", judge.text)
    score = min(int(m.group()), 100) if m else 0

    # numeric backstop: when both answers are numbers, exact agreement is ground truth —
    # it catches rounding regressions the fuzzy 0-100 judge would wave through.
    tn, on = _as_number(teacher_ans), _as_number(o.answer)
    numeric_match = (tn is not None and on is not None and abs(tn - on) < 1e-6)

    return {
        "query": query,
        "teacher": teacher_ans,
        "orchestrator": o.answer,
        "score": score,
        "model_calls": model_calls(o),
        "numeric_match": numeric_match,
        "trace": o.trace,
    }

"""Fidelity: does a coded orchestrator reproduce the teacher's conclusion?

This is the gate that lets the graph grow without silently rotting. We run the query
through both the orchestrator (fast, thinking-off) and the teacher (slow, thinking-on),
then have a one-shot judge score their agreement on the substantive conclusion.
"""

from __future__ import annotations

import re

from .compiled import model_calls
from .engine import one_shot
from .registry import Orchestrator, load_orchestrators
from .teachers import get_teacher

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _as_number(text: str):
    """Extract a single number from an answer (strips $/, etc.), else None."""
    m = _NUM.search(text.replace(",", ""))
    return float(m.group()) if m else None


def agrees(orch_answer: str, teacher_answer: str) -> bool:
    """Coded agreement check (no model judge, no sidecar): the coded answer matches the
    teacher's on the substantive value. Tries normalized-string containment first (the
    teacher is often verbose: "18.02 g/mol" vs "≈ 18.02 g/mol"), then a tolerant numeric
    compare (1% relative — coded answers round, teachers vary in precision)."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    o, t = norm(orch_answer), norm(teacher_answer)
    if o and (o in t or t in o):
        return True
    on, tn = _as_number(orch_answer), _as_number(teacher_answer)
    if on is not None and tn is not None:
        return abs(on - tn) <= 1e-6 + 0.01 * abs(tn)
    return False


def batch_fidelity(*, teacher_backend: str = "claude", orchestrators=None) -> list[dict]:
    """Run every coded (model-free) orchestrator's first example through both it and the
    teacher, and check agreement with the coded comparator. No sidecar needed: coded
    orchestrators make 0 model calls and the comparator is deterministic.
    """
    orchs = orchestrators if orchestrators is not None else load_orchestrators()
    teacher = get_teacher(teacher_backend)
    results = []
    for o in orchs:
        if not o.model_free or not o.example_queries:
            continue
        query = o.example_queries[0]
        r = o.run(query)
        t = teacher.trace(query)
        results.append({
            "id": o.id,
            "query": query,
            "orchestrator": r.answer,
            "teacher": t.answer.replace("\n", " ")[:60],
            "model_calls": model_calls(r),
            "agrees": agrees(r.answer, t.answer),
        })
    return results


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

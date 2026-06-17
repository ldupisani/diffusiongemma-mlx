"""Patterns of thinking, run by code.

Each pattern is pure control flow that composes atomic one-shot (thinking-off) calls.
The *reasoning* — decomposition, sequencing, ranking, voting — is here in Python; the
model only fills atomic slots. Patterns take injectable `shot`/`shot_n` callables
(defaulting to the engine) so they can be unit-tested with fakes, and each returns a
PatternResult carrying the answer plus a structured trace of what it did.
"""

from __future__ import annotations

import json as _json
from collections import Counter
from dataclasses import dataclass, field

from .engine import one_shot, one_shot_n

# The deterministic parsing primitives now live in bind.py (so the skill/compiled layers
# can share them without importing patterns); re-exported here for backward compatibility.
from .bind import _parse_int, _parse_list, _try_json  # noqa: F401


@dataclass
class PatternResult:
    answer: str
    trace: list = field(default_factory=list)
    data: object = None


# --- the patterns ---------------------------------------------------------------------
def decompose_solve_combine(query, *, max_subq=5, quality="balanced", shot=one_shot):
    """Split into independent sub-questions, answer each, synthesize."""
    trace = []
    dec = shot(
        f"Break this request into at most {max_subq} independent sub-questions whose "
        "answers together fully address it. Output ONLY a numbered list, one per line.\n\n"
        f"REQUEST: {query}",
        max_tokens=256, quality=quality,
    )
    subqs = _parse_list(dec.text)[:max_subq]
    trace.append({"step": "decompose", "output": subqs})
    if not subqs:  # decomposition failed → degrade to a single direct answer
        direct = shot(query, max_tokens=512, quality=quality)
        trace.append({"step": "fallback_direct", "output": direct.text})
        return PatternResult(answer=direct.text, trace=trace)

    pairs = []
    for i, sq in enumerate(subqs):
        a = shot(f"Answer concisely:\n{sq}", max_tokens=256, quality=quality)
        pairs.append((sq, a.text))
        trace.append({"step": f"solve[{i}]", "subquestion": sq, "output": a.text})

    combined = shot(
        "Using these sub-answers, write a complete answer to the original request.\n\n"
        f"REQUEST: {query}\n\n"
        + "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs),
        max_tokens=512, quality=quality,
    )
    trace.append({"step": "combine", "output": combined.text})
    return PatternResult(answer=combined.text, trace=trace)


def map_reduce(items, *, map_instruction, reduce_instruction, quality="balanced", shot=one_shot):
    """Apply a one-shot transform to each item (map), then aggregate (reduce)."""
    trace = []
    mapped = []
    for i, item in enumerate(items):
        r = shot(f"{map_instruction}\n\nITEM: {item}", max_tokens=256, quality=quality)
        mapped.append(r.text)
        trace.append({"step": f"map[{i}]", "item": item, "output": r.text})
    red = shot(
        f"{reduce_instruction}\n\n" + "\n".join(f"- {m}" for m in mapped),
        max_tokens=512, quality=quality,
    )
    trace.append({"step": "reduce", "output": red.text})
    return PatternResult(answer=red.text, trace=trace)


def generate_rank_select(
    query, *, n=3, criteria="correctness and completeness", quality="balanced",
    base_seed=0, shot=one_shot, shot_n=one_shot_n,
):
    """Generate n seed-diverse candidates, then pick the best (best-of-N)."""
    cands = shot_n(query, n, base_seed=base_seed, max_tokens=384, quality=quality)
    texts = [c.text for c in cands]
    trace = [{"step": "generate", "seeds": [c.seed for c in cands], "candidates": texts}]
    listing = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
    pick = shot(
        f"Choose the single best response by {criteria}. Output ONLY its number.\n\n"
        f"REQUEST: {query}\n\nRESPONSES:\n{listing}",
        max_tokens=8, quality="fast", temperature=0.0,
    )
    idx = min(max(_parse_int(pick.text, 1), 1), len(texts)) - 1
    trace.append({"step": "rank", "raw": pick.text, "chosen": idx + 1})
    return PatternResult(answer=texts[idx], trace=trace)


def self_consistency(query, *, n=5, quality="balanced", base_seed=0, normalize=None, shot_n=one_shot_n):
    """Sample n seed-diverse answers and take the majority (best for short answers)."""
    normalize = normalize or (lambda s: " ".join(s.lower().split()))
    cands = shot_n(query, n, base_seed=base_seed, max_tokens=128, quality=quality)
    buckets: dict[str, list[str]] = {}
    for c in cands:
        buckets.setdefault(normalize(c.text), []).append(c.text)
    counts = Counter({k: len(v) for k, v in buckets.items()})
    best_key = counts.most_common(1)[0][0]
    trace = [
        {"step": "sample", "answers": [c.text for c in cands]},
        {"step": "vote", "counts": {k: len(v) for k, v in buckets.items()}},
    ]
    return PatternResult(answer=buckets[best_key][0], trace=trace)


def extract_validate_repair(query, *, schema, validate, max_repairs=2, quality="balanced", shot=one_shot):
    """NL → typed data: extract JSON, validate with a coded predicate, repair on failure.

    `validate(obj) -> (ok: bool, errors: list[str])`. This is the 'bind' helper that turns
    a messy goal into the structured input a domain orchestrator/solver needs.
    """
    trace = []
    out = shot(
        "Extract the requested fields and output ONLY valid JSON, no prose.\n"
        f"FIELDS: {schema}\n\nINPUT: {query}",
        max_tokens=256, quality=quality, temperature=0.0,
    )
    trace.append({"step": "extract", "output": out.text})
    for attempt in range(max_repairs + 1):
        obj, perr = _try_json(out.text)
        if obj is not None:
            ok, errors = validate(obj)
            if ok:
                trace.append({"step": "valid", "attempt": attempt})
                return PatternResult(answer=_json.dumps(obj), data=obj, trace=trace)
        else:
            errors = [f"not valid JSON: {perr}"]
        if attempt == max_repairs:
            break
        out = shot(
            "Your previous output had problems. Fix them; output ONLY valid JSON.\n"
            f"FIELDS: {schema}\nPREVIOUS: {out.text}\nPROBLEMS: {errors}\nINPUT: {query}",
            max_tokens=256, quality=quality, temperature=0.0,
        )
        trace.append({"step": f"repair[{attempt}]", "problems": errors, "output": out.text})
    return PatternResult(answer=out.text, data=None, trace=trace)


# Name → callable, for registry builders and the classify-branch router.
PATTERNS = {
    "decompose_solve_combine": decompose_solve_combine,
    "map_reduce": map_reduce,
    "generate_rank_select": generate_rank_select,
    "self_consistency": self_consistency,
    "extract_validate_repair": extract_validate_repair,
}

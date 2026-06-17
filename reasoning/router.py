"""Recognize an incoming query's type and dispatch to its orchestrator.

Recognition is the make-or-break layer (semantic structure hides under surface words),
so the matcher is pluggable:
  - LexicalMatcher   : token-overlap baseline, dependency-free. The v0 default.
  - EmbeddingMatcher : stub — the scale-out path (embed example_queries, cosine rank).
  - ClassifyMatcher  : one-shot model call picks the best strategy (Classify-Branch as
                       a router). Selection is itself a pattern.

recognize() filters by each orchestrator's precondition guard, then asks the matcher to
select. None means "no known type" → the caller falls back to the teacher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .registry import Orchestrator, load_orchestrators

_STOP = set(
    "the a an of to and or for in on with is are be how what which who whom whose when "
    "where why i my me we our us your you it its this that these those do does can should "
    "would could please give me list find".split()
)


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 1 and w not in _STOP}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))  # overlap coefficient (forgiving for short queries)


@dataclass
class Match:
    orchestrator: Orchestrator
    score: float = 1.0
    via: str = ""


class Matcher(Protocol):
    def select(self, query: str, orchestrators: list[Orchestrator]) -> Optional[Match]: ...


class LexicalMatcher:
    """Token-overlap against example_queries, boosted by author-declared keywords.

    Surface words diverge across paraphrases, so example overlap alone is weak — the
    distinctive `keywords` an orchestrator declares (e.g. compare/vs/difference) carry
    most of the signal. One keyword hit → 0.5, two+ → strong. This is a deliberately
    crude baseline; EmbeddingMatcher is the real scale-out path.
    """

    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

    def select(self, query, orchestrators):
        q = _tokens(query)
        best: Optional[Match] = None
        best_rank: tuple = (-1.0, -1.0)
        for o in orchestrators:
            example_overlap = max(
                (_overlap(q, _tokens(c)) for c in [*o.example_queries, o.description] if c),
                default=0.0,
            )
            kw_hits = len(q & {k.lower() for k in o.keywords})
            kw_score = min(1.0, 0.5 * kw_hits)
            s = max(example_overlap, kw_score)
            # break ties by keyword score: declared keywords are the author's specificity
            # signal, so a discount+tax query prefers percent_discount_tax over the generic
            # numeric_consensus when both example-overlap to 1.0.
            rank = (s, kw_score)
            if rank > best_rank:
                best, best_rank = Match(o, s, via="lexical"), rank
        return best if best and best.score >= self.threshold else None


class EmbeddingMatcher:
    """Scale-out path (not built in v0)."""

    def select(self, query, orchestrators):
        raise NotImplementedError(
            "EmbeddingMatcher is a stub. v0 uses LexicalMatcher; the scale-out path is to "
            "embed each orchestrator's example_queries and rank by cosine similarity."
        )


class ClassifyMatcher:
    """Selection-as-a-pattern: one thinking-off call picks the best strategy (or none)."""

    def __init__(self, quality: str = "fast"):
        self.quality = quality

    def select(self, query, orchestrators):
        from .engine import one_shot

        listing = "\n".join(
            f"[{i + 1}] {o.query_type}: {o.description}" for i, o in enumerate(orchestrators)
        )
        r = one_shot(
            "Pick the number of the strategy that best fits the request, or 0 if none fit. "
            f"Output ONLY the number.\n\nREQUEST: {query}\n\nSTRATEGIES:\n{listing}",
            max_tokens=8, quality=self.quality, temperature=0.0,
        )
        m = re.search(r"\d+", r.text)
        idx = int(m.group()) if m else 0
        if 1 <= idx <= len(orchestrators):
            return Match(orchestrators[idx - 1], 1.0, via="classify")
        return None


def recognize(
    query: str, *, matcher: Optional[Matcher] = None,
    orchestrators: Optional[list[Orchestrator]] = None,
) -> Optional[Match]:
    matcher = matcher or LexicalMatcher()
    orchs = orchestrators if orchestrators is not None else load_orchestrators()
    orchs = [o for o in orchs if o.precondition(query)]
    if not orchs:
        return None
    return matcher.select(query, orchs)

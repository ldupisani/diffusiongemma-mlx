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

import math
import re
import zlib
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

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


# --- embedding backend -----------------------------------------------------------------
def _features(text: str) -> list[str]:
    """Word tokens + character 3/4/5-grams. Char n-grams give morphology robustness for
    free (kilometres≈kilometers, plurals) without a stoplist."""
    t = text.lower()
    feats = re.findall(r"[a-z0-9]+", t)
    s = f" {t} "
    for n in (3, 4, 5):
        feats.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return feats


def hashing_embed(texts: list[str], dim: int = 1024) -> list[list[float]]:
    """Dependency-free embedding via the feature-hashing trick: hash each feature into a
    fixed-width vector with a random sign, then L2-normalize. Deterministic (crc32, not the
    salted built-in hash), local, instant — no model, no sidecar. The default backend.

    Not semantic (it can't know two unrelated words mean the same thing), but it is a true
    cosine vector space over n-gram content, which is strictly stronger than exact-token
    overlap. Swap in a neural `embed` (sentence-transformers / MLX) for real semantics.
    """
    out = []
    for text in texts:
        vec = [0.0] * dim
        for f in _features(text):
            b = f.encode()
            vec[zlib.crc32(b) % dim] += 1.0 if (zlib.crc32(b + b"#") & 1) else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are L2-normalized → dot == cosine


class EmbeddingMatcher:
    """Rank orchestrators by cosine similarity of the query to their example_queries.

    The scale-out recognition path: as the library grows, keyword matching collides on
    shared surface words ("product", "mean", "year") — vector similarity over the whole
    example sentence disambiguates by context instead. `embed` is pluggable: the default
    `hashing_embed` is dependency-free and local; pass a neural embedder for true semantics.
    Per-text vectors are cached, so repeated routing is cheap.
    """

    def __init__(self, embed: Callable[[list[str]], list[list[float]]] = hashing_embed,
                 threshold: float = 0.30):
        self.embed = embed
        self.threshold = threshold
        self._cache: dict[str, list[float]] = {}

    def _vec(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.embed([text])[0]
        return self._cache[text]

    def select(self, query, orchestrators):
        qv = self.embed([query])[0]
        best: Optional[Match] = None
        for o in orchestrators:
            refs = [*o.example_queries, o.description]
            score = max((_cosine(qv, self._vec(r)) for r in refs if r), default=0.0)
            if best is None or score > best.score:
                best = Match(o, score, via="embedding")
        return best if best and best.score >= self.threshold else None


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
    all_orchs = orchestrators if orchestrators is not None else load_orchestrators()
    orchs = [o for o in all_orchs if o.precondition(query)]
    # Two tiers: a specific orchestrator always wins over a generic fallback (e.g.
    # numeric_consensus, whose always-true precondition + "many/count" keywords would
    # otherwise tie every new orchestrator). Fall to fallbacks only when nothing specific
    # matches above threshold.
    specific = [o for o in orchs if not o.fallback]
    # Three tiers. (1) Precise DOMAIN orchestrators win first — so a domain query that merely
    # contains "5 times 2" isn't hijacked by arithmetic. (2) BIND-GATED orchestrators
    # (arithmetic/expression) whose precondition passed a conservative parse are dispositive
    # when no domain orchestrator claimed the query — route on the parse, ignoring the score
    # threshold (a pure-symbolic "(5+3)*2-4" has no words to match). (3) Generic fallbacks.
    domain = [o for o in specific if not o.bind_gated]
    match = matcher.select(query, domain) if domain else None
    if match is not None:
        return match
    gated = [o for o in specific if o.bind_gated]
    if gated:
        return matcher.select(query, gated) or Match(gated[0], score=1.0, via="bind")
    # Clarify tier: nothing can ANSWER, but a known query type may just be missing a field —
    # ask the user rather than declining. Checked across ALL orchestrators (the strict
    # precondition excludes the under-specified query the probe is meant to catch).
    for o in all_orchs:
        if o.clarify_probe and o.clarify_probe(query):
            return Match(o, score=1.0, via="clarify")
    fallback = [o for o in orchs if o.fallback]
    return matcher.select(query, fallback) if fallback else None

"""Word-problem benchmark harness — measures the DETERMINISTIC engine's 0-model-call
accuracy on a held-out natural-language dataset (GSM8K and the like).

Design intent — this module is an *instrument*, not training data:

  * The dataset lives in the gitignored data dir (`$DIFG_DATA_DIR`, default <repo>/data),
    imported once as handles. Question text never enters version control.
  * This module contains NO question-specific logic. It is a generic route→compare loop.
    The only things we ever commit are this harness and (later) GENERAL capabilities.
  * Firewall against overfitting (you are the optimiser; you overfit by reading individual
    questions and adding code that happens to route them):
        - DEV slice  = GSM8K `train`  → inspectable. Mine its misses for *general* gaps.
        - TEST slice = GSM8K `test`   → AGGREGATE ONLY. Read the totals; never tune on,
          or even read, an individual test item. `run(..., slice="test")` refuses to emit
          per-item detail for exactly this reason.

The headline metric is `solved_0call` on the TEST slice: fraction answered CORRECTLY at
0 model calls — difg's code-only ethos made measurable. Grow it with general domains,
never with per-question handling.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from . import datasets

# GSM8K gold lives on the final line as "#### <number>" (commas allowed, e.g. "1,024").
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
# A signed decimal, optionally with thousands separators — used to scrape the engine's
# free-text answer. We deliberately ignore exponents/units (see the units blind-spot in
# docs/adding-domains.md §4): GSM8K answers are plain O(1)–O(1e4) magnitudes.
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_SLICE_HANDLE = {"dev": "gsm8k_train", "test": "gsm8k_test"}


def gold_answer(answer_text: str) -> float | None:
    """The reference numeric answer from a GSM8K `answer` field (text after `####`)."""
    m = _GOLD_RE.search(answer_text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def extract_value(engine_answer: str | None) -> float | None:
    """Pull the comparable number out of the engine's free-text answer.

    DESIGN CHOICE: GSM8K answers are a single number, but the engine replies in prose
    ("The area is 18.0 square meters", "$1,200.00", "9 eggs"). We take the LAST number in
    the string — the computed value almost always trails the sentence, and this naturally
    skips a leading restatement of an input. Commas/currency are stripped. Refine here if
    a class of answers puts the value elsewhere (the harness localises this judgment).
    """
    if not engine_answer:
        return None
    nums = _NUM_RE.findall(engine_answer)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(engine_answer: str | None, gold: float | None,
               *, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    """Tolerant numeric match (engine rounds; GSM8K golds are exact). Both must be present."""
    if gold is None:
        return False
    v = extract_value(engine_answer)
    if v is None:
        return False
    return abs(v - gold) <= max(abs_tol, rel_tol * abs(gold))


# GSM8K gold answers annotate every arithmetic step as a calculator marker "<<a op b=c>>".
# Counting them reads the problem's *arithmetic depth* straight off the gold — no LLM, no
# question-specific logic. The step-count histogram IS the reachable-ceiling curve: a perfect
# general N-step solver caps at the fraction of problems with <= N steps.
_STEP_RE = re.compile(r"<<([^>]*)>>")
_OP_CHARS = set("+-*/")
DEPTH_ORDER = ["single", "double", "multi(3-4)", "deep(5+)", "none"]


def structure(gold_text: str) -> dict:
    """Arithmetic structure of a GSM8K problem, read from its gold's calculator markers:
    step count, the operator set used, and a coarse depth bucket."""
    steps = _STEP_RE.findall(gold_text or "")
    n = len(steps)
    ops = set()
    for s in steps:
        expr = s.split("=", 1)[0]
        ops |= {c for c in expr if c in _OP_CHARS}
    depth = ("single" if n == 1 else "double" if n == 2 else
             "multi(3-4)" if 3 <= n <= 4 else "deep(5+)" if n >= 5 else "none")
    return {"steps": n, "ops": ops, "depth": depth}


def depth_profile(slice: str = "dev", limit: int | None = None):
    """The arithmetic-depth histogram over a slice's golds — NO routing, so it's cheap over the
    full slice. Returns (Counter depth->count, total). This is the reachable-ceiling diagnostic:
    intrinsic to the data, identical (modulo sampling) for dev and test."""
    if slice not in _SLICE_HANDLE:
        raise ValueError(f"slice must be one of {list(_SLICE_HANDLE)}")
    hist: Counter = Counter()
    for _q, answer in _rows(_SLICE_HANDLE[slice], limit):
        hist[structure(answer)["depth"]] += 1
    return hist, sum(hist.values())


@dataclass
class Outcome:
    matched: bool          # a coded domain (or compose) claimed the query
    model_calls: int       # 0 for a pure coded answer
    domain: str | None     # which orchestrator claimed it (or None)
    answer: str | None     # the engine's text answer (None if it needed input/LLM)


def solve(query: str) -> Outcome:
    """Route a query through the engine CODE-ONLY, mirroring mcp_server.ask exactly
    (specific model-free domain → 0-call compound split). Never invokes the LLM."""
    from . import router
    from .compiled import model_calls, needs_input
    from .decompose import decompose_code_first

    m = router.recognize(query)
    if m and m.orchestrator.model_free and not m.orchestrator.fallback:
        r = m.orchestrator.run(query)
        if needs_input(r):                       # matched but asked us for more input
            return Outcome(True, model_calls(r) or 0, m.orchestrator.id, None)
        return Outcome(True, model_calls(r) or 0, m.orchestrator.id, r.answer)
    composed = decompose_code_first(query)
    if composed is not None:
        return Outcome(True, model_calls(composed) or 0, "compose", composed.answer)
    return Outcome(False, 0, m.orchestrator.id if m else None, None)


def classify(o: Outcome, correct: bool) -> str:
    """The miss taxonomy — what converts a failure into a capability work-item.

    DESIGN CHOICE: these buckets are the gap report's vocabulary.
      correct         — solved at 0 calls (the win condition)
      wrong_answer    — a domain routed AND answered, but the value is wrong
                        (a binding/parsing/formula bug — usually the cheapest to fix)
      needs_llm       — matched a type that has no code answer (a fallback/clarify path)
      no_route        — nothing coded claimed it (a missing general capability)
    """
    if correct:
        return "correct"
    if not o.matched:
        return "no_route"
    if o.answer is None:
        return "needs_llm"
    return "wrong_answer"


def _rows(handle: str, limit: int | None):
    import pyarrow.parquet as pq
    path = datasets.data_dir() / f"{handle}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset '{handle}' in {datasets.data_dir()} — import GSM8K first "
            f"(see reasoning/bench.py docstring).")
    tbl = pq.read_table(path, columns=["question", "answer"])
    data = list(zip(tbl.column("question").to_pylist(), tbl.column("answer").to_pylist()))
    return data[:limit] if limit else data


@dataclass
class Report:
    slice: str
    total: int
    solved_0call: int           # correct AND 0 model calls — the headline
    buckets: Counter            # classify() -> count
    by_domain: Counter          # domain -> count routed (any outcome)
    domain_correct: Counter     # domain -> count correct
    no_route_depth: Counter     # depth bucket -> count among no_route misses
    solved_depth: Counter       # depth bucket -> count among solved@0call
    misses: list                # [(question, gold, Outcome)] — populated for DEV only

    @property
    def accuracy(self) -> float:
        return self.solved_0call / self.total if self.total else 0.0

    @property
    def coverage(self) -> float:
        claimed = self.total - self.buckets.get("no_route", 0)
        return claimed / self.total if self.total else 0.0


def run(slice: str = "test", limit: int | None = None, *, keep_misses: int = 40) -> Report:
    """Route every question in the slice code-only and score it against the gold.

    DEV keeps a sample of misses for inspection; TEST keeps none (aggregate-only firewall)."""
    if slice not in _SLICE_HANDLE:
        raise ValueError(f"slice must be one of {list(_SLICE_HANDLE)}")
    rows = _rows(_SLICE_HANDLE[slice], limit)

    buckets: Counter = Counter()
    by_domain: Counter = Counter()
    domain_correct: Counter = Counter()
    no_route_depth: Counter = Counter()
    solved_depth: Counter = Counter()
    misses: list = []
    solved = 0

    for question, answer in rows:
        gold = gold_answer(answer)
        depth = structure(answer)["depth"]
        o = solve(question)
        correct = o.model_calls == 0 and is_correct(o.answer, gold)
        bucket = classify(o, correct)
        buckets[bucket] += 1
        if o.matched and o.domain:   # a genuine code-only claim (not a rejected fallback)
            by_domain[o.domain] += 1
            if correct:
                domain_correct[o.domain] += 1
        if correct:
            solved += 1
            solved_depth[depth] += 1
        else:
            if bucket == "no_route":
                no_route_depth[depth] += 1
            if slice == "dev" and len(misses) < keep_misses:
                misses.append((question, gold, o))

    return Report(slice=slice, total=len(rows), solved_0call=solved, buckets=buckets,
                  by_domain=by_domain, domain_correct=domain_correct,
                  no_route_depth=no_route_depth, solved_depth=solved_depth, misses=misses)

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

import contextlib
import re
from collections import Counter
from dataclasses import dataclass
from urllib.error import URLError

from . import datasets


class _ModelCallAttempted(Exception):
    """Raised when routing/binding tries to reach the LLM sidecar during a code-only run.
    Some orchestrators declare model_free but have binds that fall back to the model on hard
    input (e.g. extract_validate_repair); for benchmarking that IS a non-code-only outcome."""


@contextlib.contextmanager
def _code_only_guard():
    """Block every model call at the choke point (engine._post) so a code-only bench run never
    touches the network — any attempt raises _ModelCallAttempted, caught by solve()."""
    from . import engine
    orig = engine._post

    def blocked(*_a, **_k):
        raise _ModelCallAttempted()

    engine._post = blocked
    try:
        yield
    finally:
        engine._post = orig

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

    try:
        m = router.recognize(query)
        if m and m.orchestrator.model_free and not m.orchestrator.fallback:
            r = m.orchestrator.run(query)
            if needs_input(r):                   # matched but asked us for more input
                return Outcome(True, model_calls(r) or 0, m.orchestrator.id, None)
            return Outcome(True, model_calls(r) or 0, m.orchestrator.id, r.answer)
        composed = decompose_code_first(query)
        if composed is not None:
            return Outcome(True, model_calls(composed) or 0, "compose", composed.answer)
        return Outcome(False, 0, m.orchestrator.id if m else None, None)
    except (_ModelCallAttempted, URLError):
        # A "model_free" path reached for the LLM on this input → not code-only solvable.
        return Outcome(True, 1, None, None)


def is_decline(answer: str | None) -> bool:
    """A coded NON-answer: the engine claimed the query but produced no usable value. Mirrors
    decompose._solve_clause — a "(no result: …)" string (or clarify) starts with "(" / is None."""
    return answer is None or answer.strip().startswith("(")


def classify(o: Outcome, correct: bool) -> str:
    """The miss taxonomy — what converts a failure into a capability work-item.

    DESIGN CHOICE: these buckets are the gap report's vocabulary, ordered by *harm*.
      correct       — solved at 0 calls (the win condition)
      wrong_answer  — committed to a WRONG NUMBER (the real safety bug: confident & wrong)
      needs_llm     — a model call was attempted (a leaky "model_free" path or fallback)
      declined      — claimed the query but honestly bailed ("(no result: …)" / clarify);
                      low harm — same end-state as no_route, but a routing-precision smell
      no_route      — nothing coded claimed it (a missing general capability)
    """
    if correct:
        return "correct"
    if not o.matched:
        return "no_route"
    if o.model_calls > 0:
        return "needs_llm"
    if is_decline(o.answer):
        return "declined"
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
    domain_wrong: Counter       # domain -> count of confident-WRONG numbers (safety target)
    no_route_depth: Counter     # depth bucket -> count among no_route misses
    solved_depth: Counter       # depth bucket -> count among solved@0call
    misses: list                # [(question, gold, Outcome)] — populated for DEV only

    @property
    def accuracy(self) -> float:
        return self.solved_0call / self.total if self.total else 0.0

    @property
    def answered(self) -> int:
        """Questions where the engine committed to a NUMBER (right or wrong) — excludes
        declines/no_route. The denominator for precision."""
        return self.solved_0call + self.buckets.get("wrong_answer", 0)

    @property
    def precision(self) -> float:
        """When the engine commits to a number, how often is it right? The safety metric:
        hardening routing should push this toward 1.0 by killing confident-wrong answers."""
        return self.solved_0call / self.answered if self.answered else 0.0

    @property
    def coverage(self) -> float:
        claimed = self.total - self.buckets.get("no_route", 0)
        return claimed / self.total if self.total else 0.0


@dataclass
class TranslateReport:
    slice: str
    total: int
    solved_1call: int           # correct after the LLM→program→execute path (1 model call)
    buckets: Counter            # solved / wrong_value / program_error / no_response
    misses: list                # [(question, gold, out)] sample (dev only)
    # unused-given confidence signal vs gold-derived correctness (over answered questions)
    flag_caught: int = 0        # answered-WRONG and flagged (true positive)
    flag_falsealarm: int = 0    # answered-CORRECT but flagged (false positive)
    answered_wrong: int = 0     # produced a number that was wrong (the denominator for recall)

    @property
    def accuracy(self) -> float:
        return self.solved_1call / self.total if self.total else 0.0

    @property
    def flag_recall(self) -> float:
        """Of answered-wrong cases, the fraction the unused-given flag caught."""
        return self.flag_caught / self.answered_wrong if self.answered_wrong else 0.0

    @property
    def flag_false_alarm(self) -> float:
        """Of answered-correct cases, the fraction the flag wrongly tripped on."""
        return self.flag_falsealarm / self.solved_1call if self.solved_1call else 0.0


def run_translate(slice: str = "dev", limit: int | None = 25, *, asker=None,
                  model: str = "haiku", backend: str = "claude",
                  keep_misses: int = 25) -> TranslateReport:
    """Hybrid LLM-as-translator over a slice: 1 model call per question turns prose into an
    arithmetic program, executed deterministically (translate.solve). Reports solved@1call.

    Makes REAL model calls unless `asker` is injected — keep `limit` small (cost). DEV only keeps
    a miss sample; TEST stays aggregate-only (firewall)."""
    from . import translate
    if slice not in _SLICE_HANDLE:
        raise ValueError(f"slice must be one of {list(_SLICE_HANDLE)}")
    rows = _rows(_SLICE_HANDLE[slice], limit)

    buckets: Counter = Counter()
    misses: list = []
    solved = flag_caught = flag_falsealarm = answered_wrong = 0
    for question, answer in rows:
        gold = gold_answer(answer)
        out = translate.solve(question, asker=asker, model=model, backend=backend)
        ans = out["answer"]
        flagged = bool(out.get("unused_givens"))
        correct = ans is not None and is_correct(str(ans), gold)
        if correct:
            solved += 1
            buckets["solved"] += 1
            if flagged:
                flag_falsealarm += 1               # correct answer the flag wrongly tripped on
        else:
            if out["error"] == "no response from the model":
                bucket = "no_response"
            elif out["error"]:
                bucket = "program_error"   # program was malformed/unsafe/undefined
            else:
                bucket = "wrong_value"     # program ran, but the translation was wrong
                answered_wrong += 1                # a confident-wrong NUMBER (the silent-wrong risk)
                if flagged:
                    flag_caught += 1
            buckets[bucket] += 1
            if slice == "dev" and len(misses) < keep_misses:
                misses.append((question, gold, out))

    return TranslateReport(slice=slice, total=len(rows), solved_1call=solved, buckets=buckets,
                           misses=misses, flag_caught=flag_caught, flag_falsealarm=flag_falsealarm,
                           answered_wrong=answered_wrong)


def run_consistency(slice: str = "dev", limit: int | None = 25, *, k: int = 5,
                    threshold: float = 0.8, sampler=None, backend: str = "sidecar",
                    model: str = "haiku") -> TranslateReport:
    """Self-consistency translator: draw K programs per question, majority-vote (translate
    .solve_consistent), score the vote vs gold. The confidence flag is `agreement < threshold`
    (default 1.0 ⇒ flag any non-unanimous answer); reports its recall / false-alarm vs gold,
    exactly like the unused-given check. Costs K model calls per question."""
    from . import translate
    if slice not in _SLICE_HANDLE:
        raise ValueError(f"slice must be one of {list(_SLICE_HANDLE)}")
    rows = _rows(_SLICE_HANDLE[slice], limit)

    buckets: Counter = Counter()
    misses: list = []
    solved = flag_caught = flag_falsealarm = answered_wrong = 0
    for question, answer in rows:
        gold = gold_answer(answer)
        out = translate.solve_consistent(question, k=k, sampler=sampler, backend=backend,
                                         model=model)
        ans = out["answer"]
        flagged = out["agreement"] < threshold        # not enough of the K draws agreed
        correct = ans is not None and is_correct(str(ans), gold)
        if correct:
            solved += 1
            buckets["solved"] += 1
            if flagged:
                flag_falsealarm += 1
        elif ans is None:
            buckets["no_answer"] += 1                 # all K draws failed to produce a program
        else:
            buckets["wrong_value"] += 1
            answered_wrong += 1
            if flagged:
                flag_caught += 1
            if slice == "dev" and len(misses) < 25:
                misses.append((question, gold, out))

    return TranslateReport(slice=slice, total=len(rows), solved_1call=solved, buckets=buckets,
                           misses=misses, flag_caught=flag_caught, flag_falsealarm=flag_falsealarm,
                           answered_wrong=answered_wrong)


def run(slice: str = "test", limit: int | None = None, *, keep_misses: int = 40) -> Report:
    """Route every question in the slice code-only and score it against the gold.

    DEV keeps a sample of misses for inspection; TEST keeps none (aggregate-only firewall)."""
    if slice not in _SLICE_HANDLE:
        raise ValueError(f"slice must be one of {list(_SLICE_HANDLE)}")
    rows = _rows(_SLICE_HANDLE[slice], limit)

    buckets: Counter = Counter()
    by_domain: Counter = Counter()
    domain_correct: Counter = Counter()
    domain_wrong: Counter = Counter()
    no_route_depth: Counter = Counter()
    solved_depth: Counter = Counter()
    misses: list = []
    solved = 0

    with _code_only_guard():            # offline-safe: any model-call attempt → needs_llm
        for question, answer in rows:
            gold = gold_answer(answer)
            depth = structure(answer)["depth"]
            o = solve(question)
            correct = o.model_calls == 0 and is_correct(o.answer, gold)
            bucket = classify(o, correct)
            buckets[bucket] += 1
            if o.matched and o.domain:  # a genuine code-only claim (not a rejected fallback)
                by_domain[o.domain] += 1
                if correct:
                    domain_correct[o.domain] += 1
                elif bucket == "wrong_answer":
                    domain_wrong[o.domain] += 1
            if correct:
                solved += 1
                solved_depth[depth] += 1
            else:
                if bucket == "no_route":
                    no_route_depth[depth] += 1
                if slice == "dev" and len(misses) < keep_misses:
                    misses.append((question, gold, o))

    return Report(slice=slice, total=len(rows), solved_0call=solved, buckets=buckets,
                  by_domain=by_domain, domain_correct=domain_correct, domain_wrong=domain_wrong,
                  no_route_depth=no_route_depth, solved_depth=solved_depth, misses=misses)

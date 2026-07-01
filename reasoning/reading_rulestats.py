"""Rule-usage frequency for ``reading.solve`` — which RULE produced each correctly-solved question, and
how often. A read-only analysis tool over the SAME GSM8K slices as ``reading_bench``/``reading_diff``.

A "rule" is one of the named arithmetic system-solvers (the ``_EARLY_SYSTEMS`` / ``_LATE_SYSTEMS``
registries), a ``solve()``-level recognizer (``partition`` / ``container_giveaway`` / ``equal_share`` /
``backward_inversion`` / ``backward_depletion``), or the general ``answer()`` fold (everything the
registries didn't claim — holdings, event-count, remainder, combined-Σ, difference, age algebra, …).

HOW ATTRIBUTION WORKS: ``solve()`` dispatches in a fixed order and returns the FIRST mechanism that
produces a value (short-circuit), so at most one rule "wins" per solve. We instrument every rule with a
recording wrapper INSIDE a self-restoring context (``_instrumented``) — importing or running this module
leaves ``reading.py``'s dispatch completely untouched outside the ``with`` block — then, for each solve,
attribute the answer to the highest-priority rule that fired (priority = ``solve()``'s dispatch order).

Reuses ``reading_bench._load`` / ``_gold`` so the slice partition + gold parsing are IDENTICAL to the
bench/diff harnesses (a divergence would silently misreport). Firewall-respecting: the counts are an
aggregate tally, so ``--slice test`` is a safe aggregate metric (no per-question TEST inspection). Never
imported by the unit tests. 0 model calls.

CLI:  python -m reasoning.reading_rulestats [--slice dev|dev-train|dev-holdout|test] [--limit N] [--commits]
      --commits  also tally WRONG commits per rule (precision column); default counts correct solves only.
"""
import argparse
import contextlib
from collections import Counter

from reasoning import reading
from reasoning.reading_bench import _load, _gold

# Recognizers ``solve()`` calls DIRECTLY (not through the two registries) — each needs its own wrapper.
# ``backward_inversion`` / ``equal_share`` ALSO appear in ``_LATE_SYSTEMS``; wrapping both the registry
# entry and the module function records the SAME canonical label (``setdefault`` keeps the first), so the
# GUARD-8 / answer-step direct calls are captured too without double-counting.
_DIRECT = ("_partition", "_container_giveaway", "_equal_share", "_backward_inversion", "_backward_depletion")

_RECORD: dict = {}   # canonical-label -> result, reset per solve (short-circuit ⇒ the winning rule)


@contextlib.contextmanager
def _instrumented():
    """Swap in recording wrappers around every rule, then RESTORE on exit — reading.py's dispatch is
    pristine outside this block (the tool has no import-time or lingering side effects on the engine)."""
    early0, late0 = list(reading._EARLY_SYSTEMS), list(reading._LATE_SYSTEMS)
    direct0 = {n: getattr(reading, n) for n in _DIRECT if hasattr(reading, n)}

    def _wrap_registry(reg, arity):
        wrapped = []
        for lbl, fn in reg:
            def make(lbl, fn):
                if arity == 2:
                    def w(premises, question):
                        r = fn(premises, question)
                        if r is not None:
                            _RECORD.setdefault(lbl, r)
                        return r
                else:
                    def w(premises, question, context, readings):
                        r = fn(premises, question, context, readings)
                        if r is not None:
                            _RECORD.setdefault(lbl, r)
                        return r
                return w
            wrapped.append((lbl, make(lbl, fn)))
        reg[:] = wrapped

    try:
        _wrap_registry(reading._EARLY_SYSTEMS, 2)
        _wrap_registry(reading._LATE_SYSTEMS, 4)
        for name, fn in direct0.items():
            def make(name, fn):
                def w(*a, **k):
                    r = fn(*a, **k)
                    if r is not None:
                        _RECORD.setdefault(name.strip("_"), r)
                    return r
                return w
            setattr(reading, name, make(name, fn))
        yield
    finally:
        reading._EARLY_SYSTEMS[:] = early0
        reading._LATE_SYSTEMS[:] = late0
        for name, fn in direct0.items():
            setattr(reading, name, fn)


def _priority() -> list:
    """The order ``solve()`` tries the rules: EARLY systems, then the GUARD-8 backward pair, then LATE
    systems, then the answer-step recognizers. First rule in this order that fired IS the winner."""
    order = [l for l, _ in reading._EARLY_SYSTEMS] \
        + ["backward_inversion", "backward_depletion"] \
        + [l for l, _ in reading._LATE_SYSTEMS] \
        + ["partition", "container_giveaway", "equal_share"]
    seen, out = set(), []
    for lbl in order:                                    # dedup (backward_inversion/equal_share appear twice)
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _winner(priority: list) -> str:
    for lbl in priority:
        if lbl in _RECORD:
            return lbl
    return "general_fold (answer)"


def _fold_subrule(question: str) -> str:
    """Coarsely bucket a general-fold solve by its ``answer()`` branch (question-shape heuristic — the
    fold's branches are inline, not separately dispatched, so this is an approximation, not exact)."""
    ql = question.lower()
    if any(w in ql for w in ("old", "age", "ages")):
        return "  · age algebra"
    if "average" in ql:
        return "  · average (Σ/N)"
    if any(w in ql for w in ("left", "remaining", "leftover", "change", " rest")):
        return "  · remainder (net holdings)"
    if ("more" in ql or "less" in ql or "fewer" in ql) and "than" in ql:
        return "  · difference (X − Y)"
    if any(w in ql for w in ("in total", "altogether", "combined", "in all", " total")):
        return "  · combined fold (Σ)"
    if any(w in ql for w in ("price", "cost", "worth")):
        return "  · wh-value (price/measure)"
    if any(w in ql for w in ("how much", "how many")):
        return "  · holdings / event-count"
    return "  · other"


def run(slice_name: str = "dev", limit=None, commits: bool = False) -> str:
    rows = _load(slice_name)
    if limit:
        rows = rows[:limit]
    priority = _priority()
    correct_by, commit_by, fold_sub = Counter(), Counter(), Counter()
    total_correct = total_commit = 0
    with _instrumented():
        for row in rows:
            q, gold = row["question"], _gold(row["answer"])
            _RECORD.clear()
            try:
                res = reading.solve(q)
            except Exception:
                continue
            if not isinstance(res, (int, float)) or isinstance(res, bool):
                continue                                 # a decline / verdict — not a committed number
            rule = _winner(priority)
            total_commit += 1
            commit_by[rule] += 1
            if gold is not None and abs(res - gold) < 1e-6:
                total_correct += 1
                correct_by[rule] += 1
                if rule.startswith("general_fold"):
                    fold_sub[_fold_subrule(q)] += 1

    out = [f"reading.solve rule-usage — slice={slice_name}  n={len(rows)}  (0 model calls)",
           f"  correct solves: {total_correct}   commits: {total_commit}   "
           f"precision: {100 * total_correct / total_commit:.1f}%" if total_commit else
           f"  correct solves: {total_correct}   commits: 0",
           "",
           f"  {'rule':32s} {'correct':>8s}" + (f" {'commits':>8s} {'prec':>6s}" if commits else ""),
           "  " + "-" * (42 + (16 if commits else 0))]
    # every rule that ever committed, ranked by correct solves (then commits)
    for rule in sorted(commit_by, key=lambda r: (-correct_by[r], -commit_by[r])):
        c, t = correct_by[rule], commit_by[rule]
        line = f"  {rule:32s} {c:8d}"
        if commits:
            line += f" {t:8d} {(100 * c / t if t else 0):5.0f}%"
        out.append(line)
    if fold_sub:
        out += ["", "  general_fold (answer) — correct-solve breakdown by branch (approx.):"]
        for lbl, n in fold_sub.most_common():
            out.append(f"  {lbl:32s} {n:8d}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rule-usage frequency for reading.solve (0 model calls).")
    ap.add_argument("--slice", default="dev", choices=["dev", "dev-train", "dev-holdout", "test"],
                    help="dev = all train rows (dev-train + dev-holdout); test = held-out (aggregate only).")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of questions (debug).")
    ap.add_argument("--commits", action="store_true",
                    help="also tally wrong commits + per-rule precision (default: correct solves only).")
    args = ap.parse_args(argv)
    print(run(args.slice, args.limit, args.commits))


if __name__ == "__main__":
    main()

"""GSM8K coverage harness for the model-free obligation parser (``reasoning/reading.py``).

An INDEPENDENT eval — never imported by the unit tests, so the parser is never tuned to the
benchmark's individual items (the overfitting firewall). Mirrors ``bench.py``'s convention:

  * DEV (= the gsm8k_train handle) is INSPECTABLE — it prints sample solves/misses + a blocker
    taxonomy so we can mine the next gap.
  * TEST (= the gsm8k_test handle) is AGGREGATE-ONLY — it reports numbers, never individual items.

Runs ``reading.solve`` over a slice and scores against the gold (the number after ``####``). 0 model
calls. The dataset lives in the gitignored data dir, so questions never enter version control.

PRECISION is THE safety metric (commit a number ⇒ be right; ``wrong`` must be 0). It is only honest
over the WHOLE slice — a habitual ``--limit 300`` silently turned the firewall into a 300-row prefix
and hid 66 confident-wrongs that live past it (precision 100%@300 vs 30.5%@full). So ``--limit`` now
prints a PARTIAL-FIREWALL warning, and ``--wrongs`` dumps every confident-wrong (the worklist to drive
to zero). Default runs the full slice.

    python -m reasoning.reading_bench --slice dev  [--limit N] [--show N] [--wrongs]
    python -m reasoning.reading_bench --slice test [--limit N]
"""
import os
import sys

from . import reading

DATA_DIR = os.environ.get(
    "DIFG_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


def _load(slice_name: str) -> list:
    import pyarrow.parquet as pq
    handle = "gsm8k_train" if slice_name == "dev" else "gsm8k_test"
    path = os.path.join(DATA_DIR, f"{handle}.parquet")
    return pq.read_table(path).to_pylist()


def _gold(answer_text: str):
    """The gold number — the value after the final ``####`` marker (commas/$ stripped)."""
    if "####" not in answer_text:
        return None
    raw = answer_text.rsplit("####", 1)[1].strip().replace(",", "").replace("$", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _as_number(result):
    """The committed numeric answer, or None if the engine declined (verdict string / deficiency)."""
    if isinstance(result, bool):
        return None
    return float(result) if isinstance(result, (int, float)) else None


def _unknown_words(text: str) -> set:
    """The set of unrecognized words across a paragraph — the actionable vocabulary gap."""
    out = set()
    for s in reading._sentences(text):
        for d in reading.deficiencies(reading.analyze(s)):
            if d.kind == "unknown-word":
                out.add(d.unit.surface.strip(".,?!:;$").lower())
    return out


def _blocker(text: str, unknown: set) -> str:
    """A coarse reason the engine couldn't commit. ($ and % now PARSE, so they're no longer blockers
    — a declining problem with them is blocked by its remaining unknowns or by unsupported structure.)"""
    if unknown:
        return "unknown-word"
    return "no-fact / unsupported structure"


def run(slice_name: str = "dev", limit=None, show: int = 6, all_wrongs: bool = False) -> str:
    rows = _load(slice_name)
    if limit:
        rows = rows[:limit]
    from collections import Counter
    n = len(rows)
    correct, wrong, declined, errored = [], [], [], []
    blockers: Counter = Counter()
    unknown_freq: Counter = Counter()
    fully_read = 0                                       # problems with EVERY word recognized

    for row in rows:
        q, gold = row["question"], _gold(row["answer"])
        unknown = _unknown_words(q)
        if not unknown:
            fully_read += 1
        try:
            result = reading.solve(q)
        except Exception as e:                          # a parse blow-up counts as a decline, tallied
            errored.append((q, gold, repr(e)))
            continue
        val = _as_number(result)
        if val is None:
            declined.append((q, gold, result))
            blockers[_blocker(q, unknown) if unknown else "fully-read but unsolved (structure)"] += 1
            unknown_freq.update(unknown)
        elif gold is not None and abs(val - gold) < 1e-6:
            correct.append((q, gold, val))
        else:
            wrong.append((q, gold, val))

    answered = len(correct) + len(wrong)
    out = [f"GSM8K reading-engine coverage — slice={slice_name}  n={n}  (0 model calls)",
           f"  correct:   {len(correct):4d}  ({100 * len(correct) / n:.1f}% of all)",
           f"  wrong:     {len(wrong):4d}",
           f"  declined:  {len(declined):4d}",
           f"  errored:   {len(errored):4d}"]
    if limit is not None:                                # the firewall is only honest on the WHOLE
        out.append(f"  ⚠ PARTIAL FIREWALL: limited to the first {limit} rows — precision/wrong below "
                   f"reflect a PREFIX, not all of {slice_name}. Run without --limit for the real number.")
    out.append(f"  committed: {answered:4d}"
               + (f"  →  precision {100 * len(correct) / answered:.1f}%  (THE safety metric: "
                  f"wrong must be 0)" if answered else ""))
    out.append(f"  coverage:  {100 * answered / n:.1f}%  (committed to a number)")
    out.append(f"  fully-read:{fully_read:4d}  ({100 * fully_read / n:.1f}%)  "
               f"— every word recognized (the vocabulary frontier; the ceiling for coverage)")

    if slice_name == "dev":                              # firewall: only DEV is inspectable
        out.append("\n  blockers among declines (where to aim next):")
        for b, c in blockers.most_common():
            out.append(f"    {c:4d}  {b}")
        out.append("\n  most frequent unknown words (the actionable vocabulary list):")
        out.append("    " + "  ".join(f"{w}×{c}" for w, c in unknown_freq.most_common(30)))
        if correct:
            out.append("\n  ✓ solved (sample):")
            out += [f"    [{v:g}=={g:g}] {q[:96]}" for q, g, v in correct[:show]]
        if wrong:
            # The confident-wrongs ARE the precision worklist: every one violates the core invariant
            # (commit a number ⇒ be right). `--wrongs` dumps the full list so none hide past the prefix.
            shown = wrong if all_wrongs else wrong[:show]
            head = "full list — the precision worklist" if all_wrongs else "sample — translation/structure slips"
            out.append(f"\n  ✗ wrong number ({head}):")
            out += [f"    [got {v:g}, gold {g:g}] {' '.join(q.split())[:96]}" for q, g, v in shown]
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    slice_name = args[args.index("--slice") + 1] if "--slice" in args else "dev"
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    show = int(args[args.index("--show") + 1]) if "--show" in args else 6
    print(run(slice_name, limit, show=show, all_wrongs="--wrongs" in args))

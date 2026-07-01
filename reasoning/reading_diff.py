"""Before/after diff harness for ``reading.solve`` — the per-increment SAFETY NET.

The aggregate coverage number (``reading_bench``) and the unit-test suite BOTH hide
confident-wrongs: an increment can post a +1 net dev gain while silently turning a
handful of previously-correct answers into wrong numbers (a real case this project hit:
a subject-elision cut broke 14 age problems under a net +1). Only a per-question
before/after comparison catches that. This module is that comparison, promoted from a
throwaway scratchpad script into a first-class, always-available tool.

It runs ``reading.solve`` over the INSPECTABLE dev slices (dev-train + dev-holdout — never
TEST, which stays aggregate-only), scoring each question against the gold, and buckets the
delta between two snapshots into:

  * SOLVED     — was decline/wrong, now correct              (the wins)
  * BROKE      — was correct, now wrong or declining         (regressions — must be 0)
  * DECL→WRONG — was declining (None), now a WRONG number    (new confident-wrongs — must be 0)

An increment SHIPS iff BROKE and DECL→WRONG are both empty and NET (solved − broke) > 0.
A diff-neutral change (0 net) is recorded, not committed (see the handoff memory).

It reuses ``reading_bench._load`` / ``_gold`` so it shares the EXACT holdout partition and
gold parsing — a divergence there would be a silent firewall leak.

Workflow (one increment):

    # 1. snapshot the current HEAD BEFORE editing
    python -m reasoning.reading_diff snapshot before.json
    # 2. make the edit
    # 3. snapshot the working tree
    python -m reasoning.reading_diff snapshot after.json
    # 4. read the verdict
    python -m reasoning.reading_diff diff before.json after.json

Snapshots default to ``$DIFG_DATA_DIR/reading_diff/{before,after}.json`` when no path is
given, so the two subcommands pair up with no arguments in the common case.

0 model calls. Never imported by the unit tests (firewall).
"""
import json
import os
import sys

from . import reading
from .reading_bench import _gold, _load

DATA_DIR = os.environ.get(
    "DIFG_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
_SNAP_DIR = os.path.join(DATA_DIR, "reading_diff")

# the two slices we diff over — the INSPECTABLE halves of DEV. TEST is deliberately excluded
# (aggregate-only; peeking at per-item TEST results would burn the single go/no-go look).
_SLICES = ("dev-train", "dev-holdout")


def _default_path(name: str) -> str:
    return os.path.join(_SNAP_DIR, f"{name}.json")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _correct(pred, gold) -> bool:
    p, g = _num(pred), _num(gold)
    return p is not None and g is not None and abs(p - g) < 1e-6


def _predict(question: str):
    """reading.solve → the committed number, or None on decline/error (declines are honest).

    solve() returns a BARE number when it commits, else a verdict string ("undetermined"/…) or None
    when it declines — mirror reading_bench._as_number so the two harnesses agree on what "committed"
    means (a bool is a logic verdict, not a number, and must not count)."""
    try:
        r = reading.solve(question)
    except Exception as e:                       # a crash is a decline for diffing purposes
        return f"ERR:{type(e).__name__}"
    if isinstance(r, bool):
        return None
    return float(r) if isinstance(r, (int, float)) else None


def snapshot(out_path: str = None) -> str:
    """Run reading.solve over the dev slices, dump {question: {slice, pred, gold}} to JSON."""
    out_path = out_path or _default_path("before")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    res = {}
    cases = [(s, r["question"], _gold(r["answer"])) for s in _SLICES for r in _load(s)]
    for i, (slc, q, gold) in enumerate(cases):
        res[q] = {"slice": slc, "pred": _predict(q), "gold": gold}
        if i % 200 == 0:
            print(f"  {i}/{len(cases)}", file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(res, f)
    return f"wrote {len(res)} predictions → {out_path}"


def diff(before_path: str = None, after_path: str = None) -> str:
    """Compare two snapshots; bucket the delta into SOLVED / BROKE / DECL→WRONG."""
    before_path = before_path or _default_path("before")
    after_path = after_path or _default_path("after")
    b = json.load(open(before_path))
    a = json.load(open(after_path))
    solved, broke, decl_wrong = [], [], []
    for q, av in a.items():
        bv = b.get(q)
        if bv is None:
            continue
        bp, ap, gold = bv["pred"], av["pred"], av["gold"]
        bc, ac = _correct(bp, gold), _correct(ap, gold)
        if not bc and ac:
            solved.append((av["slice"], q, ap, gold))
        elif bc and not ac:
            broke.append((av["slice"], q, bp, ap, gold))
        elif not bc and not ac and bp is None and ap is not None and not str(ap).startswith("ERR"):
            decl_wrong.append((av["slice"], q, ap, gold))

    out = []
    out.append(f"SOLVED (decline/wrong → correct): {len(solved)}")
    for s, q, ap, g in solved:
        out.append(f"  [{s}] {ap:g}=={g:g}  {' '.join(q.split())[:80]}")
    out.append(f"\nBROKE (correct → wrong/decline): {len(broke)}   ← MUST be 0")
    for s, q, bp, ap, g in broke:
        bp_s = f"{bp:g}" if _num(bp) is not None else bp
        ap_s = f"{ap:g}" if _num(ap) is not None else ap
        out.append(f"  [{s}] {bp_s}→{ap_s} (gold {g:g})  {' '.join(q.split())[:80]}")
    out.append(f"\nDECL→WRONG (None → wrong number): {len(decl_wrong)}   ← MUST be 0")
    for s, q, ap, g in decl_wrong:
        out.append(f"  [{s}] {ap:g} (gold {g:g})  {' '.join(q.split())[:80]}")
    net = len(solved) - len(broke)
    verdict = "SHIP" if (not broke and not decl_wrong and net > 0) else \
              ("NEUTRAL (record, don't commit)" if not broke and not decl_wrong else "REVERT")
    out.append(f"\nNET dev change: {net:+d}   VERDICT: {verdict}")
    return "\n".join(out)


def collisions(slice_arg: str = None, limit: int = None) -> str:
    """Audit the `_EARLY_SYSTEMS` registry for cross-solver COLLISIONS — a problem that MORE THAN ONE
    early system claims. The live dispatch returns the FIRST claimer and silently masks the rest, so a
    second claimer that DISAGREES on the value is a latent bug: reordering the registry (or adding a
    solver ahead of it) would flip the answer. This is exactly the failure class behind this domain's
    `_name_count` family-isolation bug and its 5 TEST regressions — a masked collision the aggregate
    coverage number and the unit suite both hide.

    Runs every early system (no short-circuit) on the IDENTICAL (premises, question) the live dispatch
    computes (`reading._parse_problem`), mirroring solve()'s unknown-word precision guard so only
    LIVE-REACHABLE problems count. Reports how many reachable problems have ≥2 claimers, and flags any
    where the claimers disagree (which MUST be 0).

    FIREWALL: this compares solvers to EACH OTHER, never to the gold, so it leaks no coverage-fitting
    signal and is safe to run on TEST as an aggregate robustness check. But per "never inspect
    individual TEST items", the per-problem listing is shown for DEV slices only; a TEST run prints
    counts only. `slice_arg`: None/"dev" → both inspectable dev slices; "test"/"dev-train"/"dev-holdout"
    → that slice."""
    slices = _SLICES if slice_arg in (None, "dev") else (slice_arg,)
    detail = "test" not in slices                     # firewall: no per-item listing on TEST
    cases = [(s, r["question"]) for s in slices for r in _load(s)]
    if limit:
        cases = cases[:limit]
    reached, multi = 0, []
    for slc, text in cases:
        parsed = reading._parse_problem(text)
        if parsed is None:
            continue
        premises, question = parsed
        # mirror solve()'s unknown-word guard: a problem that trips it returns before the early
        # dispatch runs, so any overlap there is not reachable by the live path.
        readings = [reading.analyze(c) for c in premises + [question]]
        if any(d.kind == "unknown-word" for rr in readings for d in reading.deficiencies(rr)):
            continue
        reached += 1
        claims = []
        for label, fn in reading._EARLY_SYSTEMS:
            try:
                r = fn(premises, question)
            except Exception as e:                      # a crashing solver is a decline for this audit
                r = None
            if r is not None:
                claims.append((label, r))
        if len(claims) >= 2:
            multi.append((slc, question, claims))
    disagree = [(s, q, c) for s, q, c in multi if len({str(v) for _, v in c}) > 1]
    out = [f"early-system collision audit over {'+'.join(slices)} (reached early dispatch: {reached})",
           f"  problems with >=2 claimers: {len(multi)}",
           f"  of those, claimers DISAGREE on the value: {len(disagree)}   <- latent bugs, MUST be 0"]
    if detail:
        for slc, q, claims in multi:
            vals = {str(v) for _, v in claims}
            tag = "DISAGREE" if len(vals) > 1 else "agree   "
            shown = ", ".join(f"{lab}={v}" for lab, v in claims)
            out.append(f"  [{slc:11}] {tag}: {shown}  {' '.join(q.split())[:64]}")
    else:
        out.append("  (per-item listing suppressed on TEST — aggregate-only firewall)")
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "diff"
    rest = args[1:]
    if cmd == "snapshot":
        print(snapshot(rest[0] if rest else None))
    elif cmd == "diff":
        print(diff(rest[0] if len(rest) > 0 else None, rest[1] if len(rest) > 1 else None))
    elif cmd == "collisions":
        # collisions [slice] [limit] — slice ∈ dev|dev-train|dev-holdout|test (default dev)
        slice_arg = rest[0] if rest and not rest[0].isdigit() else None
        lim = next((int(a) for a in rest if a.isdigit()), None)
        print(collisions(slice_arg, lim))
    else:
        print(__doc__)
        sys.exit(2)

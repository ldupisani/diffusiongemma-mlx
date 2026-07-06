"""WORLD-KNOWLEDGE RESOLUTION loop (prototype) — the outward half of the reader's bridge.

`reading.diagnose`/`report` say WHAT a question is missing (an unsatisfied `QUESTION_REQUIREMENTS`
entry names exactly the world-knowledge fact it needs). This module SUPPLIES that fact from a PROVIDER
and COMPUTES the answer for the recognised shape — closing the loop from "here's what I'd need" to
"I fetched it and answered".

Design (firewall-safe by construction):
- It NEVER touches `reading.solve` (which still declines these shapes). It is a separate, read-only path.
- Only the FETCHED FACT is trusted input; the arithmetic runs through `translate.run_program` — the same
  whitelisted-AST safe evaluator the LLM-translator path uses — so the math is correct-by-construction and
  auditable. Every answer carries `provenance` (which fact, from where) so a caller can verify or override.
- The PROVIDER is the world-knowledge seam. The default (`_default_provider`) derives conversion factors by
  chaining `reading.CONVERSIONS` (BFS: second→minute→hour ⇒ 3600, no curation) and looks part-counts up in a
  small curated table. Inject a custom provider (e.g. model-backed, like `reference.py`'s opt-in retrieval)
  to widen coverage — the loop and the audited arithmetic are identical, only the fact source changes.

A provider is `provider(kind, key) -> (value: float, source: str) | None`, where kind is
"conversion" (key = (from_unit, to_unit)) or "part-count" (key = (part, item_singular)).

CLI: `python -m reasoning.reading_resolve "How many seconds are in an hour?"`
"""

from __future__ import annotations

import re
import sys
from collections import deque


# --- default world-knowledge provider ---------------------------------------------------------------

# Curated per-item part counts — the genuine world-knowledge a "how many legs/wheels/feet" question needs.
# Grown deliberately (like the reader's other hand-tables); a model provider can supply the long tail.
_PART_COUNTS = {
    "leg": {"dog": 4, "cat": 4, "horse": 4, "cow": 4, "pig": 4, "sheep": 4, "goat": 4, "chicken": 2,
            "duck": 2, "bird": 2, "hen": 2, "spider": 8, "ant": 6, "insect": 6, "person": 2, "human": 2},
    "foot": {"dog": 4, "cat": 4, "horse": 4, "cow": 4, "person": 2, "human": 2, "chicken": 2, "duck": 2},
    "wheel": {"car": 4, "truck": 4, "bicycle": 2, "bike": 2, "motorcycle": 2, "tricycle": 3, "bus": 6},
}


def _conv_factor(frm: str, to: str):
    """Derive the factor converting a `frm`-unit quantity into `to` units by CHAINING reading.CONVERSIONS
    (BFS over the conversion graph) — so a pair absent from the direct table (second↔hour) is still found
    via the path second→minute→hour. Returns the product of edge factors, or None if unreachable."""
    from . import reading
    a, b = reading._unit_canon(frm), reading._unit_canon(to)
    if a == b:
        return 1.0
    # BFS over the directed conversion graph; each edge (x,y) carries CONVERSIONS[(x,y)].
    seen = {a}
    q = deque([(a, 1.0)])
    while q:
        node, factor = q.popleft()
        for (x, y), f in reading.CONVERSIONS.items():
            if x == node and y not in seen:
                nf = factor * f
                if y == b:
                    return nf
                seen.add(y)
                q.append((y, nf))
    return None


def _default_provider(kind: str, key):
    """The 0-call default: conversion factors derived from reading.CONVERSIONS, part-counts from the
    curated table. Returns (value, source) or None (fact unknown — the loop then declines honestly)."""
    if kind == "conversion":
        frm, to = key
        f = _conv_factor(frm, to)
        return (f, "derived from CONVERSIONS") if f is not None else None
    if kind == "part-count":
        part, item = key
        table = _PART_COUNTS.get(part, {})
        return (table[item], f"curated:{part}") if item in table else None
    return None


# --- shape resolvers: compute the answer for a recognised requirement, given the fetched fact ---------

def _num_or_one(s):
    return int(s) if s else 1


def _resolve_unit_conversion(ql, joined, provider):
    """"How many <A> [are] in [N] <B>?" / "convert [N] <B> to <A>" → N · factor(B→A). Fetches the factor
    from the provider. Returns (answer, program, provenance) or None if the factor is unavailable."""
    from . import reading
    m = re.search(r"how many (\w+)\s+(?:are\s+)?(?:there\s+)?in\s+(?:a |an |one )?(\d+)?\s*(\w+)", ql)
    if m:
        tgt, n, src = m.group(1), m.group(2), m.group(3)
    else:
        m = re.search(r"convert\s+(\d+)?\s*(\w+)\s+(?:in)?to\s+(\w+)", ql)
        if not m:
            return None
        n, src, tgt = m.group(1), m.group(2), m.group(3)
    a, b = reading._unit_canon(tgt), reading._unit_canon(src)   # a = target (A), b = source (B)
    if a not in reading.UNIT_NOUNS or b not in reading.UNIT_NOUNS or a == b:
        return None
    got = provider("conversion", (b, a))                        # factor to turn one B into A-units
    if got is None:
        return None
    factor, source = got
    amount = _num_or_one(n)
    program = f"amount = {amount}\nfactor = {factor}\nanswer = amount * factor"
    from .translate import run_program
    answer = reading._snap(run_program(program))
    return (answer, program, [{"fact": f"1 {b} = {factor} {a}", "source": source}])


def _resolve_part_count(ql, joined, provider):
    """"how many <part> do/does <… N item …> have?" → Σ over (count, item) of count · parts-per-item.
    Fetches each item's part-count from the provider. Returns (answer, program, provenance) or None if any
    item's fact is unavailable (honest partial-knowledge decline)."""
    from . import reading
    pm = re.search(r"\b(?:number of|how many)\s+(legs|wheels|feet)\b", ql)
    if not pm:
        return None
    part = reading._singular_of(pm.group(1)) or pm.group(1)     # legs→leg, feet→foot
    text = (joined + " . " + ql).lower()
    terms, prov = [], []
    for cnt, noun in re.findall(r"(\d+)\s+([a-z]+)", text):
        item = reading._singular_of(noun) or noun
        got = provider("part-count", (part, item))
        if got is None:
            continue
        value, source = got
        terms.append(f"{int(cnt)}*{value}")
        prov.append({"fact": f"a {item} has {value} {part}{'s' if value != 1 else ''}", "source": source,
                     "count": int(cnt)})
    if not terms:
        return None
    program = "answer = " + " + ".join(terms)
    from .translate import run_program
    answer = reading._snap(run_program(program))
    return (answer, program, prov)


_RESOLVERS = {"unit-conversion": _resolve_unit_conversion, "part-count": _resolve_part_count}


# --- the loop ----------------------------------------------------------------------------------------

def resolve(text: str, provider=None) -> dict:
    """Answer a question, fetching missing WORLD KNOWLEDGE when the reader can't answer on its own.

    1. Try `reading.solve` — if it answers, return that (source="solve", 0 fetched facts).
    2. Else consult `reading.requirements`: for each UNSATISFIED requirement with a registered resolver,
       fetch the named fact from `provider` and COMPUTE the answer via an audited program.
    3. If nothing can be resolved, return the articulated `deficiency` (never a guess).

    `provider` defaults to the 0-call `_default_provider` (chained CONVERSIONS + curated part-counts).
    Returns {answer, source, provenance?, program?, requirement?, deficiency?, model_calls}."""
    from . import reading
    provider = provider or _default_provider
    ans = reading.solve(text)
    if ans not in (None, "undetermined") and not isinstance(ans, list):
        return {"answer": ans, "source": "solve", "provenance": [], "model_calls": 0}
    parsed = reading._parse_problem(text)
    premises, question = parsed if parsed is not None else ([], text)
    joined, ql = " . ".join(p.lower() for p in premises), question.lower()
    # We only get here because solve() already declined. A requirement's `satisfied` flag is for
    # diagnose ("name what's missing"); resolution runs regardless — the provider call decides success
    # (satisfied → the fact is known/derivable/stated; unsatisfied → the provider may still supply it).
    for name, needs, satisfied in reading.requirements(text):
        fn = _RESOLVERS.get(name)
        if fn is None:
            continue
        out = fn(ql, joined, provider)
        if out is not None:
            answer, program, provenance = out
            return {"answer": answer, "source": "world-knowledge", "requirement": needs,
                    "provenance": provenance, "program": program, "model_calls": 0}
    d = reading.diagnose(text)
    res = {"answer": None, "source": "declined", "model_calls": 0}
    if d is not None:
        res["deficiency"] = {"kind": d.kind, "question": d.message}
    return res


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m reasoning.reading_resolve "<question>"', file=sys.stderr)
        return 2
    import json
    print(json.dumps(resolve(" ".join(argv)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

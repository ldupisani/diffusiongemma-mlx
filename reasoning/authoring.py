"""Author a new reasoning domain with Claude (high-reasoning), pre-loaded with the context it
needs: the full playbook (docs/adding-domains.md), the skill-registry contract, a worked exemplar
domain (all four pieces), and the list of existing query_types to avoid colliding with.

It builds a context-rich prompt, calls the model (Haiku by default — cheapest/fastest, and shown
to author a correct draft of a well-specified domain; --model opus for harder ones), and parses
the returned files from a delimiter format. This is CODE GENERATION — the playbook's own rule applies: review the
output, run the tests (`pytest` + `cli eval-all`), and route-test both directions before committing.
Nothing here runs the generated code; it only writes files (new ones; never clobbers existing).
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PLAYBOOK = "docs/adding-domains.md"
_BASE = "reasoning/skills/base.py"
# a clean, full worked example: skills + bind + orchestrator
_EXEMPLAR = ["reasoning/skills/logistics.py", "reasoning/bind/logistics.py",
             "reasoning/store/orchestrators/logistics.py"]

_FORMAT = """\
OUTPUT FORMAT — emit ONLY blocks delimited EXACTLY like this (no surrounding markdown fences):

<<<FILE reasoning/skills/<name>.py>>>
<file contents>
<<<END>>>

Produce these blocks in order:
 1. reasoning/skills/<name>.py     — pure @register'd skills, each with an EXACT example table
                                      (compute every value; never guess a float).
 2. reasoning/bind/<name>.py        — bind_<name> with a CONSERVATIVE gate; for fixed-arity numeric
                                      ops, wrap with guard_residue / decline on unexplained residue (§3d).
 3. reasoning/store/orchestrators/<name>.py — META / precondition / run; choose bind_gated per §2.
 4. <<<FILE __wiring__>>>           — the exact line to add to reasoning/skills/__init__.py, the line
                                      for reasoning/bind/__init__.py, and 3–6 CASES tuples
                                      (query, expected, orch_id) for tests, incl. ≥1 collision guard. <<<END>>>
 5. <<<FILE __notes__>>>            — collisions you guarded (both directions), and whether a dataset
                                      generator (§8) or a synthesis orchestrator (§9) is appropriate. <<<END>>>"""


def _read(rel: str) -> str:
    p = _REPO / rel
    return p.read_text() if p.exists() else f"(missing: {rel})"


def gather_context() -> str:
    """The full context block: playbook + registry contract + worked exemplar + existing domains."""
    from reasoning import introspect
    domains = ", ".join(sorted({d["type"] for d in introspect.domains()}))
    exemplar = "\n\n".join(f"### {f}\n```python\n{_read(f)}\n```" for f in _EXEMPLAR)
    return "\n\n---\n\n".join([
        f"# THE PLAYBOOK — follow it exactly ({_PLAYBOOK}):\n\n{_read(_PLAYBOOK)}",
        f"# THE SKILL-REGISTRY CONTRACT ({_BASE}):\n\n```python\n{_read(_BASE)}\n```",
        f"# A WORKED EXEMPLAR — the `logistics` domain, all pieces:\n\n{exemplar}",
        f"# EXISTING query_types — your new domain must NOT collide with these (gate so neither "
        f"steals the other; add decline-guards both directions per §3):\n{domains}",
    ])


def build_prompt(description: str, context: str | None = None) -> str:
    ctx = gather_context() if context is None else context
    return (
        "You are extending a deterministic, model-free reasoning engine: coded orchestrators answer "
        "queries at ZERO model calls. Implement a NEW domain by following the playbook below to the "
        "letter — conservative gating, exact example tables, both-direction routing, decline on "
        "unexplained residue.\n\n"
        f"DOMAIN TO IMPLEMENT:\n{description}\n\n{_FORMAT}\n\n---\n\n{ctx}")


def parse_files(text: str) -> dict:
    """Parse the delimiter format into {path: contents}. Meta blocks use __wiring__ / __notes__."""
    return {m.group(1).strip(): m.group(2)
            for m in re.finditer(r"<<<FILE (.+?)>>>\n(.*?)\n<<<END>>>", text, re.DOTALL)}


def author_domain(description: str, *, model: str = "haiku", effort: str | None = None,
                  timeout: int = 600, context: str | None = None, asker=None) -> dict:
    """Call Claude to implement the domain; return {files, raw, error}. `model`/`effort` pick the
    Claude model and reasoning level (effort: low/medium/high/xhigh/max, None → CLI default).
    Default `model` is haiku — it authored a correct draft of a fresh test domain (modulo a
    one-line example-rounding fix the test suite flags), at ~5x lower cost than opus; bump to
    `model="opus"` for harder or under-specified domains. The result also carries `usage` (token
    counts) and `cost_usd` (the CLI-computed dollar cost), populated on the real model path (None
    when a stub `asker` is injected). `asker(prompt)->str|None` is injectable for testing."""
    meter = {}
    if asker is None:
        from . import teachers

        def asker(p):
            r = teachers.ask_claude_usage(p, model=model, timeout=timeout, effort=effort)
            meter.update(usage=r["usage"], cost_usd=r["cost_usd"])
            return r["text"]
    raw = asker(build_prompt(description, context))
    base = {"usage": meter.get("usage"), "cost_usd": meter.get("cost_usd")}
    if not raw:
        return {"files": {}, "raw": None, **base,
                "error": "no response from the model (is the `claude` CLI installed and authed?)"}
    files = parse_files(raw)
    if not files:
        return {"files": {}, "raw": raw, **base,
                "error": "could not parse any file blocks from the response"}
    return {"files": files, "raw": raw, "error": None, **base}


_VERIFY_PREFIX = "_authoring_verify_"     # transient skill modules used only during verification
_verify_counter = itertools.count()


def _verify_examples(skill_src: str) -> list[str]:
    """Import the authored skill module in-process and run its @register example tables. Returns a
    list of human-readable failure strings (empty == all examples pass). Catches the registry's
    import-time arity check (bare-tuple examples) as an 'import error'. Uses a UNIQUE module name per
    call (and clears its bytecode) so successive repair rounds never load a stale `.pyc` — rewriting
    one filename within the same second otherwise hands back round-0's cached example table."""
    import importlib
    import sys
    import unittest
    from reasoning import skills
    from reasoning.skills.tests._helpers import check_examples

    tag = f"{_VERIFY_PREFIX}{next(_verify_counter)}"
    dest = _REPO / "reasoning" / "skills" / f"{tag}.py"
    pycache = _REPO / "reasoning" / "skills" / "__pycache__"
    modname = f"reasoning.skills.{tag}"
    tc = unittest.TestCase()

    def _cleanup():
        sys.modules.pop(modname, None)
        for k in [k for k, v in skills.SKILLS.items()
                  if getattr(v.fn, "__module__", "").endswith(tag)]:
            del skills.SKILLS[k]
        dest.unlink(missing_ok=True)
        for pyc in pycache.glob(f"{tag}.*"):
            pyc.unlink(missing_ok=True)

    fails: list[str] = []
    try:
        dest.write_text(skill_src)
        importlib.invalidate_caches()
        try:
            importlib.import_module(modname)           # @register runs here (arity check included)
        except Exception as e:
            return [f"import error: {type(e).__name__}: {e}"]
        checked = 0
        for name, sk in list(skills.SKILLS.items()):
            if getattr(sk.fn, "__module__", "").endswith(tag):
                checked += 1
                try:
                    check_examples(tc, sk)
                except Exception as e:                  # AssertionError or a raising fn
                    fails.append(f"{name}: {e}")
        if checked == 0:
            fails.append("no skills were registered — check the @register names and file body")
    finally:
        _cleanup()
    return fails


def _repair_prompt(skill_path: str, skill_src: str, fails: list[str]) -> str:
    """A focused, single-file repair prompt. Example verification only touches the skill module, so
    we ask the model to re-emit JUST that file — a small, reliable payload (re-emitting every file
    risks the model mangling the multi-block format) and cheaper."""
    return (
        "The skill module you wrote has FAILING example-table checks. In nearly every case the "
        "FORMULA is correct and only the EXPECTED VALUE in an @register example table is wrong — or a "
        "single-argument example wasn't wrapped as a 1-tuple (write `((0.5,), expected)`, not "
        "`((0.5), expected)`). For each failure below, mentally RUN the function on its example args "
        "and replace the expected value with the real computed output, rounded exactly as the function "
        "rounds it. Never re-guess a float; compute it. Do not change a formula unless the failure is a "
        "genuine logic error.\n\n"
        "FAILURES:\n" + "\n".join(f"  - {f}" for f in fails) +
        f"\n\nRe-emit ONLY this one file, corrected, in EXACTLY this delimiter format and nothing "
        f"else:\n\n<<<FILE {skill_path}>>>\n<the corrected file body>\n<<<END>>>\n\n"
        f"Current {skill_path}:\n\n<<<FILE {skill_path}>>>\n{skill_src}\n<<<END>>>")


def author_domain_verified(description: str, *, model: str = "haiku", effort: str | None = None,
                           timeout: int = 600, max_rounds: int = 3, context: str | None = None,
                           asker=None) -> dict:
    """author_domain + an author→test→repair loop over the SKILL example tables (§5). After the
    first draft, the authored skill module is imported and its example tables are run; any failures
    (wrong expected value, un-tupled single arg, import/arity error) are fed back to the model, which
    re-emits corrected files. Repeats up to `max_rounds`. Returns the author_domain result plus
    `verified` (bool) and `rounds` (per-round failure lists). NOTE: this verifies skills only — it does
    NOT exercise bind gating or routing (those need full registration + generated probe queries; that
    is the next layer). `asker` is injectable for testing."""
    spent = {"cost_usd": None}
    if asker is None:
        from . import teachers

        def asker(p):
            r = teachers.ask_claude_usage(p, model=model, timeout=timeout, effort=effort)
            if r["cost_usd"] is not None:
                spent["cost_usd"] = (spent["cost_usd"] or 0.0) + r["cost_usd"]
            return r["text"]

    raw = asker(build_prompt(description, context))
    files = parse_files(raw) if raw else {}
    rounds: list[dict] = []
    if not files:
        return {"files": {}, "raw": raw, "error": "no parseable file blocks from the response",
                "cost_usd": spent["cost_usd"], "verified": False, "rounds": rounds}
    skill_path = next((p for p in files if p.startswith("reasoning/skills/")), None)
    for i in range(max_rounds + 1):
        fails = _verify_examples(files[skill_path]) if skill_path else ["no skill file was emitted"]
        rounds.append({"round": i, "fails": fails})
        if not fails:
            return {"files": files, "raw": raw, "error": None,
                    "cost_usd": spent["cost_usd"], "verified": True, "rounds": rounds}
        if i == max_rounds or not skill_path:
            break
        fix = asker(_repair_prompt(skill_path, files[skill_path], fails))
        new = parse_files(fix) if fix else {}
        if skill_path not in new:                      # repair didn't come back in the right format
            rounds[-1]["repair_unparseable"] = True
            break
        files[skill_path] = new[skill_path]            # splice the corrected skill file back in
    return {"files": files, "raw": raw,
            "error": f"example checks still failing after {len(rounds) - 1} repair round(s)",
            "cost_usd": spent["cost_usd"], "verified": False, "rounds": rounds}


def write_files(files: dict, *, dry_run: bool = False) -> tuple[list, list]:
    """Write the real (non-meta) files. Never clobbers an existing file. Returns (written, skipped)."""
    written, skipped = [], []
    for path, content in files.items():
        if path.startswith("__"):              # __wiring__ / __notes__ are not files
            continue
        dest = _REPO / path
        if dest.exists():
            skipped.append(path)               # refuse to overwrite existing code
            continue
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        written.append(path)
    return written, skipped

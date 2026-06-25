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
 2. reasoning/bind/<name>.py        — bind_<name> with a CONSERVATIVE gate. Pull each numeric
                                      argument with num_near(query, "<label>", "<synonym>", …) from
                                      reasoning.bind._common (already wildcard-imported): it returns
                                      the number NEAREST an anchor on EITHER side, so it reads both
                                      "demand 800" and "480 minutes" / "0.000001 kg" correctly — do
                                      NOT hand-roll forward-only regex (the logistics exemplar predates
                                      this helper; prefer num_near). List the most specific anchor
                                      first. For fixed-arity numeric ops, wrap with guard_residue /
                                      decline on unexplained residue (§3d).
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


# Probe script run inside the throwaway worktree: route the new orchestrator's example_queries
# (must hit it) and the existing test CASES (must NOT be stolen). Emits one JSON line.
_ROUTING_PROBE = r'''
import json, sys, importlib.util
from reasoning import router
spec = importlib.util.spec_from_file_location("_newrch", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
meta = getattr(mod, "META", {}); oid = meta.get("id"); examples = meta.get("example_queries", [])
def route(q):
    try:
        m = router.recognize(q)
    except Exception:
        return None
    return m.orchestrator.id if m else None
forward = []
for q in examples:
    rid = route(q); rec = {"q": q, "routed": rid, "ok": rid == oid}
    if rid == oid:
        try:
            ans = router.recognize(q).orchestrator.run(q).answer
            rec["answer"] = ans
            # routed here but the bind declined (residue/parse miss) — a non-answer, not a pass
            if ans is None or (isinstance(ans, str) and ("no result" in ans or "could not parse" in ans)):
                rec["ok"] = False; rec["error"] = "routed correctly but run() declined: %r" % ans
        except Exception as e:
            rec["ok"] = False; rec["error"] = "%s: %s" % (type(e).__name__, e)
    forward.append(rec)
stolen = []
try:
    from reasoning.tests.test_orchestrators import CASES
except Exception:
    CASES = []
for row in CASES:
    q, orch_id = row[0], row[2]
    if orch_id == oid:
        continue
    if route(q) == oid:
        stolen.append({"q": q, "expected": orch_id})
print(json.dumps({"oid": oid, "forward": forward, "stolen": stolen}))
'''


def _routing_failures(data: dict) -> list[str]:
    oid = data.get("oid")
    fails = []
    for rec in data.get("forward", []):
        if rec.get("ok"):
            continue
        if rec.get("error"):
            fails.append(f"example query {rec['q']!r} routes to {oid} but run() errored: {rec['error']}")
        else:
            fails.append(f"example query {rec['q']!r} routed to {rec.get('routed')} instead of {oid} "
                         f"— the bind didn't claim it, or another domain outscored it on keywords")
    for s in data.get("stolen", []):
        fails.append(f"this domain wrongly CLAIMS existing query {s['q']!r} (must route to "
                     f"{s['expected']}) — tighten the gate to decline it")
    return fails


def _verify_routing(files: dict) -> list[str] | None:
    """Wire the candidate domain into a throwaway git worktree (HEAD checkout) and route-test it.
    Returns failure strings, or None if routing verification couldn't run (not a git repo, worktree
    failed, or no orchestrator file) — None means 'skipped', distinct from [] which means 'passed'."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    orch_path = next((p for p in files if p.startswith("reasoning/store/orchestrators/")), None)
    skill_path = next((p for p in files if p.startswith("reasoning/skills/")), None)
    bind_path = next((p for p in files if p.startswith("reasoning/bind/")), None)
    if not (orch_path and skill_path):
        return None
    wt = tempfile.mkdtemp(prefix="difg_verify_")
    try:
        add = subprocess.run(["git", "-C", str(_REPO), "worktree", "add", "--detach", "--force", wt],
                             capture_output=True, text=True)
        if add.returncode != 0:
            return None
        wtp = Path(wt)
        for path, content in files.items():
            if path.startswith("__"):
                continue
            dest = wtp / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        # patch the two registries so the skill registers and `from reasoning.bind import bind_X` works
        with (wtp / "reasoning/skills/__init__.py").open("a") as f:
            f.write(f"\nfrom . import {Path(skill_path).stem}  # noqa: F401 (verify)\n")
        if bind_path:
            with (wtp / "reasoning/bind/__init__.py").open("a") as f:
                f.write(f"\nfrom reasoning.bind.{Path(bind_path).stem} import *  # noqa (verify)\n")
        probe = wtp / "_routing_probe.py"
        probe.write_text(_ROUTING_PROBE)
        env = {**os.environ, "PYTHONPATH": wt}
        pr = subprocess.run([sys.executable, str(probe), str(wtp / orch_path)],
                            cwd=wt, env=env, capture_output=True, text=True, timeout=180)
        if pr.returncode != 0 or not pr.stdout.strip():
            return [f"routing probe failed to run: {(pr.stderr or '').strip()[-300:]}"]
        return _routing_failures(json.loads(pr.stdout.strip().splitlines()[-1]))
    finally:
        subprocess.run(["git", "-C", str(_REPO), "worktree", "remove", "--force", wt],
                       capture_output=True, text=True)
        shutil.rmtree(wt, ignore_errors=True)


def _routing_repair_prompt(files: dict, fails: list[str]) -> str:
    bind_path = next((p for p in files if p.startswith("reasoning/bind/")), None)
    orch_path = next((p for p in files if p.startswith("reasoning/store/orchestrators/")), None)
    skill_path = next((p for p in files if p.startswith("reasoning/skills/")), None)
    reemit = "\n\n".join(f"<<<FILE {p}>>>\n{files[p]}\n<<<END>>>" for p in (bind_path, orch_path) if p)
    skill_ref = (f"\n\nFor reference only (do NOT re-emit) — the skill module these route to:\n"
                 f"<<<FILE {skill_path}>>>\n{files[skill_path]}\n<<<END>>>") if skill_path else ""
    return (
        "The domain's ROUTING is wrong. The router picks a domain by (a) the orchestrator META's "
        "keywords / example_queries lexical match and (b) the bind precondition gate. Fix the bind "
        "gating and/or the META to resolve each failure below:\n"
        "  - if an example query routed ELSEWHERE: broaden or correct the bind gate so this domain "
        "claims it, and/or add distinctive keywords/example_queries to META;\n"
        "  - if this domain STOLE another domain's query: tighten the bind gate to DECLINE it "
        "(require a keyword distinctive to THIS domain; bail when a foreign keyword is present).\n\n"
        "FAILURES:\n" + "\n".join(f"  - {f}" for f in fails) +
        "\n\nRe-emit ONLY the bind and orchestrator files, corrected, each in EXACTLY this delimiter "
        "format and nothing else:\n\n" + reemit + skill_ref)


def author_domain_verified(description: str, *, model: str = "haiku", effort: str | None = None,
                           timeout: int = 600, max_rounds: int = 3, context: str | None = None,
                           asker=None, verify_routing: bool = True) -> dict:
    """author_domain + an author→test→repair loop. Two phases, each looping up to `max_rounds`:

    1. EXAMPLES (§5) — import the authored skill module, run its @register example tables; feed any
       failure (wrong expected value, un-tupled single arg, arity/import error) back for a corrected
       single-file re-emit.
    2. ROUTING — wire the domain into a throwaway git worktree and route-test it: the orchestrator's
       example_queries must hit the new domain, and the existing test CASES must not be stolen. Feed
       mis-routes / thefts back for a bind+orchestrator re-emit. Skipped (not failed) when it can't
       run (not a git repo). Disable with verify_routing=False.

    Returns the author_domain result plus `verified` (bool), `verified_routing` (True/False/None=
    skipped), and `rounds` (per-round, tagged by phase). `asker` is injectable for testing."""
    spent = {"cost_usd": None}
    if asker is None:
        from . import teachers

        def asker(p):
            r = teachers.ask_claude_usage(p, model=model, timeout=timeout, effort=effort)
            if r["cost_usd"] is not None:
                spent["cost_usd"] = (spent["cost_usd"] or 0.0) + r["cost_usd"]
            return r["text"]

    def result(verified, verified_routing, rounds, error):
        return {"files": files, "raw": raw, "error": error, "cost_usd": spent["cost_usd"],
                "verified": verified, "verified_routing": verified_routing, "rounds": rounds}

    raw = asker(build_prompt(description, context))
    files = parse_files(raw) if raw else {}
    if not files and raw:                  # draft ignored the delimiter format (stochastic) — retry once
        raw = asker(build_prompt(description, context) + "\n\nIMPORTANT: your previous reply did not "
                    "use the required format. Emit ONLY the <<<FILE path>>> … <<<END>>> blocks "
                    "specified above — no markdown code fences, no prose before or after.")
        files = parse_files(raw) if raw else {}
    rounds: list[dict] = []
    if not files:
        return result(False, None, rounds, "no parseable file blocks from the response (after retry)")
    skill_path = next((p for p in files if p.startswith("reasoning/skills/")), None)
    bind_path = next((p for p in files if p.startswith("reasoning/bind/")), None)
    orch_path = next((p for p in files if p.startswith("reasoning/store/orchestrators/")), None)

    # Phase 1 — skill example tables
    ok_examples = False
    for i in range(max_rounds + 1):
        fails = _verify_examples(files[skill_path]) if skill_path else ["no skill file was emitted"]
        rounds.append({"phase": "examples", "round": i, "fails": fails})
        if not fails:
            ok_examples = True
            break
        if i == max_rounds or not skill_path:
            break
        new = parse_files(asker(_repair_prompt(skill_path, files[skill_path], fails)) or "")
        if skill_path not in new:
            rounds[-1]["repair_unparseable"] = True
            break
        files[skill_path] = new[skill_path]
    if not ok_examples:
        return result(False, None, rounds, f"example checks still failing after {max_rounds} round(s)")

    # Phase 2 — bind gating + routing (worktree-isolated)
    ok_routing = None
    if verify_routing:
        for i in range(max_rounds + 1):
            rfails = _verify_routing(files)
            if rfails is None:
                rounds.append({"phase": "routing", "round": i, "skipped": True})
                break
            rounds.append({"phase": "routing", "round": i, "fails": rfails})
            if not rfails:
                ok_routing = True
                break
            if i == max_rounds:
                ok_routing = False
                break
            new = parse_files(asker(_routing_repair_prompt(files, rfails)) or "")
            spliced = [p for p in (bind_path, orch_path) if p and p in new]
            if not spliced:
                rounds[-1]["repair_unparseable"] = True
                ok_routing = False
                break
            for p in spliced:
                files[p] = new[p]

    verified = ok_examples and ok_routing in (True, None)
    err = None if verified else "routing checks still failing"
    return result(verified, ok_routing, rounds, err)


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

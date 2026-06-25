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
                                      decline on unexplained residue (§3d). GATE CALIBRATION (§3f),
                                      the #1 source of mis-routes: (a) the domain's top-level context
                                      allow-list MUST contain a trigger word for EVERY op — if an op
                                      only branches on "continuity"/"grover", that word must admit the
                                      domain, or the op is declined before its branch runs; (b) a
                                      collision-decline guard against a neighbour must require ALL of
                                      the neighbour's distinctive signal with AND, never a single
                                      shared word with OR — e.g. to decline physics "force/area
                                      pressure" use `"force" in q and "area" in q`, NOT `or`, because
                                      "area" alone is a legitimate input to your own flow/continuity
                                      ops and an OR-guard would decline them.
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

    # When EXTENDING a domain, the skill source re-registers names already in SKILLS (the live
    # module). register() raises on duplicates, so displace those entries first and restore them
    # afterwards — the temp module re-registers them fresh (existing ops + the new ones) for the run.
    names = re.findall(r'@register\(\s*"([^"]+)"', skill_src)
    displaced = {n: skills.SKILLS.pop(n) for n in names if n in skills.SKILLS}

    def _cleanup():
        sys.modules.pop(modname, None)
        for k in [k for k, v in skills.SKILLS.items()
                  if getattr(v.fn, "__module__", "").endswith(tag)]:
            del skills.SKILLS[k]
        skills.SKILLS.update(displaced)              # restore the live entries we displaced
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


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _floats(s) -> list[float]:
    return [float(x) for x in _FLOAT_RE.findall(str(s))]


def _skill_example_rows(stem: str) -> list[tuple]:
    """For every @register example of the domain's skills (those whose fn lives in module `stem`),
    a (frozenset-of-arg-numbers, expected-first-number) row. Used to cross-check that an NL example
    query whose numbers match a row actually produces that row's value (else a wrong sub-skill ran)."""
    from reasoning import skills
    rows = []
    for _name, sk in skills.SKILLS.items():
        if not getattr(sk.fn, "__module__", "").endswith("." + stem):
            continue
        for ex in getattr(sk, "examples", None) or []:
            if not (isinstance(ex, (tuple, list)) and len(ex) >= 2):
                continue
            argnums, expnums = _floats(ex[0]), _floats(ex[1])
            if argnums and expnums:
                rows.append((frozenset(round(a, 6) for a in argnums), expnums[0]))
    return rows


def _answer_mismatch(answer, query: str, rows: list[tuple]) -> str | None:
    """When the query's number-set EXACTLY matches a skill example row, the answer must equal that
    row's value (within tolerance). Returns an error string when it matches a row but the answer
    doesn't — the signal that the wrong sub-skill ran (acoustics' 'wave speed … 440 Hz … 0.78 m'
    → 1, the wavelength op, when the table says (440,0.78)→343.2). Returns None when nothing matches
    (can't verify) or the answer is consistent. Exact-set match keeps false positives low: a query
    that doesn't reuse a row's exact numbers is simply skipped, never flagged."""
    qset = frozenset(round(x, 6) for x in _floats(query))
    expected = [exp for argset, exp in rows if argset == qset]
    anums = _floats(answer)
    if not expected or not anums:
        return None
    a = anums[0]
    if any(abs(a - exp) <= 1e-3 * max(1.0, abs(exp)) for exp in expected):
        return None
    return (f"answer {answer!r} doesn't match the skill example for inputs {sorted(qset)} "
            f"(expected {sorted(set(expected))}) — the wrong sub-skill likely ran")


# Probe script run inside the throwaway worktree: route the new orchestrator's example_queries
# (must hit it AND produce the skill-table answer) and the existing test CASES (must NOT be stolen).
# Emits one JSON line. argv: [orchestrator_path, skill_module_stem].
_ROUTING_PROBE = r'''
import json, sys, importlib.util
from reasoning import router
from reasoning.authoring import _skill_example_rows, _answer_mismatch
spec = importlib.util.spec_from_file_location("_newrch", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
meta = getattr(mod, "META", {}); oid = meta.get("id"); examples = meta.get("example_queries", [])
rows = _skill_example_rows(sys.argv[2]) if len(sys.argv) > 2 else []
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
            else:
                mismatch = _answer_mismatch(ans, q, rows)   # wrong sub-skill / wrong NL answer
                if mismatch:
                    rec["ok"] = False; rec["error"] = mismatch
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
            fails.append(f"example query {rec['q']!r} routed to {oid} but did NOT verify: {rec['error']}")
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
        pr = subprocess.run([sys.executable, str(probe), str(wtp / orch_path), Path(skill_path).stem],
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


def _verify_and_repair(files: dict, asker, *, max_rounds: int, verify_routing: bool = True):
    """The two-phase author→test→repair loop over `files` (mutated in place with repaired versions).
    Phase 1: skill example tables (single-file skill re-emit on failure). Phase 2: worktree-isolated
    routing (bind+orchestrator re-emit on mis-route/theft/wrong-answer). Returns
    (verified: bool, ok_routing: True|False|None-skipped, rounds: list)."""
    skill_path = next((p for p in files if p.startswith("reasoning/skills/")), None)
    bind_path = next((p for p in files if p.startswith("reasoning/bind/")), None)
    orch_path = next((p for p in files if p.startswith("reasoning/store/orchestrators/")), None)
    rounds: list[dict] = []

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
        return False, None, rounds

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

    return ok_examples and ok_routing in (True, None), ok_routing, rounds


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
    if not files:
        return result(False, None, [], "no parseable file blocks from the response (after retry)")
    verified, ok_routing, rounds = _verify_and_repair(files, asker, max_rounds=max_rounds,
                                                       verify_routing=verify_routing)
    err = None if verified else ("example checks still failing" if ok_routing is None
                                 else "routing checks still failing")
    return result(verified, ok_routing, rounds, err)


def gather_domain_files(domain: str) -> dict | None:
    """{path: source} for an existing domain's skill + bind + orchestrator, located by file stem
    == `domain`. The orchestrator stem can differ from the skill/bind stem (e.g. thermo vs
    thermodynamics), so if `<domain>.py` isn't in store/orchestrators it's found by scanning for the
    file that imports `bind_<domain>`. Returns None if skill, bind, or orchestrator can't be found."""
    skill = _REPO / "reasoning/skills" / f"{domain}.py"
    bind = _REPO / "reasoning/bind" / f"{domain}.py"
    if not (skill.exists() and bind.exists()):
        return None
    orch = _REPO / "reasoning/store/orchestrators" / f"{domain}.py"
    if not orch.exists():
        for p in sorted((_REPO / "reasoning/store/orchestrators").glob("*.py")):
            if f"bind_{domain}" in p.read_text():
                orch = p
                break
    if not orch.exists():
        return None
    return {str(p.relative_to(_REPO)): p.read_text() for p in (skill, bind, orch)}


_EXTEND_FORMAT = """\
OUTPUT FORMAT — re-emit the THREE existing files IN FULL (same paths), each delimited EXACTLY like
this, with your additions folded in and every existing op preserved verbatim:

<<<FILE reasoning/skills/<domain>.py>>>
<full updated skill file>
<<<END>>>
<<<FILE reasoning/bind/<domain>.py>>>
<full updated bind file>
<<<END>>>
<<<FILE reasoning/store/orchestrators/<domain>.py>>>
<full updated orchestrator file>
<<<END>>>

Then a <<<FILE __wiring__>>> block with 1–2 new CASES tuples (query, expected, "<orch_id>") covering
the NEW ops only. <<<END>>>"""


def build_extend_prompt(domain: str, description: str, files: dict, context: str | None = None) -> str:
    ctx = gather_context() if context is None else context
    existing = "\n\n".join(f"<<<FILE {p}>>>\n{c}\n<<<END>>>" for p, c in files.items())
    return (
        "You are ADDING new operation(s) to an EXISTING domain in a deterministic, model-free "
        "reasoning engine. RE-EMIT the three existing files below IN FULL with your additions folded "
        "in: one new @register skill per op (EXACT example table — compute every value, never guess), "
        "a matching bind branch per op (extract args with num_near; gate on a phrase distinctive to "
        "the NEW op so it collides with neither the domain's EXISTING ops nor other domains), and the "
        "orchestrator META extended with each new op's example_query + keyword. GATE CALIBRATION "
        "(§3f): the context allow-list must admit every new op's keyword. PRESERVE every existing op "
        "EXACTLY — do not remove, rename, or change any existing skill, example value, or bind "
        "branch.\n\n"
        f"DOMAIN: {domain}\nNEW OPERATION(S) TO ADD:\n{description}\n\n{_EXTEND_FORMAT}\n\n"
        f"=== THE EXISTING DOMAIN FILES (re-emit all three, with your additions) ===\n{existing}\n\n"
        f"---\n\n{ctx}")


def extend_domain(domain: str, description: str, *, model: str = "haiku", effort: str | None = None,
                  timeout: int = 600, max_rounds: int = 3, context: str | None = None,
                  asker=None, verify: bool = True) -> dict:
    """Add one or more skills to an EXISTING domain: feed the model the domain's three files, have it
    re-emit them with the new ops folded in, then (with verify) run the same example+routing loop —
    which also guards that the additions don't break the domain's existing ops or steal neighbours.
    Returns {files, verified, verified_routing, rounds, cost_usd, error, domain_paths}; the caller
    OVERWRITES the domain_paths (these files already exist). `asker` is injectable for testing."""
    files0 = gather_domain_files(domain)
    if files0 is None:
        return {"files": {}, "raw": None, "verified": False, "verified_routing": None, "rounds": [],
                "cost_usd": None, "domain_paths": [],
                "error": f"domain {domain!r} not found — need reasoning/skills/{domain}.py + bind + "
                         f"an orchestrator (by stem or importing bind_{domain})"}
    domain_paths = list(files0)
    spent = {"cost_usd": None}
    if asker is None:
        from . import teachers

        def asker(p):
            r = teachers.ask_claude_usage(p, model=model, timeout=timeout, effort=effort)
            if r["cost_usd"] is not None:
                spent["cost_usd"] = (spent["cost_usd"] or 0.0) + r["cost_usd"]
            return r["text"]

    prompt = build_extend_prompt(domain, description, files0, context)
    raw = asker(prompt)
    files = parse_files(raw) if raw else {}
    if not files and raw:
        raw = asker(prompt + "\n\nIMPORTANT: re-emit ONLY the <<<FILE …>>> … <<<END>>> blocks, no "
                    "markdown fences, no prose.")
        files = parse_files(raw) if raw else {}

    def result(verified, ok_routing, rounds, error):
        return {"files": files, "raw": raw, "verified": verified, "verified_routing": ok_routing,
                "rounds": rounds, "cost_usd": spent["cost_usd"], "domain_paths": domain_paths,
                "error": error}

    if not files:
        return result(False, None, [], "no parseable file blocks from the response (after retry)")
    missing = [p for p in domain_paths if p not in files]
    if missing:                                   # incomplete re-emit — refuse to apply a partial domain
        return result(False, None, [], f"re-emit is missing required files: {missing}")
    if not verify:
        return result(False, None, [], None)
    verified, ok_routing, rounds = _verify_and_repair(files, asker, max_rounds=max_rounds)
    err = None if verified else ("example checks still failing" if ok_routing is None
                                 else "routing checks still failing")
    return result(verified, ok_routing, rounds, err)


def write_files(files: dict, *, dry_run: bool = False, overwrite: bool = False) -> tuple[list, list]:
    """Write the real (non-meta) files. By default never clobbers an existing file (new-domain
    authoring); pass overwrite=True to write existing files too (extending a domain). Returns
    (written, skipped)."""
    written, skipped = [], []
    for path, content in files.items():
        if path.startswith("__"):              # __wiring__ / __notes__ are not files
            continue
        dest = _REPO / path
        if dest.exists() and not overwrite:
            skipped.append(path)               # refuse to overwrite existing code
            continue
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        written.append(path)
    return written, skipped

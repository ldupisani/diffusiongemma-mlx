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
    `model="opus"` for harder or under-specified domains. `asker(prompt)->str|None` is injectable
    for testing."""
    if asker is None:
        from . import teachers
        asker = lambda p: teachers.ask_claude(p, model=model, timeout=timeout, effort=effort)  # noqa: E731
    raw = asker(build_prompt(description, context))
    if not raw:
        return {"files": {}, "raw": None,
                "error": "no response from the model (is the `claude` CLI installed and authed?)"}
    files = parse_files(raw)
    if not files:
        return {"files": {}, "raw": raw, "error": "could not parse any file blocks from the response"}
    return {"files": files, "raw": raw, "error": None}


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

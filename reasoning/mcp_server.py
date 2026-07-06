"""MCP server exposing the reasoning project as a service.

Surfaces the deterministic engine (route → orchestrator → answer) plus its introspection,
dataset, and reference capabilities as MCP tools over stdio. It answers from CODE ONLY — every
tool here runs at 0 model calls and never contacts the sidecar/teacher — so the service is
self-contained: `ask` routes to a model-free orchestrator (or a 0-call compound split) and
reports the domain and call count; a query that would need the LLM is reported as such rather
than silently invoking it. When nothing coded matches a word problem, `ask` also asks the
model-free READER what it would need and surfaces that as a `deficiency` (query rather than
assume); the `diagnose` tool exposes the reader's full epistemic read — the answer if the text
determines it, the facts it knows, the world-knowledge a question requires, and the specific
gap when it can't answer.

Run:        python -m reasoning.mcp_server
Register:   point an MCP client at the command above (stdio transport).
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("difg-reasoning")

# --- live reload -------------------------------------------------------------
# The stdio server is long-lived and imports reasoning.* once, so engine edits wouldn't show up
# without a client reconnect. Instead, before each tool runs we drop cached reasoning.* modules
# IF any source file changed; the tools' lazy imports then re-read the current code. Net effect:
# edit a file, call a tool, see the change — no reconnect. (Editing THIS server file still needs a
# reconnect, since it's the running process.) Unchanged → nothing dropped → calls stay fast.
_PKG_DIR = Path(__file__).resolve().parent
_loaded_mtime = 0.0


def _source_mtime() -> float:
    latest = 0.0
    for p in _PKG_DIR.rglob("*.py"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            pass
    return latest


def _refresh_if_changed() -> None:
    global _loaded_mtime
    mt = _source_mtime()
    if mt <= _loaded_mtime:
        return
    first = _loaded_mtime != 0.0
    # Drop the package AND its submodules (except this running server). Dropping the parent
    # `reasoning` too is essential: otherwise `from reasoning import X` returns the stale submodule
    # still cached as an attribute on the package object, silently skipping the re-import.
    for name in [n for n in list(sys.modules)
                 if (n == "reasoning" or n.startswith("reasoning.")) and n != __name__]:
        del sys.modules[name]                  # next lazy import in a tool re-reads from disk
    _loaded_mtime = mt
    if first:
        print("[difg-reasoning] source changed — reloaded engine modules", file=sys.stderr)


def _live(fn):
    """Wrap a tool so it reloads changed engine modules before running."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _refresh_if_changed()
        return fn(*args, **kwargs)
    return wrapper


def _translate_enabled(translate) -> bool:
    """Opt-in gate for the LLM-as-translator fallback. Explicit `translate=` wins; otherwise the
    env var DIFG_TRANSLATE_FALLBACK (1/true/yes/on). Default OFF — `ask` stays 0-call by default."""
    if translate is not None:
        return bool(translate)
    return os.environ.get("DIFG_TRANSLATE_FALLBACK", "").lower() in ("1", "true", "yes", "on")


def _translate_fallback(query: str) -> dict | None:
    """Last-resort: 1 model call translates the prose into an arithmetic program, executed
    deterministically (translate.solve). Returns a result dict (with the program, for audit) or
    None if it produced no answer. The arithmetic stays code — only the parse is the model's."""
    from .translate import solve as translate_solve
    out = translate_solve(query)
    if out.get("answer") is None:
        return None
    val = out["answer"]
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    result = {"matched": True, "answer": str(val), "domain": "translate",
              "model_calls": out.get("model_calls", 1), "program": out.get("program")}
    unused = out.get("unused_givens") or []
    if unused:   # a stated quantity went unused → the translation may be wrong (label-free signal)
        result["low_confidence"] = True
        result["unused_givens"] = unused
        result["warning"] = (f"the program ignored {len(unused)} number(s) given in the question "
                             f"({', '.join(str(u) for u in unused)}) — the answer may be wrong")
    return result


def ask(query: str, translate: bool | None = None) -> dict:
    """Answer a question with the deterministic reasoning engine (0 model calls by default).

    Routes the query to a coded domain (arithmetic, dates, geometry, physics, finance, logic,
    statistics, geography, …) and runs it. Returns the answer, which domain handled it, and the
    model-call count. If the query needs more information, returns the clarifying question(s).

    If nothing coded matches AND the LLM-as-translator fallback is enabled (`translate=True`, or
    env DIFG_TRANSLATE_FALLBACK=1), one model call turns the prose into an arithmetic program that
    is executed deterministically — `domain="translate"`, `model_calls=1`, and the audited
    `program` is included. Otherwise returns matched=False with a hint to call `capabilities`.
    """
    from . import router
    from .compiled import model_calls, needs_input
    from .decompose import decompose_code_first
    m = router.recognize(query)
    if m and m.orchestrator.model_free and not m.orchestrator.fallback:
        r = m.orchestrator.run(query)
        missing = needs_input(r)
        if missing:
            return {"matched": True, "domain": m.orchestrator.id, "answer": None,
                    "needs_input": {f: q for f, (q, _t) in missing.items()}}
        return {"matched": True, "answer": r.answer, "domain": m.orchestrator.id,
                "query_type": m.orchestrator.query_type, "model_calls": model_calls(r)}
    # no model-free single match → try the 0-call compound split ("X and Y")
    composed = decompose_code_first(query)
    if composed is not None:
        return {"matched": True, "answer": composed.answer, "domain": "compose",
                "model_calls": model_calls(composed)}
    # no coded answer → opt-in LLM-as-translator (1 call; arithmetic still done by code)
    if _translate_enabled(translate):
        translated = _translate_fallback(query)
        if translated is not None:
            return translated
    # No coded answer (either an LLM-needing fallback type, or nothing matched at all). Rather than
    # assume, ask the model-free READER what it would NEED and attach it — this keeps `ask`'s precision
    # contract (we never return the reader's own answer here) but turns a dead end into a well-formed
    # clarifying question the caller can act on. `diagnose(query)` gives the full read (incl. any answer).
    if m:  # matched a type that needs the LLM (e.g. a fallback) — reported, never invoked
        out = {"matched": True, "domain": m.orchestrator.id, "answer": None,
               "note": "this query type needs the LLM backend; enable the translator fallback "
                       "(translate=True or DIFG_TRANSLATE_FALLBACK=1) to answer it"}
    else:
        out = {"matched": False, "answer": None,
               "hint": "No coded domain matched. Call capabilities() or search_capabilities(term), "
                       "or enable the translator fallback (translate=True) for word problems."}
    return _with_reader_deficiency(query, out)


def _with_reader_deficiency(query: str, out: dict) -> dict:
    """Attach the model-free reader's DEFICIENCY (what it would need to answer, phrased as a question) to
    an otherwise-unanswered `ask` response. Best-effort: never let the reader break `ask`."""
    try:
        from .reading import diagnose as _reading_diagnose
        d = _reading_diagnose(query)
    except Exception:
        d = None
    if d is not None:
        out["deficiency"] = {"kind": d.kind, "question": d.message}
        out["hint"] = ("No coded domain answered this. The model-free reader identified what it would "
                       "need — see `deficiency`. Call diagnose(query) for the full read (known facts + "
                       "requirements + any reader answer), or enable translate=True.")
    return out


def _fact_str(f) -> str:
    """A compact, JSON-safe rendering of a reading fact (Quantity/Relation/Predication/…) for a
    consumer. Handles the common shapes explicitly; falls back to repr for the rest."""
    name = type(f).__name__
    if name == "Quantity":
        meas = f.measure[0] if isinstance(getattr(f, "measure", None), tuple) else getattr(f, "measure", "")
        amt = getattr(f, "amount", None)
        if isinstance(amt, float) and amt.is_integer():
            amt = int(amt)
        tag = " (derived)" if getattr(f, "derived", False) else ""
        return f"{f.entity} {f.verb} {amt} {meas}{tag}".strip()
    if name == "Relation":
        return f"{f.subject} {f.relation} {f.object}"
    if name in ("Predication", "Membership", "Subset"):
        return " ".join(str(v) for v in vars(f).values())
    return repr(f)


def diagnose(query: str) -> dict:
    """The model-free READER's epistemic read of a word problem (0 model calls) — the bridge surface for
    the greater engine to consume. Unlike `ask` (high-precision coded orchestrators only), this reports
    everything the reader can honestly say WITHOUT guessing:

    - `answer` — the value IF the text determines it (else null; the reader declines rather than guess);
    - `known` — the facts it extracted, threaded across sentences and closed under synthesis
      ("here are the things we do know about the question");
    - `requirements` — question-shapes it recognises and whether the world-knowledge they need is present
      (an unsatisfied one names exactly the fact to supply);
    - `deficiency` — when it can't answer, the specific gap phrased as a QUESTION to answer
      (unknown-word / ambiguous-base / missing-knowledge / insufficient).

    Use this when `ask` returns matched=False on a word problem, or whenever you want the facts + the
    missing piece rather than just a final number. Its precision is the reading engine's (~90% on held-out
    GSM8K), so treat its `answer` as a reader estimate, not a coded-orchestrator guarantee."""
    from .reading import report
    rep = report(query)
    ans = rep["answer"]
    solved = ans not in (None, "undetermined")
    out = {"query": query, "solved": bool(solved), "model_calls": 0,
           "answer": (ans if solved else None),
           "known": [_fact_str(f) for f in rep["known"]],
           "requirements": [{"shape": n, "needs": needs, "satisfied": bool(sat)}
                            for (n, needs, sat) in rep["requirements"]]}
    d = rep["deficiency"]
    if d is not None:
        out["deficiency"] = {"kind": d.kind, "question": d.message}
    return out


def capabilities() -> dict:
    """List what the system can do: headline counts plus every reasoning domain and its
    one-line description. Start here to discover available query types."""
    from . import introspect
    return {"summary": introspect.summary(),
            "domains": [{"id": d["id"], "description": d["description"]}
                        for d in introspect.domains()]}


def describe_domain(domain: str) -> dict:
    """Show a domain's example queries and keywords — i.e. what to ask it. `domain` is an id or
    query_type from `capabilities`."""
    from . import introspect
    d = introspect.domain(domain)
    return d or {"error": f"no domain '{domain}'", "hint": "call capabilities() for the list"}


def search_capabilities(term: str) -> dict:
    """Find domains, skills, and dataset generators whose name/description/examples mention a
    term (e.g. 'prime', 'cost', 'distance')."""
    from . import introspect
    return introspect.search(term)


def list_generators() -> dict:
    """List the dataset generators grouped by domain (what datasets can be created)."""
    from . import datagen
    return {dom: [{"generator": n, "doc": doc} for n, doc in gens]
            for dom, gens in datagen.list_domains().items()}


def generate_dataset(request: str) -> str:
    """Generate a dataset from natural language (e.g. 'generate the first 1000 prime numbers',
    'generate 500 normal samples') and save it as a queryable handle; returns its schema + head."""
    from . import datagen
    from .bind import bind_dataset
    text = request if any(w in request.lower() for w in ("generate", "synthesize")) else f"generate {request}"
    ok, d, _, _ = bind_dataset(text)
    if ok and d.get("op") == "generate":
        return datagen.generate(d["gen"], d["save_as"], **d["params"])
    return f"could not parse a generator from {request!r}; call list_generators() to see options"


def list_datasets() -> dict:
    """List available dataset handles: user/generated datasets and the bundled reference tables."""
    from . import datasets
    return {"datasets": datasets.list_datasets(),
            "reference_tables": sorted(datasets._reference_tables())}


def dataset_info(name: str) -> dict:
    """Describe a dataset handle (row count, column schema) and how to query it."""
    from . import datasets
    try:
        sch = datasets.schema(name)
    except FileNotFoundError as e:
        return {"error": str(e)}
    nums = [c for c, t in sch if any(k in t.upper() for k in ("INT", "DOUBLE", "DEC", "FLOAT", "REAL"))]
    info = {"name": name, "columns": [{"name": c, "type": t} for c, t in sch],
            "example_sql": f"SELECT * FROM {name} LIMIT 5"}
    if nums:
        info["example_aggregate"] = f"SELECT avg({nums[0]}) FROM {name}"
    return info


def query_dataset(sql: str) -> str:
    """Run read-only SQL over the dataset/reference handles (DuckDB). Returns a scalar or a small
    table; a large result is saved to a new handle and previewed."""
    from . import datasets
    try:
        return datasets.query(sql)
    except Exception as e:  # bad SQL / unknown handle → a readable error, not a crash
        return f"query failed: {type(e).__name__}: {e}"


# Register every tool, wrapped so it picks up engine edits without a reconnect (the module-level
# functions stay unwrapped and directly callable — for tests and reuse).
for _fn in (ask, diagnose, capabilities, describe_domain, search_capabilities, list_generators,
            generate_dataset, list_datasets, dataset_info, query_dataset):
    mcp.tool()(_live(_fn))


# Prompts — user-invokable templates that surface as slash commands in MCP clients
# (/mcp__difg-reasoning__ask, …__capabilities, …__query). Distinct function names avoid shadowing
# the tool functions above; the slash-command name is set explicitly via `name=`. Each returns a
# message that directs the assistant to call the matching tool, so the user gets a one-step entry
# point into the engine.

@mcp.prompt(name="ask", title="Ask the reasoning engine",
            description="Answer a question with the difg-reasoning engine (deterministic, 0 model calls).")
def ask_prompt(query: str) -> str:
    return (f"Use the difg-reasoning `ask` tool to answer the following, then report the answer, "
            f"the domain that handled it, and the model-call count:\n\n{query}")


@mcp.prompt(name="capabilities", title="Reasoning capabilities",
            description="Summarize what the reasoning engine can do.")
def capabilities_prompt() -> str:
    return ("Call the difg-reasoning `capabilities` tool and give a concise overview: the domain "
            "count and a sampling of the notable topics I can ask about.")


@mcp.prompt(name="query", title="Query datasets with SQL",
            description="Run SQL over the dataset and bundled reference handles.")
def query_prompt(sql: str) -> str:
    return (f"Run this SQL over the difg-reasoning datasets and reference tables using the "
            f"`query_dataset` tool, and show the result:\n\n{sql}")


def main():
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()

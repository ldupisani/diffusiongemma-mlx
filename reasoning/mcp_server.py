"""MCP server exposing the reasoning project as a service.

Surfaces the deterministic engine (route → orchestrator → answer) plus its introspection,
dataset, and reference capabilities as MCP tools over stdio. It answers from CODE ONLY — every
tool here runs at 0 model calls and never contacts the sidecar/teacher — so the service is
self-contained: `ask` routes to a model-free orchestrator (or a 0-call compound split) and
reports the domain and call count; a query that would need the LLM is reported as such rather
than silently invoking it.

Run:        python -m reasoning.mcp_server
Register:   point an MCP client at the command above (stdio transport).
"""

from __future__ import annotations

import functools
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


def ask(query: str) -> dict:
    """Answer a question with the deterministic reasoning engine (0 model calls).

    Routes the query to a coded domain (arithmetic, dates, geometry, physics, finance, logic,
    statistics, geography, …) and runs it. Returns the answer, which domain handled it, and the
    model-call count. If the query needs more information, returns the clarifying question(s); if
    nothing coded matches, returns matched=False with a hint to call `capabilities`.
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
    if m:  # matched a type that needs the LLM (e.g. a fallback) — reported, never invoked
        return {"matched": True, "domain": m.orchestrator.id, "answer": None,
                "note": "this query type needs the LLM backend; this service answers code-only "
                        "(0-call) queries"}
    return {"matched": False, "answer": None,
            "hint": "No coded domain matched. Call capabilities() or search_capabilities(term) "
                    "to see what can be asked."}


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
for _fn in (ask, capabilities, describe_domain, search_capabilities, list_generators,
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

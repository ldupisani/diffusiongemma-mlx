"""Best-effort code-only answer for a query — WITHOUT the LLM teacher fallback and WITHOUT
interactive clarification. The caller's own LLM is the fallback (the chat REPL's full
generation, or cli `run`'s teacher), so this is what lets a chat front-end try the
deterministic engine first and only generate a full LLM reply when the engine can't answer.

`shot` is the one-shot model callable (engine.one_shot for the CLI; backend.backend_shot(...)
for an in-process backend) — so the same ladder works from the CLI, the Python REPL, and the
sidecar's /solve endpoint.
"""

from __future__ import annotations

from . import router
from .compiled import needs_input
from .decompose import decompose_code_first
from .reformulate import reformulate_with_model


def try_coded(query: str, *, shot, matcher=None):
    """Return a code-computed PatternResult for `query`, or None to defer to the caller's LLM.

    Ladder, cheapest first:
      1. a clean DIRECT route (0 model calls) — but never the generic fallback orchestrator,
         and never a match that would need interactive clarification;
      2. a 0-call compound split on conjunctions;
      3. a ≤2-call single-question reformulation (rewrite → coded answer → verify).
    """
    m = router.recognize(query, matcher=matcher)
    # Only model_free orchestrators on the direct path: a non-model_free one would call the
    # model via engine.one_shot (the sidecar) instead of our `shot`, which deadlocks when
    # try_coded runs *inside* the single-threaded sidecar's /solve handler.
    if m and m.orchestrator.model_free and not m.orchestrator.fallback and m.via != "clarify":
        r = m.orchestrator.run(query)
        if not needs_input(r) and not r.answer.startswith("("):
            return r
    return decompose_code_first(query) or reformulate_with_model(query, shot=shot)

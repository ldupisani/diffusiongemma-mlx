"""The growing graph of stored orchestrators.

Each orchestrator is a real Python module under store/orchestrators/ — so it's
inspectable, git-versioned, and can import patterns or call sibling orchestrators
(that's the "graph"). A module exposes:

    META = {"id", "query_type", "description", "example_queries", "sub_orchestrators"}
    def precondition(query) -> bool       # optional guard; default True
    def run(query) -> PatternResult       # the coded reasoning flow

The registry just discovers and indexes them; the router reads the metadata to match
an incoming query to a query_type.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

STORE = Path(__file__).parent / "store"
ORCH_DIR = STORE / "orchestrators"
SAMPLE_DIR = STORE / "samples"


@dataclass
class Orchestrator:
    id: str
    query_type: str
    description: str
    example_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sub_orchestrators: list[str] = field(default_factory=list)
    precondition: Callable[[str], bool] = lambda q: True
    run: Optional[Callable] = None
    source: str = ""
    model_free: bool = False  # declared intent (runtime trace tally is authoritative)
    fallback: bool = False    # only matched when no specific orchestrator claims the query


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"reasoning_orch_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def load_orchestrators() -> list[Orchestrator]:
    out: list[Orchestrator] = []
    if not ORCH_DIR.exists():
        return out
    for path in sorted(ORCH_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = _load_module(path)
        meta = getattr(mod, "META", {})
        out.append(
            Orchestrator(
                id=meta.get("id", path.stem),
                query_type=meta.get("query_type", path.stem),
                description=meta.get("description", ""),
                example_queries=meta.get("example_queries", []),
                keywords=meta.get("keywords", []),
                sub_orchestrators=meta.get("sub_orchestrators", []),
                precondition=getattr(mod, "precondition", lambda q: True),
                run=getattr(mod, "run", None),
                source=str(path),
                model_free=meta.get("model_free", False),
                fallback=meta.get("fallback", False),
            )
        )
    return out


def get(orch_id: str) -> Optional[Orchestrator]:
    for o in load_orchestrators():
        if o.id == orch_id:
            return o
    return None

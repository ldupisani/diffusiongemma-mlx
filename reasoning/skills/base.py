"""The skill descriptor + registry — the growing library of deterministic operations.

A *skill* is a small, pure, typed Python function that does ONE mechanical thing
(one arithmetic op, one date calculation, one string/collection operation). Skills are
the code that replaces model one-shot calls: a step that resolves to a skill never
contacts the sidecar, so a fully-coded orchestrator runs with the model cold.

The registry is the same "discoverable index" idiom as patterns.PATTERNS — a name→Skill
dict — so the compiled interpreter (compiled.py) and the derive matcher can look skills
up by name. Skills MUST stay pure: no I/O, no model calls, no imports of engine. A test
enforces that invariant.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Skill:
    name: str                       # unique key, dotted by family, e.g. "arith.apply_tax"
    fn: Callable                    # the pure function
    doc: str                        # one-line description (used by derive's line→skill match)
    signature: str = ""             # human-readable typed signature (auto from inspect)
    examples: list = field(default_factory=list)  # list[(args_tuple, expected)] — tests + docs

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


SKILLS: dict[str, Skill] = {}


def register(name: str, doc: str, examples: list | None = None):
    """Decorator: register a pure function as a named Skill.

    `examples` is a list of (args_tuple, expected_result); it documents the skill AND
    drives the generated unit tests (every skill must have at least one example).
    """
    def deco(fn: Callable) -> Callable:
        if name in SKILLS:
            raise ValueError(f"duplicate skill name {name!r}")
        try:
            sig = str(inspect.signature(fn))
        except (ValueError, TypeError):
            sig = ""
        SKILLS[name] = Skill(name=name, fn=fn, doc=doc, signature=sig,
                             examples=examples or [])
        return fn  # return the bare function so the module can still call it directly
    return deco


def get(name: str) -> Skill:
    if name not in SKILLS:
        raise KeyError(f"no skill {name!r} (have: {sorted(SKILLS)})")
    return SKILLS[name]


def call(name: str, *args, **kwargs):
    """Invoke a skill by name — the interpreter's entry point."""
    return get(name).fn(*args, **kwargs)

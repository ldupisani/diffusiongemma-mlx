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
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            sig = None
        _check_example_arity(name, sig, examples or [])
        SKILLS[name] = Skill(name=name, fn=fn, doc=doc, signature=str(sig) if sig else "",
                             examples=examples or [])
        return fn  # return the bare function so the module can still call it directly
    return deco


def _check_example_arity(name: str, sig, examples: list) -> None:
    """Catch the single-arg gotcha AT IMPORT: an example's args must be a TUPLE whose length
    fits the function's positional arity. A bare value ('x^2' instead of ('x^2',)) unpacks
    into len('x^2') positional args at call time — a confusing test failure, caught here."""
    if sig is None:
        return
    params = [p for p in sig.parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
        return  # *args → variable arity, can't check
    required = sum(1 for p in params if p.default is p.empty)
    total = len(params)
    pnames = tuple(p.name for p in params)
    for args, _expected in examples:
        if not isinstance(args, tuple):
            raise ValueError(
                f"skill {name!r}: example args {args!r} is not a tuple — wrap a single "
                f"argument as a 1-tuple, e.g. (({args!r},), expected)")
        if not (required <= len(args) <= total):
            hint = (f" — likely a missing tuple wrap: (({args[0]!r},), expected)"
                    if len(args) > total and len(args) and total == 1 else "")
            raise ValueError(
                f"skill {name!r}: example args {args!r} has {len(args)} positional value(s) "
                f"but {name}{pnames} takes {required}–{total}{hint}")


def get(name: str) -> Skill:
    if name not in SKILLS:
        raise KeyError(f"no skill {name!r} (have: {sorted(SKILLS)})")
    return SKILLS[name]


def call(name: str, *args, **kwargs):
    """Invoke a skill by name — the interpreter's entry point."""
    return get(name).fn(*args, **kwargs)

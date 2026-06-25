"""Introspection skills — let higher-level reasoning check what the system can do BEFORE acting
(e.g. a planner probing "is there a domain/skill/generator for X?"). Pure: they read the
in-memory capability registries only — no model, no file I/O. Imports are lazy to avoid the
skills ← registry ← orchestrators import cycle.
"""

from __future__ import annotations

from .base import register


@register("introspect.domain_exists", "whether a reasoning domain (orchestrator id/type) exists",
          [(("number_theory",), True), (("no_such_domain",), False)])
def domain_exists(domain: str) -> bool:
    from reasoning import introspect
    return introspect.domain(domain) is not None


@register("introspect.skill_exists", "whether a named skill exists in the library",
          [(("numtheory.is_prime",), True), (("nope.nope",), False)])
def skill_exists(name: str) -> bool:
    from reasoning.skills import SKILLS
    return name in SKILLS


@register("introspect.generator_exists", "whether a dataset generator exists",
          [(("primes",), True), (("nope",), False)])
def generator_exists(name: str) -> bool:
    from reasoning import datagen
    return name in datagen.GENERATORS


@register("introspect.can_answer", "whether any reasoning domain covers a topic (capability probe)",
          [(("prime",), True), (("xyzzy_nonsense_topic",), False)])
def can_answer(topic: str) -> bool:
    from reasoning import introspect
    return bool(introspect.search(topic)["domains"])

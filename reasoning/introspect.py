"""Self-introspection — what the system can do.

A single source of truth over the three capability registries: the orchestrator registry
(reasoning domains), the skills library (pure operations), and the dataset generators. Used by
the CLI `help` command, the `introspect` orchestrator (NL meta-queries), and the introspection
skills. All reads are in-memory and deterministic (no model, no file I/O). Imports are lazy
because the registry pulls in orchestrators → binds → skills, and importing those at module load
would create a cycle.
"""

from __future__ import annotations


def _orchestrators():
    from .registry import load_orchestrators
    return load_orchestrators()


def domains() -> list[dict]:
    """Every reasoning domain (orchestrator), sorted by id."""
    return [{"id": o.id, "type": o.query_type, "description": o.description or "",
             "examples": len(o.example_queries), "model_free": bool(o.model_free),
             "fallback": bool(o.fallback)}
            for o in sorted(_orchestrators(), key=lambda o: o.id)]


def domain(name: str) -> dict | None:
    """Full detail for one domain (matched by id or query_type), or None."""
    for o in _orchestrators():
        if name in (o.id, o.query_type):
            return {"id": o.id, "type": o.query_type, "description": o.description or "",
                    "examples": list(o.example_queries), "keywords": list(o.keywords),
                    "model_free": bool(o.model_free)}
    return None


def search(term: str) -> dict:
    """Domains, skills, and generators whose name/description/examples/keywords mention `term`."""
    t = term.lower().strip()
    doms = [o.id for o in _orchestrators()
            if t in o.id.lower() or t in (o.description or "").lower()
            or any(t in k.lower() for k in o.keywords)
            or any(t in e.lower() for e in o.example_queries)]
    from .skills import SKILLS
    sk = [n for n, s in SKILLS.items() if t in n.lower() or t in s.doc.lower()]
    from . import datagen
    gens = [n for n, (_, doc) in datagen.GENERATORS.items() if t in n.lower() or t in doc.lower()]
    return {"domains": sorted(set(doms)), "skills": sorted(sk), "generators": sorted(gens)}


def skills(prefix: str | None = None) -> list[tuple[str, str]]:
    """(name, doc) for skills, optionally filtered to a namespace prefix (e.g. 'numtheory.')."""
    from .skills import SKILLS
    return sorted((n, s.doc) for n, s in SKILLS.items() if not prefix or n.startswith(prefix))


def summary() -> dict:
    """Headline counts across the capability registries."""
    from . import datagen
    from .skills import SKILLS
    orch = _orchestrators()
    return {"domains": len(orch),
            "model_free_domains": sum(1 for o in orch if o.model_free),
            "skills": len(SKILLS),
            "generators": len(datagen.GENERATORS),
            "generator_domains": len(datagen.DOMAINS)}

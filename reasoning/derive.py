"""Manual-assisted derivation: scaffold an orchestrator module from a captured trace.

Two modes:
  - "pattern": map the whole trace to ONE base pattern (decompose_solve_combine, etc.) and
    scaffold a run() that calls it. Best when the query type is genuinely fuzzy.
  - "steps":  map EACH atomic line of the teacher's trace onto a deterministic skill where
    one matches (leaving unmatched lines as model steps), and scaffold a compiled Plan.
    This is the skill-coverage flywheel: every unmatched line is a candidate for a NEW
    skill — add it to skills/, re-derive, and the coded coverage ratio climbs.

Derivation stays *assisted*: the scaffold prefills structure and coverage, and the human
wires each step's args and the bind. Automatic arg-binding is a later phase.
"""

from __future__ import annotations

import json
import pprint
from dataclasses import dataclass
from pathlib import Path

from . import skills
from .bind import _parse_list
from .registry import ORCH_DIR
from .router import _tokens

# --- pattern mode (unchanged) ---------------------------------------------------------
_IMPORTS = {
    "decompose_solve_combine": "from reasoning.patterns import decompose_solve_combine",
    "generate_rank_select": "from reasoning.patterns import generate_rank_select",
    "self_consistency": "from reasoning.patterns import self_consistency",
}
_RUN_BODIES = {
    "decompose_solve_combine": '    return decompose_solve_combine(query, quality="balanced")',
    "generate_rank_select": '    return generate_rank_select(query, n=3, '
    'criteria="correctness and completeness")',
    "self_consistency": "    return self_consistency(query, n=5)",
}

_PATTERN_TEMPLATE = '''"""Auto-scaffolded orchestrator (pattern mode) — EDIT to finalize.

Suggested pattern: __PATTERN__
TODO: tighten META (description / example_queries / keywords), refine run(), then run
`python -m reasoning.cli eval __ID__` to check fidelity against the teacher.
"""
__IMPORT__

META = __META__


def precondition(query):
    # TODO: define recognition. Until then the scaffold claims NOTHING — an unfinished
    # orchestrator must stay inert so it can't hijack routing for every other query.
    # Tighten to your query's signal (keywords, or `bind(query) != {}` once bind() works).
    return False


def run(query):
__BODY__
'''


def suggest_pattern(reasoning: str) -> str:
    r = reasoning.lower()
    if any(k in r for k in ["option", "candidate", "alternative", "better", " vs ", "compare"]):
        return "generate_rank_select"
    if any(k in r for k in ["check again", "verify", "double-check", "recompute", "let me re"]):
        return "self_consistency"
    return "decompose_solve_combine"  # default: most goals decompose


# --- step mode (the skill-coverage flywheel) ------------------------------------------
# generic doc/trace words that carry no skill signal — excluded from matching
_GENERIC = {
    "two", "number", "numbers", "value", "values", "list", "string", "items", "item",
    "date", "dates", "amount", "result", "step", "places", "the", "from", "into", "by",
}


def match_skill(line: str):
    """Map one atomic trace line to a Skill by cue-word overlap with its doc. None if weak.

    Cue words are the skill's doc tokens minus generic filler; a line that shares one or
    more distinctive cues with a skill's doc matches it (best-by-count). Adding a skill
    automatically extends coverage — that's the flywheel.
    """
    lt = _tokens(line)
    best, best_hits = None, 0
    for s in skills.SKILLS.values():
        cues = _tokens(s.doc) - _GENERIC
        hits = len(lt & cues)
        if hits > best_hits:
            best, best_hits = s, hits
    return best if best_hits >= 1 else None


_STEP_TEMPLATE = '''"""Auto-scaffolded compiled orchestrator (step mode) — EDIT to finalize.

Skill coverage: __MATCHED__/__TOTAL__ atomic steps mapped to skills (__GAP__ model step(s) remain).
TODO: (1) implement bind() to parse typed inputs from the query; (2) wire each CodeStep's
args using Inp("name") / Step("prior_out") / Lit(const); (3) set answer_ref/answer_format.
Each unmatched line below is a candidate for a NEW skill — add it to reasoning/skills/ and
re-derive to raise coverage. Then: `python -m reasoning.cli eval __ID__`.
"""

from reasoning import engine
from reasoning.compiled import CodeStep, Inp, Lit, ModelStep, Plan, Step, run_plan

META = __META__

PLAN = Plan(
    steps=[
__STEPS__
    ],
    answer_ref="__ANSWER_REF__",  # TODO: which step output is the answer
    answer_format=None,           # e.g. "${:.2f}" for money
)


def precondition(query):
    # TODO: define recognition. Until then the scaffold claims NOTHING — an unfinished
    # orchestrator must stay inert so it can't hijack routing for every other query.
    # Tighten to your query's signal (keywords, or `bind(query) != {}` once bind() works).
    return False


def bind(query):
    """TODO: parse typed inputs from the query (coded), e.g. with regex; or fall back to
    reasoning.patterns.extract_validate_repair for the model bind. Return a dict."""
    return {}


def run(query):
    # coded steps run model-free; switch to engine.one_shot if any ModelStep remains.
    shot = engine.one_shot if __HAS_MODEL__ else engine._no_model
    return run_plan(PLAN, bind(query), shot=shot)
'''


@dataclass
class ScaffoldResult:
    path: Path
    mode: str
    matched: int = 0
    total: int = 0


def _meta(sample: dict, query_type: str | None) -> dict:
    query = sample["query"]
    oid = sample["id"]
    return {
        "id": oid,
        "query_type": query_type or oid.replace("-", "_"),
        "description": f"TODO describe this query type. Derived from: {query!r}",
        "example_queries": [query],
        "keywords": sorted(_tokens(query))[:6],
        "sub_orchestrators": [],
        "provenance_sample": oid,
    }


def _scaffold_pattern(sample, pattern, query_type) -> str:
    pattern = pattern or suggest_pattern(sample.get("reasoning", ""))
    meta = _meta(sample, query_type)
    return (
        _PATTERN_TEMPLATE.replace("__META__", json.dumps(meta, indent=4))
        .replace("__IMPORT__", _IMPORTS[pattern])
        .replace("__BODY__", _RUN_BODIES[pattern])
        .replace("__PATTERN__", pattern)
        .replace("__ID__", sample["id"])
    )


def _scaffold_steps(sample, query_type) -> tuple[str, int, int]:
    lines = _parse_list(sample.get("reasoning", ""))
    rendered, matched = [], 0
    for i, line in enumerate(lines, start=1):
        skill = match_skill(line)
        safe = line.replace('"', "'")
        if skill:
            matched += 1
            rendered.append(f'        # [matched: {skill.name}] {safe}')
            rendered.append(f'        CodeStep("{skill.name}", [], out="s{i}"),  # TODO bind args')
        else:
            rendered.append(f'        # [model] {safe}')
            rendered.append(f'        ModelStep("{safe}", out="s{i}"),')
    total = len(lines)
    meta = _meta(sample, query_type)
    meta["model_free"] = (matched == total and total > 0)
    # pprint emits a valid Python literal (True/False/None), unlike json (true/false/null)
    src = (
        _STEP_TEMPLATE.replace("__META__", pprint.pformat(meta, indent=4, sort_dicts=False, width=88))
        .replace("__STEPS__", "\n".join(rendered) or "        # (no atomic steps parsed)")
        .replace("__ANSWER_REF__", f"s{total}" if total else "")
        .replace("__HAS_MODEL__", "True" if matched < total else "False")
        .replace("__MATCHED__", str(matched))
        .replace("__TOTAL__", str(total))
        .replace("__GAP__", str(total - matched))
        .replace("__ID__", sample["id"])
    )
    return src, matched, total


def scaffold(sample: dict, *, mode: str | None = None, pattern: str | None = None,
             query_type: str | None = None, force: bool = False) -> ScaffoldResult:
    """Scaffold an orchestrator module. mode auto-selects 'steps' when ≥1 line maps to a
    skill, else 'pattern'. Pass mode explicitly to override."""
    if mode is None:
        lines = _parse_list(sample.get("reasoning", ""))
        mode = "steps" if any(match_skill(l) for l in lines) else "pattern"

    if mode == "steps":
        src, matched, total = _scaffold_steps(sample, query_type)
    else:
        src, matched, total = _scaffold_pattern(sample, pattern, query_type), 0, 0

    ORCH_DIR.mkdir(parents=True, exist_ok=True)
    path = ORCH_DIR / f"{sample['id']}.py"
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (use force=True to overwrite)")
    path.write_text(src)
    return ScaffoldResult(path=path, mode=mode, matched=matched, total=total)

"""Compiled orchestrators: run a decomposition as a typed step list, code or model.

A `Plan` is a linear sequence of steps where each step is either a CodeStep (invoke a
deterministic skill — no model) or a ModelStep (one thinking-off `shot` call — fuzzy).
Each step names its output (`out`); later steps reference earlier outputs by name. This
is the executable form of the teacher's atomic trace, with the crucial property that a
Plan containing only CodeSteps NEVER contacts the sidecar.

The interpreter records per step whether it ran via code or model, and appends a summary
with the model-call tally — so "did this answer use the model?" is observable per run,
not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import skills
from .engine import one_shot
from .patterns import PatternResult


# --- argument references: how a step names its inputs --------------------------------
@dataclass(frozen=True)
class Inp:
    """Reference a bound input by name (from the bind layer)."""
    name: str


@dataclass(frozen=True)
class Step:
    """Reference a prior step's output by its `out` name."""
    ref: str


@dataclass(frozen=True)
class Lit:
    """A literal constant value."""
    value: object


def _resolve(arg, env: dict):
    if isinstance(arg, Inp):
        return env[arg.name]
    if isinstance(arg, Step):
        return env[arg.ref]
    if isinstance(arg, Lit):
        return arg.value
    return arg  # bare literal (int/str/etc.)


# --- step kinds ----------------------------------------------------------------------
@dataclass
class CodeStep:
    skill: str                      # key into skills.SKILLS
    args: list                      # positional ArgRefs / literals
    out: str                        # name to bind the result under
    kwargs: dict = field(default_factory=dict)


@dataclass
class ModelStep:
    prompt_template: str            # str.format template; filled from `args`
    out: str
    args: dict = field(default_factory=dict)   # name -> ArgRef (fills the template)
    max_tokens: int = 256
    quality: str = "balanced"


@dataclass
class Plan:
    steps: list
    answer_ref: str                 # which env name holds the final answer
    answer_format: str | None = None  # optional format template, e.g. "${:.2f}"


def run_plan(plan: Plan, bound_inputs: dict, *, shot=one_shot,
             prelude_trace: list | None = None, prelude_model_calls: int = 0) -> PatternResult:
    """Execute a Plan over the bound inputs; return a PatternResult with provenance.

    The trace's final entry is a summary: {"step":"summary","model_calls":N,"code_steps":M}.
    A fully-coded plan run with shot=engine._no_model proves zero model calls.

    `prelude_trace`/`prelude_model_calls` fold in work done BEFORE the plan — chiefly the
    bind step, which may itself call the model. This keeps the model-call tally honest: a
    model-based bind followed by coded steps still reports the bind's calls, not zero.
    """
    env = dict(bound_inputs)
    trace: list = list(prelude_trace or [])
    model_calls = prelude_model_calls
    for st in plan.steps:
        if isinstance(st, CodeStep):
            args = [_resolve(a, env) for a in st.args]
            val = skills.call(st.skill, *args, **st.kwargs)
            env[st.out] = val
            trace.append({"step": st.out, "via": "code", "skill": st.skill,
                          "args": args, "output": val})
        elif isinstance(st, ModelStep):
            filled = {k: _resolve(v, env) for k, v in st.args.items()}
            prompt = st.prompt_template.format(**filled)
            r = shot(prompt, max_tokens=st.max_tokens, quality=st.quality)
            env[st.out] = r.text
            model_calls += 1
            trace.append({"step": st.out, "via": "model", "prompt": prompt, "output": r.text})
        else:
            raise TypeError(f"unknown step type: {type(st).__name__}")

    code_steps = sum(1 for t in trace if t.get("via") == "code")
    trace.append({"step": "summary", "model_calls": model_calls, "code_steps": code_steps})

    ans = env[plan.answer_ref]
    answer = plan.answer_format.format(ans) if plan.answer_format else str(ans)
    return PatternResult(answer=answer, trace=trace, data=env)


def model_calls(result: PatternResult) -> int | None:
    """Model-call count from a compiled PatternResult's trace summary.

    Returns None for a non-compiled result (a pattern orchestrator with no per-step
    provenance) — we must NOT claim "0 model calls" when we simply can't see them.
    """
    for entry in reversed(result.trace):
        if entry.get("step") == "summary":
            return entry.get("model_calls", 0)
    return None  # unknown: not a compiled plan

"""Pluggable teacher backends — the 'slow path' the compiler distills from.

A teacher takes a query and returns a (reasoning, answer) trace. The compiler never
runs the teacher at student-time; it records the trace ONCE per query type so derive.py
can compile it into a coded orchestrator of thinking-off one-shot calls.

Two backends, same interface:
  - DiffusionTeacher: the local model with thinking ON (self-distillation; what v0 used).
  - ClaudeTeacher:    shells out to the `claude` CLI (-p/--print). A *stronger* teacher,
                      so fidelity then measures whether the coded DiffusionGemma pipeline
                      can reproduce a frontier model's answer — a true capability-gap gauge.

Claude's print mode returns final text (not extended-thinking blocks), so we ask it to
emit an explicit `REASONING:` / `ANSWER:` structure and split on that — the same
(reasoning, answer) shape capture.py already stores.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class TeacherTrace:
    query: str
    reasoning: str
    answer: str
    raw: str
    backend: str


# --- The teacher instruction (the key tuning point) ----------------------------------
# This shapes the ENTIRE compiler. A naive "just answer this" makes Claude reason with
# Claude-level competence per step — steps DiffusionGemma can't reproduce one-shot, so
# fidelity stays low. We instead ask for a decomposition into ATOMIC sub-steps, each of
# which a small model could plausibly do in a single thinking-off call. That is what
# makes the trace *compilable* onto the weak student.
CLAUDE_TEACHER_SYSTEM = (
    "You are a reasoning teacher for a compiler. Your reasoning will be turned into CODE "
    "that orchestrates a much smaller, weaker model (the 'student'). The student runs each "
    "step in ONE shot with NO chain of thought and a short (~256-token) output. Your job is "
    "to produce a decomposition the student can actually execute step by step.\n\n"
    "The student is RELIABLE at, per single step: one factual recall; ONE arithmetic "
    "operation (a single +, -, ×, ÷, or %); extracting a value from text you quote in the "
    "step; rephrasing or drafting a short piece of text; choosing the best of a few options "
    "you list. The student is UNRELIABLE at: chaining multiple operations in one step; "
    "multi-hop logic; remembering anything not written in the step itself; long outputs.\n\n"
    "Rules for your sub-steps:\n"
    "- Use AT MOST 7 steps. If a problem needs more, group it differently — don't exceed 7.\n"
    "- Each step is ONE atomic operation from the reliable list above. Never combine two.\n"
    "- Each step must be self-contained: restate every input it needs (including results "
    "from earlier steps as concrete values, e.g. 'Multiply 24 (from step 1) by 0.07').\n"
    "- Don't pad: if the answer is a single recall or one operation, use one step.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "REASONING:\n<numbered atomic sub-steps, one per line>\n"
    "ANSWER:\n<the final answer only, no explanation>"
)


def _result_obj(data) -> dict:
    """`claude -p --output-format json` may return the result object directly, or (in
    verbose setups) a JSON array of stream messages whose final `type:"result"` element
    holds the answer. Normalise both to that result dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        results = [x for x in data if isinstance(x, dict) and x.get("type") == "result"]
        if results:
            return results[-1]
    raise RuntimeError("could not find a result object in claude CLI output")


def ask_claude(prompt: str, model: str = "sonnet", timeout: int = 60) -> str | None:
    """One-shot Claude CLI call returning the raw answer text (no REASONING/ANSWER split).
    Returns None if the `claude` CLI is absent or the call errors — callers treat that as
    'retrieval unavailable'. Used by the reference-retrieval fallback."""
    if shutil.which("claude") is None:
        return None
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        result = _result_obj(json.loads(proc.stdout))
    except (subprocess.SubprocessError, ValueError, RuntimeError):
        return None
    return None if result.get("is_error") else (result.get("result") or "").strip()


def _split_reasoning_answer(text: str) -> tuple[str, str]:
    """Split Claude's structured output into (reasoning, answer). Prefer the LAST ANSWER:
    marker; if the model ignored the format, fall back to the final non-empty line (the
    final answer is conventionally last) rather than dumping the whole reasoning into the
    answer field — which otherwise leaves a mid-reasoning fragment as the 'answer'."""
    matches = list(re.finditer(r"\bANSWER\s*:\s*", text, re.IGNORECASE))
    if matches:
        last = matches[-1]
        reasoning = re.sub(r"^\s*REASONING\s*:\s*", "", text[: last.start()].strip(),
                           flags=re.IGNORECASE).strip()
        return reasoning, text[last.end():].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", text.strip()
    return "\n".join(lines[:-1]), lines[-1]


class DiffusionTeacher:
    """The local model, thinking ON. Self-distillation (the original v0 teacher)."""

    name = "diffusion"

    def trace(self, query: str, *, seed: int = 0) -> TeacherTrace:
        from .engine import split_trace, teacher  # lazy: avoids the import on claude-only runs

        r = teacher(query, seed=seed)
        reasoning, answer = split_trace(r.text)
        return TeacherTrace(query, reasoning, answer, r.text, self.name)


class ClaudeTeacher:
    """The `claude` CLI as a stronger teacher, via headless print mode.

    Uses the user's existing Claude Code auth (no API key, no extra plumbing). `seed` is
    accepted for interface parity but ignored — Claude is not seeded here.

    Defaults to the `sonnet` model alias (the current Claude Sonnet) so fidelity runs are
    reproducible regardless of the CLI's configured default. Pass `model=None` to defer to
    the CLI's default instead, or another id/alias to pin a specific model.
    """

    name = "claude"
    DEFAULT_MODEL = "sonnet"  # CLI alias -> the current Claude Sonnet (tracks version bumps)

    def __init__(self, *, model: str | None = DEFAULT_MODEL, system: str = CLAUDE_TEACHER_SYSTEM,
                 timeout: int = 180):
        if shutil.which("claude") is None:
            raise RuntimeError("`claude` CLI not found on PATH — install Claude Code or use "
                               "the diffusion teacher.")
        self.model = model
        self.system = system
        self.timeout = timeout

    def trace(self, query: str, *, seed: int = 0) -> TeacherTrace:
        cmd = ["claude", "-p", query, "--output-format", "json",
               "--append-system-prompt", self.system]
        if self.model:
            cmd += ["--model", self.model]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({proc.returncode}): {proc.stderr.strip()}")
        result = _result_obj(json.loads(proc.stdout))
        if result.get("is_error"):
            raise RuntimeError(f"claude returned an error: {result.get('result')}")
        raw = result.get("result", "")
        reasoning, answer = _split_reasoning_answer(raw)
        return TeacherTrace(query, reasoning, answer, raw, self.name)


def get_teacher(name: str = "diffusion", **kw) -> DiffusionTeacher | ClaudeTeacher:
    if name == "claude":
        return ClaudeTeacher(**kw)
    if name == "diffusion":
        return DiffusionTeacher()
    raise ValueError(f"unknown teacher backend {name!r} (use 'diffusion' or 'claude')")

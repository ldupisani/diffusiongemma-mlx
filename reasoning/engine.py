"""Atomic, thinking-OFF calls to the diffusion model via the infer_server sidecar.

These two primitives — one_shot and one_shot_n — are the ONLY way the reasoning
patterns touch the model. The sidecar (infer_server.py) holds the model; this module
is a stdlib-only HTTP client, so the reasoning CLI starts instantly and never imports
MLX itself.

The defining constraint: every call sets enable_thinking=False. The model fills a
single atomic cognitive slot in one shot; the *reasoning* (decomposition, sequencing,
selection) lives in code, not in the model's tokens.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

SERVER = os.environ.get("DIFG_SERVER", "http://127.0.0.1:8088")

# Even with thinking off the template emits an empty <|channel>thought ... <channel|>;
# strip it so callers get clean text. (Mirrors chat_stream.strip_thought, inlined to
# keep this client free of the heavy mlx/mcp imports chat_stream pulls in.)
_THOUGHT_RE = re.compile(r"<\|channel>thought.*?<channel\|>\s*", re.DOTALL)


def strip_thought(text: str) -> str:
    return _THOUGHT_RE.sub("", text).strip()


# Minimal diffusion presets (mirror chat.choose_diffusion_params). Reasoning sub-tasks
# favour low temperature for stability; diversity for generate/vote comes from the SEED
# (the diffusion canvas's "noise sample"), not from temperature.
_PRESETS = {
    "fast": {"max_denoising_steps": 12, "temperature": 0.2},
    "balanced": {"max_denoising_steps": 32, "temperature": 0.3},
    "quality": {"max_denoising_steps": 48, "temperature": 0.4},
}


@dataclass
class Result:
    text: str
    finish: str | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    peak_memory: float | None = None
    seed: int | None = None


def _post(body: dict, server: str) -> Result:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        server + "/generate", data=data, headers={"Content-Type": "application/json"}
    )
    text = ""
    res = Result(text="")
    with urllib.request.urlopen(req) as resp:  # NDJSON: one JSON object per line
        for raw in resp:
            line = raw.decode().strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") == "chunk":
                text += ev["text"]
            elif ev.get("type") == "done":
                res.finish = ev.get("finish")
                res.prompt_tokens = ev.get("prompt_tokens", 0) or 0
                res.generation_tokens = ev.get("generation_tokens", 0) or 0
                res.peak_memory = ev.get("peak_memory")
    res.text = text  # raw (callers strip if they want the thought channel gone)
    return res


def _no_model(*_a, **_k):
    """A `shot` stand-in for orchestrators that declare themselves model-free.

    Passing this as `shot` turns "this path uses zero model calls" from a hope into an
    invariant: if a future edit slips a ModelStep into a supposedly-coded plan, it fails
    loudly here instead of silently waking the ~19 GB sidecar.
    """
    raise RuntimeError("model-free orchestrator attempted a model call")


def one_shot(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 256,
    quality: str = "balanced",
    temperature: float | None = None,
    seed: int | None = None,
    server: str = SERVER,
) -> Result:
    """One atomic, thinking-off completion. The reasoning patterns compose these."""
    preset = _PRESETS.get(quality, _PRESETS["balanced"])
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {
        "messages": messages,
        "enable_thinking": False,  # the whole point — no chain-of-thought in tokens
        "max_tokens": max_tokens,
        "block_length": max(256, max_tokens),
        "max_denoising_steps": preset["max_denoising_steps"],
        "diffusion_sampler": "entropy-bound",
        "temperature": preset["temperature"] if temperature is None else temperature,
    }
    if seed is not None:
        body["seed"] = int(seed)
    r = _post(body, server)
    r.text = strip_thought(r.text)
    r.seed = seed
    return r


def teacher(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    quality: str = "quality",
    seed: int | None = None,
    server: str = SERVER,
) -> Result:
    """The 'slow path': run WITH thinking on, returning RAW text (thought channel kept).

    Used only by capture/eval — this is the reasoning we are trying to compile away.
    """
    preset = _PRESETS.get(quality, _PRESETS["quality"])
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {
        "messages": messages,
        "enable_thinking": True,
        "max_tokens": max_tokens,
        "block_length": max(256, max_tokens),
        "max_denoising_steps": preset["max_denoising_steps"],
        "diffusion_sampler": "entropy-bound",
        "temperature": preset["temperature"],
    }
    if seed is not None:
        body["seed"] = int(seed)
    r = _post(body, server)
    r.seed = seed
    return r


def split_trace(raw: str) -> tuple[str, str]:
    """Split a thinking-on output into (reasoning, answer) on the thought channel."""
    m = re.search(r"<\|channel>thought(.*?)<channel\|>(.*)", raw, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", strip_thought(raw)


def one_shot_n(prompt: str, n: int, *, base_seed: int = 0, **kw) -> list[Result]:
    """n completions of the same prompt, differing only by seed.

    The seed sets the random initial canvas, so distinct seeds yield genuinely
    different one-shot trajectories even at low temperature — the diffusion analog
    of sampling different noise. This is what powers generate-rank and self-consistency.
    """
    return [one_shot(prompt, seed=base_seed + i, **kw) for i in range(n)]

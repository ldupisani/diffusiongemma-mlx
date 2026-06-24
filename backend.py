#!/usr/bin/env python3
"""Pluggable inference backends, so the toolkit isn't tied to one model family.

DiffusionGemma was the early experiment this repo started from. But the reasoning
engine (reasoning/) answers most queries deterministically with *zero* model calls
— the LLM is only a fallback — so a small 3B instruct model is plenty, at a fraction
of the 26B diffusion model's memory footprint.

Two backends share ONE interface (load / apply_template / stream / parse_tool_calls
/ strip_thought), so every consumer (the HTTP sidecar, the Python + TS REPLs) is
written once and works with either:

  * DiffusionBackend       — mlx-vlm, the original DiffusionGemma path. Non-autoregressive
                             (denoising-canvas sampling), Gemma-DSL tool calls, a <thought>
                             reasoning channel. Knobs: max_denoising_steps, block_length, …
  * AutoregressiveBackend  — mlx-lm, for standard small instruct models (Qwen2.5, Llama-3.2,
                             …). Left-to-right token streaming, JSON tool calls (Hermes /
                             Qwen style), no thought channel. Small memory footprint.

Both backends yield response objects with the SAME attributes (.text, .finish_reason,
.prompt_tokens, .generation_tokens, .peak_memory) — mlx-vlm and mlx-lm happen to agree
here — which is exactly why the streaming consumers don't care which one is behind them.

Selection (most specific wins):
  1. explicit `name=` argument to get_backend()
  2. $DIFG_BACKEND   ("diffusion" | "autoregressive")
  3. auto-detect from the model id (contains "diffusiongemma" -> diffusion, else autoregressive)

Imports of mlx-vlm / mlx-lm are deferred into each backend, so you only need the
package for the backend you actually use.
"""

from __future__ import annotations

import os

# The out-of-the-box default: a small, capable, tool-calling instruct model.
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
# The original experiment — still selectable via DIFG_MODEL (auto-routes to the diffusion backend).
DEFAULT_DIFFUSION_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def context_limit(config):
    """Dig max_position_embeddings out of a (possibly nested) model config."""
    for obj in (getattr(config, "text_config", None), config):
        if obj is None:
            continue
        get = obj.get if isinstance(obj, dict) else (lambda k, d=None: getattr(obj, k, d))
        val = get("max_position_embeddings")
        if val:
            return int(val)
    return None


# --- speed/quality presets -------------------------------------------------

# Diffusion: the genuinely interesting knobs are denoising passes + canvas size.
_DIFFUSION_PRESETS = {
    "fast": {"max_denoising_steps": 12, "diffusion_sampler": "entropy-bound", "temperature": 0.2},
    "balanced": {"max_denoising_steps": 32, "diffusion_sampler": "entropy-bound", "temperature": 0.4},
    "quality": {"max_denoising_steps": 48, "diffusion_sampler": "entropy-bound", "temperature": 0.6},
}
# Autoregressive: there is no canvas/denoising; the preset only nudges sampling temperature.
_AR_TEMP = {"fast": 0.0, "balanced": 0.4, "quality": 0.7}


def choose_diffusion_params(quality: str, max_tokens: int) -> dict:
    """Map a 'quality' preference to diffusion sampling params (canvas >= max_tokens)."""
    p = dict(_DIFFUSION_PRESETS.get(quality, _DIFFUSION_PRESETS["balanced"]))
    p["block_length"] = max(256, max_tokens)  # the canvas can't emit more tokens than it has slots
    return p


def choose_params(quality: str, max_tokens: int, backend: "Backend") -> dict:
    """Per-backend generation params for a 'quality' preset. Diffusion gets the full
    canvas config; autoregressive gets just a temperature (other keys are ignored)."""
    if getattr(backend, "name", "") == "diffusion":
        return choose_diffusion_params(quality, max_tokens)
    return {"temperature": _AR_TEMP.get(quality, 0.4)}


# --- backends --------------------------------------------------------------

class Backend:
    """Common interface + shared one-shot helper. Subclasses implement load / apply_template
    / stream / parse_tool_calls / strip_thought and set the capability flags."""

    name = "base"
    supports_thinking = False
    tool_call_format = "json"   # how this model emits tool calls: "gemma-dsl" | "json"

    def __init__(self):
        self.model = None
        self.processor = None     # mlx-vlm processor OR mlx-lm tokenizer
        self.config = None
        self.context_window = None
        self.model_id = None

    def generate(self, prompt, *, max_tokens=256, enable_thinking=False, params=None):
        """One-shot convenience: drain stream() into (text, last_chunk)."""
        text, last = "", None
        for chunk in self.stream(prompt, max_tokens=max_tokens,
                                 enable_thinking=enable_thinking, params=params):
            text += chunk.text
            last = chunk
        return text, last

    def info(self) -> dict:
        """The /info payload the sidecar reports so clients know how to parse output."""
        return {
            "model": self.model_id,
            "context_window": self.context_window,
            "backend": self.name,
            "supports_thinking": self.supports_thinking,
            "tool_call_format": self.tool_call_format,
        }


class DiffusionBackend(Backend):
    name = "diffusion"
    supports_thinking = True
    tool_call_format = "gemma-dsl"

    def load(self, model_id):
        from mlx_vlm import load as _load
        print(f"Loading {model_id} (diffusion / mlx-vlm) ...", flush=True)
        self.model, self.processor = _load(model_id)
        self.config = self.model.config
        self.context_window = context_limit(self.config)
        self.model_id = model_id
        return self

    def apply_template(self, messages, *, tools=None, enable_thinking=False):
        from mlx_vlm.prompt_utils import apply_chat_template
        return apply_chat_template(
            self.processor, self.config, messages, num_images=0,
            tools=tools or None, enable_thinking=enable_thinking,
        )

    def stream(self, prompt, *, max_tokens=256, enable_thinking=False, params=None):
        from mlx_vlm import stream_generate
        params = params or {}
        yield from stream_generate(
            self.model, self.processor, prompt,
            max_tokens=max_tokens, enable_thinking=enable_thinking,
            max_denoising_steps=params.get("max_denoising_steps", 32),
            block_length=params.get("block_length", 256),
            diffusion_sampler=params.get("diffusion_sampler", "entropy-bound"),
            temperature=params.get("temperature", 0.4),
        )

    def parse_tool_calls(self, text):
        from tools_util import parse_tool_calls
        return parse_tool_calls(text)

    def strip_thought(self, text):
        from tools_util import strip_thought_channel
        return strip_thought_channel(text)


class AutoregressiveBackend(Backend):
    name = "autoregressive"
    supports_thinking = False
    tool_call_format = "json"

    def load(self, model_id):
        from mlx_lm import load as _load
        print(f"Loading {model_id} (autoregressive / mlx-lm) ...", flush=True)
        self.model, self.processor = _load(model_id)   # processor == tokenizer here
        self.config = getattr(self.model, "args", None)
        self.context_window = self._ctx_window()
        self.model_id = model_id
        return self

    def _ctx_window(self):
        args = getattr(self.model, "args", None)
        for attr in ("max_position_embeddings", "max_seq_len"):
            val = getattr(args, attr, None) if args is not None else None
            if val:
                return int(val)
        return 32768  # sane default (e.g. Qwen2.5) when the config doesn't expose it

    def apply_template(self, messages, *, tools=None, enable_thinking=False):
        # The tokenizer's own chat template knows this model's tool-call format. We do
        # NOT pass enable_thinking: standard instruct models (Qwen2.5, Llama-3.2) have
        # no thought channel, and some templates reject the kwarg.
        return self.processor.apply_chat_template(
            messages, tools=tools or None,
            add_generation_prompt=True, tokenize=False,
        )

    def stream(self, prompt, *, max_tokens=256, enable_thinking=False, params=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        params = params or {}
        sampler = make_sampler(temp=float(params.get("temperature", 0.0)))
        yield from stream_generate(
            self.model, self.processor, prompt,
            max_tokens=max_tokens, sampler=sampler,
        )

    def parse_tool_calls(self, text):
        from tools_util import parse_tool_calls_json
        return parse_tool_calls_json(text)

    def strip_thought(self, text):
        return text.strip()  # no thought channel to strip


def _auto_backend(model_id: str) -> str:
    mid = (model_id or "").lower()
    return "diffusion" if ("diffusiongemma" in mid or "diffusion-gemma" in mid) else "autoregressive"


def get_backend(model_id: str = DEFAULT_MODEL, name: str | None = None) -> Backend:
    """Resolve and return an (unloaded) backend. Call .load(model_id) before use."""
    name = (name or os.environ.get("DIFG_BACKEND") or _auto_backend(model_id)).lower()
    if name == "diffusion":
        return DiffusionBackend()
    if name in ("autoregressive", "ar", "mlx-lm", "mlx_lm"):
        return AutoregressiveBackend()
    raise ValueError(f"unknown DIFG_BACKEND={name!r} (use 'diffusion' or 'autoregressive')")

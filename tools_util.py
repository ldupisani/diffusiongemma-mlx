#!/usr/bin/env python3
"""Shared tool-calling helpers, kept separate so the one-shot agent (mcp_agent.py),
the interactive REPL (chat_stream.py), and the backends (backend.py) can use them
without a circular import.

Two tool-call wire formats are supported, one per backend family:
  * Gemma DSL (diffusion backend):  <|tool_call>call:NAME{key:<|"|>value<|"|>,n:42}<tool_call|>
  * JSON (autoregressive backend):  <tool_call>{"name": NAME, "arguments": {...}}</tool_call>
Both may emit several blocks back-to-back (parallel calls).
"""

import json
import re

QUOTE = "<|\"|>"  # Gemma's string-quote token inside the tool-call DSL
_OPEN = "<|tool_call>call:"  # each call starts here; the <tool_call|> close is optional

# DiffusionGemma wraps reasoning in <|channel>thought ... <channel|>. We show it live
# but strip it from stored history so it doesn't accumulate as context on later turns.
_THOUGHT_RE = re.compile(r"<\|channel>thought.*?<channel\|>\s*", re.DOTALL)


def strip_thought_channel(text: str) -> str:
    """Remove DiffusionGemma's ephemeral <thought> channel from text."""
    return _THOUGHT_RE.sub("", text).strip()


def mcp_tool_to_model(tool):
    """MCP tool (name/description/inputSchema) -> the model's tools= entry.

    MCP's inputSchema is already JSON Schema, exactly what the chat template's
    tool declaration consumes — so this is a thin reshape, not a translation.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def split_top_level(body: str):
    """Split a `{...}` body on top-level commas (ignores commas inside braces)."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return [p for p in parts if p.strip()]


def _parse_args(body: str) -> dict:
    """Parse a `{k:v,...}` body. Values are quoted strings (QUOTE token) or bare
    literals (int / float / bool). Covers flat argument objects."""
    args = {}
    for pair in split_top_level(body):
        if ":" not in pair:
            continue
        key, val = pair.split(":", 1)
        key, val = key.strip(), val.strip()
        if val.startswith(QUOTE) and val.endswith(QUOTE):
            args[key] = val[len(QUOTE):-len(QUOTE)]
        elif val in ("true", "false"):
            args[key] = val == "true"
        else:
            try:
                args[key] = int(val)
            except ValueError:
                try:
                    args[key] = float(val)
                except ValueError:
                    args[key] = val.replace(QUOTE, "")
    return args


def _balanced_brace_end(text: str, open_idx: int):
    """Index of the `}` that closes the `{` at open_idx, or -1 if unbalanced."""
    depth = 0
    for k in range(open_idx, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return -1


def parse_tool_calls(text: str):
    """Return a list of (name, args) for every tool call in the output.

    Robust to two things models actually do wrong: (1) omitting the <tool_call|>
    terminator, and (2) emitting several calls back-to-back. We never rely on the
    closing token — we find each `<|tool_call>call:` opener, read the name up to
    `{`, then scan BALANCED braces for the argument body (which also handles
    nested-object args). A malformed, unterminated call no longer swallows the
    next one or corrupts its arguments.
    """
    calls, i = [], 0
    while True:
        start = text.find(_OPEN, i)
        if start == -1:
            return calls
        brace = text.find("{", start + len(_OPEN))
        if brace == -1:
            return calls
        end = _balanced_brace_end(text, brace)
        if end == -1:
            return calls  # truncated mid-args; stop
        name = text[start + len(_OPEN):brace].strip()
        calls.append((name, _parse_args(text[brace + 1:end])))
        i = end + 1


def _loads_first_object(blob: str):
    """json.loads the first balanced {...} object in blob, or None.

    Tolerates redundant outer brace layers: Qwen2.5's bundled chat template shows the
    tool-call example as `{{"name": ..., "arguments": ...}}` (a Jinja-escape artifact),
    and the model copies the doubled braces verbatim. We peel matched outer layers until
    it parses to an object — which also hardens us against similar LLM malformations.
    """
    brace = blob.find("{")
    if brace == -1:
        return None
    end = _balanced_brace_end(blob, brace)
    if end == -1:
        return None
    candidate = blob[brace:end + 1].strip()
    for _ in range(4):
        try:
            obj = json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            obj = None
        if isinstance(obj, dict):
            return obj
        if candidate.startswith("{") and candidate.endswith("}"):
            candidate = candidate[1:-1].strip()  # peel one redundant brace layer
        else:
            return None
    return None


def parse_tool_calls_json(text: str):
    """Return [(name, args)] for the JSON tool-call format that standard instruct
    models (Qwen2.5, Llama-3.2, Hermes-style) emit:

        <tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>

    Robust to the same two real-world failure modes as the Gemma parser: a missing
    </tool_call> terminator (fall back to the balanced JSON object after the opener)
    and several calls back-to-back. `arguments` may itself arrive as a JSON string.
    """
    OPEN, CLOSE = "<tool_call>", "</tool_call>"
    calls, i = [], 0
    while True:
        start = text.find(OPEN, i)
        if start == -1:
            return calls
        body_start = start + len(OPEN)
        close = text.find(CLOSE, body_start)
        if close == -1:  # no terminator: take the balanced JSON object after the opener
            brace = text.find("{", body_start)
            if brace == -1:
                return calls
            end = _balanced_brace_end(text, brace)
            if end == -1:
                return calls  # truncated mid-args; stop
            blob, i = text[brace:end + 1], end + 1
        else:
            blob, i = text[body_start:close], close + len(CLOSE)
        obj = _loads_first_object(blob)
        if not obj or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("parameters", {})) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, json.JSONDecodeError):
                args = {}
        calls.append((obj["name"], args if isinstance(args, dict) else {}))

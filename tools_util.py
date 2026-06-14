#!/usr/bin/env python3
"""Shared helpers for DiffusionGemma tool calling.

Kept separate so both the one-shot agent (mcp_agent.py) and the interactive
REPL (chat_stream.py) can use them without a circular import.

The model emits tool calls in Gemma's native DSL, NOT JSON:
    <|tool_call>call:NAME{key:<|"|>value<|"|>,n:42}<tool_call|>
and may emit several such blocks in one turn (parallel calls).
"""

QUOTE = "<|\"|>"  # Gemma's string-quote token inside the tool-call DSL
_OPEN = "<|tool_call>call:"  # each call starts here; the <tool_call|> close is optional


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

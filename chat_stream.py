#!/usr/bin/env python3
"""Interactive streaming chat over a pluggable MLX backend (see backend.py).

Works with either backend family. With the diffusion backend, output arrives in
canvas blocks as they finish denoising rather than smoothly left-to-right; with the
autoregressive backend it streams token-by-token. The REPL itself doesn't care —
it only consumes the shared chunk interface.

Commands inside the REPL:
  /quality fast|balanced|quality   switch the speed/quality preset
  /think    toggle the model's <thought> reasoning channel (diffusion backend only)
  /continue extend a truncated reply OR resume a capped tool loop
  /auto <n> budget for both text continuations AND tool-call rounds per turn
  /tools    list MCP tools available to the model
  /reset    clear conversation history
  /exit     quit

Tool calling: by default the REPL launches mcp_server_demo.py over MCP and lets
the model call its tools mid-conversation. Pass --no-tools to disable.
"""

import asyncio
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend import DEFAULT_MODEL, backend_shot, get_backend
from reasoning.compiled import model_calls
from reasoning.solve import try_coded
from tools_util import mcp_tool_to_model

# Default tool-round budget per turn when /auto is OFF (runaway guard). When
# /auto N is set, N becomes the budget — so the same "how long do I keep going?"
# dial governs both text continuations and tool-call rounds.
DEFAULT_TOOL_STEPS = 6

# Reasoning + answer need ~3-4x a normal answer's budget; the model still stops
# early at EOS if it finishes sooner, so this is a ceiling, not a fixed cost.
THINK_MAX_TOKENS = 1024


def print_context(last, ctx_max):
    """Show context-window occupancy and peak memory after a generation.

    On a Mac the honest ceiling is memory, not the token window, so peak_memory
    (GB of unified memory at high-water mark) is the number that actually tells
    you how close you are to trouble.
    """
    if last is None:
        return
    used = (last.prompt_tokens or 0) + (last.generation_tokens or 0)
    ctx = f"{used:,} / {ctx_max:,} tokens — {used / ctx_max:.1%}" if ctx_max else f"{used:,} tokens"
    mem = getattr(last, "peak_memory", None)
    mem_str = f", peak mem {mem:.1f} GB" if mem else ""
    print(f"(context: {ctx}{mem_str})\n")


def stream_reply(backend, formatted, effective_max, quality, think):
    """Stream one generation to stdout. Returns (raw_text, finish_reason, last_chunk)."""
    params = backend.gen_params(quality, effective_max)
    text, finish, last = "", None, None
    for chunk in backend.stream(formatted, max_tokens=effective_max,
                                enable_thinking=think, params=params):
        text += chunk.text
        finish = chunk.finish_reason
        last = chunk
        print(chunk.text, end="", flush=True)
    print()
    return text, finish, last


def extend_reply(backend, history, last_prompt, last_raw, last_max,
                 last_think, quality, ctx_max):
    """Run ONE continuation of the last (truncated) reply.

    Resume by raw concatenation: original templated prompt + the partial reply,
    fed straight to the engine. We deliberately do NOT re-run apply_template here,
    because the chat template strips the ephemeral <thought> channel on re-render —
    which would delete the continuation anchor whenever the cut lands mid-thought.
    Concatenation keeps the partial text (open thought and all) intact.
    Returns (updated_last_raw, finish_reason).
    """
    formatted = last_prompt + last_raw
    print("difg> ", end="", flush=True)
    more, finish, last = stream_reply(backend, formatted, last_max, quality, last_think)
    print()
    print_context(last, ctx_max)
    last_raw += more
    history[-1] = {"role": "assistant", "content": backend.strip_thought(last_raw)}
    return last_raw, finish


def auto_extend(backend, history, last_prompt, last_raw, last_finish,
                last_max, last_think, quality, auto_max, ctx_max):
    """Extend repeatedly until EOS or auto_max extensions are used."""
    n = 0
    while last_finish == "length" and n < auto_max:
        n += 1
        print(f"(auto-continue {n}/{auto_max})")
        last_raw, last_finish = extend_reply(
            backend, history, last_prompt, last_raw, last_max,
            last_think, quality, ctx_max,
        )
    if last_finish == "length":
        print(f"(stopped at {auto_max} auto-extensions — /continue or raise /auto)\n")
    return last_raw, last_finish


async def agent_turn(backend, history, tools, session,
                     effective_max, quality, think, ctx_max, budget):
    """Run (or resume) one user turn, looping over tool calls until a final answer.

    Runs up to `budget` tool-call rounds. Returns (last_prompt, last_raw,
    last_finish): for a real answer last_finish is the text finish_reason; if the
    budget is exhausted with calls still pending it is the sentinel "tool_cap",
    which the REPL treats as resumable (re-enter with the same history, which
    already holds every tool result, to keep going).

    Tool-call/response messages are appended to `history` in OpenAI style; the
    chat template renders them back into the model's native tool blocks each round.
    """
    formatted = ""
    for step in range(1, budget + 1):
        formatted = backend.apply_template(history, tools=tools or None, enable_thinking=think)
        print("difg> ", end="", flush=True)
        text, finish, last = stream_reply(backend, formatted, effective_max, quality, think)
        print_context(last, ctx_max)

        calls = backend.parse_tool_calls(text) if (tools and session) else []
        if not calls:
            # Final answer — store the clean text (no ephemeral thought channel).
            history.append({"role": "assistant", "content": backend.strip_thought(text)})
            return formatted, text, finish

        # Execute every call this round (the model may issue several at once).
        tool_calls, tool_replies = [], []
        for j, (name, args) in enumerate(calls):
            cid = f"s{step}_{j}"
            print(f"  [tool] {name}({args})")
            try:
                res = await session.call_tool(name, args)
                payload = res.content[0]
                value = getattr(payload, "text", str(payload))
            except Exception as e:
                value = f"ERROR: {type(e).__name__}: {e}"
            print(f"  [tool] -> {value}\n")
            tool_calls.append({
                "id": cid, "type": "function",
                "function": {"name": name, "arguments": args},
            })
            tool_replies.append({
                "role": "tool", "tool_call_id": cid, "name": name, "content": value,
            })
        history.append({"role": "assistant", "tool_calls": tool_calls})
        history.extend(tool_replies)

    return formatted, "", "tool_cap"


async def turn_with_followups(backend, history, tools, session,
                              effective_max, quality, think, ctx_max, auto_max):
    """Run a turn (or resume one) and apply the matching follow-up policy.

    Unifies both 'keep going' limits under /auto: auto_max (or DEFAULT_TOOL_STEPS
    when off) is the tool-round budget, and also drives text auto-continuation.
    """
    budget = auto_max if auto_max > 0 else DEFAULT_TOOL_STEPS
    last_prompt, last_raw, last_finish = await agent_turn(
        backend, history, tools, session,
        effective_max, quality, think, ctx_max, budget,
    )
    if last_finish == "length":  # the text answer was cut off
        if auto_max > 0:
            last_raw, last_finish = auto_extend(
                backend, history, last_prompt, last_raw, last_finish,
                effective_max, think, quality, auto_max, ctx_max,
            )
        else:
            print("(reply hit the token cap — type /continue to extend it)\n")
    elif last_finish == "tool_cap":  # ran out of tool rounds mid-task
        print(f"(used {budget} tool steps without finishing — "
              "/continue to keep going, or raise /auto)\n")
    return last_prompt, last_raw, last_finish


async def run_repl(backend, ctx_max, session, tools):
    quality, think, auto_max = "balanced", False, 0
    max_tokens = 256
    history = []
    # State for /continue: original templated prompt + raw text of the last reply.
    last_prompt, last_raw, last_finish = "", "", None
    last_max, last_think = max_tokens, False
    tool_names = [t["function"]["name"] for t in tools]
    coded_shot = backend_shot(backend)  # in-process one-shot for the code engine's reformulation

    while True:
        try:
            user = (await asyncio.to_thread(input, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)\n")
            continue
        if user == "/tools":
            print(f"(tools: {', '.join(tool_names) if tool_names else 'none'})\n")
            continue
        if user.startswith("/quality"):
            parts = user.split()
            if len(parts) == 2 and parts[1] in ("fast", "balanced", "quality"):
                quality = parts[1]
                print(f"(quality -> {quality})\n")
            else:
                print("usage: /quality fast|balanced|quality\n")
            continue
        if user == "/think":
            if not backend.supports_thinking:
                print(f"(the {backend.name} backend has no <thought> channel — /think ignored)\n")
                continue
            think = not think
            print(f"(thinking -> {'on' if think else 'off'})\n")
            continue
        if user.startswith("/auto"):
            parts = user.split()
            if len(parts) == 2 and parts[1].isdigit():
                auto_max = int(parts[1])
                print(f"(auto -> {auto_max} text-continuations / tool-rounds per turn"
                      f"{' (off)' if auto_max == 0 else ''})\n")
                # Apply the new budget to whatever is currently paused.
                if auto_max > 0 and last_finish == "length":
                    last_raw, last_finish = auto_extend(
                        backend, history, last_prompt, last_raw, last_finish,
                        last_max, last_think, quality, auto_max, ctx_max,
                    )
                elif auto_max > 0 and last_finish == "tool_cap":
                    last_prompt, last_raw, last_finish = await turn_with_followups(
                        backend, history, tools, session,
                        last_max, quality, last_think, ctx_max, auto_max,
                    )
            else:
                print("usage: /auto <max>   (0 disables)\n")
            continue
        if user == "/continue":
            if last_finish == "length":
                last_raw, last_finish = extend_reply(
                    backend, history, last_prompt, last_raw,
                    last_max, last_think, quality, ctx_max,
                )
            elif last_finish == "tool_cap":
                # Resume the tool loop: history already holds every tool result.
                last_prompt, last_raw, last_finish = await turn_with_followups(
                    backend, history, tools, session,
                    last_max, quality, last_think, ctx_max, auto_max,
                )
            else:
                print("(nothing to continue — last reply finished naturally)\n")
            continue

        # --- try the deterministic engine first; the LLM is the fallback ---
        # (recognize → 0-call compound split → ≤2-call reformulation+verify). Returns None
        # for anything non-computational, which then proceeds to a normal generation.
        try:
            coded = try_coded(user, shot=coded_shot)
        except Exception:
            coded = None  # never let the code path break the chat
        if coded is not None:
            print(f"difg> {coded.answer}")
            print(f"(coded answer — {model_calls(coded)} model call(s), no full generation)\n")
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": coded.answer})
            last_finish = None
            continue

        # --- a normal user turn (may involve tool calls) ---
        turn_start = len(history)
        history.append({"role": "user", "content": user})
        # Thinking opens a verbose <thought> channel that shares the fixed canvas
        # with the answer, so give thinking turns a much larger budget.
        effective_max = THINK_MAX_TOKENS if think else max_tokens
        last_max, last_think = effective_max, think
        try:
            last_prompt, last_raw, last_finish = await turn_with_followups(
                backend, history, tools, session,
                effective_max, quality, think, ctx_max, auto_max,
            )
        except Exception as e:
            # Never let one bad turn kill the REPL; roll back the whole turn.
            print(f"\n[turn error: {type(e).__name__}: {e}]\n")
            del history[turn_start:]
            last_finish = None


async def run(model_id, use_tools):
    backend = get_backend(model_id)
    backend.load(model_id)
    ctx_max = backend.context_window

    async with AsyncExitStack() as stack:
        session, tools = None, []
        if use_tools:
            try:
                server = StdioServerParameters(
                    command=sys.executable, args=["mcp_server_demo.py"]
                )
                read, write = await stack.enter_async_context(stdio_client(server))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                mcp_tools = (await session.list_tools()).tools
                tools = [mcp_tool_to_model(t) for t in mcp_tools]
                print(f"MCP connected — tools: {[t.name for t in mcp_tools]}")
            except Exception as e:
                print(f"(MCP unavailable: {type(e).__name__}: {e} — continuing without tools)")
                session, tools = None, []

        tool_note = "with tools" if tools else "no tools"
        print(f"Ready ({backend.name} backend, context window: {ctx_max:,} tokens, {tool_note}). "
              "Commands: /quality, /think, /continue, /auto <max>, /tools, /reset, /exit.\n")
        await run_repl(backend, ctx_max, session, tools)


def main():
    args = [a for a in sys.argv[1:] if a != "--no-tools"]
    use_tools = "--no-tools" not in sys.argv[1:]
    model_id = args[0] if args else DEFAULT_MODEL
    asyncio.run(run(model_id, use_tools))


if __name__ == "__main__":
    main()

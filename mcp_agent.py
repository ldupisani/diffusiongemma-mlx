#!/usr/bin/env python3
"""Minimal MCP agent loop over a pluggable MLX backend (see backend.py).

Wires three things together:
  1. An MCP server (mcp_server_demo.py), launched as a subprocess over stdio.
  2. An MLX model (diffusion or autoregressive, per backend.py) that emits tool
     calls in its native format.
  3. The glue: translate MCP tool schemas -> the model's `tools=` format, parse
     the model's tool call, execute it over MCP, feed the result back, repeat.

The model never "speaks MCP" — this file is the host/agent that owns the MCP
connection and the call/execute/continue loop. Swap the demo server for any MCP
server and nothing here changes.

Run:  python mcp_agent.py "What's the weather in Paris, and what is 21 + 21?"
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend import DEFAULT_MODEL, get_backend
from tools_util import mcp_tool_to_model

MAX_STEPS = 6  # guard against a model that loops on tool calls forever


async def run(user_prompt: str):
    backend = get_backend(DEFAULT_MODEL)
    backend.load(DEFAULT_MODEL)

    server = StdioServerParameters(command=sys.executable, args=["mcp_server_demo.py"])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [mcp_tool_to_model(t) for t in mcp_tools]
            print(f"Connected to MCP server. Tools: {[t.name for t in mcp_tools]}\n")

            messages = [{"role": "user", "content": user_prompt}]
            print(f"user> {user_prompt}\n")

            for step in range(1, MAX_STEPS + 1):
                formatted = backend.apply_template(
                    messages, tools=tools, enable_thinking=False,
                )
                # Generate on the main thread (NOT via asyncio.to_thread): MLX's GPU
                # stream is thread-local. This blocks the event loop during generation,
                # which is fine — model inference and MCP I/O never overlap in this loop.
                text, _ = backend.generate(
                    formatted, max_tokens=256, params={"temperature": 0.0},
                )
                calls = backend.parse_tool_calls(text)

                if not calls:
                    print("difg>", backend.strip_thought(text))
                    return

                # Execute every call this turn (the model may issue several at
                # once), collecting one tool_call entry and one tool reply each.
                tool_calls, tool_replies = [], []
                for j, (name, args) in enumerate(calls):
                    cid = f"c{step}_{j}"
                    print(f"[step {step}] tool call -> {name}({args})")
                    tool_result = await session.call_tool(name, args)
                    # MCP returns a list of content blocks; take the first's text.
                    payload = tool_result.content[0]
                    value = getattr(payload, "text", str(payload))
                    print(f"[step {step}] tool result <- {value}\n")
                    tool_calls.append({
                        "id": cid, "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                    tool_replies.append({
                        "role": "tool", "tool_call_id": cid,
                        "name": name, "content": value,
                    })

                # Record the exchange in OpenAI-style messages; the chat template
                # renders these back into Gemma's native tool-call/response blocks.
                messages.append({"role": "assistant", "tool_calls": tool_calls})
                messages.extend(tool_replies)

            print("(hit MAX_STEPS without a final answer)")


def main():
    prompt = " ".join(sys.argv[1:]) or "What's the weather in Paris, and what is 21 + 21?"
    asyncio.run(run(prompt))


if __name__ == "__main__":
    main()

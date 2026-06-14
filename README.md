# DiffusionGemma on Apple Silicon (MLX)

Running `google/diffusiongemma-26B-A4B-it` — a 26B Mixture-of-Experts **diffusion**
language model (~3.8B active params) — locally via `mlx-vlm` on an M3 Max (36 GB).

## What's different from a normal LLM
DiffusionGemma is **not autoregressive**. It fills a fixed-size *canvas* of masked
tokens and refines the whole block in parallel over N denoising steps. Consequences:
- Cost scales with **denoising steps**, not just tokens generated.
- `block_length` (canvas size) must be **≥ max_tokens** — it can't emit more tokens than it has slots.
- Reference config (Google's guide): **48 steps**, **256-token canvas**, `entropy-bound` sampler.

## Setup
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -U "mlx-vlm>=0.6.3"
```

## Run — CLI
```bash
source .venv/bin/activate
python -m mlx_vlm.generate \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --prompt "Why is the sky blue? Answer in two sentences." \
  --max-tokens 128 \
  --max-denoising-steps 48 \
  --block-length 256 \
  --diffusion-sampler entropy-bound \
  --temperature 0.4
```

## Authenticate (optional but recommended — faster, rate-limit-free downloads)
```bash
source .venv/bin/activate
hf auth login        # paste a READ token from https://huggingface.co/settings/tokens
# or, for CI/one-off:  export HF_TOKEN=hf_xxx
```
Saved to `~/.cache/huggingface/token`; picked up automatically by mlx-vlm and hf.

## Run — reusable script
```bash
python chat.py "Give me three tips for naming variables." --quality fast --max-tokens 120
```

## Run — interactive streaming chat
```bash
python chat_stream.py                 # uses 4-bit default; optional arg: model id
```
Multi-turn REPL with live streaming output. **Tool calling is built in**: it
launches `mcp_server_demo.py` over MCP and lets the model call tools mid-chat
(pass `--no-tools` to disable). In-session commands:
`/quality fast|balanced|quality` · `/think` (toggle reasoning channel) ·
`/continue` (extend a reply that hit the token cap) · `/auto <max>` ·
`/tools` (list MCP tools) · `/reset` · `/exit`.

The REPL is async: each user turn runs a tool loop (generate → parse tool calls →
execute over MCP → feed results back → repeat) before returning a final answer.
`generate()` stays on the main thread (MLX GPU stream is thread-local); only `input()`
and MCP I/O are awaited. Tool-call/response messages live in history in OpenAI format,
so context (and `/continue`, `/auto`) work across them.

**One budget, `/auto N`, governs both "keep going" limits:** how many times a truncated
reply is auto-extended AND how many tool-call rounds a turn may run (default
`DEFAULT_TOOL_STEPS=6` when `/auto` is off). If a long tool sequence exhausts the budget
mid-task, the turn pauses in a resumable `tool_cap` state — `/continue` (or raising
`/auto`) picks up exactly where it left off, since history already holds every tool
result. So a task needing ~30 sequential tool calls completes in one turn with `/auto 100`.

- `/think`: model streams its `<thought>` reasoning before the answer; the thought
  block is stripped from stored history (kept ephemeral per turn). Thinking turns
  auto-use a larger budget (`THINK_MAX_TOKENS=1024`) so reasoning + answer both fit.
- `/continue`: when a reply ends with `finish_reason="length"` (hit the cap, not EOS),
  the REPL prints a hint. `/continue` resumes by **raw concatenation** — the original
  templated prompt + the partial reply, fed straight to the engine — so generation
  picks up mid-sentence with no re-greeting or repetition. (We deliberately avoid
  re-running the chat template here: it strips the ephemeral `<thought>` channel on
  re-render, which would corrupt a continuation when the cut lands mid-thought.)
  Repeat as needed for very long output.
- `/auto <max>`: automate the above. After each reply, auto-`/continue` until the model
  hits EOS or `<max>` extensions are used — whichever comes first. It uses the *minimum*
  needed (stops the moment EOS is reached), with `<max>` as a runaway safety cap.
  `/auto 0` disables. Setting it while a reply is already truncated applies it immediately.

After every reply the REPL prints a readout, e.g.
`(context: 113 / 262,144 tokens — 0.0%, peak mem 19.5 GB)`.
The window (262K) is read from the model's `max_position_embeddings`; peak mem is the
unified-memory high-water mark for the turn. Note: on a 36 GB Mac you'll hit memory
pressure (watch the GB figure) long before the token ceiling — the percentage is a
usage gauge, the GB is the real limit. Of the 30 layers, 25 use sliding-window attention (1024-token
window) and only 5 use full attention, so KV-cache growth with context is dominated by
those 5 global layers.
Note: the diffusion **text** path is text-only when streaming (no images), and
output arrives in canvas-block chunks rather than smooth token-by-token.
`--quality` presets (see `choose_diffusion_params` in `chat.py`):
| preset    | steps | temp | use for |
|-----------|-------|------|---------|
| fast      | 12    | 0.2  | structured / constrained tasks |
| balanced  | 32    | 0.4  | everyday default |
| quality   | 48    | 0.6  | open-ended prose |

## Measured on M3 Max (36 GB), 4-bit quant
- Peak memory: **~18.9 GB**
- `quality` (48 steps): ~10.8 tok/s
- `fast` (12 steps): ~30 tok/s  (≈3× faster — the speed lever is *steps*, not model size)

## MCP tool calling (`mcp_agent.py`)
DiffusionGemma supports tool calling natively (Gemma's `<|tool_call>call:NAME{...}<tool_call|>`
DSL — not JSON). The model never "speaks MCP"; `mcp_agent.py` is the host that owns the
MCP connection and the call→execute→continue loop.

```bash
uv pip install mcp                      # one-time
python mcp_agent.py "What's the weather in Paris, and what is 21 + 21?"
```
It launches `mcp_server_demo.py` (a real stdio MCP server exposing `get_weather`, `add`,
and `random_int`), translates the server's tool schemas into the model's `tools=` format,
parses the model's tool calls, executes them over MCP, feeds results back, and loops until
a final answer.

Notes:
- The model can emit **multiple tool calls in one turn** (parallel); the parser handles all.
- The parser (`tools_util.parse_tool_calls`) does NOT rely on the `<tool_call|>` terminator
  (models sometimes drop it) — it scans balanced braces from each `<|tool_call>call:` opener,
  so a malformed/unterminated call can't swallow the next one or corrupt its args. Truncated
  mid-argument calls parse to nothing rather than garbage.
- Match the tool to the task: a `roll_die` (1-6) tool for a "random 1-10" request sends the
  model into a spiral trying to bridge the gap. `random_int(low, high)` fits directly.
- `generate()` must run on the main thread (MLX's GPU stream is thread-local) — do NOT wrap
  it in `asyncio.to_thread`.
- Swap `mcp_server_demo.py` for any MCP server (filesystem, HTTP, DB) — the agent is unchanged.

## TypeScript REPL (`ts/`)
A TypeScript port of `chat_stream.py` with full feature parity (streaming, `/quality`,
`/think`, `/continue`, `/auto`, `/tools`, MCP tool calling, context readout).

**Why it needs a Python sidecar:** DiffusionGemma only runs through `mlx-vlm` (Python) —
there is no MLX runtime for JS, and the diffusion sampler isn't supported by JS-native
engines (transformers.js / llama.cpp are autoregressive). So inference is delegated to a
tiny local HTTP server (`infer_server.py`) that wraps the exact same `stream_generate` call;
the TypeScript REPL owns everything else and talks to it over NDJSON streaming.

**One command (recommended)** — `run.sh` boots the sidecar, waits for it, launches the REPL,
and stops the sidecar on exit (an already-running sidecar is reused and left alone):
```bash
./run.sh                # tools on; pass-through flags go to the REPL, e.g. ./run.sh --no-tools
# or from the ts/ dir:  npm run dev
```
Env overrides: `DIFG_PORT` (8088), `DIFG_MODEL` (4-bit), `DIFG_PYTHON` (`.venv/bin/python`).

**Or run the two pieces by hand:**
```bash
# 1. start the MLX inference sidecar (Python, loads the model)
source .venv/bin/activate && python infer_server.py        # listens on 127.0.0.1:8088

# 2. in another terminal, run the TS REPL
cd ts && npm install && npm start                          # --no-tools to disable MCP
```

The TS REPL spawns `mcp_server_demo.py` for tools (via the MCP TypeScript SDK), so it uses
the project venv's Python — override with `DIFG_PYTHON=/path/to/python` if needed. Point at a
different sidecar with `npm start -- --server http://host:port`. Layout: `ts/src/tools.ts`
(DSL parser + presets, port of `tools_util.py`), `ts/src/infer.ts` (sidecar client),
`ts/src/chat_stream.ts` (REPL + agent loop).

## Model variants (mlx-community)
`-4bit` (default, ~13 GB) · `-5bit` · `-6bit` · `-8bit` (~26 GB) · `-bf16`.
On 36 GB, 4-bit is the safe default; 6-bit fits with headroom; 8-bit is tight.

## Notes
- Set `HF_TOKEN` for faster/rate-limited downloads (unauthenticated works, just slower).
- The model is multimodal (image-text-to-text): pass `--image <path|url>` to either entry point.
- It emits a `<|channel>thought ... <channel|>` reasoning block; add `--enable-thinking` to steer it.

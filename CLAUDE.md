# difg — local reasoning toolkit on Apple Silicon (MLX)

A deterministic reasoning engine with a small LLM as fallback. Most queries are answered
by coded orchestrators at **0 model calls**; the model only runs when no orchestrator claims
a query. Originally an experiment around DiffusionGemma — the model is now a pluggable backend.

## Architecture

- **`reasoning/`** — the engine. Pipeline: `router` → `decompose` → `bind` → `derive` →
  `synthesize`. Orchestrators are individual Python modules under
  `reasoning/store/orchestrators/`, each exposing `META`, optional `precondition(query)`,
  and `run(query)`. The registry (`registry.py`) discovers and indexes them; the router
  matches an incoming query to a `query_type`. Composition between orchestrators is via
  `SubStep(..., orchestrator="<id>")` in a `Plan` (see `compiled.py`), NOT via metadata.
- **`backend.py`** — pluggable inference backend (see below).
- **`infer_server.py`** — tiny HTTP sidecar (`/info`, `/generate` NDJSON) wrapping the active
  backend. Single-threaded on purpose (MLX's GPU stream is thread-local).
- **`chat_stream.py`** (Python REPL), **`mcp_agent.py`** (one-shot agent), **`chat.py`**
  (explicit diffusion demo CLI) — all generation flows through a backend handle.
- **`tools_util.py`** — tool-call parsers (Gemma-DSL + JSON) and thought-channel strip.
- **`ts/`** — TypeScript REPL. Talks to the sidecar over HTTP only; adapts behavior off
  `/info` (tool-call format, thinking support). Mirrors `tools_util.py` in `ts/src/tools.ts`.

## Backends (`backend.py`)

Two backends share ONE interface (`load` / `apply_template` / `stream` / `parse_tool_calls`
/ `strip_thought`), so every consumer is written once and works with either. Both yield
response objects with identical fields (`.text`, `.finish_reason`, `.prompt_tokens`,
`.generation_tokens`, `.peak_memory`).

- **`AutoregressiveBackend`** (`mlx-lm`, **default**) — small standard instruct models.
  Default `mlx-community/Qwen2.5-3B-Instruct-4bit` (~1.86 GB). JSON tool calls
  (`<tool_call>{...}</tool_call>`), no thought channel.
- **`DiffusionBackend`** (`mlx-vlm`) — the original `diffusiongemma-26B-A4B-it-4bit`
  (~18.5 GB). Denoising-canvas sampling, Gemma-DSL tool calls, a `<thought>` channel.

Selection (most specific wins): explicit `name=` arg → `$DIFG_BACKEND`
(`diffusion`|`autoregressive`) → auto-detect from the model id (`diffusiongemma` ⇒ diffusion,
else autoregressive). `mlx-vlm`/`mlx-lm` imports are deferred per backend.

**Notes / gotchas:**
- The sidecar `/info` reports the actually-loaded model + `{backend, supports_thinking,
  tool_call_format}`; clients use it to pick a parser and gate `/think`.
- Qwen2.5's bundled chat template shows the tool-call example with doubled braces
  (`{{...}}`, a Jinja-escape artifact) and the model copies them. The JSON parser
  (`parse_tool_calls_json` / TS `loadsFirstObject`) peels redundant outer brace layers.
- Diffusion-only knobs (`max_denoising_steps`, `block_length`, `diffusion_sampler`) are
  ignored by the autoregressive backend, which uses only `temperature`.

## Commands

```bash
./run.sh                                  # boot sidecar + TS REPL (default Qwen2.5-3B)
DIFG_MODEL=mlx-community/diffusiongemma-26B-A4B-it-4bit ./run.sh   # diffusion instead
python chat_stream.py                     # Python REPL (optional arg: model id)
python -m pytest reasoning/tests -q       # 76 tests (use system python; .venv lacks pytest)
python -m reasoning.cli eval-all          # fidelity of every coded orchestrator vs teacher
```

Env: `DIFG_PORT` (8088), `DIFG_MODEL`, `DIFG_BACKEND`, `DIFG_PYTHON` (`.venv/bin/python`).

## Conventions

- Adding an orchestrator: new module in `reasoning/store/orchestrators/`, add a collision
  guard / `precondition`, and keep it model-free where possible. Run `eval-all` after changes.
- Keep the sidecar HTTP contract (`/info` + `/generate` NDJSON) stable — it's the seam that
  insulates the TS REPL from the Python internals.

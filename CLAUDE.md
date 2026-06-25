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
- **`reasoning/datasets.py`** — dataset subsystem for larger-than-text data (separate from the
  pure no-I/O `skills/`). DuckDB is the single engine over CSV/Parquet/JSON/SQLite/DuckDB, plus
  Arrow/Feather and NumPy `.npy` (read via pyarrow/numpy, registered into DuckDB):
  datasets are handles (files in `$DIFG_DATA_DIR`, default `<repo>/data`) registered as views, so
  ops are SQL. Results stay small (scalar / ≤20-row table / new handle + preview). Reached via the
  `dataset` orchestrator (NL) and `python -m reasoning.cli data …` (import/list/describe/query/
  generate/convert). 0 model calls.
- **`reasoning/reference.py`** + **`reasoning/store/reference/seed.py`** — bundled, constant,
  version-controlled reference data (geography: countries/population-by-year/cities). The seed
  (diffable Python) builds lazily into a SQLite DB (gitignored artifact; `$DIFG_REFERENCE_DIR`
  overrides location), auto-registered read-only into the dataset engine so reference tables are
  queryable handles. Lookups + `import_csv` (replace/extend with an authoritative source);
  reached via the `geography` orchestrator (NL) and `reasoning.cli reference`. A MISS (unknown
  place / missing year) can fall back to a validated, provenance-flagged Claude Sonnet retrieval
  — opt-in via `DIFG_REFERENCE_RETRIEVAL=1` (off by default; records `model_calls=1`). Retrieved
  values are persisted to a `retrieved` log table (repeat query → 0 calls, "previously
  retrieved"); batch requests ("all cities in France", "populations of A, B and C") go through
  `retrieve_set` → a CSV from the model saved as a model-sourced dataset handle.
- **`reasoning/datagen.py`** — domain dataset generators (primes/Fibonacci, random samples,
  logistic trajectories, projectile motion, amortization, …) that produce rows (reusing pure
  skills + numpy/SymPy) and save via `datasets.save_rows`. Registry `GENERATORS`; reached via
  `data gen <name> <handle> key=value …` and NL ("generate the first 100 primes").
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
/ `strip_thought` / `gen_params` / `params_from_request`), so every consumer is written once
and works with either. Both yield response objects with identical fields (`.text`,
`.finish_reason`, `.prompt_tokens`, `.generation_tokens`, `.peak_memory`).

Each backend owns its own generation params — callers never branch on backend type:
`gen_params(quality, max_tokens)` is the REPL/CLI preset path; `params_from_request(req)` is
the HTTP `/generate` path. The base (autoregressive) returns just a `temperature`;
`DiffusionBackend` overrides both to add its canvas knobs.

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
- Diffusion-only knobs (`max_denoising_steps`, `block_length`, `diffusion_sampler`) only
  exist on `DiffusionBackend`; the autoregressive backend's `params_from_request`/`gen_params`
  pull only `temperature`, so those keys in a request are simply never read.

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

- Adding a domain: follow `docs/adding-domains.md` — skill module + bind + orchestrator, gate
  conservatively, and test routing in BOTH directions (a new domain must not steal or be stolen
  from). New module in `reasoning/store/orchestrators/`; keep it model-free. Run `eval-all` after.
- After finishing a set of domains, update `docs/adding-domains.md` if the work taught something
  new: a fresh collision class or decline-guard, a parsing gotcha, a routing/tier subtlety, or a
  cleaner step. Add it as a concrete "Learning" (the real bug + how it was caught) or fold it into
  the checklist — keep the playbook ahead of the codebase. If nothing new came up, leave it.
- Adding a backend: subclass `Backend`, set the capability flags, and override `gen_params`
  / `params_from_request` if it needs knobs beyond `temperature`. No call site should branch
  on backend type — keep that logic inside the backend.
- Keep the sidecar HTTP contract (`/info` + `/generate` NDJSON) stable — it's the seam that
  insulates the TS REPL from the Python internals.

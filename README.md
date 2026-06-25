# DiffusionGemma on Apple Silicon (MLX)

![reasoning fidelity vs Claude](https://img.shields.io/badge/fidelity_vs_Claude-63%2F63-brightgreen)
![model-free](https://img.shields.io/badge/model--free-63%2F63-blue)
![coded skills](https://img.shields.io/badge/skills-246-informational)

A local reasoning toolkit on Apple Silicon (MLX). It started as an experiment running
`google/diffusiongemma-26B-A4B-it` — a 26B Mixture-of-Experts **diffusion** language model
(~3.8B active params) — via `mlx-vlm`. Since the reasoning engine answers most queries
deterministically (0 model calls), the LLM is only a fallback, so the model is now a
**pluggable backend** and the default is a small autoregressive instruct model.

**Backends** (see `backend.py`) — picked by `$DIFG_BACKEND`, else auto-detected from the model id:
- **autoregressive** (`mlx-lm`, *default*) — small standard instruct models. Default
  `mlx-community/Qwen2.5-3B-Instruct-4bit` (~1.8 GB): left-to-right streaming, JSON tool calls.
- **diffusion** (`mlx-vlm`) — the original `diffusiongemma-26B-A4B-it-4bit` (~14 GB):
  denoising-canvas sampling, Gemma-DSL tool calls, a `<thought>` reasoning channel.

> **Reasoning compiler:** 63/63 coded orchestrators reproduce the Claude teacher's answer
> ([`reasoning.cli eval-all`](#reasoning-compiler-reasoning)), each at **0 model calls** (sidecar
> cold). Static badges — regenerate with `eval-all` after changes.

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
uv pip install -r requirements.txt   # mlx-lm (default backend) + mlx-vlm (diffusion) + mcp + sympy
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
python chat_stream.py                 # default small AR model; optional arg: model id
# diffusion instead:  python chat_stream.py mlx-community/diffusiongemma-26B-A4B-it-4bit
```
Multi-turn REPL with live streaming output. **Code-first**: each turn first tries the
deterministic engine (`reasoning.solve.try_coded`: recognize → 0-call compound split →
≤1-call reformulation); only non-computational turns fall through to a full LLM generation, so
e.g. "how much surface does a circular table of radius 5 m have?" answers `78.54` at 1 model
call. **Tool calling is built in**: it launches `mcp_server_demo.py` over MCP and lets the model
call tools mid-chat (pass `--no-tools` to disable). In-session commands:
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
Both backends support tool calling, each in its native wire format — JSON
(`<tool_call>{"name":…,"arguments":…}</tool_call>`) for autoregressive models, Gemma's
`<|tool_call>call:NAME{…}<tool_call|>` DSL for diffusion. The active backend picks the
matching parser (`tools_util.py`). The model never "speaks MCP"; `mcp_agent.py` is the host
that owns the MCP connection and the call→execute→continue loop.

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
`/think`, `/continue`, `/auto`, `/tools`, MCP tool calling, context readout, code-first turns).

**Why it needs a Python sidecar:** MLX only runs through Python — there is no MLX runtime
for JS. So inference is delegated to a tiny local HTTP server (`infer_server.py`) that wraps
whichever backend is active; the TypeScript REPL owns everything else and talks to it over
NDJSON streaming. The REPL adapts to the sidecar's `/info` (backend, `tool_call_format`,
`supports_thinking`) — so it parses JSON or Gemma-DSL tool calls and gates `/think` automatically.
**Code-first** like the Python REPL: each turn first calls the sidecar's `POST /solve` (the
deterministic engine, run with an in-process shot to avoid re-entrant generation) and only
generates a full reply when nothing coded answers.

**One command (recommended)** — `run.sh` boots the sidecar, waits for it, launches the REPL,
and stops the sidecar on exit (an already-running sidecar is reused and left alone):
```bash
./run.sh                # tools on; pass-through flags go to the REPL, e.g. ./run.sh --no-tools
# or from the ts/ dir:  npm run dev
```
Env overrides: `DIFG_PORT` (8088), `DIFG_MODEL` (default Qwen2.5-3B-Instruct-4bit),
`DIFG_BACKEND` (`diffusion`|`autoregressive`; auto-detected if unset), `DIFG_PYTHON` (`.venv/bin/python`).

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

## Reasoning compiler (`reasoning/`)
Replace the model's open-ended **thinking mode** with **coded reasoning patterns**: a request is
decomposed into sub-tasks the model handles in **one shot with thinking OFF** ("patterns of
thinking, run by code"). Longer term it's a *compiler* — use the reasoning model as a teacher
once per query type, distill its trace into a coded orchestrator, store it, and route future
matching queries to the orchestrator instead of thinking. Mental model: trace-based JIT for
cognition (teacher = slow path, orchestrator = compiled fast path, registry = cache, router =
cache-key lookup, miss = compile trigger).

The model-driven paths require the sidecar running (`python infer_server.py`); the coded paths
below do not (see "Replacing model calls with code"). Then:
```bash
python -m reasoning.cli run "What is 17 percent of 240?"        # → numeric_consensus (vote)
python -m reasoning.cli run "Compare a Civic and a Corolla..."  # → decompose-solve-combine
python -m reasoning.cli run "Write me a poem"                   # no match → teacher fallback
python -m reasoning.cli capture "<query>"   # record a teacher (thinking-on) trace
python -m reasoning.cli derive <sample-id>  # scaffold an orchestrator from it (then edit)
python -m reasoning.cli eval <orch-id>      # score one orchestrator's fidelity vs the teacher
python -m reasoning.cli eval-all            # fidelity of EVERY coded orchestrator vs the teacher
python -m reasoning.cli list                # stored orchestrators
```
`eval-all` runs each coded orchestrator's example through both it and the teacher (default
`--teacher claude`) and checks agreement with a deterministic comparator (numeric tolerance +
normalized-string) — **no sidecar needed**: coded orchestrators make 0 model calls and the
comparator uses no model. It reports per-orchestrator agreement and confirms every coded answer
used 0 model calls — the capability-gap gate across the whole graph (currently 63/63 agree, 63/63
model-free vs Claude). The comparator is tolerant (numeric/normalized-string/yes-no verdict); the
teacher's final answer is taken as its last line when it skips the `ANSWER:` marker.

**Multi-step reasoning, minimal LLM.** Beyond single-orchestrator leaves, the graph now composes:
- **Compound arithmetic** — `expression` evaluates full order-of-operations ("15% of 200 plus 10",
  "(5+3)*2-4") via a recursive-descent parser (`skills/expr.py`). 0 model calls.
- **Sub-orchestrator composition** — a `SubStep` lets one plan delegate a sub-question to another
  orchestrator (via `SubStep(..., orchestrator=...)`); e.g. `converted_cost` delegates a unit
  conversion then multiplies. Model-free iff its parts are (the sub-run's call tally folds in).
- **Ask for missing info** — a recognized type missing a field returns a `clarify` result
  (`{field: (question, template)}`); `cli run` prompts and re-runs by appending the answer in a
  form the existing bind already parses. A new router tier, below answering, above fallback.
- **Cross-domain decomposition** — `decompose.py` splits a compound query and routes each clause to
  a coded orchestrator: 0 calls (code-first split) or ≤1 (a single model call only SPLITS; coded
  orchestrators solve the parts — vs the old N+2). The router is a five-tier cascade:
  compose → domain → bind-gated → clarify → fallback.
- **Reformulate-then-route** (`reformulate.py`) — the single-question sibling of decomposition:
  when a question doesn't route as-written, one model call *rewrites* it into a phrasing the coded
  engine recognizes (e.g. "how much room is inside a circle of radius 5?" → "area of a circle with
  radius 5"), a coded orchestrator answers it (0 further calls), and a **deterministic** guard
  confirms the rewrite invented no number absent from the original (with a word→digit map so "five"
  matches "5") — chosen over a model self-verify, which a small model does unreliably. A cheap
  `_looks_computational` pre-gate (digit / number-word / domain keyword) skips the call entirely on
  conversational turns. `cli run`'s fallback ladder is: 0-call compound split → reformulate+verify →
  model-assisted compound split → teacher.
- **Goal-directed plan synthesis** (`synthesize.py`) — BFS over a typed-op library finds the
  shortest operation path from typed knowns to a goal. Generalized across domains in one search
  pool (`ALL_OPS`): re-derives finance discount→tax, a 3-op physics chain (force→work→power), the
  distance–speed–time triangle (goal-directed op selection), and simple interest — all model-free.
  Roles live on ops (not skills), so disjoint role vocabularies keep domains from interfering
  (a finance goal can't pull a physics op); ambiguous goals (two shortest paths) are detectable.
  Memoized, and LIVE across many domains, each via a bind that maps NL → {typed knowns, goal}:
  synthesized_physics (force→work→power, geometry area→pressure, motion→kinetic energy),
  synthesized_finance, synthesized_geometry (area/perimeter/volume→cost), synthesized_chemistry
  (formula→molar mass→moles/mass/molarity), synthesized_materials (force + a shape's area→stress→
  Young's modulus), synthesized_geo (haversine→travel cost / flight time), and priced_cost
  (labour, electricity). **Cross-domain composition**: a cost produced by ANY domain funnels into
  a generic `cost` role and inherits the finance terminals (split / tax / discount) once — so
  geometry→finance ("tile a floor … split 3 ways"), chemistry→finance→split (a 5-op chain), and
  physics→finance (kWh→cost) compose automatically. **Unit-aware roles** carry a canonical unit so
  "5 km" / "20 minutes" feed correctly, and the planner inserts a coercion op (g→kg) between
  same-dimension roles in different domains. **Inverse synthesis** solves backward for an input
  given the output (sympy.solve over a symbolic mirror of the chain) — "what discount yields a
  $200 final price?". Cross-domain seams pass unique=True, so an ambiguous derivation declines
  rather than guesses. All 0 model calls.

Modules: `engine.py` (one-shot / one-shot-N thinking-off calls via the sidecar; N varies the
**seed** for diverse candidates), `patterns.py` (decompose-solve-combine, map-reduce,
generate-rank-select, self-consistency, extract-validate-repair), `registry.py` (orchestrators as
git-versioned Python modules under `store/` — the graph), `router.py` (pluggable recognition:
`LexicalMatcher` default, `EmbeddingMatcher` for scale — cosine over example-query vectors,
`ClassifyMatcher` one-shot; recognition is two-tier — a specific orchestrator always beats a
generic `"fallback": True` one like `numeric_consensus`), and
`capture`/`derive`/`eval` (the teacher→orchestrator compile loop). Selecting among a handful of
patterns needs no embeddings — a keyword/lexical or one-shot classify router suffices. As the
library grows, keyword routing starts colliding on shared surface words ("product", "mean",
"year"), and a collision is a *confident wrong answer*, not a miss — so `EmbeddingMatcher`
(`--matcher embedding`) routes by cosine similarity of the query to each orchestrator's
example_queries instead. Its `embed` backend is pluggable: the default `hashing_embed` is
dependency-free, local, and never touches the sidecar (so coded paths stay model-free); pass a
neural embedder (sentence-transformers / MLX) for true semantic recognition at scale.

**Teacher backend (`teachers.py`).** The teacher (the slow reasoning we distill from) is pluggable:
`--teacher diffusion` (the local model with thinking ON — self-distillation) or `--teacher claude`
(the `claude` CLI, a stronger teacher). With Claude, fidelity becomes a **capability-gap** metric:
can a coded pipeline of weak one-shot calls reproduce a frontier model's answer? `CLAUDE_TEACHER_SYSTEM`
asks Claude to decompose into *atomic single-operation steps a small model could each do in one shot*,
so the trace is compilable onto the student.

### Replacing model calls with code (`skills/`, `compiled.py`, `bind.py`)
One level deeper: replace the *one-shot calls themselves* with deterministic Python where a step is
mechanical (arithmetic, dates, string/collection ops). A fully-coded orchestrator issues **zero model
calls — so the ~19 GB sidecar never loads** and the answer is computed in megabytes. The model is kept
only for fuzzy steps (open generation, judgement, recall) and ambiguous NL parsing.

- **`skills/`** — a growing library of small, pure, **typed, example-tested** functions (`arith.*`,
  `date.*`, `text.*`, `stats.*`, `units.*`, `finance.*`, `geometry.*`, `physics.*`, `numtheory.*`,
  `chem.*`, `radix.*`, `trig.*`, `coord.*`, `clock.*`, `sets.*`, `prob.*`, `ratio.*`, `health.*`,
  `logic.*`, `expr.*`; university math — `setops.*`, `group.*`, `topology.*` (pure stdlib), plus
  `algebra.*`/`calculus.*` (SymPy-backed); and applied/life sciences — `geo.*` (geospatial:
  haversine/bearing/DMS), `genetics.*` (Hardy–Weinberg, Punnett), `materials.*` (stress/modulus/
  packing factor), `discrete.*` (graphs + Catalan/Fibonacci/Stirling/Bell), `linalg.*` (SymPy
  matrices & vectors), `pchem.*` (ideal gas, % composition), `molec.*` (Avogadro, unsaturation),
  `biochem.*` (GC content, transcription, Henderson–Hasselbalch, Michaelis–Menten); and
  number systems beyond the reals — `padic.*` (p-adic valuation/abs/expansion/distance),
  `hyper.*` (quaternions), `ord.*` (ordinals, Cantor normal form), `card.*` (cardinals ℵ₀/𝔠),
  `hreal.*` (hyperreals: ε/ω, standard part), `surr.*` (surreals: {L|R}, birthday); and advanced
  computable domains — `crypto.*` (RSA/Diffie–Hellman/Caesar/totient), `numerical.*` (Newton/
  bisection/Simpson/Euler), `game.*` (2×2 Nash/zero-sum/payoff), `quantum.*` (qubit gates/
  measurement/⟨X⟩,⟨Z⟩), `ml.*` (losses/activations/precision-recall-F1/entropy), `optimize.*`
  (min/max + gradient descent), `dynamics.*` (logistic map/Lyapunov/Mandelbrot), `diffgeo.*`
  (curvature/arc length); plus extensions — combinatorics & graph theory in `discrete.*`
  (derangements/multinomial/stars-and-bars/partitions; bipartite/Eulerian/MST), `prob.*`
  (binomial, Bayes), `stats.*` (correlation/regression/z-score), and a `funcanalysis.*` fragment
  (vector norms, inner product, Cauchy–Schwarz); and `logistics.*` (operations research — EOQ,
  reorder point, safety stock, inventory turnover, Little's Law, trucks needed, fuel cost) —
  all still 0 model calls — 368 and counting), indexed in a `SKILLS` registry. Skills never import the engine (a test enforces it). Rounding goes through one
  `arith.round_to`/`round_money` chokepoint (`Decimal`+`ROUND_HALF_UP`). Concept-level skills
  (`stats.mean` over any-length list, `finance.loan_payment`, `units.f_to_c`) capture an operation as
  one named step, so a single orchestrator generalizes — vs. the teacher's brittle fixed chain of
  primitive add/multiply steps. The strongest case: `units.convert` is one skill over a canonical
  base-unit factor table that spans length/mass/volume/time/speed/temperature — replacing the
  N×(N−1)-per-dimension explosion of pairwise converters; a new unit is one table row. The `finance.*` family (compound/simple interest, present value, CAGR,
  annuity FV, loan payment, margin, depreciation, break-even) was grown by capturing a *variety* of
  Claude traces, then choosing the concept-level abstractions from the evidence: ROI and markup turned
  out to be `arith.percent_change`, tip-and-split composes `apply_tax`+`divide` (no new skill), and the
  error-prone 7-step interest chains each collapse to one formula.
- **`compiled.py`** — a `Plan` is a typed step list; each step is a `CodeStep` (invoke a skill, no model)
  or a `ModelStep` (one `shot` call). `run_plan` records per step whether it ran via **code or model** and
  emits a summary tally — so "did this answer use the model?" is *observable per run*, not assumed. A
  model-free orchestrator runs with `shot=engine._no_model`, turning "0 model calls" into an invariant
  (an accidental `ModelStep` raises instead of silently waking the sidecar).
- **`bind.py`** — natural language → typed args. A **coded parser** (regex) runs first; only on failure
  does it fall back to the model (`extract_validate_repair`). The bind reports whether it used the model,
  and that cost is folded into the tally — a model-based bind followed by coded steps still reports its
  calls, never zero. The parser is deliberately conservative: it bails to the model rather than guess.

Try it (the coded orchestrators run with the **sidecar stopped**):
```bash
python -m reasoning.cli run "What is 17% off \$240 then 8% tax?"     # → $215.14, 0 model calls
python -m reasoning.cli run "How many days are between 2024-03-03 and 2024-07-15?"  # → 134, 0 calls
python -m reasoning.cli run "How many times does 'a' appear in 'banana'?"           # → 3, 0 calls
python -m reasoning.cli run "What is the average of 4, 8, 15, 16, and 42?"          # → 17, 0 calls
python -m reasoning.cli run "What is the percentage increase from 50 to 75?"        # → 50%, 0 calls
python -m reasoning.cli run "Convert 100 degrees Fahrenheit to Celsius."            # → 37.78°C, 0 calls
```

**Introspection / help.** The system can describe its own capabilities (0 model calls):
```bash
python -m reasoning.cli help                       # overview: every domain + one-line description
python -m reasoning.cli help geometry              # a domain's example queries + keywords
python -m reasoning.cli help --search prime        # domains/skills/generators mentioning a term
python -m reasoning.cli run "What can you do?"      # same, as an NL meta-query
python -m reasoning.cli run "What can I ask about geometry?"
```
The `introspect` orchestrator answers meta-questions in-engine, and pure `introspect.*` skills
(`domain_exists`, `skill_exists`, `generator_exists`, `can_answer`) let higher-level reasoning
probe what's available before acting — all reading the same `reasoning/introspect.py` API over the
orchestrator, skill, and generator registries.

**As an MCP service.** `reasoning/mcp_server.py` exposes the engine over the Model Context
Protocol (stdio) — **code-only, 0 model calls**, so it never needs the sidecar. Run it directly
or register the command `python -m reasoning.mcp_server` with any MCP client. Tools: `ask` (route
→ coded answer + domain + model-call count; compound queries split at 0 calls; unmatched returns
a hint), `capabilities` / `describe_domain` / `search_capabilities` (introspection), and
`list_generators` / `generate_dataset` / `list_datasets` / `dataset_info` / `query_dataset` (the
dataset + bundled-reference layer). A query that would need the LLM is *reported as such*, never
silently invoked. The server **hot-reloads** on each call: if any `reasoning/*.py` changed since
the last call it drops the cached modules and re-imports, so engine edits show up **without a
client reconnect** (only changes to `mcp_server.py` itself need a restart). It also exposes **prompts** — user-invokable slash commands in the client:
`/mcp__difg-reasoning__ask <query>`, `…__capabilities`, and `…__query <sql>` — a one-step entry
point that routes straight to the matching tool.

`derive` now grows the library: it maps each atomic line of a teacher trace onto a skill where one
matches (`--mode steps`), leaving unmatched lines as model steps and reporting coverage
(`matched 3/4 atomic steps to skills`). Each remaining model step is a candidate for the next skill —
add it to `skills/`, re-derive, and coded coverage climbs. `eval` reports fidelity *and* the model-call
count, with an exact numeric backstop that catches rounding regressions the fuzzy judge would miss.
Skill tests: `python -m unittest discover -s reasoning -p "test_*.py" -t .` (no sidecar, no network).

## Datasets — larger-than-text data (`reasoning/datasets.py`)
For inputs, scratch storage, or outputs too big to inline as text. A **separate I/O subsystem**
(the `skills/` library stays pure): **DuckDB is the single engine** over CSV/TSV, Parquet,
JSON/JSONL, SQLite, and DuckDB files — plus **Arrow/Feather** and **NumPy `.npy`** (read via
pyarrow/numpy and registered into DuckDB; a 2-D `.npy` becomes columns `c0…cN`). A dataset is a
**handle** — a file in `$DIFG_DATA_DIR`
(default `<repo>/data`) registered as a DuckDB view, so every operation is just SQL over
`SELECT … FROM <handle>`. **Results stay small**: a scalar, a compact table (≤20 rows), or — when
the output is large — a *new handle* plus a preview (schema, row count, head). Still **0 model
calls**.

```bash
python -m reasoning.cli data import sales.csv --name sales   # input (also Parquet/JSON/SQLite/…)
python -m reasoning.cli data describe sales                  # schema + row count + head
python -m reasoning.cli data query "SELECT status, count(*) FROM sales GROUP BY status"
python -m reasoning.cli data query "SELECT * FROM sales WHERE price > 100" --save-as big   # output handle
python -m reasoning.cli data generate 1000000 synth         # generate a large dataset (SQL)
python -m reasoning.cli data convert sales sales_pq parquet  # format conversion
python -m reasoning.cli data domains                        # what domains can create datasets
python -m reasoning.cli data gen primes myprimes n=1000     # domain generator → handle (explicit)
python -m reasoning.cli data gen "generate first 1000 prime numbers"   # …or natural language
python -m reasoning.cli data list                           # what datasets exist
python -m reasoning.cli data info primes                    # schema + how to query it
```
**Discovery flow.** `data domains` lists each domain and its dataset-creation tools; `data gen
"<phrase>"` runs one from natural language (*"generate first 1000 prime numbers"*); `data list` +
`data info <handle>` show what exists and print copy-paste SQL/NL queries for it.
**Domain generators** (`reasoning/datagen.py`) produce a domain-relevant dataset and save it as
a handle — **29 generators across 14 domains**: number theory (primes, Fibonacci, factorials,
partitions), statistics & probability (uniform/normal samples, coin/dice rolls, synthetic
regression/classification, binomial PMF), dynamical/numerical/physics (logistic trajectory,
bifurcation, `f(x)` over `[a,b]`, Euler ODE path, projectile motion), finance (amortization,
compound growth), logistics (EOQ curve, inventory simulation), logic (truth tables), trigonometry
(sin/cos/tan tables), group theory (Cayley tables), discrete math (catalan/bell/derangement
sequences), calculus (`f, f', f''` tables), health (BMI tables), chemistry (pH dilution series),
geometry (regular-polygon tables), and geospatial (great-circle waypoints). Each reuses the pure
domain skills + numpy/SymPy and returns a handle + preview; random ones take a `seed`.
NL also routes through the engine via the `dataset` orchestrator: *"What is the average of the
price column in sales?"* → `20`; *"How many rows in sales where status = 'active'?"* → `2`;
*"Describe the dataset sales."* It's gated on an explicit dataset reference, so a plain
*"average of 10, 20, 30"* still goes to the stats domain. Needs `duckdb` (in `requirements.txt`).

**Reference data** (`reasoning/reference.py`, seed in `store/reference/seed.py`) — bundled,
constant, version-controlled datasets. A curated geography seed (countries with capital/
continent/area/coordinates, population by year, and major cities) builds lazily into a SQLite DB
that's auto-registered **read-only** into the dataset engine — so reference tables are queryable
like any handle (`data query "SELECT name FROM cities ORDER BY population DESC LIMIT 5"`), and a
user dataset of the same name shadows it. NL routes through the `geography` orchestrator:
*"What is the population of France in 2020?"* → `67000000`, *"coordinates of Paris"* → `48.8566,
2.3522`, *"capital of Japan"* → `Tokyo`, *"Where is New Delhi?"*. Seed numbers are round,
sanity-checkable approximations; `python -m reasoning.cli reference import <csv> <table>` (or
`reference.import_csv`) replaces/extends them with an authoritative source (GeoNames, World Bank).

**Retrieval fallback** — a data point *not* in the curated set (an unknown country, or a year
the seed lacks) can be **retrieved via a Claude Sonnet call**, opt-in with
`DIFG_REFERENCE_RETRIEVAL=1` (off by default, so the engine stays 0-model-call). The retrieved
value is **validated** (a population must be a positive integer; coordinates must be in range)
and returned **flagged as model-sourced** — *"3170000 — retrieved via Claude Sonnet (not in the
curated reference set; verify)"* — with `model_calls=1` recorded, so it's never silently mixed
with verified reference data. Retrieved values are **persisted** to a separate `retrieved` log
table (queryable: `data query "SELECT * FROM retrieved"`); a repeat of the same query is served
from the log at **0 model calls** (*"… — previously retrieved"*), and the log is the audit trail
for later review/promotion into the curated seed.

**Year-aware + trend projection** — population is stored by year (1990–2023). An exact seed year
answers plainly; a year *not* in the seed gets a deterministic **trend estimate** from the place's
by-year series (interpolated within range, extrapolated by CAGR beyond it) — *0 model calls*,
clearly labeled: *"What is the projected population for London in 2030?"* → `9422865 (projected
from the 1990–2023 trend; estimate, not curated)`. It never silently presents an estimate as a
fact, and never fabricates beyond a place's own trend. A two-year **change** query —
*"how much did the population of London grow from 2010 to 2020?"* → `+900000 (+11.25%)` — returns
the delta and percent (and the single-year lookup steps aside so it can't silently answer just the
first year).

**Batch retrieval** — *"List all cities in France."* or *"What are the populations of France,
Germany and Italy?"* asks Claude for a **CSV table**, validates it, and saves it as a
model-sourced **dataset handle** (e.g. `list_all_cities_in_france`) — queryable like any other
(`data query`, `aggregate`). Also opt-in via `DIFG_REFERENCE_RETRIEVAL`.

**Review & promote** — `reasoning.cli reference review` lists the model-retrieved log;
`reasoning.cli reference promote <op> <place> [year]` (e.g. `promote population Mongolia 2015`)
merges a reviewed value into the **version-controlled curated data** (`store/reference/
promoted.json`, merged into the DB at build alongside `seed.py`). The promoted point becomes a
**verified, 0-model-call** lookup — resolvable even with retrieval disabled — and its log row is
cleared.

## Model variants (mlx-community)
`-4bit` (default, ~13 GB) · `-5bit` · `-6bit` · `-8bit` (~26 GB) · `-bf16`.
On 36 GB, 4-bit is the safe default; 6-bit fits with headroom; 8-bit is tight.

## Notes
- Set `HF_TOKEN` for faster/rate-limited downloads (unauthenticated works, just slower).
- The model is multimodal (image-text-to-text): pass `--image <path|url>` to either entry point.
- It emits a `<|channel>thought ... <channel|>` reasoning block; add `--enable-thinking` to steer it.

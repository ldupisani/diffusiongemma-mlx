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
  `retrieve_set` → a CSV from the model saved as a model-sourced dataset handle. `reference
  review`/`promote` move a logged value into version-controlled curated data
  (`store/reference/promoted.json`, merged into the DB at build alongside `seed.py`) — it becomes
  a verified 0-call lookup and its log row clears.
- **`reasoning/datagen.py`** — domain dataset generators (primes/Fibonacci, random samples,
  logistic trajectories, projectile motion, amortization, …) that produce rows (reusing pure
  skills + numpy/SymPy) and save via `datasets.save_rows`. Registry `GENERATORS`, grouped by
  `DOMAINS` (`list_domains`/`domain_of`). Reached via `data gen <name> <handle> key=value …`,
  natural language (`data gen "generate first 1000 prime numbers"`, or the NL engine), with
  discovery via `data domains` (what can create datasets) and `data list`/`data info <handle>`
  (what exists + how to query it; `datasets.schema` gives columns+types).
- **`reasoning/introspect.py`** — self-introspection over the three capability registries
  (orchestrators / skills / generators): `domains`, `domain(id)`, `search(term)`, `skills`,
  `summary`. In-memory, deterministic, lazy imports (the registry pulls in orchestrators → binds
  → skills). Surfaced three ways: the `introspect` orchestrator (NL meta-queries — "what can you
  do?", "what can I ask about geometry?" — 0 calls, bind-gated and digit-free so it never steals a
  computational query), the CLI (`reasoning.cli help [domain] [--search term]`), and pure
  `introspect.*` skills (`domain_exists`/`skill_exists`/`generator_exists`/`can_answer`) for
  higher-level reasoning to probe availability before acting.
- **`reasoning/authoring.py`** — "author a domain" tool: builds a context-rich prompt (the full
  playbook + the skill-registry contract + a worked exemplar (`logistics`) + the existing
  query_types) and calls Claude to implement a NEW domain, parsing the returned files from a
  `<<<FILE …>>>` delimiter format. Model + reasoning effort are configurable (`--model`/`--effort`);
  default is **Haiku** (cheapest/fastest — empirically authored a correct draft of a fresh
  well-specified domain, modulo a one-line example-rounding fix the suite flags; use `--model opus`
  for harder or under-specified domains). Code-gen only — writes new module files (never clobbers),
  prints the wiring + CASES to apply, and tells you to review + run the tests. `--verify` adds a
  two-phase author→test→repair loop (each phase loops up to `--max-rounds`, default 3): **(1) examples**
  — import the authored skill module, run its @register tables, feed failures (wrong value, un-tupled
  single arg, arity/import error) back for a single-file re-emit; **(2) routing** — wire the domain
  into a throwaway git worktree and route-test it (the orchestrator's example_queries must hit the new
  domain; the existing test CASES must not be stolen), feeding mis-routes/thefts back for a bind+orch
  re-emit. The live tree is never touched; routing is skipped (not failed) outside a git repo. Still
  review the result before committing (the loop checks routing + example math, not every edge case).
  CLI: `python -m reasoning.cli author-domain "<name + ops>" [--model opus] [--effort high] [--verify]
  [--max-rounds N] [--dry-run]`.
- **`reasoning/bench.py`** — word-problem benchmark harness (GSM8K and the like): routes a held-out
  set CODE-ONLY (mirrors `mcp_server.ask`; an offline guard blocks model calls at `engine._post` so a
  "model_free" bind that leaks to the LLM is bucketed honestly) and scores against the gold. The
  dataset lives in the gitignored data dir (handles `gsm8k_train`/`gsm8k_test`) — questions never
  enter version control. Firewall against overfitting: DEV (=train) is inspectable for gap-mining,
  TEST (=test) is aggregate-only. Reports a harm-ordered miss taxonomy (`wrong_answer` = confident
  wrong number; `declined` = honest "(no result)"; `no_route`), **precision** (when it commits to a
  number, how often right — the routing-safety metric), per-domain confident-wrong (the §3f-hardening
  target list), and an arithmetic-depth *reachable-ceiling* curve read from the gold's `<<…>>` calc
  markers. CLI: `python -m reasoning.cli bench --slice dev|test [--limit N]`. 0 model calls.
- **`reasoning/translate.py`** — hybrid **LLM-as-translator** (Program-of-Thought) for parse-limited
  word problems (the path the bench diagnostic pointed to: GSM8K is parse-limited, not depth-limited).
  The model is a TRANSLATOR, not a solver — one call turns prose into a tiny arithmetic *program*,
  which `run_program` EXECUTES deterministically (a security-critical **safe evaluator**: `ast.parse`
  + an operator/function whitelist, never eval/exec — a program can do arithmetic and nothing else).
  The model never does the math; the engine's re-execution is what's trusted and reported, so the
  ARITHMETIC is correct-by-construction and auditable (1 model call, 0 arithmetic calls) — but the
  TRANSLATION can be silently wrong, hence the confidence signals below. `solve(question, asker=…)`
  has an injectable asker; `backend=` picks the model path — `claude` (the Claude CLI, default) or
  `sidecar` (the local MLX model via `engine.one_shot` — the project's own small LLM, free/offline).
  **Prompt `style=`:** `program` (default — direct prose→program), `analyze` (the model first lays
  out facts/relations/answer-type, then the program — structured CoT demonstrated in the exemplars;
  helps relationship/coreference problems but is net-negative overall on the small model), or `auto`
  (route only relationship-cue questions to `analyze`, plain `program` otherwise — a near-free
  no-regret opt-in; ≈ `program` within noise on held-out test, so `program` stays the default).
  Two label-free confidence signals: `unused_givens` (a question number absent from the program)
  and `solve_consistent` (K-draw majority vote; `agreement` is the signal — also lifts accuracy).
  Measured via `bench --translate [--backend …] [--style …] [--consistency K]` and the preprocessing
  `bench --ablation`. Wired into `ask` as an OPT-IN fallback (default off — keeps the service 0-call).
- **`reasoning/reading.py`** — the **model-free obligation parser** (a separate, bottom-up effort to
  REPLACE the LLM translator): grammar lives in the words, each carrying an *obligation* it discharges
  by binding; parsing, sense-disambiguation, deficiency-detection and reasoning are all one
  coverage/saturation mechanism. Ingests statements into facts (`Predication`/`Membership`/`Subset`/
  `Relation`/`Quantity`/`Frequency` + count instances) and answers questions with a four-verdict
  epistemic square (yes/no/undetermined/conflict) plus count/wh/which/aggregation and **arithmetic**
  (values, operators, quantities, accumulation with signed verbs, coreference, relational quantities,
  question-relative polarity). `solve(text)` reads a whole word-problem paragraph (premises + trailing
  question); arithmetic executes through `translate.run_program` (the audited safe evaluator). A
  **relational temporal model**: time references are NODES (`time` sense) carrying `associates` (the
  value they timestamp → `Quantity.time`) and `orders` obligations; ORDER is tracked as `Before` facts
  read off the text (never a calendar rank — days wrap). A **read–synthesize loop** (`synthesize` +
  the `SYNTHESIZERS` registry) derives implied facts to a fixpoint (first: transitive ordering),
  generalizing `solve()`'s fixpoint — new reasoning is REGISTERED, never special-cased. 0 model
  calls; every step a reversible `Decision`, auditable via `--trace`. Strategy doc:
  `docs/logical-reading-playbook.md`; tests `reasoning/tests/test_reading.py` (curated/synthetic, never
  GSM8K). CLI: `python -m reasoning.reading "<text>" [--trace]`.
- **`reasoning/reading_bench.py`** — INDEPENDENT GSM8K coverage harness for `reading.solve` (mirrors
  `bench.py`'s firewall: DEV=train inspectable, TEST=test aggregate-only; never imported by the unit
  tests). Scores against the gold `#### N`, reports coverage / precision / a decline-blocker taxonomy
  (currency / percentage / unknown-word / structure) and the **most-frequent unknown words** (the
  data-driven vocabulary worklist). State (2026-07-03, after campaigns #180–248): dev-train
  **1018 correct / 0 wrong (~27.4% of dev-train)**, dev-holdout 142/2, TEST **39 correct / 5 wrong
  (88.6% precision, hit-rate 14.4%; TEST not re-checked since #246 — #247/#248 are holdout-clean)**. The
  #239–246 vocab-singles campaign crossed 1000 dev-train but the TEST checkpoint gained +1 correct /
  +1 wrong (a fold under-read on a 'carrots'-unlocked problem) — the vein is now ~TEST-neutral;
  general machinery is what moves TEST. #247/#248 kept widening the generic age-equations solver:
  #247 added shifted-time forms (+3 dev-train); #248 made every shifted form accept WORD-NUMBER time
  offsets ("in seven years"), added an E8 same-age-as-B-today equality form + a "how many years old is
  X" ask head (+1 dev-train) — the general lever.
  Coverage = fully-read ceiling × structure hit-rate (see docs/reading-coverage-methodology.md §9;
  the bench prints both) — judge generality by the HOLDOUT hit-rate, never dev-train's. 0 model calls. DEV has a stable-hash **train/holdout split** (`--slice dev-train|dev-holdout`, the anti-overfit
  refinement: tune on dev-train, validate on dev-holdout BEFORE the single TEST look — `dev-holdout` is a
  cheap TEST proxy). The full selection+validation process is in **`docs/reading-coverage-methodology.md`**.
  CLI: `python -m reasoning.reading_bench --slice dev|dev-train|dev-holdout|test [--limit N]`.
- **`reasoning/reading_diff.py`** — the per-increment **safety net** (before/after diff over dev-train +
  dev-holdout). The aggregate `reading_bench` number AND the unit suite both HIDE confident-wrongs: an
  increment can post a +1 net gain while silently turning previously-correct answers wrong. This runs
  `reading.solve` over the two inspectable dev slices and buckets the delta into SOLVED / BROKE /
  DECL→WRONG; an increment SHIPS iff BROKE and DECL→WRONG are both 0 and NET > 0 (diff-neutral → record,
  don't commit). Reuses `reading_bench._load`/`_gold` for the SAME holdout partition + gold parsing (a
  divergence would silently leak the firewall). Snapshots default to `$DIFG_DATA_DIR/reading_diff/`
  (gitignored — GSM8K questions never enter version control); never imported by the unit tests. 0 model
  calls. Workflow: `snapshot before.json` (at HEAD) → edit → `snapshot after.json` → `diff`. CLI:
  `python -m reasoning.reading_diff snapshot [path]` and `… diff [before.json] [after.json]`. Also
  `… collisions` — audits the `_EARLY_SYSTEMS` registry for cross-solver COLLISIONS (a dev problem
  claimed by ≥2 arithmetic system-solvers whose values disagree — the `_name_count` family-isolation
  failure mode; must be 0). Required after adding/reordering any system-solver; see the playbook's
  "arithmetic system-solvers are a REGISTRY" corollary.
- **`reasoning/reading_rulestats.py`** — read-only **rule-usage frequency** for `reading.solve`: which RULE
  produced each correctly-solved question, and how often. A "rule" is a named system-solver (`_EARLY_SYSTEMS`/
  `_LATE_SYSTEMS`), a `solve()`-level recognizer (partition/container_giveaway/equal_share/backward_*), or the
  general `answer()` fold (holdings/event-count/remainder/combined-Σ/difference/age-algebra — sub-bucketed by a
  question-shape heuristic). Attribution exploits `solve()`'s short-circuit dispatch (one rule wins per solve):
  it wraps every rule with a recording shim INSIDE a self-restoring context (`_instrumented`), so importing or
  running it leaves the engine's dispatch untouched. Reuses `reading_bench._load`/`_gold` (same partition + gold
  parsing); counts are an aggregate tally so `--slice test` is firewall-safe; never imported by the unit tests.
  0 model calls. Use it to see where solve's coverage actually comes from (the anchored family dominates; age
  algebra silently carries ~⅓ of the general fold) and to sanity-check a new solver actually fires. CLI:
  `python -m reasoning.reading_rulestats [--slice dev|dev-train|dev-holdout|test] [--limit N] [--commits]`.
- **`backend.py`** — pluggable inference backend (see below).
- **`reasoning/mcp_server.py`** — exposes the reasoning project as an MCP service (FastMCP, stdio):
  `ask` (route → coded answer + domain + model-call count; 0-call compound split; non-coded types
  reported, never invoked — UNLESS the opt-in **translator fallback** is on: `ask(query,
  translate=True)` or env `DIFG_TRANSLATE_FALLBACK=1` routes an otherwise-unanswered word problem
  through `translate.solve` (1 model call, `domain="translate"`, the audited `program` returned;
  default OFF so `ask` stays 0-call) plus introspection
  (`capabilities`/`describe_domain`/`search_capabilities`)
  and dataset/reference tools (`list_generators`/`generate_dataset`/`list_datasets`/`dataset_info`/
  `query_dataset`). Also exposes user-invokable **prompts** (slash commands in clients):
  `ask`/`capabilities`/`query`. Code-only (0 model calls) so it needs no sidecar. Hot-reloads
  changed `reasoning/*.py` per call (drops cached modules — incl. the parent package, else the
  stale submodule attribute defeats it), so engine edits land without a client reconnect; only
  edits to `mcp_server.py` itself need a restart. Run: `python -m reasoning.mcp_server`.
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
python -m pytest reasoning/tests -q       # ~1150 tests (use system python; .venv lacks pytest)
python -m reasoning.cli eval-all          # fidelity of every coded orchestrator vs teacher
python -m reasoning.cli bench --slice dev # GSM8K word-problem accuracy of the code-only engine
```

Env: `DIFG_PORT` (8088), `DIFG_MODEL`, `DIFG_BACKEND`, `DIFG_PYTHON` (`.venv/bin/python`).

## Conventions

- Adding a domain: either hand-write it or scaffold it with `reasoning.cli author-domain "<spec>"`
  (calls Claude — Haiku by default, `--model opus` for harder domains — pre-loaded with the
  playbook + exemplar; review + test its output). Follow
  `docs/adding-domains.md` — skill module + bind + orchestrator, gate
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

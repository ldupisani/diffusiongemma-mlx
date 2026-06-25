# Adding a domain safely

How to extend the reasoning engine with a new subject area (p-adics, cryptography, game
theory, …) **without breaking existing routing**. Written from building ~14 domains; every
"learning" below is a mistake that actually happened and how it was caught.

---

## 0. The one rule that decides everything

**This engine answers by computing a definite value with code, at zero model calls.** A subject
only belongs here if its questions have *deterministic numeric/symbolic answers*. Before writing
anything, classify the request:

- **Computable** (build it): p-adic valuation, RSA, quaternion norm, Nash equilibrium, curvature,
  Lyapunov exponent. Clear inputs → one correct output.
- **Conceptual** (don't): model theory, proof theory, the bulk of functional analysis. These are
  "explain / prove" territory — forcing them into a calculator yields thin or misleading
  orchestrators. Leave them to the LLM teacher. (You *can* carve out a computable *fragment* — e.g.
  functional analysis → vector norms, inner products, Cauchy–Schwarz — and skip the rest.)
- **Already covered** (extend, don't duplicate): before adding "statistics" or "combinatorics",
  grep the existing skills (`reasoning/skills/`) and orchestrators. Add new *operations* to the
  existing domain rather than a parallel one that fights it for routing.

> **Learning.** Survey existing coverage first. `stats.*` already had variance/stdev/percentile;
> `numtheory.*` already had nCr/nPr. Half of a "18 new domains" request was already implemented or
> conceptual. Building blind would have duplicated coverage and produced dead conceptual stubs.

---

## 1. Anatomy of a domain — four pieces

| Piece | Location | Job |
|---|---|---|
| **Skill module** | `reasoning/skills/<name>.py` | Pure, typed functions with inline example tables. The actual math. |
| **Bind module** | `reasoning/bind/<name>.py` | `bind_<name>(query)` — recognize the NL query and extract typed arguments. |
| **Orchestrator** | `reasoning/store/orchestrators/<name>.py` | `META` + `precondition` + `run`. Registers the domain with the router. |
| **Tests** | example tables (auto-run) + a row in `reasoning/tests/test_orchestrators.py::CASES` | Skill correctness + end-to-end routing. |

### 1a. Skill module

Pure functions registered with `@register("<ns>.<op>", "<doc>", [examples])`. The example table
**is** the unit test — `reasoning/skills/tests/test_all_examples.py` runs every registered skill's
examples automatically, so a new family needs no new test file.

```python
"""<domain> — one-paragraph description of what's computable and the conventions used."""
from __future__ import annotations
from .base import register

@register("padic.valuation", "p-adic valuation v_p(n): exponent of p dividing n",
          [((24, 2), 3), ((54, 3), 3), ((100, 5), 2)])
def valuation(n: int, p: int) -> int:
    ...
```

Then add the module to the import list in `reasoning/skills/__init__.py` (import-for-side-effects;
that's what runs `@register`).

Rules:
- **Pure stdlib or SymPy.** No engine imports (a test enforces it). No I/O, no randomness.
- **Every example must be exactly correct.** Floats are compared at `places=6`. Don't eyeball an
  expected value — *run the function* and paste its real output (see Learning below).
- Return a `str` for non-numeric answers (formatted expressions, verdicts); the orchestrator marks
  those `round=False`.

### 1b. Bind module

`bind_<name>(query) -> (ok: bool, data: dict | None, via: str, model_calls: int)`. On a match,
`data` is usually `{"skill", "dims", <dim>: value, ...}` consumed by `dispatch`.

```python
from reasoning.bind._common import *  # noqa: F401,F403

def bind_padic(query):
    q = query.lower()
    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    pm = _PRIME.search(query)            # a CONSERVATIVE gate (see §3)
    if not pm:
        return False, None, "code", 0
    ...
```

Register it in `reasoning/bind/__init__.py` (`from reasoning.bind.<name> import *`).

The `dispatch` data contract:
- `dims`: ordered argument names; `dispatch` calls the skill with those inputs.
- `round`/`places`: round a numeric result (default `round=True, places=2`); set `round=False` for
  string-returning skills.
- `suffix`: appended unit string (`"%"`, `" N"`).

### 1c. Orchestrator

```python
from reasoning.bind import bind_padic
from reasoning.compiled import dispatch

META = {
    "id": "padic", "query_type": "padic",
    "description": "...",                 # also feeds the teacher/eval
    "example_queries": [...],             # drives lexical/embedding routing — make them realistic
    "keywords": [...],
    "model_free": True,
    # "bind_gated": True | omit  → see §2, this is the most important choice
}

def precondition(query):
    return bind_padic(query)[0]

def run(query):
    return dispatch(bind_padic(query), parse_message="could not parse the … query")
```

---

## 2. The router tier model — read this before choosing `bind_gated`

`router.recognize` runs **five tiers in order** (`reasoning/router.py`); the first that matches
wins. Only orchestrators whose `precondition(query)` is true are even candidates.

1. **composer** — a query that fully decomposes into independently-answerable clauses.
2. **domain** — *specific* orchestrators with `bind_gated` **false**, chosen by the matcher
   (lexical/embedding score) above a threshold.
3. **bind-gated** — orchestrators with `bind_gated: True`; **dispositive** (route on the parse even
   at ~0 lexical score), but only reached if **no tier-2 domain matched**.
4. **clarify** — recognized type, missing a field → ask the user.
5. **fallback** — generic catch-alls (e.g. `numeric_consensus`).

**So `bind_gated` is not "better" — it's a *lower-priority* tier.** Choose by query shape:

- **`bind_gated: True`** when the query may have *no lexical signal* — pure-symbolic or
  notation-only ("2-adic valuation of 24", "ω·2"). The conservative parse is the whole signal.
  Dispositive routing protects these from threshold misses.
- **`bind_gated` omitted (tier-2 domain)** when the query is keyword-rich AND its structure (a
  matrix `[[..],[..]]`, a number list) could be grabbed by another tier-2 orchestrator. Competing
  *in* tier 2 by lexical score lets distinctive keywords win.

> **Learning (the big one).** `game_theory` was first marked `bind_gated: True`. "value of the
> zero-sum game [[1,-1],[-1,1]]" then routed to `list_statistic` (a tier-2 domain) which averaged
> the matrix entries — because tier 2 runs *before* the bind-gated tier. Fix: make `game_theory` a
> tier-2 domain (drop `bind_gated`); its keyword-rich `example_queries` then beat `list_statistic`
> on score. **Rule of thumb:** number-list / matrix domains → tier 2; notation-only domains →
> bind-gated.

> **Learning (entity-gated domains can be bind-gated).** The usual rule is "keyword-rich →
> tier 2", but a domain whose bind only fires when a **known entity** is present (e.g. `geography`
> matches only a geography keyword *plus* a country/city name from the reference data) is a strong
> `bind_gated` candidate: the entity check makes it conservative enough not to over-claim, and
> dispositive routing then handles *all* phrasings ("Where is New Delhi?") without needing an
> example per phrasing. Tier-2 domains still win their own queries first, so geometry's "area of a
> circle" isn't stolen by geography's "area". Reach for this when phrasing variance is high but the
> bind is tightly gated on a closed vocabulary.

> **Learning (tier-2 needs an example per op).** A tier-2 domain is selected by *lexical score
> against its `example_queries`*, so **every operation needs at least one representative example**
> — an op with none scores below the matcher threshold and the query silently returns `None`
> (falls through), even though its `bind` matched. This bit `game_theory`, `numerical` (Simpson),
> and `logistics` (Little's Law / days-of-supply were missing → `None` until added). When you add
> an op, add an example query for it.

---

## 3. Collision management — the actual hard part

A new domain is **not done when its own queries pass**. It's done when it neither *steals* other
domains' queries nor is *stolen from*. Test routing in **both directions**.

### 3a. Gate conservatively

The `precondition` (the bind) must only fire on queries that genuinely belong to the domain. Prefer
an unambiguous signal no other domain uses:

- p-adic → the literal `"<p>-adic"` phrasing.
- cryptography → `RSA / Diffie-Hellman / Caesar / totient / modular inverse`.
- quaternions → a `j`/`k` imaginary unit or the word "quaternion".
- game theory → `nash / payoff / zero-sum / dominant strategy / saddle`.

A broad gate (e.g. just "mod") is fine **only if** each operation branch is specific and the bind
returns `False` when none match.

### 3b. Add decline-guards to the *victim*, not just the *new* domain

When a generic orchestrator over-matches the new domain's queries, the clean fix is a narrow
decline guard on the over-eager bind. Real examples now in the codebase:

- `bind_list_statistic` declines `spanning tree | graph | edges | inner product` — it was matching
  the **"minimum"** in "minimum spanning tree" and the **"product"** in "inner product" as its stat
  operators.
- `bind_complex` declines `j`/`k`/`quaternion` — a complex literal must not claim a quaternion.
- `bind_algebra` declines `ω`/`omega`/`ordinal` — SymPy would parse the words as variables.
- `bind_ordinal` declines hyperreal signals (`ε`/`epsilon`/`infinitesimal`/`standard part`) —
  because **`ω` is overloaded** (ordinal vs. the hyperreal infinite `1/ε`); context decides, so the
  later domain forces the earlier one to yield.
- `bind_optimize` claims "minimum/maximum of X" **only if X contains a variable** — otherwise
  "minimum of 5, 3, 8" is a stat query, not a function optimization.

> **Learning.** Overloaded symbols are the sharpest trap. `ω` means an ordinal *or* an infinite
> hyperreal; the symbol can't disambiguate, so gate each domain on *context words* and make the
> later-added domain decline-guard the earlier one. Same lesson for stat-operator substrings hiding
> inside compound terms ("**min**imum spanning tree", "inner **product**").

### 3c. The composer tier eats "X of A **and** B"

Two-operand phrasings ("the product of A and B", "compare A and B") are claimed by the composer
tier before any domain. Don't bind those; either (a) leave the two-operand op as a tested-but-
unbound skill (like `cplx.multiply`, `hyper.q_multiply`), or (b) use a phrasing without "and"
("is A greater than B"). Document the unbound op so it's a choice, not a gap.

---

## 4. Parsing gotchas (each cost a debugging cycle)

- **Implicit multiplication.** `sympify("x^2 - 4x + 7")` fails on `4x`. Use
  `parse_expr(expr.replace("^","**"), transformations=standard + implicit_multiplication_application)`.
- **Possessive `'s`.** Filler-word stripping left `'s method` from "Newton's method". Strip
  `['’]s\b` before stripping filler words.
- **Sentence punctuation in char classes.** A regex char class like `[εω\d+*^.]+` greedily ate the
  trailing `.` in "classify ε." → `float("ε.")` crashed. `rstrip(".")` extracted tokens, or exclude
  `.` and handle decimals separately.
- **Output-format delimiters beat free-text extraction.** Small models append explanations despite
  "output only X". Asking for the answer between `<q>…</q>` tags (and splitting on `</q>` *first*)
  is far more robust than taking "the first line".
- **`getattr(obj, attr, default)` never raises** — it returns the default for *any* object,
  including `None`. Don't "fix" a non-bug.

---

## 5. Float examples — never eyeball them

Example tables compare floats at `places=6`. Two safe tactics:

- **Use cases with exact/clean answers**: a Simpson's rule that's exact for the polynomial degree;
  a saddle-point game value; gradient descent run long enough to round to `0.0`; Newton iterated to
  full convergence.
- **Otherwise, run the function and paste its real rounded output.** Several Lyapunov/regression
  examples were hand-guessed wrong (`0.692147` vs the true `0.6932`; an intercept of `1.5` vs `2`).
  Compute, then hardcode.

---

## 6. Verification workflow (per domain, before committing)

1. **Skill examples** — run the new family's example tables (fast, offline). Fix until green.
2. **Routing, forward** — every `example_query` routes to the new orchestrator and returns the
   right value at **0 model calls**.
3. **Routing, both directions** — a handful of *neighbouring* queries (the ones most likely to
   collide: matrices → linalg, number lists → list_statistic, `ω` → ordinal/hyperreal, "integral
   of" → calculus) still route to their *original* domain. Add decline-guards until they do.
4. **`CASES` row** — add representative `(query, expected, orch_id)` tuples (including a collision
   *guard* row like `("What is the minimum of 5, 3, 8, 1?", "1", "list_statistic")`).
5. **Full suite** — `python -m pytest reasoning/tests reasoning/skills/tests -q` (use system
   `python`; the `.venv` lacks pytest). Must stay all-green.
6. **Commit per domain** on a branch, fast-forward to `main`, push. One domain per commit keeps the
   history bisectable and each collision fix attributable.

A quick routing harness (used constantly):

```python
from reasoning import router
def route(q):
    m = router.recognize(q)
    return (m.orchestrator.id, m.orchestrator.run(q).answer) if m else (None, None)
```

> **Live-test caveat.** A `router.recognize(...).run(q)` on a *non-model-free* fallback
> (`numeric_consensus`) will try the sidecar and raise `ConnectionRefused` if it isn't running.
> That's the harness, not the domain — the routing result above the traceback is still valid.

---

## 7. Incremental rollout

For a multi-domain request: **validate one end-to-end first** (pick the cleanest — p-adic,
cryptography), commit it as the proven template, then proceed. Don't batch 8 domains blind; each
tends to surface its own collision, and per-domain commits make each fix reviewable.

---

## 8. Look for opportunities to seed new datasets

**Every new domain — ask: does it naturally produce a *sequence, sample, series, or schedule*?**
This is not an afterthought; make it a deliberate check. Two kinds of "seeding":

- **Generator** (computed rows) — add one to `reasoning/datagen.py` (registered in `GENERATORS`,
  grouped under a `DOMAINS` key) that builds rows reusing the domain's pure skills + numpy/SymPy
  and saves via `datasets.save_rows`, returning a handle + preview (never a text dump). Wire it
  into NL via the `dataset` orchestrator's `_generate` (and add a generation `example_query` so it
  routes — see the tier-2 learning in §2). Random generators take a reproducible `seed`. Shipped
  examples: primes/Fibonacci (number theory), normal/uniform samples (probability), logistic
  trajectory/projectile (dynamics/physics), amortization (finance), EOQ curve (logistics), truth
  tables (logic), Cayley tables (group theory), polygon tables (geometry).
- **Reference seed** (constant, verifiable rows) — if the domain has bundled facts worth shipping
  (like geography's countries/cities), add them to `reasoning/store/reference/seed.py`; they build
  into the read-only reference DB and become queryable handles. See the geography subsystem.

**Skip** when the answer is one number, not many rows (complex/quaternion, ordinals/cardinals/
surreals, p-adic, crypto). "Appropriate" = *many rows*. Verify the generator with `data domains`
and a spot-checked value, and add it to the introspection surface for free (it's auto-listed).

---

## 9. Look for opportunities to synthesize with other domains

**A new domain rarely lives alone — ask: what real question chains it with an existing domain?**
The engine composes orchestrators via `SubStep(..., orchestrator="<id>")` in a `Plan` (see
`compiled.py`), NOT via metadata; a `synthesized_*` orchestrator runs one domain's output into
another's input, still at 0 model calls. Shipped cross-domain syntheses: `synthesized_geometry`
(area → fencing/flooring **cost**, i.e. geometry → finance), `synthesized_physics` (distance/speed
→ travel **cost/time**, physics → geo/finance), `synthesized_geo` (distance → flight time/price),
`synthesized_logistics` (trucks needed → shipping **cost**, logistics → finance),
`synthesized_chemistry`, `synthesized_materials`.

When you add a domain, scan its neighbours for a natural hand-off:

- **a quantity it produces that another domain prices/converts/times** — area→cost, distance→time,
  mass→price, moles→mass;
- **a unit it emits that another domain consumes** — coerce at the seam (dimension-aware unit
  coercion auto-inserts `convert` ops at bind time), don't hardcode the conversion;
- **a shared entity** — a place (geography), a shape (geometry), a substance (chemistry) that two
  domains both reason about.

Build the synthesis only when the chained question is one a user would actually ask; add it as a
new `synthesized_<x>` orchestrator with its own `example_queries` and `CASES`, and **test routing
both directions** so the composite doesn't steal the atomic queries (and vice-versa) — the same
collision discipline as §5.

---

## TL;DR checklist

- [ ] Computable, not conceptual? Not already covered?
- [ ] Skill module + `@register` examples (run them; floats exact); added to `skills/__init__.py`.
- [ ] `bind_<name>` with a **conservative gate**; registered in `bind/__init__.py`.
- [ ] Orchestrator `META`/`precondition`/`run`; `bind_gated` chosen per §2 (number-list/matrix → tier 2).
- [ ] Forward routing ✓ at 0 model calls; **reverse** routing of neighbours ✓ (decline-guards added).
- [ ] `CASES` rows incl. a collision guard; full suite green.
- [ ] **Seed datasets?** generator in `datagen.py` if it yields rows/series, or a reference seed if it has constant facts (§8).
- [ ] **Synthesize?** scan neighbours for a chained question → a `synthesized_<x>` orchestrator, routing tested both ways (§9).
- [ ] One commit per domain → `main`.

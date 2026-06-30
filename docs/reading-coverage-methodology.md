# Reading-engine coverage/precision methodology

How we decide **what to work on next** in `reasoning/reading.py` (the model-free GSM8K obligation
parser) and how we **validate** each change before it ships. This is the firewall-disciplined funnel
that turns "where are we losing?" into "what is the safest available win?".

It complements two siblings: `logical-reading-playbook.md` (how the parser is *built* — obligations,
senses, synthesis) and `adding-domains.md` (the orchestrator/skill workflow). This doc is about the
*measurement and selection loop*, not the parser internals.

---

## 0. The one-paragraph version

Measure on GSM8K split into **DEV** (inspectable, where we mine and tune) and **TEST** (aggregate-only,
the honest generalization check — one go/no-go look per increment). Prioritise **precision before
coverage** (a confident-wrong is worse than an honest decline). Mine the next target from DEV with the
diagnostic harnesses, build the tightest gate that catches it with **zero DEV regression**, A/B it,
validate on **`dev-holdout`** (a cheap TEST proxy), and only then spend the **TEST** look. A TEST-precision
drop reverts the change, no matter how good the DEV numbers were.

---

## 1. The two metrics, harm-ordered

The bench (`reasoning/reading_bench.py`) reports a **harm-ordered miss taxonomy**:

| Outcome | Meaning | Cost |
|---|---|---|
| `wrong` | committed a confident wrong number | **worst** — a *precision* failure |
| `declined` | honest "(no result)" | safe — a *coverage* gap |
| `errored` | parse blow-up (counts as a decline) | safe |

- **Precision** = correct / committed. **THE safety metric**: when the engine commits to a number it
  must be right, so `wrong` must trend to 0.
- **Coverage** = committed / total. Grows only *after* the engine is decline-safe, so a new capability
  can only ever move a problem `blocked → {correct | decline}`, never `→ wrong`.

**Work order:** precision first (decline-safety), then coverage (composition). Never trade held-out
precision for a coverage gain.

---

## 2. The firewall: DEV vs TEST

- **DEV** (`--slice dev`, = the `gsm8k_train` handle) is **inspectable**. The bench prints sample
  solves, the decline-blocker taxonomy, and the frequent-unknown-word list so we can mine the next gap.
- **TEST** (`--slice test`, = `gsm8k_test`) is **aggregate-only**. We read the summary numbers and
  *never* inspect individual items.

Why: DEV is where we tune, so it inevitably becomes overfit; TEST is the honest check. Two historical
leaks shaped this:

- *The 300-row prefix.* "100% precision" was once an artifact of only ever running `--limit 300`. Full
  DEV had 66 confident-wrongs hiding past row 300. → `--limit` now prints a **PARTIAL-FIREWALL** warning;
  always read the full-slice number.
- *The offset overfit* (see §7) — which motivated the dev-holdout refinement below.

---

## 3. The DEV-internal holdout refinement

**Problem it solves.** DEV is where we tune, so a hand-written gate can be tuned (often unconsciously,
via whack-a-mole) to pass exactly the DEV problems we looked at. Such a gate shows **0 DEV wrong** yet
fails to generalize. TEST catches this, but TEST is precious — one go/no-go look, and repeated looks turn
TEST into a second training set.

**The split.** DEV is partitioned into two halves by a **stable hash of the question** (`md5`, not
Python's per-run-randomized `hash`), so a given problem always lands in the same half across runs and
machines:

- **`dev-train`** (`--slice dev-train`) — mine and tune here freely; fully inspectable.
- **`dev-holdout`** (`--slice dev-holdout`) — a **cheap TEST proxy**. Aggregate + a class-level blocker
  taxonomy by default; per-item lists only behind `--wrongs`, for *diagnosis* (never item-tuning).

The split lives in `reading_bench._holdout_bucket(question)` and is **importable**, so ad-hoc harnesses
use the same partition. `dev` (the full slice) stays available for the broad "what classes exist" survey.

**The protocol (per increment):**

1. **Mine** the target class on `dev` or `dev-train`.
2. **Tune** the gate on **`dev-train` only** — never read `dev-holdout` items while tuning.
3. **Validate** on `dev-holdout` as an **A/B delta gate**: run it on the baseline and on the change; the
   increment passes iff it adds **zero new holdout wrongs** (ideally +holdout correct). The *absolute*
   holdout wrong count includes pre-existing deferred wrongs — it is the **delta** that gates.
4. On a clean holdout delta, spend the **single TEST look**. A TEST-precision drop reverts the change.

### 3a. What the holdout DOES and DOESN'T catch — read this

The dev-holdout is a **partial** safeguard. Be honest about its limits:

- ✅ It catches **item-level tuning-overfit**: a gate tuned to specific DEV-train problems that doesn't
  generalize to *other DEV problems of the same shape* will surface as new holdout wrongs.
- ❌ It does **NOT** catch **unseen-structure failures** (distribution shift): a problem *shape* that is
  absent from all of DEV but present in TEST. No DEV-based check can see those.

This is not hypothetical. In the offset case (§7), the recognizer fired **9 correct / 0 wrong on
dev-train and 8 correct / 0 wrong on dev-holdout** — a clean holdout PASS — yet still dropped **TEST
precision 50→40%**. The TEST wrongs came from shapes not represented in DEV at all.

**Implication:** the holdout reduces one class of risk cheaply, but the only defenses against unseen
structures are (a) **specific/conservative surface patterns** (a rarer pattern has less exposure to
unseen variants — "k times as much as X" is safer than the very common "X more/less than Y"), (b)
**semantic gates** rather than surface cues, and (c) treating the **TEST look as the final arbiter** —
a drop means revert, full stop.

---

## 4. The diagnostic harnesses (surfacing candidates)

These answer different "what's next?" questions. The durable one is the bench; the rest are small,
reusable scripts (kept under the job tmp dir during active work — recreate from this doc + git history).
All should import `reading_bench._holdout_bucket` if they tune gates, so they honour the same split.

| Harness | Question it answers |
|---|---|
| `reading_bench --slice dev` | Aggregate precision/coverage + the decline-blocker taxonomy + frequent unknown words |
| **permissive scan** | If every unknown word were a noun, which problems would *commit a wrong*? → the **decline-safety** worklist (exposes fully-read-but-unhandled commits on the inspectable slice) |
| **classify** | Bucket those wrongs by signal (relational / money / each-attr / rate / …) → pick the largest **clean** class |
| **fullread-declined** | Of fully-read problems that *decline*, what structures (by gold calc-step count)? → the **coverage** worklist (no vocab blocker, fully inspectable) |
| **vocab-scan** | Does adding any single frequent unknown word yield a *verified* decline→correct gain with zero new wrongs? |
| **catch-vs-regression** | For a candidate gate: how many DEV wrongs **caught** vs how many DEV correct **broken**? Run in `dev-train` (tune) and `dev-holdout` (validate) |

The **permissive scan** is the key precision tool: temporarily registering every unknown word as a noun
*exposes* problems blocked only by vocabulary that would mis-commit if unblocked — found on the
inspectable slice, before they can bite on TEST.

---

## 5. Selection criteria — what makes a candidate "the next problem"

1. **Frequency** — pick the largest clean class (e.g. each-attribute had 11 catches; the 2-unknown
   sum class ~97 declines). Log what you drop.
2. **Semantic typing, not raw number-accounting** — "every stated number must be used" snags legitimate
   structural counts ("the 3 people"). Clean signals are *semantically typed*: a period is a
   time-unit-quantifying value; "each" is what flips an object count from enumeration to operative.
3. **Specific over common surface patterns** — prefer a rarer, more specific trigger (it has less
   exposure to unseen TEST variants; see §3a).
4. **Zero regression** — a decline guard must catch wrongs with **zero** real-correct regressions.

---

## 6. The increment lifecycle (checklist)

1. **Mine** a target class (§4) on `dev` / `dev-train`.
2. **Build** the tightest gate/recognizer that handles it. Decline-safety guards live in `solve()` after
   `answer()`, gated so special recognizers (`_partition`, …) are exempt. Recognizers that *defer to no
   fact* must run **early** (before the unused-given guard would decline them).
3. **A/B per-problem** vs a clean baseline (detached-HEAD worktree). Diff **both** the wrongs-by-question
   **and** the full solved-set — the wrongs diff alone misses `correct → decline` churn.
4. **Permissive A/B** (full wrong-set by question identity): the "NEW-after must be EMPTY" check.
5. **`dev-train`**: tuned to 0 new wrongs, ideally +correct.
6. **`dev-holdout`**: zero new-wrong **delta** (§3). If it fails, fix the gate's *logic* on `dev-train`,
   don't tune to the holdout items.
7. **Unit test**: synthetic only — *never* a GSM8K string in `test_reading.py` (the firewall: the parser
   is never tuned to the benchmark's individual items).
8. **TEST look** (single, aggregate): precision must hold. A drop → **revert**.
9. **Ship**: commit + push (auto-commit on a green objective). Update this doc / the playbook if the work
   taught a new collision class, gate, or parsing gotcha.

---

## 7. Worked case study — the offset overfit (why §3a exists)

We extended the 2-unknown solver from pure-ratio (`A = k·B`) to the linear case (`A = k·B + c`). On the
standalone DEV scan it reached **17 fires / 0 DEV wrong** after a chain of gates (non-negative-result
guard; "N times more" → multiplier not +offset; combined-total cue vs per-entity "X sold 35 total";
enumeration exclusion "the *two* of them … *eight* hoodies"). Integrated: **DEV 72→78 (+6), but TEST
50.0→40.0%** — two held-out commit-wrongs.

- **Revert**, per the firewall (held-out precision beats DEV gains).
- **Diagnosis:** the offset *surface pattern* — 2 entities + a "more/less than" comparative + a "total"
  word — is **common** and shared by many non-system problems. A surface recognizer commits on shapes it
  shouldn't, and the bad shapes were **TEST-only**.
- **Holdout post-mortem:** the recognizer split 9/0 (train) and 8/0 (holdout) — a clean holdout PASS that
  still failed TEST. This is the empirical basis for §3a: the holdout catches tuning-overfit, not unseen
  structures.
- **Lesson:** the pure-ratio recognizer (rarer, more specific pattern) ships and holds TEST at 50%; the
  offset case is deferred until a genuinely *semantic* gate (not surface cues) is found, or treated as a
  different mechanism (e.g. a forward inverse-relational synth).

---

## 8. Disposition of any candidate

- **Ship** — clean, firewall-safe, A/B-verified.
- **Defer** — needs bespoke/risky machinery for a single problem (the *example-driven-deferral*
  principle: wait until a concrete example forces the design). Leave an inline `NOTE` at the site.
- **Revert + document** — fails the firewall or proves net-negative. The negative result is a *result*:
  record it (here, in the memory, or as a `NOTE`) so the next iteration doesn't repeat it.

---

## 9. Current frontier (snapshot — update as it moves)

- Decline-safety is **exhausted** for the cheap "absent-from-log" mechanism (GUARDs 11–15).
- **Single-word vocab is dead** (`vocab-scan` found zero safe gains): coverage is **structure-blocked**,
  not vocab-blocked at the margin.
- Remaining work is **composition** (multi-step solves) — riskier per-increment, must clear the firewall.
  The 2-unknown pure-ratio solver ships; the offset variant is deferred (§7).

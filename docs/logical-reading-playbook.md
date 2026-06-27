# Logical-reading playbook

Strategies for building **model-free natural-language logical reasoning**, bottom-up and word by
word (`reasoning/reading.py`, `reasoning/tests/test_reading.py`). A living document — updated as
strategies stabilize. See also [[logical-reading-initiative]] and
[[working-method-piecemeal-annotate]] in project memory.

## Goal & stance

- Deterministic reasoning, **not an LLM**. A valid statement gets a definitive, auditable reading;
  a deficient one gets an honest "this is unmet / ambiguous" — **never a guess**. 0 model calls at
  runtime.
- **Grammar lives in the WORDS** (lexicalist). Each word carries its possible meanings and any
  *obligations* it must discharge against other words. Structure, sense-disambiguation, and
  deficiency-detection all emerge from relating words and checking coverage — not from a global
  grammar. The logical form is an *emergent output*, not an imposed taxonomy.

## Strategies (the moves we reuse)

1. **Definitive dictionaries, never heuristics.** Word knowledge (plurals, senses) is a curated
   lookup returning *facts*. Unknown stays unknown (unmarked) — a safe silent miss, never a
   confident wrong guess. Grow the dictionary deliberately; a missing entry is a *roadblock to
   extend it*, not a cue to invent a rule. Irregulars (`children→child`) are free.

2. **Annotate, don't transform.** Keep every word exactly as written (`surface` immutable). Attach
   findings as `features` (number, singular, senses, obligations). Findings only accumulate;
   nothing is overwritten, so `surface` + `features` always reconstruct the whole history.

3. **Every change is a reversible Decision.** All mutation goes through one chokepoint
   (`Reading.record`) and is logged as a feature-diff (`before → after`) on a target unit.
   Backtrack by restoring `before`. For nested objects (an `Obligation`), **replace rather than
   mutate** so the diff stays a clean, reversible snapshot.

4. **Every role carries an obligation; the unmet ones are what we still track.** An obligation
   belongs to a *role*, not a word: a quantifier ranges over a collection; a linking preposition
   relates an `anchor` (left) to an `object` (right) — by virtue of their roles. Attach unmet;
   binding **drains** the set. A statement is **well-formed (saturated)** when nothing is unmet and
   no word is an orphan; **deficient** exactly when something stays unmet — the bottom-up
   deficiency detector. ("Filter all the concepts we track" = drain the open-obligation set.)

5. **Commit only when unambiguous.** Discharge an obligation iff exactly one candidate fits. Zero
   (nothing to bind) or several (which one?) → leave it **unmet = a reportable deficiency**. This
   is "definitive-or-decline" localized to one word.

9. **Bind vs. merge; condense in passes.** Most obligations resolve by **binding** — a relation to
   another unit that stays separate (`quantifier→collection`, `preposition→object`). But when two
   words form a single richer concept — adjective + noun (`red apple` = the *restricted* set, not
   all apples), or a name (`New York`) — resolution must **merge** them into one unit (a noun
   carrying `red` as an attribute), so later passes treat it as one thing. Hence parsing is
   **multi-pass**: condense local structures into richer units, then re-resolve over the condensed
   set, iterating to a fixed point. Merge is the *structural* move; binding is the *relational* one.
   Merge only when the result is genuinely one concept — otherwise bind (strategy 5 still gates it).

6. **Normalize variant syntaxes to one operation.** `each student` and `each of the students` are
   two syntaxes for the same distributive operation and must resolve to the *same* binding. Prefer
   making a variant land on the existing logic over inventing parallel logic.

7. **Ground sense taxonomies in real lexicography** (authoring-time only; 0 runtime calls). Look up
   attested senses/uses; record the findings as comments beside the lexicon entries.

8. **Test-first, piecemeal, example-by-example.** Smallest increment; red → green. Let
   generalizations fall out of failing examples rather than designing up front. **Expect churn**:
   when a new fact (e.g. `student` entering the lexicon) breaks an older test's assumption, *retest
   the decision* (update the test) — that is the method working, not a regression. Pin
   "unknown-word" invariants with nonce words (`wug`) so they stay stable as the lexicon grows.

10. **Disambiguate by coverage.** When a word carries several senses, don't tag it — *try* each and
    keep the sense-assignment that leaves the **fewest obligations unmet** across the whole
    statement (saturation arbitrates). `"rose"` is a noun in "the rose wilted" and a verb in "the
    teacher rose" because only those readings cover. Trials run on throwaway deep-copies; the winner
    is committed as a reversible `assign-sense` decision. This is the lexicalist thesis: a word's
    meaning is whichever sense makes the whole cohere — and it needs *competing* obligations to work.

## Pass order (dependencies)

`read → annotate_number → annotate_senses → disambiguate → annotate_obligations → condense* → resolve`
(wired as `analyze`)

- **number before senses** — a plural inherits its noun sense via its singular.
- **disambiguate after senses** — it needs the full candidate sense lists; it commits one each.
- **obligations read senses**; **condense** (merge) loops to a fixed point; **resolution reads
  obligations**.

## Data model

- `Word(surface, index)` — immutable atom of input.
- `Unit(words, features)` — a unit of meaning; `surface` reconstructed from words, findings in
  `features`.
- `Obligation(need, target)` — a demand; `met` once `target` points at the unit that satisfies it.
- `Decision(seq, kind, target, before, after, reason)` — a reversible feature-diff.
- `Reading(units, decisions, facts)` — the workspace; `record()` is the single mutation door.

## Increments done

1. lossless word tracking · 2. plural annotation (definitive dictionary) · 3. word-meaning
dictionary (senses, list-valued for multi-sense) · 4. obligations (quantifier → collection) ·
5. content words in the lexicon (noun/verb) · 6. unambiguous resolver · 7. partitive
`each of the students` resolves like `each student` (singular fallback + `analyze` pipeline) ·
8. preposition link obligation (`anchor`/`object`) — `each of the students` fully binds, **no merge
needed** (binding sufficed; merge is not a parsing primitive here) · 9. **attributive merge**:
`red apple` → one noun unit `(head: apple, attributes: {red})`, multi-pass to a fixed point,
reversible *structural* decision; a quantifier ranges over the compound so the restriction survives ·
10. **every role carries an obligation** (`article→determines`, `verb→subject`, `noun→predicate`) —
so a complete statement *saturates* and a bare noun phrase is flagged a **fragment** (the noun's
`predicate` is unmet). The deficiency detector now works directly off the obligation set ·
11. **sense disambiguation by coverage** — `"rose"` resolves to noun or verb purely by which reading
saturates; committed as a reversible `assign-sense` decision. No POS tagger ·
12. **`deficiencies(reading)` / `is_well_formed`** — the answer-or-ask boundary. Returns the
unrecognized words and unmet obligations as specific, human-readable findings (`"'students' has
nothing asserted of it"`); empty ⇒ understood. This IS the tool's contract in one function ·
13. **`answer(question, context)`** — a question is a statement whose interrogative obligation we
fulfil against parsed text. `"How many red apples?"` reads the `(apple,{red})` descriptor and counts
matching instances via the subset lattice; an unknown word makes it ASK, not guess. The reasoning
loop closes: same reader, same obligations, now producing an answer. 0 model calls throughout ·
14. **CLI** — `python -m reasoning.reading "<text>"` (analyze) / `… --about "<phrase>" …` (answer);
logic in a testable `run_cli` · 15. **`instances(statement)`** — context is now PARSED, not
hand-fed: `"There are three red apples"` → three `(apple,{red})` instances (a number is a quantifier
with a VALUE). The reader that answers questions is the same one that ingests facts ·
16. **yes/no questions** — `"Did the student pass?"` checks a `Predication((student,{}), "pass")`
against context; tense is lemmatized so `"passed"` == `"pass"`. `learn(statement)` is the one
ingestion door (count / predication / bare phrase); `answer()` dispatches on the interrogative kind ·
17. **syllogistic inference** — `"All students passed" + "Bob is a student" ⊢ "Did Bob pass?" = yes`,
by **universal instantiation**: a capitalized unknown word is a proper noun (named individual);
`"X is a Y"` is a `Membership`; the answer is *derived* (subject → its collections → predicate),
not looked up. Lookup became reasoning, still 0 model calls ·
18. **paragraph mode (CLI)** — one string is split into sentences; the statements are `learn`ed and
the trailing question answered, so `"All students passed. Bob is a student. Did Bob pass?"` → `yes`
works directly (no `--about` needed) ·
19. **`--trace` + `explain()`** — exposes the analyzed data and reasoning: per statement/question,
the parsed units (senses + obligation bindings), the reversible decision trail, the facts learned,
and the derivation behind the verdict. Surfacing the trail caught a hard-coded `oblige` reason
(every obligation said "quantifier ranges over a collection") — now reports the real need ·
20. **membership questions + Barbara syllogism** — `"Is X a Y?"` (membership question) and
`"Every X is a Y"` (a `Subset` fact). `_is_member` checks membership by reachability over `Subset`
edges (transitive), so "Every teacher is a person. Bob is a teacher. Is Bob a person?" → yes.

A **collection descriptor** = `(head noun, set of atomic attribute predicates)`. `apple`=`(apple,{})`,
`red apple`=`(apple,{red})`, `rotten green apple`=`(apple,{green,rotten})`. Sub-collection = same
head, attribute-superset; a sub-group's count = instances whose attributes ⊇ the query's. We store
the shape now (free); aggregation/counting is *derived later* when a question's obligation needs it.

## Open threads / next

- **Give `boolean` (and other deferred senses) their obligations** — now unblocked: disambiguation
  by coverage means `no`-as-quantifier no longer wrongly inherits `boolean`'s demand. `boolean→`
  (negates a clause?) can be added and will only apply when that reading covers best.
- **"Forego when subordinated"**: a noun used as an object shouldn't independently demand a
  `predicate`; refine so object-nouns don't read as false fragments.
- **Open-class words** to add as we meet them (probe surfaced: verbs `sells`/`bought`/`failed`,
  nouns `car`/`dog`, etc.) — each with a `VERB_LEMMA` entry where needed.
- **Negation**: `"not"` (`"Bob did not pass"`), negative universals (`"No students failed"`).
- **Predicate adjectives**: `"The apple is red"` / `"Is the apple red?"` — copula + adjective as a
  property fact, distinct from membership.
- **wh-questions**: `who` / `what` / `which` — each a new interrogative obligation.
- **Comparatives**: `"taller than"` — a relational fact (the deferred relations thread).
- **Verb lemmas** are a small hand-list (`VERB_LEMMA`); grows by deliberate addition like the other
  dictionaries (no stemming guesswork).
- **Overlapping facts**: count statements are currently assumed disjoint (subset-matching sums
  them). "3 red apples" + "1 rotten red apple" would double-count — needs an inclusion model.
- **Aggregation beyond count** (sum, max, "which has the most") over collection descriptors.

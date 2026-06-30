# Logical-reading playbook

Strategies for building **model-free natural-language logical reasoning**, bottom-up and word by
word (`reasoning/reading.py`, `reasoning/tests/test_reading.py`). A living document — updated as
strategies stabilize. See also [[logical-reading-initiative]] and
[[working-method-piecemeal-annotate]] in project memory.

## THE CORE PRINCIPLE — every word is an obligation

**Word obligation is the PRIMARY feature of this parser. Everything else exists to serve it.**

A word does not passively "have a meaning" — it carries a **demand on the rest of the sentence**:
something it must FIND, by binding, to be satisfied. A quantifier must find its collection; a verb
its subject; an article its noun; a preposition its object; a negator the verb it scopes. **Parsing
IS the discharge of these obligations** — there is no separate grammar. Everything we do is a view
of one mechanism:

- **Structure** = the bindings that discharge obligations.
- **Sense disambiguation** = choose the sense-assignment whose obligations cover best.
- **Deficiency / "ask, don't guess"** = an obligation left unmet (the answer-or-ask boundary).
- **Reasoning** = facts read off the saturated obligation graph.

**The non-negotiable rule when adding ANY word or capability: give the word an obligation and let it
BIND. Never special-case a word with a global flag or an ad-hoc check** — that is a corner being
cut, and it *will* produce a silent bug. The obligation is what *scopes* a word to a specific
target; a global flag cannot.

> **Cautionary lesson (increment #26).** `"not"` was first handled as a global "is a negator
> present?" flag instead of an obligation. Because nothing *bound* it to a specific predicate,
> `_membership` silently ignored it and `"Bob is not a student"` was learned as a *positive*
> membership — a confident false `"yes"`. Giving `"not"` a `negates` obligation that binds to its
> verb fixed the design and the bug at the same time. When a fact comes out wrong, suspect a word
> that isn't carrying (or binding) its obligation.

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

4. **Every role carries an obligation; the unmet ones are what we still track.** *(This is THE core
   principle — see the section above; it is restated here because it is the move we reach for most.)*
   An obligation belongs to a *role*, not a word: a quantifier ranges over a collection; a linking
   preposition relates an `anchor` (left) to an `object` (right) — by virtue of their roles. Attach
   unmet; binding **drains** the set. A statement is **well-formed (saturated)** when nothing is unmet and
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
edges (transitive), so "Every teacher is a person. Bob is a teacher. Is Bob a person?" → yes ·
21. **negation** — `Predication.negated` polarity, set at INGESTION when a negator (`not`/`never`/
`no`) is present, so `"Bob did not pass"` / `"No students failed"` are never learned as positives.
The positive query then doesn't match → `"no"`, and the false-positive (the real danger) is avoided ·
22. **predicate adjectives** — `"The apple is red"` is a property fact (predicative adjective, didn't
merge), distinct from attributive `"red apple"`. `"Is the apple red?"` queries it; reuses
`Predication` with the adjective as predicate ·
23. **wh-questions** — `"Who passed?"` returns the subjects with that predicate; `"What is Bob?"`
returns Bob's collections. These return ENTITIES, not yes/no. (`which` over instances still queued.) ·
24. **comparatives** — `"Alice is taller than Bob"` is a `Relation` fact; comparisons are
TRANSITIVE (`_related` walks the edges), so A>B>C ⊢ A>C. The first true relation (not set membership) ·
25. **open-class vocabulary** — a curated batch of nouns/verbs/adjectives (with plurals + lemmas),
grown deliberately like every other dictionary, so common sentences parse and reason ·
26. **negation is an obligation** — `"not"`/`"never"` carry a `negates` obligation that BINDS to the
verb/copula they scope (no longer a global flag). Polarity then attaches to the right fact, so
`Membership` gained a polarity and `"Bob is not a student"` is no longer a positive membership (fixed
a silent false-`yes`) ·
27. **(alignment) scoped `"no"` negation** — `"no"`-quantifier negation now reads the BINDINGS too:
a verb is negated iff its `subject` is the collection a `"no"` quantifier bound to — not "is `no`
present?". Removes the global flag (latent multi-clause over-negation). Behavior-preserving today ·
28. **(alignment) the copula binds a `complement`** — `is`/`are`/`was`/`were` now carry BOTH a
`subject` and a `complement` obligation (inversion-aware: subject left in statements, right in
fronted questions). `_membership`/`_property`/`_subset` read those bindings instead of COUNTING
nouns/adjectives — the keystone that dissolved the unit-counting from the fact-extraction layer ·
29. **(alignment) comparative + wh obligations** — a `comparative` is the copula's complement and
binds its `standard` (after "than"), so `_relation` reads off bindings; `who`/`what` bind to what
they ask about (`asks_predicate` → the verb / `asks_individual` → the name). The last counting is
gone; every meaningful role now BINDS. (`_find_target` also simplified to compute pools by role.) ·
30. **`some` ≠ `all` (+ multi-hop)** — a `Predication` is `universal` only when its subject is bound
by `all`/`every` (read from the bindings); universal-instantiation uses ONLY universal facts, routed
through `_is_member` (transitive Subset). So `"Some students passed"` no longer instantiates to Bob
(probe bug B — a false `yes`), AND multi-hop `"all teachers are persons; every person passed"` ⊢
"Bob (a teacher) passed" works (probe bug C) — the same fix closes both ·
31. **the "undetermined" verdict** — a yes/no question is now `"yes"` (derivable true), `"no"`
(derivable FALSE — an explicit `"Bob did not pass"` or universal negative `"No students passed"`,
or comparative asymmetry), or **`"undetermined"`** (we lack the information). No more closed-world
`"no"`: `"Some students passed. … Did Bob pass?"` → `undetermined`. Applies to all yes/no branches
(predication / membership / property / comparative). This IS the tool's contract: answer, decline,
or say precisely what can't be concluded ·
32. **universal property instantiation** (probe bug E) — a property IS a `Predication` (adjective as
predicate), so `_property` carries the `universal` flag and the property question routes through
`_predication_verdict`. `"Every student is happy. Bob is a student. Is Bob happy?"` → `yes`; `"Some
…"` → `undetermined`. The bug-C fix, one layer up (verbs → adjectives) ·
33. **negative subset / disjointness** (probe bug F) — `"No X are Y"` → `Subset(X, Y, negated=True)`
(disjoint); a member of X is then definitely NOT a Y (`_disjoint` check) → a definite `no`. Surfaced
a deeper fix: a quantifier's `collection` is now its **nearest noun to the right** (its head), not
"the unique noun anywhere" — which had silently broken on objects/complements and mis-disambiguated
`"No"` to `boolean`. (`_is_member` refactored via `_collections_of`.) ·
34. **contradiction detection** (probe bug D) — a 4th verdict, **`"conflict"`**, when BOTH `yes` and
`no` are derivable (`_verdict(yes, no)`). Detection is DERIVABLE: it counts inference, so `"All
students passed. Bob did not pass. Bob is a student."` is a conflict, not just direct `P`∧`¬P`. The
four verdicts now span the epistemic space: **yes / no / undetermined / conflict** = true / false /
unknown / inconsistent. Applies to predication, property, membership, comparative branches ·
36. **`explain()` for all question types** — property and comparative questions now derive in
`--trace` (a shared `_verdict_steps` helper), instead of falling through to "not a question I can answer" ·
37. **transitive verbs / objects** — a verb carries an OPTIONAL `dobj` obligation (binds the noun to
its right; intransitive verbs leave it unmet without a deficiency, via `Obligation.optional`).
`Predication` gained an `object`, so "Mary owns a dog" ≠ "owns a cat"; "Does Mary own a dog?" → yes,
"… a cat?" → undetermined ·
38. **forego when subordinated** — a unit that FILLS another word's slot (a verb's `dobj`, a copula's
`complement`) has its own obligations foregone in `deficiencies()`, so "Mary owns a dog" / "Bob is a
student" / "The apple is red" are *understood*, not false fragments — while a real noun phrase
("Each of the students", no predicate anywhere) is still flagged ·
39. **wh-object** — "What does Mary own?" returns the direct OBJECTS of (Mary, own) (the `"what"`
branch detects the auxiliary+verb pattern); "What is Bob?" still returns memberships ·
40. **which-questions** — "Which <noun> <predicate>?" returns the named members of that noun that
satisfy the predicate (action verb or predicative adjective). "Which students passed?" → the
students who passed (a teacher who passed is excluded) ·
41. **aggregation (count comparison)** — "Are there more X than Y?" compares the counts of two
collections over the instance context via the subset lattice. "more"/"fewer" with "than" ·
42. **arithmetic (math step 1)** — toward replacing the LLM word-problem translator with this parser.
A digit token is a `value` (with a numeric `value` feature); operator words (`plus`/`minus`/`times`/
`divided`) carry an `op` and bind `left`/`right` operands (same shape as the comparative's `standard`).
"What is 3 plus 2?" → 5, built into a one-line program and run through `translate.run_program` (the
audited safe evaluator — the model is gone, the correct-by-construction arithmetic stays) ·
43. **Quantity facts (math step 2)** — "Tom has 3 apples" → `Quantity("tom", (apple,{}), 3)`: a value
gains a `quantifies` obligation (binds the noun to its right; OPTIONAL so a bare operand quantifies
nothing). "How many apples does Tom have?" reads the amount back (entity fronted after the auxiliary);
an unknown entity → `undetermined`, not a fabricated count.  ·
44. **aggregation across entities (math step 3)** — "How many apples do Tom and Mary have
altogether?" collects the conjoined entities (names between the auxiliary and the verb) and folds
their `Quantity` amounts (sum). Multiple entities ⇒ sum by default; `altogether`/`total` reinforce.
An unknown conjunct → `undetermined`. NOTE: `and`/`altogether` are recognized but don't yet carry
obligations (positional collection for now) — a principled conjunction-binds-conjuncts refinement
is queued for when conjunction appears in more roles (objects, "3 apples and 2 oranges").  ·
45. **accumulation on one entity (math step 4)** — repeated `Quantity` statements for the same
entity+measure ACCUMULATE (signed sum): `Quantity` now carries its `verb`, and `_verb_sign` gives
the polarity (`LOSS_VERBS` = lose/sell/give/spend/eat/use/drop → −; everything else incl. the `have`
baseline → +). "Tom has 3 apples. Tom buys 4. Tom gives 2 away." → 3+4−2 = 5. The first arithmetic
INTENT carried by a verb rather than an operator word — the bridge to GSM8K phrasing. `LOSS_VERBS`
is the human-owned semantic knob.  ·
46. **coreference as an obligation (math step 5a)** — a pronoun (`he`/`she`/`it`/`they`) carries a
`refers` obligation, discharged CROSS-SENTENCE against the discourse (entity names threaded through
`learn`, most-recent last): `_pick_antecedent` takes the most recent gender-compatible entity
(`PRONOUN_GENDER` × `NAME_GENDER`), bound to a `Referent` stand-in and logged as a reversible Decision
(shown in `--trace`). `_entity()` reads the resolved name into facts. "Tom has 3 apples. He buys 2."
→ 5; "Mary … Tom … She buys 2" → She=Mary (gender skips the nearer Tom). A pronoun with no antecedent
is just an unmet obligation → honest clarification, not a guess. (Subject resolution now accepts a
pronoun.) Still singular-only; `they`/object pronouns and relational quantities (5b) are next.  ·
47. **value operand = forego, not optional** (principle correction, prompted by review) — a value's
`quantifies` was marked `optional`, a flag-shortcut the core principle forbids. Truthful model:
`quantifies` is REQUIRED but FOREGONE when the value is an operator's operand (forego-when-subordinated,
#37 — `left`/`right` join `complement`/`dobj` as subordinating roles). "3 plus 2" stays well-formed
(operands excused); a STRAY value "Tom has 3" is now honestly flagged ("what does 3 count?"), which the
`optional` shortcut had silently swallowed. Lesson: reach for the existing obligation mechanism
(subordination) before adding an exception flag.  ·
48. **value quantifies a verb's frequency (multipliers)** — a value can measure an ACTION, not only a
noun. `once`/`twice`/`thrice` are lexicalized multipliers (value + frequency) carrying a `scales`
obligation that binds the verb to their LEFT → a `Frequency(entity, action, count)` fact. "Mary jumps
twice" → Frequency(mary, jump, 2); "How many times does Mary jump?" → 2. `times` is now multi-sense
(`operator` | `noun`): the collision resolves by coverage — values flanking it ("4 times 3") tie and
fall to the operator by sense-order, a verb-to-bind with no second value ("how many times … jump")
wins as the noun. SIDE-FIX: `_unmet_count` (disambiguation cost) now reuses `deficiencies` so coverage
and the answer-or-ask boundary share ONE notion of "unmet" (optional/subordinated foregone) — the
operator reading no longer loses to a `times`-as-noun that opportunistically fills a copula's slot.  ·
49. **relational quantities (math step 5b)** — "Mary has twice as many" is a DERIVED `Quantity` =
`factor × referent`, the referent resolved like coreference but pointing at a prior QUANTITY (the
most recent in `context`, threaded into `learn`). The multiplier now `scales` a quantifier to its
RIGHT ("twice as MANY") — else a verb to its left (frequency); the referent supplies BOTH the measure
and the amount. The relational STRUCTURE (multiplier-on-quantifier) is committed-or-declined: no
referent → `[]`, never a bare predication. Composes: "Tom has 4 apples. He buys twice as many." → 12
(coreference + relational + signed accumulation, zero glue). `half`=×0.5. Refinements queued: explicit
referent ("as many as Tom"), same-measure referent selection, "many"'s collection → referent measure.  ·
50. **question-relative polarity (events vs. holdings)** — a quantity's SIGN depends on the QUESTION,
not just its verb. "How many does X HAVE?" folds holdings (baseline + gains − losses, via
`_verb_sign`/`LOSS_VERBS`); "How many did X SELL?" COUNTS the matching-verb events (positively,
filtered to that verb). The same `Quantity(..., sell)` is −amount toward holdings but +amount toward a
sales count — the fact is neutral; the query interprets it. Unblocks the most common GSM8K shape and
gets **GSM8K problem #1 (Natalia) end-to-end at 0 model calls**: "Natalia sold 48 clips. She sold half
as many. How many clips did Natalia sell?" → 72 (coreference + relational quantity + event-count).  ·
51. **GSM8K coverage harness** (`reading_bench.py`) — an INDEPENDENT eval over `reading.solve`,
firewall-respecting (DEV inspectable, TEST aggregate-only; never in the unit tests). Honest baseline:
**0% on real GSM8K** — the clean shapes the parser handles ("Tom sold 8 apples") are far simpler than
real problems ("Natalia sold clips to 48 of her friends in April"). The value is the decline-blocker
taxonomy + the most-frequent-unknown-word list, which turns "what next?" from a guess into a measurement:
top blockers are FUNCTION-WORD vocabulary (`to`/`in`/`for`/`on`/`per`/`with`/`from`…), then currency
(`$`, ~32%) and percentage (~8%). Lesson: the gap analysis guessed "the hardest gaps are coreference &
relational referents" (true, and now built) — but the FIRST wall is mundane: the parser can't even
recognize the function words to attempt the structure. Measure before filling.  ·
52. **function-word vocabulary (data-driven) + a precision fix** — added a curated, correctly-TYPED
batch from the harness's unknown-word frequency list: content nouns (time/money/measure + plurals),
common prepositions (`to`/`in`/`on`/`at`/`from`/`with`/`for`/`after`), and possessive/object pronouns
(`his`/`her`/`him`/`them`/`their`/`its` — which also extend coreference). DEFERRED the
construction-dependent words (`per`→rates, `left`/`remaining`→subtraction, `will`/`can` modals,
`that`/`then`/ordinals) rather than tag them empty. Effect: the unknown-word frequency list shifted to
that next tier (the loop working), coverage stayed ~flat (structure still blocks — as predicted). The
batch EXPOSED a precision leak — an unblocked partial parse reached the count-instances fallback and
fabricated a confident `0`. Fix: no instance data to count → `undetermined`, never a fabricated 0.
Wrong dropped 1→0, **precision back to 100%**. Lesson: coverage is a vanity metric here; precision is
the safety metric, and unblocking parses must not come at its expense.  ·
53. **left/remaining subtraction** — "how many are left / remaining" is NOT a new operation: it's the
HOLDINGS question (baseline + gains − losses, `_holdings`) in another surface form. The subtraction was
already there (loss verbs); the remainder words just ASK FOR THE NET. Form A ("does X have left")
routes through the existing holdings fold once `left`/`remaining`/`rest`/… are recognized; Form B
("are left", no aux) folds the implicit UNIQUE owner (ambiguous → decline); the measure can be implicit
too ("how many are left" → the latest quantity's measure). DISPATCH FIX: lifted the `has_interrogative`
guard above the copula yes/no branches so a "how many … are <adjective>?" count is no longer stolen by
the property branch ("are apples left?") — a latent ordering bug. Composes with loss-verb polarity; 214
tests; precision stays 100% on the dev slice.  ·
54. **currency + the precision GUARD (a key lesson)** — "$12" is a self-measuring VALUE (12 of the
`MONEY` measure; the "$" discharges its `quantifies`, so no dobj noun needed). "X verb $N" → a money
`Quantity`; "how much" routes like "how many" with the measure fixed to MONEY, so holdings/event-count
polarity composes. BUT currency unblocked PARTIAL parses of complex multi-step problems, which
committed to confident-WRONG numbers — precision crashed 100%→17% (5 wrongs). The fix is the
philosophy made literal: **`solve` now DECLINES if any word in the problem is unrecognized** — a
partial parse may have silently dropped the fact that mattered, so "definitive-or-decline" applies at
the whole-problem level, not just per-word. Wrongs → 0, precision → 100%. Consequence: coverage only
rises when a problem is FULLY readable — so `reading_bench` now reports **fully-read** (every word
recognized; the vocabulary frontier and the ceiling for coverage). Honest baseline: **0/300 fully-read**
— GSM8K's vocabulary is vast; the currency capability is correct (unit-proven) but real problems stay
blocked. Lesson: a new capability that unblocks parsing can REGRESS precision by enabling fragment
answers; guard commitment on complete understanding.  ·
55. **per → rates (money)** — "$12 per hour" is a `Rate(entity, amount, measure, per_unit, verb)`:
"per" binds the rate value (left) and the unit noun (right). The TOTAL isn't stored — it's composed at
answer time as `rate.amount × _holdings(entity, per_unit)`, slotting into the existing event-count fold
as a FALLBACK (no direct money quantity → try a rate). "Weng earns $12 per hour. Weng works 3 hours.
How much did Weng earn?" → 36. Declines when the per-unit quantity is unknown (the per-unit `_holdings`
→ None). Because it reuses `_holdings`, a rate composes with accumulation for free (hours add up first).
Same lesson as polarity: facts stay neutral, the query computes. Precision stays 100%; "per" leaves the
unknown frontier. Slice 1 is money; "N items per unit" (non-money rate measure) is the follow-on.  ·
56. **percentage % (arithmetic)** — "20%" is a fraction (0.20; a `percent` sense parsed before `value`
so "%" doesn't read as a plain number) carrying a `base` obligation: "What is 20% of 50?" → 0.20 × 50 =
10 (a third arithmetic answer-branch beside operators and rates). "% of them / of a holding" (referent)
is the follow-on. HARNESS HONESTY: with `$` and `%` now PARSING, they're no longer blockers — so
`reading_bench` dropped them from the decline taxonomy (a capability must stop being counted as a gap,
or the metric flatters progress). The taxonomy is now stark: **300/300 declines are unknown-word** —
not one problem is even fully-read. Vocabulary breadth is the universal wall; the frequency list is the
worklist. Precision stays 100%; coverage 0%.  ·
57. **vocabulary grind (safe) + two structural-confidence guards** — added ~60 frequency-list words in
correct roles (modal auxiliaries `will`/`can`; common verbs `make`/`pay`/`take`/`go`/…; nouns
`store`/`bill`/`marble`/…; ordinals/adjectives; adverbs/conjunctions/prepositions). This moved
**fully-read 0→12 (4%)** — but full readability REMOVES the unknown-word guard, so 2 of the 12 misparsed
into confident-WRONG (precision 100%→0%). The grind is only safe with structural-confidence signals, so
`solve` now has THREE precision layers: decline what we can't (1) READ (unknown word), (2) ACCOUNT FOR
(**unused-given**: every stated data-number must surface as a fact amount — "Oliver has 10 $20 and 3 $5
bills" dropped three), or (3) read without MISREADING a relation (**lost-comparison**: a `than` premise
that collapsed to a flat Quantity, no Relation — "25 times more stickers than Kristoff"). Precision back
to 100%; clean problems still solve. Lesson: each new capability that unblocks parsing can regress
precision; guard commitment on completeness, not just recognition. Nonce-pin unknown-word tests
(`wuggle`) so they survive the grind.  ·
58. **comparative quantities (diagnosed from "how much more money does Oliver have than William?")** —
"A has N more/less X than B" and "how much more X does A have than B?" → a comparison off the NAMED
standard. Surfaced a LATENT BUG: the `object` resolver was noun-only (`right("noun")`) and merged with
`determines`, so prepositions NEVER bound names ("than Bob", "of Mary" all silently failed). Fix: split
them — `object` binds noun OR name; `determines` stays noun-only. Also fixed `standard` to resolve
THROUGH `than` (skipping the measure: "more MONEY than WILLIAM" → William, not money). Question form →
`holdings(A) − holdings(B)` (`less`/`fewer` → B−A); statement form "Tim has 30 less apples than Martha"
→ a derived `Quantity` = `holdings(Martha) ± N` (`_comparative_quantity`, sequential referent only —
forward refs decline). Currency words (`dollar`/`cent`/`money`) normalize to one `MONEY` measure so "$30"
/ "30 dollars" / "how much money" agree. Precision guards REFINED, not just reused: unused-given exempts
comparative/multiplier premises (their value is a consumed offset/factor, not a stored amount), and
lost-comparison checks the amount is a VERBATIM stated value (a derived ± is fine). 237 tests; precision
100%; the "?" the diagnosis started from was a non-issue (`_key` strips it; lossless surface is by design).  ·
59. **fronted-question argument binding (diagnosed from "have ['verb'] subject->?, dobj->?")** — the
verb's subject/object weren't binding because the resolvers only knew DECLARATIVE order (subject-left,
object-right). A fronted question rearranges it: `<aux> SUBJ <verb>` with the wh-OBJECT fronted before
the aux. Fix: the `subject` resolver takes the noun/name/pronoun BETWEEN the auxiliary and the verb
(disambiguating "does OLIVER have" from the fronted "money" — the very ambiguity that left it unmet);
the `dobj` resolver, when no noun is to the right, takes the noun fronted before the auxiliary ("how
much MONEY does Oliver have" → money, NOT "than"/William — `than` is a connective, and `more`'s
`standard` already carries the comparison). Added `standard` to the subordination set so the comparison
reference's predicate is foregone. "How much more money does Oliver have than William?" now fully
SATURATES (no deficiencies). The answer was already correct — what changed is the PARSE binds via
structure instead of a positional sidestep. 242 tests. Lesson: bind via the obligation/structure, not a
positional shortcut in the answer (the core principle); fronting is a structural cue the resolvers can read.  ·
60. **answer reads the parse's bindings (`_question_subjects`)** — the count/quantity, frequency, and
comparative answer branches each had their own POSITIONAL rescan for "the entity after the aux"; with
the verb's `subject` now binding structurally (#59), they all route through one helper that reads
`_follow(verb, "subject")` instead. This closes the loop the core principle wants: the parse discharges
obligations by binding, and the ANSWER consumes those bindings — no re-deriving the same entity three
slightly-different ways (a drift source). The one honest fallback is a CONJOINED subject ("Tom and
Mary"): the single-target `subject` obligation can bind only one unit, so a group still needs the
positional collection — a precise marker of a missing capability (conjunction-binding into a group),
not a shortcut. 242 tests; precision unchanged. When conjunction-binding lands, the fallback disappears.  ·
61. **conjunction binding ("Tom and Mary" → a group)** — the bind-vs-merge call: a coordinated subject
acts as ONE compound subject, so "X and Y" (names) CONDENSES into a single group unit carrying both in
`members` — a collective, not a fusion (reversible, lossless: the merge records its constituents). The
payoff cascades: the group is one unit, so the `subject` obligation binds it UNIQUELY — the very
ambiguity (Tom vs Mary) that forced the positional fallback dissolves. `_question_subjects` reads
`subject.features["members"]`; the fallback (now flattening groups, for comma-lists) is no longer hit
for "Tom and Mary". Three-way "X and Y and Z" stacks through the multi-pass condense loop, like
"big red apple". A group-subject STATEMENT ("Tom and Mary have 5 apples") is ambiguous
distributive-vs-collective → declines (the precision discipline applied to a semantic ambiguity).
247 tests. Follow-on: conjoined OBJECTS ("3 apples and 2 oranges" — two quantities, not a group).  ·
62. **conjoined objects ("3 apples and 2 oranges" → two quantities)** — `_quantity` now returns a LIST,
one Quantity per value-noun pair (or per money value), so a conjoined object yields several; "how many
apples / oranges does Tom have?" → 3 / 2; composes with accumulation. PRECISION (the Oliver-bills
problem, 3rd encounter): capturing ALL values DEFEATED the unused-given guard (every number became a
fact), so "10 $20 bills" (ten twenty-dollar bills, a compound denomination) slipped to a confident-wrong.
Fix is structural at the source, not another whole-problem guard: two ADJACENT bare values ("10 $20")
are a compound we can't parse — a genuine conjoined object always SEPARATES its values with nouns + `and`
— so decline. Lesson: a capability that captures more can capture it wrong; the durable defense is to
recognize the shape you CAN'T parse and refuse it, not to keep patching the shapes you can.  ·
63. **multi-step chaining (a solve() FIXPOINT)** — instead of requiring premises in dependency order,
`solve` DEFERS a statement it can't yet resolve and RETRIES it as the context fills, iterating to a
fixed point. This turns the sequential engine into a constraint-propagating one: a FORWARD REFERENCE
("Tim has 30 less than Martha. Martha has 68 …" — base stated last) resolves, and chains
(Harry←Tim←Martha) resolve hop by hop. Three things made it precision-safe: (a) `_facts_from` DEFERS an
unresolved comparative-quantity structure (returns [] / pending) rather than falling through to a FLAT
`_quantity` that would fabricate "Tim=30"; (b) `_comparative_quantity` REFUSES "N times more" (it looks
additive like "30 less" but is multiplicative — the Riku trap; refusing means the fixpoint never
resolves it → declines, never the plausible-wrong 85+25); (c) relational referents resolve by NAME
("half as many as TIM" → Tim's quantity, not "most recent" — makes the chain robust, not order-lucky).
A test flipped decline→38 (the capability landing). 256 tests; precision 100%. Follow-on: multi-CLAUSE
sentences (comma-joined "Tim …, and Harry …" is one premise — needs clause splitting).  ·
64. **clause splitting → FIRST real GSM8K solve** — `_split_clauses` splits a sentence into independent
clauses joined by `;`/`, and`/`and`/comma, but ONLY where BOTH sides have their own verb (so a
clause-joining "and" splits, a phrase-joining "and" — "3 apples and 2 oranges", "Tom and Mary" — does
not; same structural-honesty discriminator as everywhere). `solve` splits all sentences into clauses
and picks the `?`-clause as the question, so a leading "If P, Q?" premise embedded in the question
sentence becomes its own premise. This feeds the fixpoint, and **the first real GSM8K problem solves
end-to-end at 0 model calls**: "Tim has 30 less apples than Martha, and Harry has half as many apples as
Tim. If Martha has 68 apples, how many apples does Harry have?" → 19 — composing clause-split +
comparative + named-relational + forward-ref + fixpoint + holdings, ZERO glue. Coverage 0→0.3% (1/300),
**precision still 100% (0 wrong)** — a benchmark point earned without lowering the bar to commit.  ·
65. **verb elision ("grandparents twice as much" borrows "gave")** — a clause joined by `, and` that
LACKS a verb but has a SUBJECT before its quantity ("Mary 5 apples", "her grandparents twice as much")
inherits the prior clause's verb. `_split_clauses` now splits when the left has a verb and the right is
an `_is_elided_clause` (a name/noun BEFORE its first value/multiplier — which distinguishes it from a
bare conjoined object "2 oranges", quantity-first/no-subject); `_supply_verb` inserts the left's last
verb after the right's subject and re-analyzes. SIDE-FIX: "how much" is a MONEY question UNLESS a mass
noun sits IMMEDIATELY after it — "how much did GRANDPARENTS give" had bound `much→collection→grandparents`
(the subject) and asked "count grandparents"; the adjacency check routes it to money. 261 tests;
precision 100%. The synthesis is guarded (only fires on a real elided clause), so no confident-wrongs.  ·
66. **vocabulary push + GUARD 4 (number → fact)** — added ~150 data-driven words in correct roles
(demonstratives `that`/`this`/`those` → article; temporal `before`/`during`/`while`/`when`;
comparatives `younger`/`faster`/`cheaper`; common verbs `decided`/`increases`/`weighs`; units
`feet`/`inches`/`kilograms`; nouns `bank`/`weight`/`average`/…). `fully-read` rose 12→20 (4%→6.7%). But
8 more readable problems exposed 2 confident-wrongs (Bobby's "increases by 50%" → no fact, answered from
baseline; Colton's "three friends" → spurious instance-tuples). GUARD 4 closes it, complementing GUARD 3:
a premise that STATES a number (value/percent/multiplier) but produces NO quantity fact dropped
information → decline. Precision back to 100%. The safe grind: vocabulary lifts `fully-read` (the
ceiling), structure determines `coverage`, and the four guards keep the gap always DECLINED. Side-fix
("how much" adjacency, #65) was needed because the new subject nouns let "much" mis-bind.  ·
67. **each/per distribution** — "gave 4 stickers EACH to 3 friends" → 4 × 3 = 12. "each" after a
value-noun makes that per-recipient amount MULTIPLY the recipient count (the other value-noun), instead
of `_quantity` producing two separate conjoined quantities. `_distributed_quantity` runs before
`_quantity`; the unused-given guard exempts "each" premises (the two values multiply into a product, so
they don't appear verbatim — same exemption as comparative/multiplier). Precision 100%. Real "each"
GSM8K problems (Colton) are multi-step and still decline via the guards. Follow-on: "each box has 5
apples" (each BEFORE the noun).  ·
68. **percentage-of-a-holding + compounds + question/group coreference** (coverage 1→2). (a) "puts 20%
more money" → `_percent_quantity` = 0.2 × the matching prior Quantity (a percent with NO base value —
its base is the referent). (b) MULTI-WORD COMPOUNDS: "piggy bank" merges into ONE indivisible unit
(`_merge_compounds` before analysis; `COMPOUNDS` = lexicon keys with a space) — the original design's
"indivisible units". (c) The QUESTION's pronouns now resolve against the discourse (`answer(…, discourse)`),
so "how much does SHE have" works — which made the real Susie solve. (d) That exposed a wrong: a plural
"they" had resolved to ONE entity ("how many do THEY have in total" → just the last) — now "they"/"them"
resolves to the GROUP of all prior entities (a `members` list, reusing the conjoined-subject machinery),
so Paddington (180+140) → 320. **Second real GSM8K problem solves.** Precision held at 100% throughout
(the group fix turned a confident-wrong back into a correct). 269 tests.  ·
69. **age / "how old" questions** — map onto the comparative machinery: "X is N years old" → an age
`Quantity` (measure `year`, verb `be`); "X is N years younger/older than Y" → a comparative age
(`COMPARATIVE_SIGN`: younger/shorter/lighter… → −, older/taller… → +, replacing the hardcoded
less/fewer); "how old is/will X be?" reads the year-holdings (entity via `_question_subjects` or the
copula subject). FIX: `_membership` now skips when a VALUE is present ("Kris is 24 years old" is a
QUANTITY, not Membership(kris, year)). PRECISION: a future/past SHIFT ("how old will X be NEXT YEAR")
declines — caught by checking the shift word's ATTRIBUTE too, since "next year" condenses to year{next}.
Precision 100% (the shift check turned a confident-wrong back into a decline). 274 tests.  ·
70. **relational temporal foundation + the read–SYNTHESIZE loop** (a NEW architecture, user-designed).
Two ideas. (a) TIME is RELATIONAL, not calendar: a time reference (`monday`..`sunday`, `yesterday`/
`today`/months) is a NODE (lexicon sense `time`); ORDER is tracked as separate `Before(earlier, later)`
facts read off the TEXT, never a calendar rank — days WRAP ("Tue → Sat → Mon" makes that Mon later than
that Tue). Driven by OBLIGATIONS: a `time` word carries `associates` (the value it timestamps — nearest
verb, reusing the `negates` resolver; `Quantity` gains `time` from this binding) and `orders` (its order
vs another time, bound only when an after/before cue is present; direction read at fact-time). (b) The
read–synthesize LOOP: reading BINDS what's present; some obligations have no word to bind (target exists
only by IMPLICATION) → a `synthesize(facts)` driver derives implied facts to a FIXPOINT, dispatching a
SYNTHESIZER REGISTRY (`SYNTHESIZERS`, keyed by fact kind; outstanding obligations are the agenda). First
synthesizer = transitive ordering (`_before` closure: a<b ∧ b<c ⟹ a<c, iterated over rounds → multi-step
reasoning front-loaded). `solve()`'s `while progress` fixpoint is GENERALIZED to call it (one loop). NEW
capability = REGISTER a synthesizer, never a one-off check. PRECISION: making `Monday` a time removed an
ambiguous-subject block, exposing Buddy ("Mon 30 + Wed 12", dropping the Tue halving → 42 wrong) → GUARD 5
declines any problem whose value@time facts span ≥2 distinct times (time-aware FOLDING/querying is the
next increment; the foundation only tracks the facts). Precision 100%; 284 tests. Memory:
`temporal-model-relational`. Next: timeline assembly + time-aware querying (state vs change).  ·
71. **timeline assembly + time-aware querying** (turns #70's GUARD-5 decline into an ANSWER). STATE vs
CHANGE: a STATE verb (`STATE_VERBS` = have/be) SETS the running value at a time (a known total); every
other verb is a CHANGE (signed delta). `_holdings_at(entity, measure, context, at_time)` folds along the
timeline — events at-or-before `at_time` (timeless = always-on baseline; strictly-after excluded; UNKNOWN
order vs the query → None/decline), linearised by `_time_order` (insertion sort over the `Before` closure;
None if any pair incomparable), then state-sets / change-adds in chronological order. A SECOND synthesizer
`_synth_narrative_order` (registered) gives the default "stated-in-sequence ⇒ chronological" — consecutive
distinct value@time nodes → `Before`, UNLESS already ordered (explicit/transitive order always wins, never
contradicted). `answer()`'s holdings branch: a HELD-question carrying a time → `_holdings_at`. GUARD 5
reworked: (a) a TIMED premise stating a value/multiplier but yielding no Quantity dropped a time-tagged
event ("loses half of them" → a misparsed Frequency) → decline; (b) ≥2 times but the question names none →
ambiguous → decline. So "On Mon Tom has 30 apples. On Tue Tom buys 5. How many on Tue?" → 35 (state 30 +
change 5); on Mon → 30; Buddy still declines (its Tuesday Frequency). Precision 100%; 291 tests. Next:
relational losses ("loses half of them" → 0.5 × held, not a Frequency) would let Buddy-class problems solve.  ·
72. **relational losses / partitives ("loses half of them")** — the partitive `of` is the signal: a
multiplier FOLLOWED BY `of` ("half **of** them") is a FRACTION of a set, not a verb multiplier (frequency).
`_partitive_quantity` (runs before `_frequency`, gated on `<mult> of …`): fraction × the current holding
of the referent measure — a noun after `of` ("half of the apples"), else a pronoun ("them"/"it") → the
subject's most-recent measure; verb-lemma sets gain/loss; carries the clause time. So "loses half of them"
→ `Quantity(.., 0.5×held, lose, <time>)`, a CHANGE the timeline folds. **Buddy now solves end-to-end**
(30 − 15 + 12 = 27); "Tom gives half of them away" → 5 (timeless). `Mary jumps twice` stays a Frequency
(no `of`). The full Buddy (with "a third of what he had on Tuesday") still declines — `third` isn't a
multiplier and the Thursday clause is a TIME-REFERENCED partitive (next target). Precision 100%; 295 tests.  ·
73. **time-referenced partitive — the full Buddy solves** (coverage 2→3, the real GSM8K item). Two parts.
(a) FRACTION WORDS: `FRACTIONS` (denominators — `half`=2,`third`=3,`quarter`/`fourth`=4,…) so "a third of"
is a unit fraction; the partitive trigger fires on any FRACTIONS word + `of` (not just a multiplier). Base
is `holding / n` (exact when it divides — dodges `1/3` float drift). (b) The of-clause sets the BASE:
`_partitive_quantity` takes the main verb to the LEFT of the fraction ("buys"; the of-clause's "had" is
embedded); if the of-clause has a verb + holder + time ("what he had on Tuesday") the base is
`_holdings_at(holder, measure, TIME)`, else the subject's current set ("of them"). So "buys a third of what
he had on Tuesday" → (1/3)×15 = 5 at Thursday; the timeline folds 30−15+12+5 = 32. FIX along the way: the
`solve()` fixpoint RETRY now passes `discourse` (was `None`), so a retried clause's "he" re-resolves —
without it the Thursday clause's holder stayed unbound and it silently dropped. Precision 100%; 298 tests.  ·
74. **"how many more/fewer … than" ACROSS TIMES** — value comparison over the timeline. When the comp
question's `than` splits TWO times ("…on Tuesday than on Monday"), compare ONE entity's holdings at the two
times: `_holdings_at(E, m, t1)` (the time before `than`) vs `…(E, m, t2)` (after) → `more`: a−b, `fewer`:
b−a. Slots into the existing comp branch before the entity-vs-entity case (which is unchanged — that one's
standard is a name, this one's "than" referent is a time). "On Mon Tom had 30, On Tue 50 — how many more on
Tue than on Mon?" → 20; composes with changes (state+buy). Precision 100%; 302 tests.  ·
75. **vocabulary push 2** (`fully-read` 7%→14.7%, 21→44/300). Two data-driven batches in correct roles:
common verbs (work/hold/go/collect/save/purchase/charge/share/receive/pay/keep/bring/send/fill/mix/order/
earn/make/take + compare/plant/add/manage/turn/split/know/come/teach/grow/build/run/walk/drive/count),
adjectives (able/different/previous/final/double/extra/expensive/…), nouns (food/shopping/school/family/
role + time-of-day + units), number words six..twelve (NUMBERS), and hyphenated unit fractions
(one-half/one-third/… → FRACTIONS). Tooling lesson: many additions COLLIDED with existing keys — a Python
dict literal silently keeps the LAST, so an `ast` duplicate-key scan caught 6 sense changes (last/building/
drink/can/plant) fixed via a multi-sense override block at the dict's end. PRECISION FIX exposed by the
new coverage: the instance-count FALLBACK fabricated a 0 when the queried TYPE was absent as instances
("how much money" amid only `son` instances) — now it declines unless the query's head noun is present as
an instance (a present type with 0 matches — "how many RED apples", none red — stays a real 0). Precision
100%; 305 tests.  ·
76. **titles + article-nearest + possessive back-link** (parse fixes from "Mr. Sam shared a certain amount
of money between his two sons, Ken and Tony"). (a) TITLES (`mr`/`mrs`/`dr`/… → a `title` sense, no
obligation) — recognised but NOT names, so "Mr. Sam shared" leaves Sam the UNIQUE name and `shared` binds
its `subject` (two competing names had left it unmet — the real blocker, not flagged in the eyeball trace).
(b) An article's `determines` now binds the NEAREST noun, not a unique one ("a certain amount of money" → a
determines `amount`, which was blocked by the second noun `money`). (c) `_resolve_possessives` (in
`analyze`, before the discourse resolver): an unresolved possessive ("his") BACK-LINKS to the clause's
agent when the subject is a NAME ("Sam … his sons" → his→Sam); a non-name subject ("his friend") is left
for discourse. The sentence's meaningful obligations are now all discharged (subject/dobj/determines/of/
refers). Precision 100%; 310 tests. NOT YET SOLVED: the aggregation "shared between A and B" ⇒ total = sum
of the shares (the "certain amount" = Ken+Tony) — the next semantic step. (User's broader idea: infer an
unmet obligation by BACK-LINKING through the obligation chain — possessive→agent is the first instance.)  ·
77. **distribution + the UNKNOWN as a first-class node** (the Sam problem solves; coverage 3→4). The
user's Option 3, chosen because it PRESERVES THE MOST INFORMATION and is most faithful to the obligation
principle (vs a sum-all heuristic or a total-as-fact-field). "Mr. Sam shared a certain amount of money
between his two sons, Ken and Tony" reads off TWO true facts: an UNKNOWN-valued node
`Quantity(sam, money, None, share)` (the "certain amount" — its value is an unmet obligation) and
`Distribution(sam, (ken,tony), money)` (the share-between relation; recipients = the named group's
members; measure = the `of`-object). `_holdings`/`_holdings_at` SKIP `amount is None` (unknowns aren't
folded until pinned). A registered synthesizer `_synth_distribution` BACK-LINKS the unknown: once every
recipient's holding is known, derive `Quantity(sam, money, Σshares, share)` — the value obligation
discharged by inference, exactly like `his`→agent one level up. The question "how much was the money
shared?" reads the pinned total (declines until ALL shares are known — never a partial guess): 1750+3500
= 5250. Precision 100%; 314 tests. The lesson: a genuine UNKNOWN deserves to be a node with an
obligation, not a hole filled at question-time — then one synthesizer handles share/divide/remainder alike.  ·
78. **equal split ("shared <total> equally among N")** — the INVERSE of #77 (total KNOWN, recipients a
COUNT), reusing the SAME Distribution relation + synthesis loop, as predicted. `Distribution` gains
`count`/`equal`/`recipient_noun`; `_distribution` now also reads an unnamed count ("among 3 sons" → the
value on the recipient noun) and a non-money measure ("30 candies" → a value-quantified noun), and accepts
a NOUN distributor ("a teacher"), not just a name. `_synth_equal_split` (registered): a KNOWN whole +
`equally`/`evenly` + count → each gets whole ÷ count, derived as a `get` Quantity per recipient (named, else
keyed to the recipient noun so "how much did each SON get" folds it). Two more "unique → nearest" relaxations
(same family as the article fix): the interrogative's `queries` binds the NEAREST quantifier ("how many
candies … EACH student" — `each` was a 2nd quantifier), and distribution premises are GUARD-3-exempt (their
total+count are consumed into the shares). "$5250 equally among 3 sons" → 1750; "30 candies among 5" → 6;
"$30 between Ken and Tony" → 15 each; #77's unknown-total case still 5250. Precision 100%; 318 tests.  ·
79. **manner adverb → verb (the method as an OBLIGATION, not a global scan)**. The #78 `equal` flag was a
global `any("equally" in units)` scan — exactly the anti-pattern the core principle forbids. Refactored: a
verb carries an OPERATION; a manner adverb PARAMETRIZES it (the same shape as negation, one slot over).
Method words (`equally`/`evenly`/`uniformly`/`randomly`/`optimally`/`proportionally`) get a `manner` sense
with a `qualifies → nearest verb` obligation (reusing the `negates` resolver); `DISTRIBUTION_METHODS` maps
the surface → method ("equal" is computable, the rest aren't). `_distribution` reads the method BOUND to
its own verb (`Distribution.method`, replacing `equal`); `_synth_equal_split` keys on `method=="equal"`.
The method drives precision: a KNOWN total with method ≠ "equal" (none/random/…) and no explicit shares →
an `under-specified` DEFICIENCY ("how was it shared?"), not a bogus number — `_underspecified_distribution`.
Also fixed a latent bug: the recipient group must be AFTER the verb (else a SUBJECT group "Tom and Mary
shared 10 apples" misfired as recipients); GUARD 3's distribution-exemption now keys on the premise
actually producing a `Distribution` fact (precise), not a verb-lemma guess. Precision 100%; 324 tests.
FUTURE (user, not now): when a verb is blocked for want of a manner word, instead of deficiency we could
(a) fall back on a CONTEXT-DEPENDENT default method, or (b) enumerate an answer PER plausible manner.
LESSON: a manner adverb is a modifier that parametrizes a verb's operation — bind it; let each operation
read the modifiers it cares about (distribution reads its method; a future rate op might read "quickly").  ·
80. **two parse near-wins from the bench (coverage 4→6, incl. GSM8K #1)**. The bench diagnostic showed
relational+aggregation ALREADY works (72 on a clean "48 in April, half as many in May, altogether"), so
several declines were math we had, blocked by ONE mis-parse. Fixed two. (a) SUBJ-AUX-VERB subject binding:
"how much money she WILL have" left `have`'s subject unbound — we handled `AUX SUBJ VERB` ("does she have")
but not the declarative `SUBJ AUX VERB` (subject before the aux). The resolver now, when the aux→verb gap
is empty, takes the nearest noun/pronoun LEFT of the aux → recovers Susie (a percentage-of-holding we
already solve) and any "X will/can/has VERB" question. (b) "sold `<X>` to N `<people>`" → N of X:
`_transfer_quantity` (before `_quantity`) — a BARE-plural dobj is the measure, the recipient count (via
"to") is the amount ("sold clips to 48 friends" → 48 clips, not 48 friends). Gated to a bare dobj (one with
its own count is the each/distribution shape), so it's precise. **Solves the canonical Natalia #1 → 72.**
SIX real GSM8K problems now solve end-to-end at 0 calls (Natalia, Sam, Tim/Harry, Paddington, Buddy, Susie).
Precision 100%; 329 tests. Method: run the bench, find the shape we're CLOSEST to, fix the one gap.  ·
81. **rates + unit conversion + result-measure binding** (cluster start; coverage 6→7, solves Weng).
Four pieces. (a) NON-MONEY rates + "a/an"=per: `_rate` now reads a value quantifying a noun ("20 pages a
day" → Rate(.., page, day)), and treats "a"/"an" before a unit noun as `per` when it sits right after a
money value or a quantified measure-noun (so a plain article "a book" is not a rate). (b) The rate's
per-unit count is read from the QUESTION ("…write in 3 DAYS?") when it's not a premise. (c) UNIT
CONVERSION: a `CONVERSIONS` table (min↔hr, g↔kg, …) lets a premise quantity in one unit fold into a rate
asked in another ("50 minutes" for an hourly rate → 50/60 hr → $10). (d) RESULT_MEASURE: the user's
"earn ≈ earnings" insight — `much` now BINDS the result-measure verb (earn/spend/give/… → MONEY) or a
money noun, read off the binding, REPLACING the global much→MONEY default. More faithful AND more precise:
"how much did he RUN" (run ∉ RESULT_MEASURE, no money noun) now DECLINES instead of answering in dollars.
Also: "do/does/did" is now multi-sense aux+verb ("she did 50 minutes of babysitting" → a `do` quantity) —
`_has_verb` ignores them so clause-splitting never breaks a "Tom and Mary" group on a stray "do". SEVEN
real GSM8K problems solve. Precision 100%; 336 tests. Follow-ons in this cluster: "N X of M each"
multiplication (sugar, James-beef), rate INVERSION (Joy, John-writes: total ÷ rate → time), cross-entity
rates ("$5.50/lb" applies to anyone's pounds), and "work" as a noun ("minutes of work").  ·
82. **containment multiplication + question-measure conversion** (cluster cont.; coverage 7→8, solves
sugar). (a) `_container_quantity`: "N <container> OF M <unit>" → N×M of the INNER unit ("20 packets of 100
grams" → 2000 grams) — two value-noun pairs straddling "of" multiply, measure = the inner unit. Runs
before `_quantity` (which would otherwise emit two separate quantities); GUARD 3 exempts it (both values
are consumed into the product). (b) Question-measure CONVERSION: the holdings fold, finding no quantity of
the asked measure, converts a held quantity in a same-family unit ("how many KILOGRAMS … sell" with grams
held → grams×1/1000). So "20 packets of 100 grams of sugar … how many kilograms?" → 2000 g → 2 kg. EIGHT
real GSM8K problems solve. Precision 100%; 341 tests. (The relative-clause forms "5 packs of beef THAT ARE
4 pounds each" / "3 books that are 400 pages each" still decline — the "that are" adds a second verb;
that's the next gap, alongside rate inversion.)  ·
83. **relative-clause multiplication + cross-entity rate → James-beef** (coverage 8→9). Five interlocking
fixes, all flushed out by one problem. (a) RELATIVE-CLAUSE multiplication: `_container`/`_distributed` use
the MAIN (non-copula) verb, so "5 packs of beef THAT ARE 4 pounds each" → 20 (the "that are" copula no
longer breaks single-verb). (b) DECIMAL-aware `_sentences`: a "." flanked by digits ("$5.50") is not a
sentence end. (c) PP-OBJECT exclusion from subjects: "the price OF beef is …" → subject `price`, not
`beef` (a noun inside a PP can't be the subject). (d) Prep `object` accepts a TIME: "on MONDAY Tom …" →
on→Monday (so the time, not the trailing name, is the object — Tom stays the subject). (e) CROSS-ENTITY
money rate: "how much did he pay" applies a `$/unit` price (any entity) to the asker's units (James's 20
lb × $5.50 = 110). And coreference now stores (entity, is_name): a gendered "he"/"she" prefers a PERSON
name over a thing — "he" → James, not "price". NINE real GSM8K problems solve. Precision 100%; 347 tests.  ·
84. **rate INVERSION** (coverage 9→10, solves John-writes). "How LONG will it take to <verb> <target>?"
→ `target ÷ rate`, answered in the rate's per_unit. The "how long" branch finds a `Rate` and the target
amount = (value on the rate's measure) × (a container count if present): "3 books that are 400 pages each"
→ 400 × 3 = 1200 pages, ÷ 20 pages/day = 60 days; an explicit "1200 pages" → 1200 ÷ 20 = 60. Tolerates the
matrix verbs ("it will TAKE him to WRITE …") since it reads values + the rate, not a single-verb clause.
TEN real GSM8K problems solve. Precision 100%; 349 tests. (Joy — "8 pages in 20 minutes … how many hours
to read 120 pages" — still declines: a per-N-unit rate "8 per 20 min" + "how many hours" (not "how long")
+ min→hr; the remaining rate gaps are the per-N-unit rate form and cross-unit "how many <unit>" duration.)  ·
85. **temporal age shift via the temporal model** (coverage 10→11, solves Herbert). The user vetoed a
per-word "next→+1" hack — do it through the temporal model. A relative-time reference has a POSITION on the
timeline relative to "now" (`_time_offset`: "next year"→+1, "last year"→−1, "in N years"→+N, "N years
ago"→−N — one general parse, not per-word), and AGE is a TIME-TRACKING quantity (advances 1/year), so the
age at a shifted time = age@now + Δ. The age branch now reads `_time_offset` and adds it (replacing the old
"shift → decline" guard). So "Kris is 24 now; Herbert is 10 years younger; how old next year?" → 14 + 1 =
15; "in 3 years" → +3; "5 years ago" → −5. (`ago`/`later`/`earlier` recognised.) Precision 100%; 355 tests.
LESSON: a "shift" isn't a special case — it's the queried time's offset on the timeline, and age is the
measure that tracks that axis. The same offset generalizes to any future time-dependent quantity.  ·
86. **word-numbers ARE values + the compound relational** (coverage 11→12, solves John-roommates). A
foundational unification: `one`..`twelve` are now `["value"]` (with the value from `NUMBERS`), so "three
apples" parses exactly like "3 apples" (a Quantity, no longer 3 instances), and "three years younger"
feeds the comparative. Count statements are now EXISTENTIAL-gated — `_facts_from` / `instances()` emit
instances only for "there ARE N <coll>", so a bare subject-verb word-number stays arithmetic. This let the
COMPOUND RELATIONAL land: `_relational_quantity` adds a comparative offset — "five MORE than twice as many
as Bob" → 2×Bob + 5 = 25 (plain "twice as many as X" has no comparative/offset, unchanged). PRECISION
fallout from the change, both fixed: (a) `_container` must not treat the SUBJECT as a container — "FOUR
people lost a total of 103 kg" is not 4×103 (skip when v1's noun is the subject); (b) GUARD 6 — "each OF
the [last two …]" is a per-MEMBER query over a restricted SUBSET of a group; we hold only the group
aggregate, so decline rather than match the total (distinct from "each <noun>", the equal-split per-each we
answer). Precision 100%; 362 tests. LESSON: a foundational re-typing (word-numbers) exposes latent
over-eager extractors — the fix is to make each extractor state its real precondition (a container isn't
the subject; a per-each isn't a group aggregate), not to revert the unification. Mel still needs "two
dozen"=24; the real John-writes-style "how many hours" (Joy) needs the per-N-unit rate.  ·
87. **number-scale words** (coverage 12→13, solves Mel). `dozen`/`hundred`/`thousand`/`million`/`billion`
are value words (a bare "a dozen" → 12); `_merge_scales` combines "N scale" into ONE value unit by
multiplying — "two dozen" → 24, "three hundred" → 300 — and recurses for chains ("two hundred thousand").
Runs right after `annotate_senses` (both are value units by then) and before obligations bind, so the
merged value quantifies its noun normally. So "Mel is three years younger than Katherine; when Katherine
is two dozen years old; how old will Mel be?" → 24 − 3 = 21. Precision 100%; 365 tests. THIRTEEN real
GSM8K problems solve. (Remaining named gaps: Joy's per-N-unit rate + "how many hours"; percentage-of-total
COMPLEMENT (Sanchez); "N times more" (Riku, declined for the 25× vs 26× ambiguity).)  ·
88. **per-N-unit rate + cross-unit duration query** (coverage 13→14, solves Joy; completes the rates
cluster). (a) `Rate` gains `per_amount` (default 1): "8 pages in 20 minutes" → `Rate(joy, 8, page, minute,
read, per_amount=20)`. `_rate` parses "N <measure> in M <time-unit>" (the "in"-PP whose object is a
TIME_UNIT) as the per. (b) The inversion branch now also fires on "how many <time-unit>" (not just "how
long"), computes `target × per_amount ÷ amount` (120 × 20 ÷ 8 = 300 min), and CONVERTS to the asked unit
(300 min → 5 hours). (c) `_container` no longer reaches across a second preposition — "8 pages of a book IN
20 minutes" is not 8×20 (vpair(False) stops at the next preposition after "of", the "each" relative clause
excepted); this latent misfire surfaced once word-numbers made more values live. (d) GUARD 3 counts
`Rate.per_amount` as a used given. Precision 100%; 369 tests. FOURTEEN real GSM8K problems solve. LESSON:
a rate is "amount per (per_amount × per_unit)"; once that shape is explicit, both the forward (×) and
inverse (÷) directions and unit conversion compose from the same four numbers.  ·
89. **perfect auxiliary "has/have/had" + "twice that many"** (coverage 14→15, solves Brandon). "How many
geckos HAS Brandon SOLD" is present-perfect — `has` is the AUXILIARY, `sold` (→ sell) the main verb. They
were verb-only, so the question had two verbs and no aux → `_question_subjects` couldn't find the subject.
Now `have/has/had` are multi-sense `["verb","auxiliary"]` (verb FIRST, so "Tom has 5 apples" stays a main
verb; the auxiliary reading wins by coverage only when a participle follows). `_lemma` already maps
sold→sell, so the question matches the context's sell-events: 86 + 172 = 258. ("twice that many" already
worked — `_relational_quantity`'s unnamed branch takes the most recent Quantity as the referent: 2×86.)
Precision 100%; 373 tests. FIFTEEN real GSM8K problems solve (5.0% of the dev slice). LESSON: the same
auxiliary/verb multi-sense pattern as do/does/did — but verb-FIRST, because have is primarily a main verb;
the obligation-coverage disambiguator flips it to the auxiliary reading exactly when a participle needs it.  ·
90. **percent of a COUNT** (capability; coverage unchanged at 15 — honest). "40% OF the students passed"
→ 0.4 × the students' quantity = 24 of 60 (the count of students who passed). `_percent_quantity` now
forks on the percent's base: a VALUE base ("20% of 50") is arithmetic (decline here); the "of"-object NOUN
("of the students") is a percent-of-count — the referent is the prior Quantity of that noun's measure, the
result a `Quantity(<noun>, <noun>, 0.4×ref, <verb>)`. The "How many students passed?" question has NO aux
(the query noun is itself the subject), so the holdings fold gained a no-aux path — gated on a matching
count-fact existing, so it never preempts the remainder / count-instances branches. The portion percent
("puts 20% more money") and "20% of 50" arithmetic are unchanged. Precision 100%; 377 tests. WHY NO
COVERAGE: the real percent problems need MORE than the partitive — Sanchez wants the category COMPLEMENT
("B and above" = not "below B" = 100%−40%), Adam wants multi-class subtraction. We DECLINE those (no
complement/category model) rather than guess — "how many failed" from "40% passed" → undetermined.
PRINCIPLE: a correct, general, tested capability is a legitimate increment even when this slice's coverage
doesn't move; forcing the complement would be a guess. The complement/category model is the next real step.  ·
91. **complement/category model** (capability; coverage unchanged at 15 — honest). "The REST <verb>" → the
population total − Σ(the stated subsets of the same MEASURE): "60 students; 40 passed; the rest failed" →
20; with a percent subset → 36. `_rest_quantity` reads the population from a have/be total and sums the
other (event) quantities of that measure — by MEASURE not entity (a plural subject like "40 students" is
keyed "students", a percent-of-count "student" — an inconsistency the measure-sum sidesteps). The
complement requires an EXPLICIT signal (`rest`/`remainder`/`others`) AND a known total — never assumed
(no total → a non-numeric fact → decline; we don't infer "failed = total − passed" without the cue).
Precision 100%; 381 tests. WHY NO COVERAGE: the motivating Sanchez has FIVE compounding parse blockers, of
which the complement is one — "Mr." splits the sentence (abbreviation), "found out THAT 40% … got below B"
embeds the category clause, "B and above" splits on "and", the total "60 students" is a forward-ref in an
"if" clause, and "below B" ⊥ "B and above" is a THRESHOLD complement (not the lexical "rest" cue). The
other real "rest" problems (tables, Henry) are 4–6 step. PRINCIPLE: build the requested model cleanly on
its explicit signal; do NOT hack through Sanchez's parse thicket. Next: the abbreviation/embedding/and-split
parse fixes (general, unblock many), then the threshold complement, would let Sanchez through.  ·
92. **title abbreviation doesn't end a sentence** (parse fix; coverage unchanged at 15). A title's period
("Mr.", "Dr.", "Prof.") is NOT a sentence terminator — `_sentences` now keeps "Mr. Sanchez …" whole
(alongside the decimal exception). `TITLES` is DERIVED from the lexicon's title-sense words (single source
of truth), not a second hardcoded list. NOTE: titles were ALREADY recognised (the #69 "title" sense frees
the verb's subject) — the only bug was the sentence split; a first stab adding a `_drop_titles` step was
WRONG (redundant, and it broke the #69 title-sense test). So this is a one-line fix in `_sentences`, not a
new mechanism. Removes one of Sanchez's five blockers (the other four: `that`-embedding, `and`-split of "B
and above", forward-ref total, threshold complement). Precision 100%; 385 tests. LESSON: before adding a
mechanism, check what already exists — the title machinery was there; only the tokenizer needed the fix.  ·
93. **trailing conditional "Q if P?"** (Sanchez chain cont.; coverage unchanged at 15). `if` is now a
clause separator, so a trailing condition becomes its own premise — and crucially it peels off BEFORE the
"and" split, so "B and above if he has 60 students" → ["… B and above", "he has 60 students"] (left of
"and" is then verbless → no mis-split). THREE interlocking fixes: (a) " if " added to the clause
separators (placed before " and "); (b) the QUESTION clause is the interrogative-bearing one (`how`/`what`/
`which`/…), not the bare "?"-clause — for "Q if P?" the "?" lands on the condition P; (c) the complement
clause DEFERS (returns []) when there's a `rest` cue but no total yet, so the fixpoint resolves it once a
later clause (a trailing "if") supplies the total. End-to-end: "40% passed; the rest failed; how many
failed if he has 60 students?" → 36. Leading "If P, Q" (comma) unchanged. Precision 100%; 388 tests. Two
Sanchez blockers remain: the `that`-clause embedding ("found out THAT 40% … got below B") and the THRESHOLD
complement ("B and above" = not "below B"). LESSON: one structural cue (`if`) often fixes several symptoms
at once — the trailing condition, the spurious "and" split, and the forward-ref total all dissolve together.  ·
94. **vocabulary push** (ceiling raise; fully-read 44→52 / 14.7%→17.3%, coverage unchanged at 15). (a)
`_as_number` parses FRACTION LITERALS "N/M" → N/M ("1/2" → 0.5) — 11 unrecognised values in the slice. (b)
A deliberate, conservatively-typed batch of frequent unknown words (nouns/verbs+forms/adverbs/pronouns
from the bench worklist), added to `LEXICON`/`PLURALS`/`VERB_LEMMA` — NO morphological guessing (the
project's deliberate-vocab principle); a wrong sense yields a DECLINE, never a confident-wrong, so precision
held at 100%. Coverage didn't move: the 8 newly-fully-read problems decline on STRUCTURE — vocab raises the
ceiling, structure is the next wall. The nearest unlocked target is Christina (3 snakes' lengths summed
with feet→inch conversion) — but it needs multi-entity "all … combined" summation and its question parses
as a `Subset`, so it's deferred (not a clean win). LESSON: a vocab push is a ceiling move, not a coverage
move — measure it by `fully-read`, and expect the freed problems to surface the NEXT structural gaps.  ·
95. **combined / grand-total question** (coverage 15→16, solves Christina — the vocab-push target). "How
many <unit> are all <X> COMBINED?" → the GRAND TOTAL: every Quantity convertible to the asked unit, summed
IN that unit (cross-ENTITY and cross-UNIT via CONVERSIONS): 2 feet + 16 in + 10 in = 50 inches. Triggered
by `combined`/`altogether`/`total`/`all … together`. GATED to a GENERIC population — `not named`: a
question with NAMED entities ("Tom and Mary altogether") keeps the per-entity holdings fold, which DECLINES
when an entity's count is unknown (the grand total would silently sum only the known one — a confident
wrong). So the distinction is generic-population (sum everything of the measure-family) vs named-entities
(check each). Precision 100%; 394 tests. SIXTEEN real GSM8K problems solve. LESSON: a "sum everything"
branch must be fenced off from "sum these specific entities" — the cue is whether the question names them;
the ceiling→coverage handoff worked exactly as predicted (vocab freed Christina, then structure solved it).  ·
96. **`Grade N` classifier label** (Sanchez chain; coverage unchanged at 16). A bare number right after a
noun is a LABEL/ordinal ("Grade 5", "Room 3", "Type 2"), NOT a quantity. `_merge_labels` (a pre-pass beside
`_merge_scales`) absorbs it — the number becomes an ATTRIBUTE of the noun — so it no longer (a) quantifies
the next noun, (b) bases the percent ("40% of his Grade 5 students" was misreading as "40% of 5"), or (c)
trips the unused-given guard. Gated to a PURE noun (skips `operator`-sense, so "4 TIMES 3" still
multiplies; and `$5` is excluded as a self-measuring money value). With the free `5` gone, the percent
of-object resolves to the of-phrase HEAD (the LAST noun before the verb — `students`, not the `Grade 5`
classifier), so "40% of his Grade 5 students" → 24 of 60. Precision 100%; 398 tests. This was the
SECOND-to-last Sanchez blocker — the percent clause now yields 24; only the THRESHOLD complement ("B and
above" = not "below B") remains, which is specialized grade semantics (deferred). LESSON: word ORDER
disambiguates number role — number-then-noun is a quantity ("5 students"), noun-then-number is a label
("Grade 5"); the same digit, opposite meaning, read off position.  ·
97. **vocabulary push, round 2** (ceiling raise; fully-read 52→65 / 17.3%→21.7%, coverage unchanged at 16).
A second deliberate batch, prioritising the "exactly ONE unknown word" problems (each is one term from
fully-read). Same conservative typing (a wrong sense → decline). Coverage held: the freed problems decline
on structure (Sanchez's threshold complement, Anne's age multi-hop). CONTRACTIONS ("what's"×5) and
hyphenated number-compounds ("120-page", "2-hour") are a separate PARSING concern (apostrophe/hyphen
splitting), deferred. Precision 100%; 400 tests. LESSON: `LEXICON.update()` OVERRIDES — an added word must
not shadow an existing MULTI-sense entry. "rose" was already `["noun","verb"]` (a disambiguation test
relies on it); adding `"rose":["noun"]` silently narrowed it and broke 4 tests. The suite caught it — but
when batch-adding vocab, scan for keys that are already multi-sense and leave them alone (a regression test
now pins `LEXICON["rose"] == ["noun","verb"]`).  ·
98. **contraction expansion at tokenization** (ceiling raise; fully-read 65→66, coverage unchanged at 16).
A contraction is the surface form of TWO words — `read()` now restores both (`_split_contraction`), so each
carries its OWN obligation and the existing machinery handles it; NO special-case flag. "what's" → what +
is, "didn't" → did + not, "they're" → they are, "won't" → will not (irregular), "I'm" → I am, "we've" → we
have, "he'll" → he will. Trailing punctuation rides on the last word ("what's?" → ["what", "is?"]). The KEY
discrimination is the apostrophe-s: a PRONOUN/interrogative "'s" (what/he/it/that/there/who/…) is the copula
"is", but a NAME's possessive ("Tom's") is NOT a contraction — it stays one token for the possessive
resolver. Only +1 fully-read (the "what's" problems carry other unknowns) but it's a correct, GENERAL
tokenization fix that composes with everything downstream. Precision 100%; 406 tests. LESSON: restore the
underlying words rather than special-case the contraction — the obligation machinery then needs no change
(the lexicalist principle: put the grammar back IN the words, don't bolt a rule onto the side).  ·
99. **hyphen number-compound split** (ceiling raise; fully-read 66→68, coverage unchanged at 16). Same move
as contractions, same seam (`read()`): a DIGIT-led hyphen compound is an adjectival number-unit phrase, so
restore its words — "120-page" → 120 + page, "2-hour" → 2 + hour — and the number quantifies the unit
through the normal machinery. GATED to a digit FIRST part (`_as_number(parts[0])`): "fast-food",
"twenty-five", "fourth-grade" stay one token, so a hyphenated number-WORD is never split into two wrong
values (worst case stays a decline, never a confident error). Composes immediately with rate inversion:
"John writes 20 pages a day; how long to write a 120-page book?" → 120 ÷ 20 = 6 (a synthetic; the real
120-page/2-hour problems are multi-step, hence coverage held). Precision 100%; 410 tests. LESSON: the two
tokenizer fixes (contractions, hyphens) share a shape — a single surface token standing for several words;
fix it ONCE at the tokenizer seam and it composes with every downstream extractor, no per-site changes.  ·
100. **growth projection** (coverage 16→17, solves Haley — mined from the now-larger fully-read set). A
quantity growing at a RATE per year, valued at a FUTURE time → `current + rate × Δ`. This is the
explicit-rate GENERALISATION of the age shift (age is just rate 1/year): "Haley grows 3 inches every year;
currently 20 inches; her height after 10 years?" → 20 + 3×10 = 50. THREE coherent pieces: (a) `every`/`each
<unit>` is a per-marker in `_rate` (like `a`/`an`), so "3 inches every year" → `Rate(.., inch, 3, year,
grow)`; (b) `_time_offset` handles "after N years" (+N, like "in N years"); (c) a projection branch — when
a future offset and a `GROWTH_VERBS` per-year rate both exist, return `_holdings(current) + rate×offset`.
Gated on the grown rate existing, so age questions (no grow-rate) still take the age branch. The query word
("height") is IGNORED — the structure (growth rate + current value + future offset) drives it, not the
noun. Precision 100%; 414 tests. SEVENTEEN real GSM8K problems solve. LESSON: the ceiling→coverage handoff
again — two vocab rounds + the tokenizer fixes raised fully-read to 22.7%, and mining that larger set
surfaced Haley as the cleanest target (2/2 premises already parsed); the win was one new ASSEMBLY step
(projection) over facts the parser already produced.  ·
101. **relational age + multi-hop chaining** (HARD TIER; coverage 17→18, solves Anne). "X is N times as
old as Y" → X's age = N × Y's age. `_relational_age`: the factor is a multiplier ("twice") or "<value>
times" (`_relational_age_factor`); the standard Y is the LAST name; the result is `Quantity(X, year,
factor × Y_age, be)`, DEFERRED (returns []) until Y's age is known — so the solve() fixpoint CHAINS a
multi-hop: Anne=2×Emile, Emile=6×Maude, Maude=8 → 48 → 96. THREE supporting fixes: (a) `_copula_unit` now
recognises the bare/perfect "be"/"been" ("Emile will BE …", "has BEEN …"), so the relational subject binds;
(b) GUARD 3 exempts a relational-age premise (the factor 2/6 is consumed into the product, never appears as
a verbatim amount); (c) the defer is structural (factor + copula subject present) so a never-resolved
standard declines rather than guesses ("Anne is twice as old as Emile. How old is Anne?" → undetermined).
Precision 100%; 418 tests. EIGHTEEN real GSM8K problems (6.0% of the dev slice). LESSON: a multi-hop is NOT
a new mechanism — each hop is one relational fact that DEFERS on a forward ref, and the existing fixpoint
already chains forward refs; the work was the per-hop fact (relational age) + making it defer, not any
chaining logic. OBLIGATION-aligned: the multiplier scales the age dimension, "as…as" binds the standard.  ·
102. **James — gerund adjunct + implicit-standard comparative + query-less combined** (HARD TIER; coverage
18→19, 6.3%). A 3-piece combine, each piece general: (a) a trailing GERUND adjunct ("spends 40 years
TEACHING") is an unbound `-ing` verb — `_quantity` ignores it, so the finite verb stands alone and "40
years" is a clean Quantity; (b) a comparative with an IMPLICIT standard ("10 years less" — no "than Y")
takes the most recent Quantity OF THE MEASURE as referent (partner = 40−10 = 30), and `_comparative_quantity`
now also tolerates the multi-verb "has been teaching" (subject from ANY bound verb); (c) a query-less
"combined" question ("How long is their combined experience?") — the `how long` inversion no longer
preempts a `combined` question, and the grand-total infers the measure as the ONE the context quantities
share (40 + 30 = 70 years). Precision 100%; 422 tests. NINETEEN real GSM8K problems (6.3%). LESSON: a hard
problem is often several INDEPENDENT general gaps stacked — fix each as its own principled piece (gerund,
implicit standard, query-less combined) and the problem falls out; none was James-specific.  ·
103. **all word-problem arithmetic routed through the AUDITED evaluator** (no coverage change — an
auditability refactor). The engine had DRIFTED from its stated design: only the two explicit-arithmetic
questions ("What is 3 plus 2?") went through `translate.run_program` (the safe evaluator); every word-
problem number was a bare inline Python operator. Now a single seam — `_compute(expr)` — builds a one-line
program and EXECUTES it through `run_program` (ast.parse + operator/function whitelist), logging
`(program, result)` to `ARITHMETIC_LOG` (reset per `solve()`). Every extractor and the answer fold emit an
expression string instead of computing inline: `_holdings` → `"1 * 30 + -1 * 10"`, relational age → `"6 *
8"`, comparative → `"40 - 10"`, the combined sum → `"40 + 30"`, rate inversion → `"((120) * 20 / 8) *
0.01666…"`. So Anne's `ARITHMETIC_LOG` reads `6*8=48, 2*48=96` — the multi-hop is legible. NOT a security
need (the engine's operands are its own, not untrusted LLM output) — the value is AUDITABILITY: a recorded,
re-executable program per answer, restoring the "correct-by-construction and auditable" guarantee the docs
claim. 425 tests; precision 100%; 19 solves unchanged (every number identical, just produced via the
evaluator). LESSON: a stated invariant ("arithmetic executes through the audited evaluator") silently rots
as code grows around it — the fix is one seam every site funnels through, so the invariant can't drift again.  ·
104. **surface the audit trail in `--trace` (+ fix the CLI/engine drift)** (tooling). `--trace` now prints
an "arithmetic (executed via the audited safe evaluator)" section — every `(program, result)` from
`ARITHMETIC_LOG`, so Anne reads `6 * 8 = 48`, `2 * 48 = 96`; James reads `40 - 10 = 30`, `40 + 30 = 70`.
Surfacing it EXPOSED a pre-existing bug: the CLI's word-problem path did a single SENTENCE pass with no
fixpoint, so it diverged from `solve()` — Anne gave `undetermined` from the CLI but `96` from `solve()`.
Could NOT just route the CLI through `solve()` (its precision GUARDS decline things the CLI's guard-free
`answer()` commits to — e.g. the logical "more red than green? → yes" case). Fix: the paragraph CLI now
CLAUSE-splits and runs the same first-pass + deferred-retry + synthesize FIXPOINT as `solve()` (so a
multi-step problem resolves), keeping `--about` single-pass and the guard-free `answer()`. So `reading "Anne
…"` now agrees with `reading.solve("Anne …")`. 427 tests; precision 100%. LESSON: adding observability (the
trail) surfaced a latent correctness gap (CLI ≠ engine) — you can't audit what you can't see, and once you
can see it the divergence is obvious.  ·
105. **possessive genitive tokenization** (ceiling raise; fully-read 68→73 / 22.7%→24.3%, coverage
unchanged). The third tokenizer in the family (contractions, hyphens, now genitives): a NAME's "'s" splits
into the name + the possessive marker — "Leo's" → `Leo` + `'s`, so the name is RECOGNISED instead of a
stray unknown `Leo's` token. The apostrophe-s now forks three ways in `_split_contraction`: a pronoun
("he's"→is, copula), a name ("Leo's"→Leo + 's, genitive), else unchanged. `'s` (key `s`) is an inert
`["possessive"]` marker — recognised (no unknown-word) but with NO obligation yet, so it can't misfire
(precision held 100%). WHY NO COVERAGE: the five freed problems are multi-step and the possessive is
INCIDENTAL — Leo's is a 3-part assignment sum, Ernesto's is temporal+relational, Matt's needs an INVERSE
comparative ("Bush is younger than Matt by 3" → solve for Matt, which the directional comparative can't do).
So this raises the ceiling (the documented mining loop) without a solve; the genitive OWNER-binding (Tom's
apples → Tom owns) is future work, needed only when a possessive problem is otherwise solvable. LESSON: the
≤2-step clean tier of the 300-slice is now EXHAUSTED (only Riku/Artemis/Sanchez remain — ambiguous or
need world knowledge); further coverage needs foundational parsing (genitive semantics, noun-noun
classifier compounds "mango trees", inverse comparatives) — each a real feature, not a quick win.

106. **bare-"together" aggregation cue** (coverage +2 on dev-train; Salvadore 2934, Anika 64). The
grand-total trigger `combined_q` fired only on `{combined, altogether, total}` or BOTH `{all, together}`
— so "how much did the two men earn TOGETHER?" (only "together", no "all") missed the sum even though the
relational quantity already resolves (Santo = 978). Add bare "together" as a sum cue. LESSON / GOTCHA: an
aggregation cue is CONFOUNDED with the collaborative-rate sense — "how long to paint the house together?"
is a joint RATE, not a sum. The rate-inversion branch is gated `not combined_q`, so firing `combined_q` on
a how-long/time question would SKIP it and sum rates into garbage. Gate the cue to NON-time questions
(`not is_long and asked_unit not in TIME_UNITS`), which the rate branch never claims. Sibling cue "in all"
was mined too but gave 0 dev-train gain (its problems also need offset-base resolution) → DROPPED per
example-driven-deferral (don't ship a trigger that only risks firing unmeasured on TEST). Firewall:
dev-train 37→39 / 0 wrong; dev-holdout 0 new-wrong delta; TEST precision held 50.0%. 820 tests.  ·
107. **bidirectional relational-multiplier resolution** (backward inversion; dev-train 39→40, holdout
35→37). `_relational_quantity` only derived subject-FROM-standard ("Mary has k× as many as Tom" fires
once Tom is known); the symmetric case — SUBJECT known, standard unknown — deferred forever, so backward
chains declined ("Tyrah 6× Sarah, Tyrah=12 → Sarah=2 → Tim 8×Sarah=16"; "Alexis 4× (Dilan+Ashley),
Alexis=60 → group=15 → all=75"). FIX: when the named standard has no quantity but the subject does, emit
`standard = subject / factor`, tagged with the relation's OWN verb (run/have) so the answer fold matches —
the fixpoint then chains forward off it. PURE-RATIO only (a comparative offset inverts ambiguously → keep
deferring), and mutually exclusive with the forward path (forward needs the standard known, backward needs
it unknown) → no double-emit, no correct→decline churn. The conjoined-group standard ("Dilan and Ashley
combined") grounds as one entity and composes with the aggregation fold. LESSON (about the FIREWALL, not
the parser): this was the first increment where TEST gained a confident-wrong (4→5) yet precision ROSE
(50.0→58.3%, +3 correct / +1 wrong). The governing gate is **precision-hold, not raw wrong-count** — a
*drop* reverts (the offset case, 50→40); a *rise* with a new wrong is the §3a distribution-shift the holdout
can't catch, but the committed-answer set got safer on average, so it ships. The new wrong is undiagnosable
without breaking the TEST firewall — the price of an honest one-look arbiter. 822 tests.  ·
108. **elided count-noun → infer the single shared measure** (dev-train 40→41, holdout 37→38). "How many
does Steven have?" omits the measure noun, so the question's `query` descriptor is None and the per-entity
fold can't run — the engine declined ("couldn't tell what to count") even though its forward chain already
computed the answer (Steven: 6×3=18, 4×18=72). FIX: when `query` is None on a "how many" question and the
context quantities share exactly ONE measure, infer it as the query — mirrors the query-less combined-fold's
measure inference (#102). The existing fold then runs; NO new arithmetic. Decline-safe: two+ distinct
measures keep query None → honest decline (the genuinely-ambiguous case). TEST precision held 58.3% (0
delta). 824 tests. LESSON: an *artificial* parse-level decline (the answer is computed, only the question's
referent is elided) is the cheapest, lowest-risk coverage win — infer the missing referent from the salient
context and reuse the existing fold, rather than adding any new solving machinery.  ·
- **backward COMPARATIVE-offset inversion — ATTEMPTED + REVERTED (holdout caught the overfit).** The additive
twin of #107: when "A is N more/less than B" has the SUBJECT known and the standard unknown, invert
(standard = subject ∓ N). Worked on the targets (Tim/Martha 68 both directions) and gained +1 dev-train
(Benjie) with 0 dev-train wrong — but dev-holdout went 3→5 wrong (**+2 new-wrong delta**), failing the §3
gate. WHY it differs from #107 (which shipped clean): "N more/less than Y" is a VERY common surface pattern,
so backward-inverting it fires on many unseen shapes where the inversion is unsound; "k times as many as Y"
(#107) is rarer/more specific and so far less exposed (§3a exactly). LESSON: a capability being the "twin" of
a safe one does NOT make it safe — safety tracks the SURFACE PATTERN's commonness, not the mechanism. The
holdout is what distinguishes them; spend it before the TEST look. Re-attempt only behind a tighter gate
(e.g. require the standard be the asked entity AND no other relation/offset in the problem).  ·
109. **average over a set** (dev-train 41→42, holdout 38→39). "What is the average height of the three
people?" = Σ(each owner's net holding) ÷ owner-count. Handled FIRST in answer() (it's a "what is …" phrasing
the wh_value dispatch — expecting "what is <name>" — would otherwise decline). Reuses the existing per-entity
holdings; the only new op is the ÷N. DENOMINATOR SAFETY (the whole risk in averaging): infer the SINGLE
shared measure (the question says "height" but holdings are keyed 'inch' — so a context with >1 measure
declines); require ≥2 owners; and if the question states the set size ("the three"), the owner count MUST
equal it, else an entity is missing/extra → DECLINE rather than divide by the wrong N. Parker's three heights
were ALREADY computed by the relational chain (60/68/64) — only the ÷3 was missing. TEST precision held
58.3%. 826 tests. CONTRAST with the reverted backward-comparative just above: "average" is a SPECIFIC cue
(safe to act on), "N more/less than Y" is ubiquitous (over-fires) — the same §3a commonness axis.  ·
110. **value-list parsing + value-list average** (dev-train 42→43, Malou). "Malou got 91, 90, and 92 in her
three quizzes. What is her average score?" declined TWO ways: (1) the comma-list "91, 90, and 92" tripped
_quantity's adjacent-bare-values guard (meant for "10 $20" denominations) → no quantity emitted; (2) the
trailing cardinal "three quizzes" (= the list length) would pollute the set / trip GUARD 3 unused-given.
THREE coordinated pieces: (a) _quantity — a COMMA after the first value marks a LIST (not a denomination),
emit one Quantity per item; (b) `_list_cardinals` — a value equal to the count of OTHER values quantifying
the SAME noun is the set size, dropped (shared by _quantity AND GUARD 3 so the cardinal is neither emitted
nor flagged unused); (c) the average branch gained a VALUE-LIST mode — one owner with several same-measure
quantities averages the INDIVIDUAL amounts (not the accumulated 273), multi-owner still averages per-owner
holdings. TEST held 58.3%; no _quantity regression (solved-set diff clean both ways). 828 tests. LESSON: a
single surface ("got 91, 90, and 92 …") can be blocked by a STACK of guards each doing its job for other
shapes (denomination guard, unused-given guard) — unblocking needs a discriminator (the comma) threaded
through every one of them, plus the helper shared so the parse and the guard agree on what the cardinal is.  ·
111. **name–noun disambiguation** (dev-train 43→45, holdout 39→41). A name that's ALSO a common noun
("Job", "Bill", "Penny", "Rose") took the lexicon's noun sense and never reached the capitalized→name
fallback, so "4 times as old as Joe"→20 but "…as Job"→undetermined. ROOT CAUSE: even made multi-sense
["name","noun"], `disambiguate` picks the fewest unmet obligations, and in "as old as Job" a NAME leaves
its own `predicate` obligation unmet (cost 1) while a NOUN covers (cost 0) → noun wins, killing the
relational-age resolver. A comparative's "than"-standard escapes this because its MET `standard` obligation
already subordinates it (predicate foregone); the bare `as…as` frame carries no comparative. FIX, two pieces:
(a) `annotate_senses` — a capitalized known common-noun becomes `["name","noun"]` (name FIRST → ties favour
it); modals/verbs/articles aren't `["noun"]`, untouched; (b) `_comparison_standard_ids` + `deficiencies` —
subordinate the noun/name after the SECOND `as` of an `as <X> as <STANDARD>` frame (it fills the comparison's
reference slot — the same "slot-filler is discharged" principle the subordinated set already encodes). Since
`disambiguate` scores via `_unmet_count`→`deficiencies`, this stops penalizing the name reading with NO change
to `disambiguate` or the resolvers. SAFE because an article-DETERMINED noun ("the bill") covers strictly
better as a noun and stays one; only the undetermined `as…as` standard flips. Held-out generalised CLEANLY
(+2 holdout, 0 new-wrong delta); TEST precision held 58.3%. 832 tests. LESSON: when `disambiguate` mis-picks a
sense, the bug is usually an obligation that's unfairly counted as "unmet" — fix the COVERAGE accounting (here,
subordinate a legitimate slot-filler) rather than bolting a position rule onto `disambiguate`; the coverage
mechanism then picks correctly on its own, and the article-vs-bare contrast does the name/noun discrimination
for free. ALSO confirmed by an external word-by-word cognitive analysis: the human reader assigns "Job" the
role "comparison reference" (a person) precisely because it fills the `as…as` standard slot.  ·
112. **bare age "X is N" (no unit) in age-typed problems** (dev-train 45→47, **TEST 58.3→61.5%**). A copula
"X is N" with no unit ("If Job is 5") states X's AGE when the problem is age-framed, but the bare value
quantifies no noun → `_quantity` emits nothing → a relational-age chain can't ground ("Stephanie is 4× as
old as Job. Job is 5 YEARS OLD"→20, but "…If Job is 5"→undetermined). A bare equation has no dimension of its
own, so INFER it from the problem's age framing. `_inject_bare_ages` (shared by `solve()` AND the CLI loop —
mirrored so they don't re-diverge, cf. #104): in a problem with age cues (old/older/younger/age), a premise
that is a bare copula "X is N" (name subject, value with no quantified noun, no quantity emitted) injects
`Quantity(X, year, N)` before the fixpoint, which then chains (Job 5 → Stephanie 20 → Freddy 18). GATED on
age cues so a bare "X is N" in a non-age problem is untouched. Builds directly on #111 — the collision-name
fix (Job resolves as a name) is what lets these chains exist at all. dev-holdout 0 delta; TEST +1 correct, 0
new wrong (precision ROSE). 835 tests. LESSON: a bare equation's DIMENSION is supplied by the discourse, not
the clause — when a value has no unit, look to the problem's framing (age cues) to type it, and gate tightly
so the inference only fires where that framing is unambiguous. The #111→#112 pair shows compounding: a
foundational disambiguation fix re-opens a whole cluster whose follow-on increments were previously unreachable.  ·
113. **possessive relational age + "how many years old" question** (dev-train 47→50: Mehki, Noah, Jerica).
Two coupled age-cluster fixes. (a) POSSESSIVE relational age: "Jordyn's age is double Zrinka's age" / "Jerica
is twice Louis' age" → X_age = k × Y_age. The possessive frame (the noun "age" owned via a name, not "as old
as") means the multiplier mis-scales the copula into a bogus Frequency, so `_possessive_age_parts` reads it
POSITIONALLY (standard = the name after the multiplier that owns an "age"); the dispatch resolves it or DEFERS
(so the fixpoint chains AND it never falls through to `_frequency`). Pure-ratio only. (b) "How many years old
is X?": `_copula_args` picks the fronted measure "years" as the subject, not the name → no holdings → decline;
the "how old" branch now prefers the NAME asked about. These are COUPLED — the possessive-age problems all use
the "how many years old" question form, so neither fix alone moves them (the recognizer alone was +1, on a
"how old"-phrased problem). Noah ("Noah's age is twice Joe's age … how old will Noah be after 10 years") now
composes possessive-age + bare-age (#112) + projection (#100) end to end. dev-holdout 0 delta; TEST held 61.5%.
839 tests. LESSON: when a recognizer is validated but yields ~0 on the bench, suspect a SECOND blocker on the
same problems (here the question form) — bisect the failing problem (it solved with "how old" but not "how many
years old"), don't assume the recognizer is wrong. The age cluster's wins keep coming in #111→#112→#113 chains.  ·
114. **total-age question + backward age** (MULTI-PIECE bundle; dev-train 50→52: Xavier, Sam). Two coupled
pieces that only move the bench TOGETHER. (a) BACKWARD age inversion in `_relational_age` (the age twin of
#107): standard's age unknown but the SUBJECT's known → standard = subject / factor ("Sam is twice as old as
Sue; Sam 12" → Sue 6); present-tense, pure-ratio, mutually exclusive with the forward path. (b) TOTAL-age
question (sibling of the #3l/109 average branch, placed before the wh_value handler): "what is the total of
their ages now?" / "what will be their total age in N years?" → Σ(each owner's age) [+ `_time_offset` ×
owner-count]. Lands Xavier 36 (future-age #2d + backward + total) and Sam 36 (backward chain + total +
projection). WHY BUNDLED: backward-age standalone was +0/+0/+0 (its problems need the total aggregation), the
total-age question needs the ages backward produces — neither moves the bench alone (documented in the
firewall memory as a deferred-ready twin until this companion landed). FIREWALL CAUGHT AN OVER-FIRE: the first
cut claimed the INVERSE "in how many years will their total age be N?" (solve for the DURATION, not the total
— Adam/Tom, +1 holdout wrong); gated with `not any("many")` (the inverse has "how many years"; the forward
total-age question does not). dev-holdout 0 new-wrong delta after the gate; TEST held 61.5%. 842 tests.
LESSON: a +0 standalone primitive isn't worthless — it can be a deferred-ready HALF of a bundle whose other
half unlocks it (here total-age); record it precisely so the companion session can re-add it in minutes. And
an aggregation branch must exclude the INVERSE "how many <unit>" question (solving for the operand, not the
aggregate) — the "many" interrogative is the discriminator.  ·
115. **N-unknown sum + ratio-CHAIN solver** (generalizes #3h; dev-train 52→55: Kayla, Amy/Jeremy, Sam/Drew).
`_two_unknown_ratio` (#3h) handled only 2 entities + one DIRECT relation and rejected 3-way; this generalizes
it to N entities whose relations CHAIN, solved by PIVOT REDUCTION (express every entity as a multiple of one
pivot, Σ the coefficients, pivot = Total / Σcoeff). Two functions: (a) `_ratio_relation(rp)` → (subject,
factor, standard) — a clause with EXACTLY two names unifies every surface form (subject before the verb,
standard the other, factor the multiplier/fraction): "k× as many as B", "k× as old as B", "1/3 the age of B",
"twice B's age", "1/4 of the X that B Ved" — and the standard must sit in a real ratio frame (after as/of/that
or a possessor "B's"), so "bought half the CDs" (a count of an object) isn't a ratio. (b) `_sum_ratio_system`
builds the coefficient graph (unique root = the non-subject pivot, propagate coeff = factor × coeff[std] to a
fixpoint, require a connected tree), finds the total, solves. The 2-entity case reproduces #3h exactly →
replaces it. THE FIREWALL CAUGHT TWO HOLDOUT OVER-FIRES in the first cut (3→5 wrong): an OFFSET-tainted system
("CJ has 5 MORE than twice …" — `_ratio_relation` drops the offset clause, leaving an incomplete system that
still "solves") and a NO-FRAME false positive ("Prince bought half the CDs" — "CDs" capitalized → a name, but
no ratio frame). Both gated (decline the whole system on a factor+comparative premise; require the standard in
a ratio frame). After gating: holdout +1 correct / 0 new-wrong delta; TEST held 61.5%. Preserved #3h's
unit/qualifier discipline (rel_mod + full descriptor: "red" vs "blue", years vs months). 846 tests. LESSON:
generalizing a TIGHTLY-GATED recognizer re-opens every gap the original gate quietly covered — the original's
specific gates (exactly-2-entities, scaled-quantifier) were load-bearing; dropping them needs REPLACEMENT
discipline (ratio-frame, offset-system rejection, unique-root tree), and the holdout is what tells you which
gates you still owe. Also: when a sub-relation is REJECTED (the offset clause), don't solve the remainder —
an incomplete system silently commits a wrong; decline the whole thing.  ·
116. **scene-depletion remainder** (dev-train 55→57: Candy, cookie-jar). "There were 349 pieces of candy in
a bowl. Talitha took 108, Solomon took 153. How many left?" → 349 − 108 − 153 = 88. The existential "there
were N X" stores N INSTANCES (a count base), not a holding Quantity, and each take is credited to the takER
(not debited from the bowl), so the remainder fold finds no single owner to net — AND the 349 reads as an
unused given, so GUARD 3 killed it. `_scene_depletion(question, context, prem_readings)`: base = #instances
of the measure; answer = base − Σ(the measure's depletion-verb Quantities). Tried early in solve() (before
GUARD 3) and mirrored into the CLI loop (#104). GATED — the firewall caught 2 dev over-fires (James-milk used
the "128 ounces in a gallon" CONVERSION factor as the base; Pizza used "8 slices in a pizza" ignoring "2
pizzas"): every measure-Quantity must be a depletion verb (an addition breaks base − Σ), and NO unused given
— every stated premise number must be the base or a depletion, so a conversion factor / per-unit count (the
base needs computing) declines. holdout 0 delta; TEST held 61.5%. 849 tests. LESSON: an existential "there
were N X" is a COUNT (instances), not a holding — a remainder over it has no owner to net and trips GUARD 3;
recognise the pool explicitly. And the no-unused-given gate is the clean discriminator between a plain pool
(every number is base/depletion) and a computed base (an extra conversion/per-unit number) — reused from
GUARD 3's principle.  ·
117. **sum+ratio system with KNOWN-absolute parts** (extends #3r/#115; dev-train 57→58: Delta). #3r solved
sum-constraint systems where every part is a ratio of the pivot; this adds a part with a stated ABSOLUTE
value: "Delta, Beta and Gamma share 40 donuts. Delta took 8. Beta took 3× as many as Gamma. Gamma?" → Gamma
= (40 − 8) / (3 + 1) = 8. `_sum_ratio_system` now collects KNOWNS (an entity with a stated value of the asked
measure, no ratio/offset); the pivot is the one entity that is neither a ratio-subject nor known; solves
pivot = (Total − Σknowns) / Σcoeff. Broadened the total cue with share/shared/altogether. Backward-compatible
(no knowns → exactly #3r's Total/Σcoeff). All #3r/#3h gates preserved (unit/qualifier match, offset-tainted
rejection) → Felipe still declines. holdout 42→43 / 0 new-wrong; TEST held 61.5%. 850 tests. LESSON: a fresh
recognizer's natural generalizations (here knowns alongside ratios) are cheap, clean follow-ons IF you
preserve its hard-won gates — the regression surface was already paid down in #115's holdout cycle.  ·

118. **offset sum-system** (the affine twin of #3h/#115; dev-train 58→61, holdout 43→45 generalized, TEST held
61.5%). #3h/#115/#117 fold only PURE-RATIO systems and explicitly DECLINE any offset-tainted one ("5 more than
twice"). This adds the affine relation subject = factor × standard + offset: "Justin + Sabrina have 50 pencils;
Justin has 8 more than twice as many as Sabrina → Sabrina = (50 − 8)/(2+1) = 14"; "Devin = ½·Tom − 5, total 70
→ Tom = (70 − (−5))/1.5 = 50"; "Antoinette = 2·Rupert − 7, together 98 → Antoinette 63". New `_offset_ratio_
relation` (POSITIONAL parse: the offset value precedes the comparative "8 **more**", the factor is a multiplier
AFTER "than" "twice"; requires an explicit factor so a bare "N more than Y" stays the firewall-risky declined
shape) + `_offset_sum_system` (2-entity solve std = (T−c)/(k+1), subj = T−std). Kept SEPARATE from
`_sum_ratio_system` to isolate the riskier offset surface; the shared `_combined_total` was factored out of #3h's
total-finder (used by both). GATES: EXACTLY one offset relation, NO pure-ratio relation alongside (a mixed/3-way
system stays out of scope → declines, preserving #115's `test_offset_tainted_system_declines`), the asked entity
among the two, and #3h's measure/qualifier match on the total. A/B both ways on BOTH slices: dev-train +3 / 0
lost / 0 new wrong; **holdout +2 (Sara/Joe, Rajesh/Hiro both generalized — strong anti-overfit signal)** / 0
new wrong; TEST unchanged 61.5%. 608 reading tests. LESSON: when an existing recognizer DECLINES a near-shape it
was deliberately gated against, the affine/offset extension is a clean SIBLING (not a generalization-in-place) —
a separate function reusing the shared total-finder isolates the new regression surface from the mature path's
hard-won gates, and the holdout proves it generalized rather than memorized.  ·

119. **"compared to <name>" difference fold** (dev-train 61→62: Scott/Anthony/Jim, holdout/TEST held). FOUND by
a FOLD-REACHABLE miner (build each fully-read decline's context to a fixpoint, then test whether the gold is a
simple combination — Σ holdings, a difference, a product — of the already-resolved quantities; the largest such
group is where parsing already succeeded and only an answer-fold is missing). "How many more pairs of shoes does
Anthony have compared to Jim?" had Anthony=21 and Jim=19 both resolved, but the difference branch declined: the
comparative "more" greedily binds its `standard` to the adjacent MEASURE noun ("more PAIRS"), not the entity, and
the real standard sits after "compared to" (which the than-only branch never reads). FIX: when the comp-bound
standard isn't a NAME and the question carries a comparison marker (than/compared/versus), recover the standard
as the non-subject NAME (Anthony − Jim = 2). The pre-existing "than <NAME>" path is untouched (its standard is
already a name → fallback skipped). dev-train +1 / 0 new wrong; holdout + TEST held (no such problem there, but
no regression). 610 tests. LESSON: the fold-reachable miner is the right tool once the clean recognizer vein is
worked out — it separates PARSE-blocked declines (the sub-facts never resolve; multi-piece, skip) from FOLD-
blocked ones (sub-facts resolve, one answer-path is missing; a cheap, low-risk win). Most "in all"/sum declines
turned out parse-blocked (e.g. "how many CANS" reads 'cans' as the modal auxiliary 'can', so the query measure
never resolves) — only the difference-fold case was genuinely fold-only.  ·

120. **pool-and-redistribute (equal-share)** (dev-train 62→64: Mark/Peter/John 30, Hugh/Tommy/Melany 7; holdout
+ TEST held). Also a FOLD-REACHABLE find: "Mark has 30 candies, Peter 25, John 35. They combine and share them
equally. How many will EACH have?" — all three holdings resolved, only Σ/N missing. Same math as the average
branch (#3l) but cued by a SHARE-EQUALLY premise + a per-each result question, so it lives in solve() (the cue
isn't in the question). `_equal_share`: a share/divide/combine + equal cue in a premise; question "how many/much
… EACH … have/get"; single shared measure with ≥2 owners; Σ(each owner's holding)/owner-count. THE FIREWALL ARC
(three over-fires caught, three principled gates added): (1) a Distribution fact in context = the distinct
distributor-among-a-COUNT shape (Mr. Sam $5250÷3) whose synthesized per-recipient share would be wrongly pooled
with the distributor's holding → defer to the existing handler; (2) "the rest/remaining/kept" = a remainder
shared to OTHERS (Harold kept 20, shared the rest among 5 FRIENDS → 80÷5, not Σ÷N) → decline; (3) a NOUN-entity
owner (entity == measure head, from "the rest" → entity 'marble') is excluded, and a NON-INTEGER result (a
relation didn't resolve — Wolfgang's 40/3 because the "2/3 of both" relation deferred) → decline. Also: "each
ONE of them" makes "one"=value 1, so the set-size safety check must ignore a stated count of 1. dev-train +2 / 0
new wrong; holdout 45/3 held (the first A/B regressed it 45/5 — the three gates recovered it); TEST held 61.5%.
861 tests. LESSON: a Σ/N pool recognizer's danger is MIS-COUNTING the denominator — exclude noun-entity "owners"
(remainder nouns) and the distributor-among-others shape, and let a non-integer result veto a half-resolved
context. The holdout caught all three; each became a reusable precision gate.  ·

121. **abstract-measure accumulation** (dev-train 64→65: Hazel; holdout + TEST held). Fold-reachable: "Hazel
walked 2 km in the first hour. She walked twice as far in the second hour. How FAR did she walk for 2 hours?" —
both facts resolved (2, 4 → hazel/kilometer) but the answer path raised "couldn't tell what to count" because
"how far" carries no measure noun (query=None). FIX in answer()'s query=None branch: when the context holds ONE
measure, ONE owner, and ≥2 quantities, return that owner's net holding. The ≥2 gate is the discriminator — it
fires only on an ACCUMULATION the measure-less question couldn't dispatch, while a SINGLE-fact "how much did Tom
run?" (5 miles) keeps DECLINING per #74's deliberate no-guess design (don't answer a lone holding when the
question's measure is unreadable). dev-train +1 / 0 new wrong; holdout + TEST held. 863 tests. LESSON: a query=None
fallback is safe only if it requires MORE than a single lookup — gate on the accumulation (≥2 facts the normal
fold missed), so it rescues a genuine sum without overriding a conservative single-fact decline.  ·

122. **label-letter entities ("Jar A" / "Class B")** — a deliberate PARSE fix (dev-train 65→66: Jar A/B/C 148;
holdout + TEST held). "Jar A has 28. Jar B has 12 more than jar A. Jar C has twice as many as jar B. How many
altogether?" declined because "Jar A"/"Jar B"/"Jar C" ALL collapse to the bare noun 'jar' ("A"/"B"/"C" read as
article/name) — so the three jars are one entity and the chain can't form. VERIFIED FIRST that the only blocker
was the name collapse: substituting distinct proper names (Alpha/Beta/Gamma) solved it via the existing forward-
chain + combined-sum machinery (148). FIX: a new merge pass `_merge_label_letters` (after `_merge_labels`, before
disambiguate) merges a NOUN/NAME + a single capital letter into one NAME unit, keyed 'jar a'/'jar b'/'jar c' like
proper names. Fires on a LOWERCASE head too ("…than jar A") so a subject ("Jar B") and a later standard ("jar B")
key identically. A sentence-initial article ("A teacher") has no preceding noun → untouched. Despite being a
BROAD parse change (every analyze()), the narrow single-uppercase-letter gate kept the blast radius tiny: 866
tests, dev-train +1 / 0 new wrong, holdout + TEST held (0 regression). LESSON: a parse fix the firewall log
flagged as "high-blast-radius" (cf. the reverted noun-noun head-walk) can be SAFE if gated on a narrow, specific
surface signal (one capital letter, not a general compound-head rule) — and ALWAYS verify the fix is the SOLE
blocker first (substitute clean names) so you don't build parse machinery for a problem that still won't solve.  ·

123. **noun-after-quantifier + "in all" cue** (dev-train 66→67: Solomon; holdout + TEST held). Two coupled
fixes for "How many cans did the boys collect in all?": Solomon=66, Juwan=66/3=22, Levi=22/2=11 → Σ=99 (the
relations + backward inversion already resolved all three — this was FOLD-blocked). Blockers: (1) "cans"
disambiguated to the MODAL 'can' (it's `["auxiliary","noun"]`) because a nearby "did" pulled it there, so "many"
bound the next noun 'boys' → the Σ found no 'boy' quantities → decline; (2) "in all" wasn't a combined-total cue
(combined_q only knew combined/altogether/total or all+together). FIX: `_prefer_noun_after_quantifier` (before
disambiguate) drops the auxiliary/modal sense from a word right after "many"/"much" that also has a noun sense
(a quantifier counts a noun, never a modal); and "in all" added to combined_q. NARROW: the noun-pref fires only
immediately after many/much (an adjective like "how many RED apples" is untouched — 'red' has no noun sense to
prefer; a true modal "she can run" is nowhere near a quantifier). dev-train +1 / 0 new wrong; holdout + TEST
held. 869 tests. LESSON: a disambiguation bug ("cans"→modal) can MISDIRECT the quantifier binding two words
downstream ("many"→'boys'), so the visible symptom (Σ over the wrong measure) is far from the cause — trace the
collection the interrogative actually bound, not just the final fold. Coupled fixes (parse + cue) often both
needed: the gq-inference covered the measure but the quantifier still mis-bound, and "in all" was a separate
gate.  ·

A **collection descriptor** = `(head noun, set of atomic attribute predicates)`. `apple`=`(apple,{})`,
`red apple`=`(apple,{red})`, `rotten green apple`=`(apple,{green,rotten})`. Sub-collection = same
head, attribute-superset; a sub-group's count = instances whose attributes ⊇ the query's. We store
the shape now (free); aggregation/counting is *derived later* when a question's obligation needs it.

## Open threads / next

- **GSM8K via this parser (replace the LLM translator)** — kept as an INDEPENDENT eval (a `bench`-style
  harness, never in `test_reading.py`). Roadmap, simplest-first: (1) pure arithmetic ✓; (2) `Quantity`
  facts ("Tom has 3 apples") ✓; (3) aggregation questions ("how many … altogether") ✓; (4) accumulation on
  one entity ✓; (5a) **coreference** (`he`/`she`) ✓; (5b) "half/twice as many" **relational
  quantities** ✓ (implicit referent); "the remaining", explicit "as many as X" queued; (6) units + **world-knowledge conversions** (hour=60min,
  year=52wk). Build a math/quantity layer in `reading.py`; reuse `translate.run_program` to evaluate.
- **"N times" STATEMENT form** ("Mary jumps 2 times") — slice 2 of frequency: the multiplier words
  (twice/thrice) ✓; the "N times" form needs the frequency-noun "times" to modify the verb to its LEFT
  (a `scales`-like obligation on the noun reading), distinct from a subject-noun's predicate. Lead-in
  to (5b) relational quantities, where the same multiplier applies to a referent quantity.

- **`boolean` sense of `no`** (decided: leave as-is): no meaningful obligation exists in this
  statement grammar — `no` reliably reads as a quantifier (its `collection` always binds), and a
  standalone polar `No` isn't ingested. Kept as the multi-sense demonstrator (#5); a forced
  obligation would never fire.
- **Verb lemmas** are a small hand-list (`VERB_LEMMA`); grows by deliberate addition like the other
  dictionaries (no stemming guesswork).
- **Overlapping facts** (documented limitation): counts assume DISJOINT sub-collections — correct
  for "3 red, 2 green" → 5, but "3 red + 1 rotten red" double-counts. A correct fix is
  underdetermined without instance identity (we can't verify overlap); would need a count-layer
  redesign to `(descriptor → count)` facts with exact-match + partition logic. Left as-is.
- **Aggregation beyond count**: sum across heads, max, "which has the most" (count comparison done).

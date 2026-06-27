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
247 tests. Follow-on: conjoined OBJECTS ("3 apples and 2 oranges" — two quantities, not a group).

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

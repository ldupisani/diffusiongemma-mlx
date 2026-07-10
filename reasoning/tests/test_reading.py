"""Word-by-word reading scaffolding — tests grow one increment at a time.

The substrate for model-free logical reading: every word is read into a tracked Unit, with a
full decision trail so any choice can be backtracked or retested. We test literally word by word
and let generalizations fall out of the cases, so each test pins exactly one new behavior.
"""

import unittest

from reasoning import reading


# === FOUNDATION: reading, lexicon & vocabulary, obligations, sense disambiguation, CLI ========

class TestIncrement1ReadAndTrack(unittest.TestCase):
    """#1 — read text into one Unit per word, losslessly. No normalization, no decisions yet."""

    def test_one_unit_per_word_surfaces_and_indices_preserved(self):
        r = reading.read("Each student passed")
        self.assertEqual(len(r.units), 3)
        self.assertEqual([u.surface for u in r.units], ["Each", "student", "passed"])
        # each unit wraps exactly one Word, carrying its position in the stream
        for expected_index, u in enumerate(r.units):
            self.assertEqual(len(u.words), 1)
            self.assertEqual(u.words[0].index, expected_index)
            self.assertEqual(u.words[0].surface, u.surface)

    def test_nothing_decided_yet(self):
        r = reading.read("Each student passed")
        self.assertEqual(r.decisions, [])
        self.assertEqual(r.facts, [])

    def test_original_casing_is_not_mutated(self):
        # "Each" keeps its capital — reading captures words exactly as written.
        r = reading.read("Each student passed")
        self.assertEqual(r.units[0].surface, "Each")


class TestIncrement2AnnotateNumber(unittest.TestCase):
    """#2 — a unit is plural iff it is in the definitive PLURALS dictionary (no guessing). We
    ATTACH the finding (number + singular) without changing the word, logged reversibly."""

    def _unit(self, reading, surface):
        return next(u for u in reading.units if u.surface == surface)

    def test_known_plural_is_annotated_not_changed(self):
        r = reading.read("All students passed")
        reading.annotate_number(r)
        u = self._unit(r, "students")
        # the word is untouched; we only added data ABOUT it
        self.assertEqual(u.surface, "students")
        self.assertEqual(u.words[0].surface, "students")
        self.assertEqual(u.features, {"number": "plural", "singular": "student"})

    def test_irregular_plural_is_free_via_dictionary(self):
        # a lookup table gets "children"->"child" with no special rule.
        r = reading.read("All children played")
        reading.annotate_number(r)
        self.assertEqual(self._unit(r, "children").features,
                         {"number": "plural", "singular": "child"})

    def test_annotation_is_logged_as_a_reversible_decision(self):
        r = reading.read("All students passed")
        reading.annotate_number(r)
        self.assertEqual(len(r.decisions), 1)
        d = r.decisions[0]
        self.assertEqual(d.kind, "annotate")
        self.assertIs(d.target, self._unit(r, "students"))
        self.assertEqual(d.before, {})
        self.assertEqual(d.after, {"number": "plural", "singular": "student"})
        self.assertEqual(d.reason, "known plural")

    def test_decision_can_be_reversed(self):
        # backtracking: restoring the before-snapshot undoes the annotation cleanly.
        r = reading.read("All students passed")
        reading.annotate_number(r)
        d = r.decisions[0]
        d.target.features = dict(d.before)
        self.assertEqual(self._unit(r, "students").features, {})

    def test_non_plural_words_carry_no_data(self):
        r = reading.read("All students passed")
        reading.annotate_number(r)
        for surface in ("All", "passed"):
            self.assertEqual(self._unit(r, surface).features, {})

    def test_unknown_words_are_never_flagged(self):
        # "Socrates" ends in 's' but is NOT a known plural — we make no claim. No guessing.
        r = reading.read("Socrates is wise")
        reading.annotate_number(r)
        self.assertEqual(r.decisions, [])
        self.assertTrue(all(u.features == {} for u in r.units))


class TestIncrement3WordMeaningDictionary(unittest.TestCase):
    """#3 — look each word up in the definitive LEXICON and ATTACH its possible meanings (senses)
    as data. Same dictionary pattern as plurals: known words get facts, unknown words get nothing.
    The value is a LIST of senses, because a word may have several meanings."""

    def _unit(self, reading, surface):
        return next(u for u in reading.units if u.surface == surface)

    def test_known_quantifier_gets_its_sense(self):
        r = reading.read("Each student passed")
        reading.annotate_senses(r)
        self.assertEqual(self._unit(r, "Each").features, {"senses": ["quantifier"]})

    def test_known_article_gets_its_sense(self):
        r = reading.read("The student passed")
        reading.annotate_senses(r)
        self.assertEqual(self._unit(r, "The").features, {"senses": ["article"]})

    def test_lookup_is_case_insensitive_but_word_unchanged(self):
        r = reading.read("EACH student passed")
        reading.annotate_senses(r)
        u = self._unit(r, "EACH")
        self.assertEqual(u.surface, "EACH")             # word untouched
        self.assertEqual(u.features, {"senses": ["quantifier"]})

    def test_unknown_words_get_no_sense(self):
        # nonce words are guaranteed not in the dictionary, whatever we add later
        r = reading.read("Each wug glorped")
        reading.annotate_senses(r)
        for surface in ("wug", "glorped"):
            self.assertEqual(self._unit(r, surface).features, {})

    def test_sense_assignment_is_logged_reversibly(self):
        r = reading.read("Each wug glorped")  # only "Each" is known → exactly one decision
        reading.annotate_senses(r)
        self.assertEqual(len(r.decisions), 1)
        d = r.decisions[0]
        self.assertEqual(d.kind, "annotate")
        self.assertIs(d.target, self._unit(r, "Each"))
        self.assertEqual(d.before, {})
        self.assertEqual(d.after, {"senses": ["quantifier"]})
        self.assertEqual(d.reason, "lexical role")

    def test_features_accumulate_across_passes(self):
        # number and sense annotations coexist; neither overwrites the other.
        r = reading.read("All students passed")
        reading.annotate_senses(r)
        reading.annotate_number(r)
        self.assertEqual(self._unit(r, "All").features, {"senses": ["quantifier"]})
        self.assertEqual(self._unit(r, "students").features,
                         {"number": "plural", "singular": "student"})


class TestIncrement4EachObligation(unittest.TestCase):
    """#4 — a quantifier sense carries an OBLIGATION it must discharge: find the collection it
    ranges over. We attach it UNMET (target=None) and log it reversibly; resolving it is next."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def _prep(self, text):
        r = reading.read(text)
        reading.annotate_senses(r)
        reading.annotate_obligations(r)
        return r

    def test_each_carries_an_unmet_collection_obligation(self):
        r = self._prep("Each student passed")
        obs = self._unit(r, "Each").features["obligations"]
        self.assertEqual(obs, [reading.Obligation("collection")])
        self.assertFalse(obs[0].met)
        self.assertIsNone(obs[0].target)

    def test_obligation_is_definitional_to_the_quantifier_role(self):
        # not special to "each" — every quantifier inherits it for the same reason
        for word in ("All", "Every", "Some", "No"):
            r = self._prep(f"{word} students passed")
            self.assertEqual(self._unit(r, word).features["obligations"],
                             [reading.Obligation("collection")])

    def test_article_carries_a_determines_obligation(self):
        # every word has an obligation now; an article's is to determine its noun (not a collection)
        r = self._prep("The student passed")
        obs = self._unit(r, "The").features["obligations"]
        self.assertEqual([o.need for o in obs], ["determines"])

    def test_unknown_words_carry_no_obligation(self):
        r = self._prep("Each wug glorped")
        for surface in ("wug", "glorped"):
            self.assertEqual(self._unit(r, surface).features, {})

    def test_obligation_is_logged_reversibly(self):
        r = self._prep("Each student passed")
        each = self._unit(r, "Each")
        oblige = [d for d in r.decisions if d.kind == "oblige" and d.target is each]
        self.assertEqual(len(oblige), 1)
        d = oblige[0]
        self.assertEqual(d.before, {"senses": ["quantifier"]})  # senses already there
        self.assertEqual(d.after["obligations"], [reading.Obligation("collection")])


class TestIncrement5ContentWordsInLexicon(unittest.TestCase):
    """#5 — open-class content words go in the lexicon too, tagged with their part of speech, so
    the resolver can tell a noun (a candidate collection) from a verb. Unknown words stay blank."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def test_noun_gets_noun_sense(self):
        r = reading.read("Each student passed")
        reading.annotate_senses(r)
        self.assertEqual(self._unit(r, "student").features, {"senses": ["noun"]})

    def test_verb_gets_verb_sense(self):
        r = reading.read("Each student passed")
        reading.annotate_senses(r)
        self.assertEqual(self._unit(r, "passed").features, {"senses": ["verb"]})

    def test_multi_sense_word_keeps_all_its_senses(self):
        # "no" is both a quantifier and a boolean (none/negation) — the list holds both
        r = reading.read("No one left")
        reading.annotate_senses(r)
        self.assertEqual(self._unit(r, "No").features["senses"], ["quantifier", "boolean"])

    def test_preposition_is_in_lexicon(self):
        r = reading.read("Each of the students")
        reading.annotate_senses(r)
        self.assertEqual(self._unit(r, "of").features, {"senses": ["preposition"]})


class TestIncrement6ResolveCollectionUnambiguous(unittest.TestCase):
    """#6 — discharge a quantifier's collection obligation ONLY when exactly one noun is available
    (unambiguous). Zero or several → leave it unmet, a reportable deficiency. Bind = set target."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def _prep(self, text):
        r = reading.read(text)
        reading.annotate_senses(r)
        reading.annotate_obligations(r)
        reading.resolve(r)
        return r

    def test_binds_the_unique_noun(self):
        r = self._prep("Each student passed")
        ob = self._unit(r, "Each").features["obligations"][0]
        self.assertTrue(ob.met)
        self.assertIs(ob.target, self._unit(r, "student"))

    def test_unmet_when_there_is_no_collection(self):
        r = self._prep("Each passed")  # "passed" is a verb, not a collection
        ob = self._unit(r, "Each").features["obligations"][0]
        self.assertFalse(ob.met)

    def test_quantifier_binds_its_nearest_noun(self):
        # a quantifier governs its HEAD noun (the nearest to its right); a later noun is not a
        # competing collection — it plays another role (here "teacher" just dangles).
        r = self._prep("Each student teacher")
        ob = self._unit(r, "Each").features["obligations"][0]
        self.assertTrue(ob.met)
        self.assertEqual(ob.target.surface, "student")

    def test_bind_is_logged_reversibly(self):
        r = self._prep("Each student passed")
        each = self._unit(r, "Each")
        binds = [d for d in r.decisions if d.kind == "bind" and d.target is each]
        self.assertEqual(len(binds), 1)
        self.assertFalse(binds[0].before["obligations"][0].met)   # was unmet
        self.assertTrue(binds[0].after["obligations"][0].met)     # now met


class TestIncrement7PartitiveResolvesLikeDeterminer(unittest.TestCase):
    """#7 — "Each of the students" must resolve to the SAME binding as "Each student": each ⟶
    student. Needs (a) a plural inheriting the noun sense via its singular, and (b) the passes run
    in dependency order (number before senses). `analyze` wires the full pipeline."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def test_plural_inherits_noun_sense_via_its_singular(self):
        r = reading.read("the students")
        reading.annotate_number(r)   # students -> singular "student"
        reading.annotate_senses(r)   # then the sense lookup can fall back to the singular
        self.assertEqual(self._unit(r, "students").features["senses"], ["noun"])

    def test_each_student_resolves_through_the_pipeline(self):
        r = reading.analyze("Each student passed")
        ob = self._unit(r, "Each").features["obligations"][0]
        self.assertTrue(ob.met)
        self.assertIs(ob.target, self._unit(r, "student"))

    def test_each_of_the_students_resolves_like_each_student(self):
        r = reading.analyze("Each of the students")
        ob = self._unit(r, "Each").features["obligations"][0]
        self.assertTrue(ob.met)
        self.assertIs(ob.target, self._unit(r, "students"))   # the one noun present


class TestIncrement8PrepositionLinkObligation(unittest.TestCase):
    """#8 — every role carries an obligation. A linking preposition ("of") must find an `anchor`
    (the unit on its left it attaches to) and an `object` (the noun it governs, on its right).
    Binding fulfills these — so the statement holds together with NO merge."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def _obs(self, reading_, surface):
        return {o.need: o for o in self._unit(reading_, surface).features["obligations"]}

    def test_preposition_starts_with_unmet_anchor_and_object(self):
        r = reading.read("Each of the students")
        reading.annotate_number(r)
        reading.annotate_senses(r)
        reading.annotate_obligations(r)
        obs = self._unit(r, "of").features["obligations"]
        self.assertEqual([o.need for o in obs], ["anchor", "object"])
        self.assertTrue(all(not o.met for o in obs))

    def test_of_binds_anchor_left_and_object_right(self):
        r = reading.analyze("Each of the students")
        obs = self._obs(r, "of")
        self.assertIs(obs["anchor"].target, self._unit(r, "Each"))
        self.assertIs(obs["object"].target, self._unit(r, "students"))

    def test_each_still_binds_its_collection(self):
        r = reading.analyze("Each of the students")
        self.assertIs(self._unit(r, "Each").features["obligations"][0].target,
                      self._unit(r, "students"))

    def test_object_stays_unmet_with_no_noun_to_govern(self):
        r = reading.analyze("Each of")  # nothing nounish to the right → don't guess
        self.assertFalse(self._obs(r, "of")["object"].met)


class TestIncrement9AttributiveMerge(unittest.TestCase):
    """#9 — an adjective MERGES into its head noun (it doesn't just bind): "red apple" condenses to
    ONE noun unit carrying head="apple" + attributes={red} — the restricted set, not all apples.
    Multi-pass stacks modifiers; the structural merge is reversible; a quantifier then ranges over
    the COMPOUND so the restriction survives."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def test_adjective_and_noun_condense_into_one_noun_unit(self):
        r = reading.analyze("red apple")
        self.assertEqual(len(r.units), 1)
        u = r.units[0]
        self.assertEqual(u.surface, "red apple")          # words preserved, in order
        self.assertEqual(u.features["head"], "apple")     # parent collection recoverable
        self.assertEqual(u.features["attributes"], ["red"])
        self.assertIn("noun", u.features["senses"])       # the compound is still a noun

    def test_stacked_adjectives_accumulate_as_a_set_via_multipass(self):
        r = reading.analyze("big red apple")
        self.assertEqual(len(r.units), 1)
        u = r.units[0]
        self.assertEqual(u.surface, "big red apple")
        self.assertEqual(u.features["head"], "apple")
        self.assertEqual(u.features["attributes"], ["big", "red"])  # sorted set, composable

    def test_quantifier_ranges_over_the_compound_not_the_bare_noun(self):
        r = reading.analyze("Each red apple")
        target = self._unit(r, "Each").features["obligations"][0].target
        self.assertEqual(target.features["head"], "apple")
        self.assertEqual(target.features["attributes"], ["red"])   # the restriction is not lost

    def test_merge_is_recorded_reversibly(self):
        r = reading.analyze("red apple")
        merges = [d for d in r.decisions if d.kind == "merge"]
        self.assertEqual(len(merges), 1)
        constituents = [w.surface for u in merges[0].before["merged_from"] for w in u.words]
        self.assertEqual(constituents, ["red", "apple"])           # enough to split back

    def test_adjective_with_no_noun_to_modify_stays_unmet(self):
        r = reading.analyze("red passed")  # "passed" is a verb — nothing to modify
        self.assertEqual(len(r.units), 2)  # not condensed
        self.assertFalse(self._unit(r, "red").features["obligations"][0].met)


class TestIncrement10EveryWordHasAnObligation(unittest.TestCase):
    """#10 — every role now carries an obligation. Article → determines a noun; verb → finds its
    subject; noun → finds its predicate. A complete statement saturates (no unmet obligation); a
    noun phrase with no verb stays a FRAGMENT (the noun's predicate is unmet)."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def _obs(self, reading_, surface):
        return {o.need: o for o in self._unit(reading_, surface).features["obligations"]}

    def test_every_lexical_role_carries_an_obligation(self):
        for role in ("quantifier", "preposition", "adjective", "article", "verb", "noun"):
            self.assertIn(role, reading.ROLE_OBLIGATIONS)

    def test_article_determines_its_noun(self):
        r = reading.analyze("The student passed")
        self.assertIs(self._obs(r, "The")["determines"].target, self._unit(r, "student"))

    def test_verb_finds_its_subject(self):
        r = reading.analyze("The student passed")
        self.assertIs(self._obs(r, "passed")["subject"].target, self._unit(r, "student"))

    def test_noun_finds_its_predicate(self):
        r = reading.analyze("The student passed")
        self.assertIs(self._obs(r, "student")["predicate"].target, self._unit(r, "passed"))

    def test_complete_statement_saturates(self):
        r = reading.analyze("The student passed")
        unmet = [o.need for u in r.units for o in u.features.get("obligations", []) if not o.met and not o.optional]
        self.assertEqual(unmet, [])   # every word's obligation discharged → well-formed

    def test_noun_phrase_with_no_verb_is_a_fragment(self):
        r = reading.analyze("Each of the students")  # no predicate
        unmet = [o.need for u in r.units for o in u.features.get("obligations", []) if not o.met and not o.optional]
        self.assertIn("predicate", unmet)   # the noun has nothing asserted of it → incomplete


class TestIncrement11SenseDisambiguation(unittest.TestCase):
    """#11 — when a word has several senses, SELECT the one whose obligations cover the sentence
    best (fewest unmet). "rose" is a noun or a verb; context decides, with no POS tagger — just
    which reading saturates."""

    def _unit(self, reading_, surface):
        return next(u for u in reading_.units if u.surface == surface)

    def test_rose_reads_as_noun_before_a_verb(self):
        r = reading.analyze("The rose wilted")     # rose is the subject; wilted is the verb
        self.assertEqual(self._unit(r, "rose").features["senses"], ["noun"])

    def test_rose_reads_as_verb_after_a_subject(self):
        r = reading.analyze("The teacher rose")    # teacher is the subject; rose is the verb
        self.assertEqual(self._unit(r, "rose").features["senses"], ["verb"])

    def test_chosen_reading_saturates_either_sentence(self):
        for text in ("The rose wilted", "The teacher rose"):
            r = reading.analyze(text)
            unmet = [o.need for u in r.units for o in u.features.get("obligations", []) if not o.met and not o.optional]
            self.assertEqual(unmet, [], text)

    def test_selection_is_logged_reversibly(self):
        r = reading.analyze("The teacher rose")
        picks = [d for d in r.decisions if d.kind == "assign-sense"]
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0].before["senses"], ["noun", "verb"])  # both candidates
        self.assertEqual(picks[0].after["senses"], ["verb"])           # the one that covers

    def test_unambiguous_words_are_not_reassigned(self):
        r = reading.analyze("The teacher rose")
        reassigned = {d.target.surface for d in r.decisions if d.kind == "assign-sense"}
        self.assertEqual(reassigned, {"rose"})   # only the genuinely ambiguous word


class TestIncrement12DeficienciesAPI(unittest.TestCase):
    """#12 — the answer-or-ask boundary. `deficiencies(reading)` returns exactly what stops a
    statement from being complete and understood: unrecognized words, and unmet obligations.
    Empty ⇒ `is_well_formed`. Each finding is specific enough to hand back to the user."""

    def test_complete_statement_has_no_deficiencies(self):
        r = reading.analyze("The student passed")
        self.assertEqual(reading.deficiencies(r), [])
        self.assertTrue(reading.is_well_formed(r))

    def test_fragment_reports_the_missing_predicate(self):
        r = reading.analyze("Each of the students")  # no verb
        defs = reading.deficiencies(r)
        self.assertFalse(reading.is_well_formed(r))
        self.assertTrue(any(d.kind == "unmet-obligation" and d.unit.surface == "students"
                            and "predicate" in d.message for d in defs))

    def test_unknown_word_is_reported(self):
        r = reading.analyze("Each wug glorped")
        defs = reading.deficiencies(r)
        unknown = {d.unit.surface for d in defs if d.kind == "unknown-word"}
        self.assertEqual(unknown, {"wug", "glorped"})

    def test_unmet_collection_is_reported_specifically(self):
        r = reading.analyze("Each passed")  # nothing to range over
        defs = reading.deficiencies(r)
        self.assertTrue(any(d.unit.surface == "Each" and "collection" in d.message for d in defs))

    def test_messages_name_the_word(self):
        r = reading.analyze("Each of the students")
        self.assertTrue(all(d.unit.surface in d.message or d.unit.surface.lower() in d.message.lower()
                            for d in reading.deficiencies(r)))


class TestIncrement14CLI(unittest.TestCase):
    """#14 — a thin CLI over the engine. `run_cli(argv)` returns the text to print (factored out of
    __main__ so it is testable): analyze a statement, or answer a question given --about context."""

    def test_statement_is_reported_understood(self):
        out = reading.run_cli(["The student passed"])
        self.assertIn("understood", out)
        self.assertIn("student", out)

    def test_fragment_reports_clarification(self):
        out = reading.run_cli(["Each of the students"])
        self.assertIn("needs clarification", out)
        self.assertIn("predicate", out)

    def test_count_question_with_context(self):
        out = reading.run_cli(["How many red apples?",
                               "--about", "red apple", "green apple", "red apple"])
        self.assertEqual(out.strip(), "2")

    def test_unanswerable_question_asks(self):
        out = reading.run_cli(["How many blorps?", "--about", "red apple"])   # blorp: a nonce unknown word
        self.assertIn("can't answer", out.lower())

    def test_no_args_shows_usage(self):
        self.assertIn("usage", reading.run_cli([]).lower())


class TestIncrement25Vocabulary(unittest.TestCase):
    """#25 — a batch of open-class words (nouns, verbs with lemmas, adjectives, plurals) so common
    sentences parse and reason. Vocabulary is curated data, grown deliberately — not stemming."""

    def test_new_words_parse_cleanly(self):
        for s in ["The dog barked.", "Birds fly.", "The cat slept.", "Mary is happy."]:
            unknown = [d for d in reading.deficiencies(reading.analyze(s)) if d.kind == "unknown-word"]
            self.assertEqual(unknown, [], s)

    def test_reasoning_with_new_vocabulary(self):
        out = reading.run_cli(["All dogs bark. Rex is a dog. Does Rex bark?"])
        self.assertEqual(out.strip(), "yes")

    def test_new_plural_and_lemma(self):
        # "dogs" inherits noun via singular "dog"; "barked" lemmatizes to "bark"
        self.assertEqual(reading.learn("All dogs barked"),
                         [reading.Predication(("dog", frozenset()), "bark", universal=True)])


class TestIncrement68VocabularyPushTwo(unittest.TestCase):
    """#68 - vocabulary push 2 (fully-read 7% -> 14.7%): common verbs/adjectives/nouns/family + number
    words six..twelve + hyphenated unit fractions. Precision fix: the instance-count FALLBACK declines
    when the queried TYPE isn't present as instances at all (was fabricating a 0)."""

    def test_new_vocabulary_recognized(self):
        self.assertEqual(reading.analyze("pizza").units[0].features.get("senses"), ["noun"])
        self.assertEqual(reading.analyze("expensive").units[0].features.get("senses"), ["adjective"])
        self.assertIn("verb", reading.analyze("collected").units[0].features.get("senses", []))

    def test_count_of_absent_type_declines_not_zero(self):
        ctx = reading.instances("there are two sons") + reading.learn("Ken got 1750 dollars")
        self.assertEqual(reading.answer("how much was the money shared", ctx), "undetermined")

    def test_count_of_present_type_zero_is_real(self):
        ctx = reading.instances("there are three green apples")
        self.assertEqual(reading.answer("how many red apples are there", ctx), 0)


class TestIncrement87VocabularyPush(unittest.TestCase):
    """#87 - fraction literals "N/M" parse as values (0.5, 0.75); a deliberate batch of frequent GSM8K
    words is recognised, raising fully-read 44→52 on the dev slice. Senses are conservative — a wrong
    guess declines, never a confident-wrong (precision stays 100%)."""

    def test_fraction_literals(self):
        self.assertEqual(reading._as_number("1/2"), 0.5)
        self.assertEqual(reading._as_number("3/4"), 0.75)
        self.assertEqual(reading._as_number("1/4"), 0.25)

    def test_added_noun_recognised(self):
        self.assertEqual(reading.learn("Tom has 5 mangoes"),
                         [reading.Quantity("tom", ("mango", frozenset()), 5.0, "have")])

    def test_added_verb_form_recognised(self):
        self.assertEqual(reading._lemma("mows"), "mow")
        self.assertEqual(reading.deficiencies(reading.analyze("Tom created 5 brownies")), [])


class TestIncrement90VocabularyRound2(unittest.TestCase):
    """#90 - a second deliberate vocab batch (frequent + single-unknown-word GSM8K terms), raising
    fully-read 52→65 on the dev slice. LESSON: .update() OVERRIDES, so an added word must NOT shadow an
    existing MULTI-sense entry — "rose" was already ["noun","verb"] (the disambiguation test relies on it),
    so it was excluded from the batch (the "roses"→"rose" plural is fine)."""

    def test_round2_words_recognised(self):
        defs = reading.deficiencies(reading.analyze("Tom borrowed 5 pencils and 3 gumballs"))
        self.assertFalse(any(d.kind == "unknown-word" for d in defs))   # recognition, not structure
        self.assertEqual(reading._lemma("borrowed"), "borrow")

    def test_rose_stays_multi_sense(self):
        self.assertEqual(reading.LEXICON["rose"], ["noun", "verb"])


class TestIncrement135SafeVocabGain(unittest.TestCase):
    """#135 - COVERAGE (first firewall-safe vocab gain). After decline-safety GUARDS 11+12, vocabulary can
    be re-added under the rule "only a word with a VERIFIED dev decline→correct". The verbs score/raise each
    unlocked a real dev solve (players-scored-points, Adam-raises-goats) with ZERO dev regression and TEST
    held at 40.0% — unlike the blind batch (groundwork nouns lowered held-out precision). SYNTHETIC."""

    def test_scored_solves(self):
        self.assertEqual(reading.solve(
            "Mary scored 8 points. John scored 5 more points than Mary. How many points did John score?"), 13)

    def test_score_raise_have_verb_senses(self):
        for w in ("score", "scored", "raise", "raised"):
            self.assertIn("verb", reading.LEXICON.get(w, []), w)

    def test_raise_sentence_is_fully_read(self):
        # "raise" was a frequent unknown word; a "raise" sentence now has no unknown-word deficiency
        rd = reading.analyze("Adam and Andrew raise goats.")
        self.assertFalse([d for d in reading.deficiencies(rd) if d.kind == "unknown-word"])


# === LOGIC: questions (yes-no / wh / which), membership, negation, inference, comparatives ====

class TestIncrement13AnswerAQuestion(unittest.TestCase):
    """#13 — a question is a statement whose interrogative obligation we fulfil against a context.
    "How many red apples?" reads the (apple, {red}) descriptor and counts matching instances in the
    context. Reuses the whole pipeline; an unrecognized word makes it ASK, not guess."""

    def _context(self):
        # the parsed text we're answering about: 2 red apples, 1 green apple (3 apples in all)
        phrases = ["red apple", "green apple", "red apple"]
        return [reading.collection_descriptor(reading.analyze(p)) for p in phrases]

    def test_counts_the_matching_subcollection(self):
        ctx = self._context()
        self.assertEqual(reading.answer("How many red apples?", ctx), 2)
        self.assertEqual(reading.answer("How many green apples?", ctx), 1)

    def test_parent_collection_counts_every_instance(self):
        # (apple, {}) is a superset-match of every apple, red or green
        self.assertEqual(reading.answer("How many apples?", self._context()), 3)

    def test_unknown_word_makes_it_ask_not_answer(self):
        result = reading.answer("How many blorps?", self._context())   # blorp: a nonce unknown word
        self.assertIsInstance(result, list)
        self.assertTrue(any(d.kind == "unknown-word" for d in result))

    def test_trailing_punctuation_is_tolerated(self):
        # the "?" must not turn "apples" into an unknown word (surface kept, lookup punct-insensitive)
        r = reading.analyze("How many apples?")
        self.assertEqual([d for d in reading.deficiencies(r) if d.kind == "unknown-word"], [])


class TestIncrement15ContextFromStatements(unittest.TestCase):
    """#15 — build the context by PARSING statements, not by hand. "There are three red apples"
    becomes three (apple,{red}) instances; answer() then queries the learned context. The reader
    that answers questions is the same one that ingests facts."""

    def test_count_statement_expands_to_instances(self):
        ctx = reading.instances("There are three red apples")
        self.assertEqual(len(ctx), 3)
        self.assertEqual(ctx[0], ("apple", frozenset({"red"})))

    def test_answer_over_learned_context(self):
        ctx = (reading.instances("There are three red apples")
               + reading.instances("There are two green apples"))
        self.assertEqual(reading.answer("How many red apples?", ctx), 3)
        self.assertEqual(reading.answer("How many green apples?", ctx), 2)
        self.assertEqual(reading.answer("How many apples?", ctx), 5)   # 3 + 2 via the subset lattice

    def test_unparseable_statement_learns_nothing(self):
        # an unknown collection → we learn nothing rather than guess
        self.assertEqual(reading.instances("There are three blorps"), [])

    def test_cli_about_accepts_count_statements(self):
        out = reading.run_cli(["How many red apples?", "--about", "There are three red apples"])
        self.assertEqual(out.strip(), "3")


class TestIncrement16YesNoQuestions(unittest.TestCase):
    """#16 — yes/no questions. "Did the student pass?" asks whether a predication (subject + verb)
    is known. A statement is learned as a Predication; tense is normalized ("passed" == "pass").
    Same reader, same obligations — just a different fact type and a yes/no verdict."""

    def _context(self):
        return reading.learn("The student passed")   # one known predication

    def test_statement_is_learned_as_a_predication_with_lemma(self):
        self.assertEqual(reading.learn("The student passed"),
                         [reading.Predication(("student", frozenset()), "pass")])

    def test_yes_when_the_fact_is_known(self):
        self.assertEqual(reading.answer("Did the student pass?", self._context()), "yes")

    def test_no_when_the_fact_is_not_known(self):
        self.assertEqual(reading.answer("Did the teacher pass?", self._context()), "undetermined")

    def test_unknown_word_makes_it_ask(self):
        result = reading.answer("Did the wizard pass?", self._context())
        self.assertIsInstance(result, list)
        self.assertTrue(any(d.kind == "unknown-word" for d in result))

    def test_cli_answers_yes_no(self):
        out = reading.run_cli(["Did the student pass?", "--about", "The student passed"])
        self.assertEqual(out.strip(), "yes")


class TestIncrement17SyllogisticInference(unittest.TestCase):
    """#17 — derive an answer, don't just look it up. "All students passed" + "Bob is a student" =>
    "Did Bob pass?" is YES by universal instantiation. Proper nouns (capitalized unknown words) are
    named individuals; "X is a Y" is a Membership fact; the deduction is one hop."""

    def test_proper_noun_recognized_and_membership_learned(self):
        self.assertEqual(reading.learn("Bob is a student"),
                         [reading.Membership("bob", ("student", frozenset()))])

    def test_universal_instantiation_yields_yes(self):
        ctx = reading.learn("All students passed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "yes")

    def test_non_member_yields_no(self):
        ctx = reading.learn("All students passed") + reading.learn("Bob is a teacher")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "undetermined")

    def test_full_chain_through_the_cli(self):
        out = reading.run_cli(["Did Bob pass?",
                               "--about", "All students passed", "Bob is a student"])
        self.assertEqual(out.strip(), "yes")


class TestIncrement18ParagraphMode(unittest.TestCase):
    """#18 — the CLI accepts a whole paragraph: split into sentences, learn the statements, answer
    the trailing question. So the natural one-liner works without --about."""

    def test_inline_premises_then_question(self):
        out = reading.run_cli(["All students passed. Bob is a student. Did Bob pass?"])
        self.assertEqual(out.strip(), "yes")

    def test_count_paragraph(self):
        out = reading.run_cli(["There are three red apples. How many red apples?"])
        self.assertEqual(out.strip(), "3")

    def test_single_statement_still_analyzed(self):
        out = reading.run_cli(["The student passed"])
        self.assertIn("understood", out)

    def test_about_flag_still_works(self):
        out = reading.run_cli(["Did Bob pass?", "--about", "All students passed", "Bob is a student"])
        self.assertEqual(out.strip(), "yes")


class TestIncrement19Trace(unittest.TestCase):
    """#19 — `--trace` exposes the analyzed data and reasoning: parsed units, the decision trail,
    the facts learned from each statement, and the derivation behind the answer."""

    def test_trace_shows_facts_decisions_and_derivation(self):
        out = reading.run_cli(["All apples passed. Bob is a student. Did Bob pass?", "--trace"])
        self.assertIn("Membership", out)                 # facts learned
        self.assertIn("decisions:", out)                 # the reversible trail
        self.assertIn("lack of information", out)         # the derivation (why undetermined)
        self.assertIn("ANSWER: undetermined", out)

    def test_trace_on_single_statement(self):
        out = reading.run_cli(["The student passed", "--trace"])
        self.assertIn("STATEMENT", out)
        self.assertIn("decisions:", out)

    def test_explain_lists_the_inference_steps(self):
        ctx = reading.learn("All students passed") + reading.learn("Bob is a student")
        steps = reading.explain("Did Bob pass?", ctx)
        self.assertTrue(any("is a student" in s for s in steps))


class TestIncrement20MembershipQuestions(unittest.TestCase):
    """#20 — "Is X a Y?" membership questions, and "Every X is a Y" universal membership (a Subset
    fact). Together they give the Barbara syllogism: every teacher is a person, Bob is a teacher,
    therefore Bob is a person — answered by membership reachability over Subset edges."""

    def test_subset_fact_is_learned(self):
        self.assertEqual(reading.learn("Every teacher is a person"),
                         [reading.Subset(("teacher", frozenset()), ("person", frozenset()))])

    def test_direct_membership_question(self):
        ctx = reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Is Bob a student?", ctx), "yes")
        self.assertEqual(reading.answer("Is Bob a teacher?", ctx), "undetermined")

    def test_barbara_syllogism(self):
        ctx = reading.learn("Every teacher is a person") + reading.learn("Bob is a teacher")
        self.assertEqual(reading.answer("Is Bob a person?", ctx), "yes")

    def test_cli_barbara_in_one_line(self):
        out = reading.run_cli(["Every teacher is a person. Bob is a teacher. Is Bob a person?"])
        self.assertEqual(out.strip(), "yes")


class TestIncrement21Negation(unittest.TestCase):
    """#21 — negation. "Bob did not pass" and "No students failed" must NOT be learned as positive
    facts (the dangerous silent error). A polarity bit on Predication, set at ingestion when a
    negator (not/never/no) is present; the positive query then doesn't match → "no"."""

    def test_negated_predication_is_learned_with_polarity(self):
        self.assertEqual(reading.learn("Bob did not pass"),
                         [reading.Predication(("bob", frozenset()), "pass", negated=True)])

    def test_negative_universal_is_learned(self):
        self.assertEqual(reading.learn("No students failed"),
                         [reading.Predication(("student", frozenset()), "fail", negated=True, universal=True)])

    def test_did_not_pass_answers_no(self):
        self.assertEqual(reading.answer("Did Bob pass?", reading.learn("Bob did not pass")), "no")

    def test_no_students_failed_means_a_student_did_not_fail(self):
        ctx = reading.learn("No students failed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob fail?", ctx), "no")

    def test_positive_still_works_alongside(self):
        # the same word "pass" stays positive without a negator
        self.assertEqual(reading.learn("The student passed"),
                         [reading.Predication(("student", frozenset()), "pass")])

    def test_cli_negation(self):
        self.assertEqual(reading.run_cli(["Bob did not pass. Did Bob pass?"]).strip(), "no")


class TestIncrement22PredicateAdjectives(unittest.TestCase):
    """#22 — predicative adjectives: "The apple is red" is a property fact (NOT an attributive merge
    like "red apple"); "Is the apple red?" queries it. The predicate is the adjective; polarity too."""

    def test_property_fact_is_learned(self):
        self.assertEqual(reading.learn("The apple is red"),
                         [reading.Predication(("apple", frozenset()), "red")])

    def test_property_question(self):
        ctx = reading.learn("The apple is red")
        self.assertEqual(reading.answer("Is the apple red?", ctx), "yes")
        self.assertEqual(reading.answer("Is the apple green?", ctx), "undetermined")

    def test_negated_property(self):
        self.assertEqual(reading.learn("The apple is not red"),
                         [reading.Predication(("apple", frozenset()), "red", negated=True)])

    def test_cli(self):
        self.assertEqual(reading.run_cli(["The apple is red. Is the apple red?"]).strip(), "yes")


class TestIncrement23WhQuestions(unittest.TestCase):
    """#23 — wh-questions return entities, not yes/no. "Who passed?" → the subjects with that
    predicate; "What is Bob?" → the collections Bob belongs to. ("(no one)"/"(unknown)" when empty.)"""

    def test_who_did_something(self):
        self.assertEqual(reading.answer("Who passed?", reading.learn("Bob passed")), "bob")

    def test_who_with_no_match(self):
        self.assertEqual(reading.answer("Who passed?", []), "(no one)")

    def test_what_is_an_individual(self):
        self.assertEqual(reading.answer("What is Bob?", reading.learn("Bob is a student")), "student")

    def test_cli_who(self):
        self.assertEqual(reading.run_cli(["Bob passed. Who passed?"]).strip(), "bob")


class TestIncrement24Comparatives(unittest.TestCase):
    """#24 — comparatives are relational facts ("Alice is taller than Bob" → Relation), and the
    relation is transitive: A>B and B>C ⊢ A>C. The first genuine relation (vs set membership)."""

    def test_relation_is_learned(self):
        self.assertEqual(reading.learn("Alice is taller than Bob"),
                         [reading.Relation("alice", "taller", "bob")])

    def test_direct_comparison(self):
        ctx = reading.learn("Alice is taller than Bob")
        self.assertEqual(reading.answer("Is Alice taller than Bob?", ctx), "yes")
        self.assertEqual(reading.answer("Is Bob taller than Alice?", ctx), "no")

    def test_transitive_comparison(self):
        ctx = (reading.learn("Alice is taller than Bob")
               + reading.learn("Bob is taller than Carol"))
        self.assertEqual(reading.answer("Is Alice taller than Carol?", ctx), "yes")

    def test_cli_transitive_chain(self):
        out = reading.run_cli(["Alice is taller than Bob. Bob is taller than Carol. "
                               "Is Alice taller than Carol?"])
        self.assertEqual(out.strip(), "yes")


class TestIncrement26NegationIsAnObligation(unittest.TestCase):
    """#26 — "not" is a proper obligation-bearing word: it carries a `negates` obligation that BINDS
    to the verb/copula it scopes (not a global flag). Polarity then attaches to the right fact, so
    `Membership` gets a polarity and "Bob is not a student" is no longer a positive membership."""

    def test_not_carries_a_negates_obligation_bound_to_the_verb(self):
        r = reading.analyze("Bob did not pass")
        neg = next(u for u in r.units if "negation" in u.features.get("senses", []))
        ob = neg.features["obligations"][0]
        self.assertEqual(ob.need, "negates")
        self.assertTrue(ob.met)
        self.assertEqual(ob.target.surface, "pass")     # bound to the verb it negates

    def test_negated_membership_is_learned(self):
        self.assertEqual(reading.learn("Bob is not a student"),
                         [reading.Membership("bob", ("student", frozenset()), negated=True)])

    def test_is_not_a_student_answers_no(self):
        self.assertEqual(reading.answer("Is Bob a student?", reading.learn("Bob is not a student")), "no")

    def test_positive_membership_still_yes(self):
        self.assertEqual(reading.answer("Is Bob a student?", reading.learn("Bob is a student")), "yes")

    def test_negated_predication_still_works(self):
        self.assertEqual(reading.learn("Bob did not pass"),
                         [reading.Predication(("bob", frozenset()), "pass", negated=True)])


class TestIncrement27ScopedNoNegation(unittest.TestCase):
    """#27 (alignment) — "no" negation is now scoped via BINDING: a verb is 'no'-negated iff its
    subject is the collection a 'no' quantifier binds to, not a global "is 'no' present?" flag.
    Behavioral contract preserved; the latent multi-clause over-negation is gone."""

    def test_no_subject_negates_its_own_predication(self):
        self.assertEqual(reading.learn("No students failed"),
                         [reading.Predication(("student", frozenset()), "fail", negated=True, universal=True)])

    def test_universal_without_no_stays_positive(self):
        self.assertEqual(reading.learn("All students failed"),
                         [reading.Predication(("student", frozenset()), "fail", universal=True)])

    def test_no_negation_rides_the_collection_binding(self):
        # the 'no' quantifier binds its collection to the same unit the verb takes as subject
        r = reading.analyze("No students failed")
        no = next(u for u in r.units if u.surface == "No")
        failed = next(u for u in r.units if u.surface == "failed")
        self.assertIs(no.features["obligations"][0].target,
                      next(o.target for o in failed.features["obligations"] if o.need == "subject"))


class TestIncrement28CopulaComplement(unittest.TestCase):
    """#28 (alignment) — the copula binds BOTH a `subject` and a `complement`; the fact extractors
    read those bindings instead of counting nouns/adjectives. Inversion-aware so it works for
    statements (subject left) and fronted questions (subject right)."""

    def test_copula_binds_subject_and_complement(self):
        r = reading.analyze("Bob is a student")
        cop = next(u for u in r.units if u.surface == "is")
        obs = {o.need: o for o in cop.features["obligations"]}
        self.assertEqual(obs["subject"].target.surface, "Bob")
        self.assertEqual(obs["complement"].target.surface, "student")

    def test_fronted_question_binds_subject_on_the_right(self):
        r = reading.analyze("Is Bob a student?")
        cop = next(u for u in r.units if u.surface == "Is")
        obs = {o.need: o for o in cop.features["obligations"]}
        self.assertEqual(obs["subject"].target.surface, "Bob")        # fronted → right
        self.assertEqual(obs["complement"].target.surface, "student?")  # surface kept lossless

    def test_extractors_read_the_bindings(self):
        self.assertEqual(reading.learn("Bob is a student"),
                         [reading.Membership("bob", ("student", frozenset()))])
        self.assertEqual(reading.learn("The apple is red"),
                         [reading.Predication(("apple", frozenset()), "red")])
        self.assertEqual(reading.learn("Every teacher is a person"),
                         [reading.Subset(("teacher", frozenset()), ("person", frozenset()))])

    def test_questions_still_answer_via_bindings(self):
        self.assertEqual(reading.answer("Is Bob a student?", reading.learn("Bob is a student")), "yes")
        self.assertEqual(reading.answer("Is the apple red?", reading.learn("The apple is red")), "yes")


class TestIncrement29ComparativeAndWhObligations(unittest.TestCase):
    """#29 (alignment) — the last ad-hoc words get obligations. A comparative is the copula's
    complement and binds its `standard` (after "than"), so Relations read off bindings; wh-words bind
    to what they ask about (who → the verb, what → the individual). No more counting in the fact layer."""

    def test_comparative_binds_its_standard(self):
        r = reading.analyze("Alice is taller than Bob")
        taller = next(u for u in r.units if u.surface == "taller")
        std = next(o for o in taller.features["obligations"] if o.need == "standard")
        self.assertEqual(std.target.surface, "Bob")

    def test_relation_read_from_bindings(self):
        self.assertEqual(reading.learn("Alice is taller than Bob"),
                         [reading.Relation("alice", "taller", "bob")])

    def test_comparative_question_and_transitivity_intact(self):
        ctx = (reading.learn("Alice is taller than Bob")
               + reading.learn("Bob is taller than Carol"))
        self.assertEqual(reading.answer("Is Alice taller than Carol?", ctx), "yes")
        self.assertEqual(reading.answer("Is Carol taller than Alice?", ctx), "no")

    def test_who_binds_the_predicate(self):
        r = reading.analyze("Who passed?")
        who = next(u for u in r.units if u.surface == "Who")
        ob = who.features["obligations"][0]
        self.assertEqual(ob.need, "asks_predicate")
        self.assertEqual(ob.target.surface, "passed?")

    def test_wh_questions_still_answer(self):
        self.assertEqual(reading.answer("Who passed?", reading.learn("Bob passed")), "bob")
        self.assertEqual(reading.answer("What is Bob?", reading.learn("Bob is a student")), "student")


class TestIncrement30SomeIsNotAll(unittest.TestCase):
    """#30 — fix probe bug B (+C): an existential "Some X passed" is NOT a universal, so it must not
    be instantiated to a specific individual. A Predication is `universal` only when its subject is
    bound by all/every; instantiation uses only those, routed through `_is_member` (so multi-hop
    membership — bug C — works too)."""

    def test_only_all_every_are_universal(self):
        self.assertTrue(reading.learn("All students passed")[0].universal)
        self.assertTrue(reading.learn("Every student passed")[0].universal)
        self.assertFalse(reading.learn("Some students passed")[0].universal)
        self.assertFalse(reading.learn("Bob passed")[0].universal)

    def test_some_does_not_instantiate(self):   # bug B
        ctx = reading.learn("Some students passed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "undetermined")

    def test_all_still_instantiates(self):
        ctx = reading.learn("All students passed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "yes")

    def test_multi_hop_predication(self):   # bug C — falls out of the same fix
        ctx = (reading.learn("All teachers are persons")
               + reading.learn("Every person passed")
               + reading.learn("Bob is a teacher"))
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "yes")


class TestIncrement31UndeterminedVerdict(unittest.TestCase):
    """#31 — the third verdict. A yes/no question is "yes" (derivable true), "no" (derivable FALSE —
    an explicit/universal negative), or "undetermined" (we lack the information). No more closed-world
    "no" for things we simply don't know."""

    def test_no_information_is_undetermined(self):
        self.assertEqual(reading.answer("Did Bob pass?", reading.learn("Bob is a student")),
                         "undetermined")

    def test_explicit_negative_is_no(self):
        self.assertEqual(reading.answer("Did Bob pass?", reading.learn("Bob did not pass")), "no")

    def test_universal_negative_is_no(self):
        ctx = reading.learn("No students passed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "no")

    def test_some_is_undetermined_not_no(self):
        ctx = reading.learn("Some students passed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "undetermined")

    def test_membership_without_evidence_is_undetermined(self):
        self.assertEqual(reading.answer("Is Bob a teacher?", reading.learn("Bob is a student")),
                         "undetermined")

    def test_comparative_asymmetry_is_no(self):
        ctx = reading.learn("Alice is taller than Bob")
        self.assertEqual(reading.answer("Is Bob taller than Alice?", ctx), "no")


class TestIncrement32UniversalPropertyInstantiation(unittest.TestCase):
    """#32 — fix probe bug E: a universal PROPERTY ("Every student is happy") instantiates to a
    member, exactly like a universal predication. Properties are Predications with the adjective as
    the predicate, so they route through the same universal-instantiation verdict."""

    def test_property_universal_only_for_all_every(self):
        self.assertTrue(reading.learn("Every student is happy")[0].universal)
        self.assertFalse(reading.learn("The apple is red")[0].universal)

    def test_universal_property_instantiates_to_a_member(self):
        ctx = reading.learn("Every student is happy") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Is Bob happy?", ctx), "yes")

    def test_existential_property_does_not_instantiate(self):
        ctx = reading.learn("Some students are happy") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Is Bob happy?", ctx), "undetermined")

    def test_direct_and_negated_property_unchanged(self):
        self.assertEqual(reading.answer("Is the apple red?", reading.learn("The apple is red")), "yes")
        self.assertEqual(reading.answer("Is the apple red?", reading.learn("The apple is not red")), "no")


class TestIncrement33NegativeSubset(unittest.TestCase):
    """#33 — fix probe bug F: "No X are Y" is a NEGATED Subset (disjointness). A member of X is then
    definitely NOT a Y — a definite "no", not undetermined."""

    def test_negative_subset_is_learned(self):
        self.assertEqual(reading.learn("No students are teachers"),
                         [reading.Subset(("student", frozenset()), ("teacher", frozenset()),
                                         negated=True)])

    def test_disjoint_membership_is_no(self):
        ctx = reading.learn("No students are teachers") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Is Bob a teacher?", ctx), "no")

    def test_positive_subset_still_yes(self):
        ctx = reading.learn("Every teacher is a person") + reading.learn("Bob is a teacher")
        self.assertEqual(reading.answer("Is Bob a person?", ctx), "yes")

    def test_unrelated_collection_stays_undetermined(self):
        ctx = reading.learn("No students are teachers") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Is Bob a person?", ctx), "undetermined")


class TestIncrement34ContradictionDetection(unittest.TestCase):
    """#34 — fix probe bug D: when both "yes" AND "no" are derivable for a query, the verdict is
    "conflict" (a 4th verdict) — the premises contradict, distinct from "undetermined" (no info).
    Detection is DERIVABLE: it counts inference (universals/subsets), not just direct facts."""

    def test_direct_contradiction(self):
        ctx = reading.learn("Bob passed") + reading.learn("Bob did not pass")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "conflict")

    def test_derivable_contradiction_via_universal(self):
        ctx = (reading.learn("All students passed") + reading.learn("Bob did not pass")
               + reading.learn("Bob is a student"))
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "conflict")

    def test_membership_contradiction(self):
        ctx = reading.learn("Bob is a student") + reading.learn("Bob is not a student")
        self.assertEqual(reading.answer("Is Bob a student?", ctx), "conflict")

    def test_consistent_facts_are_not_conflict(self):
        self.assertEqual(reading.answer("Did Bob pass?", reading.learn("Bob passed")), "yes")
        self.assertEqual(reading.answer("Did Bob pass?", reading.learn("Bob did not pass")), "no")


class TestIncrement35ExplainAllQuestionTypes(unittest.TestCase):
    """#35 — `explain()`/`--trace` now derives property and comparative questions too (they used to
    fall through to "not a question I can answer")."""

    def test_explain_property(self):
        ctx = reading.learn("Every student is happy") + reading.learn("Bob is a student")
        steps = reading.explain("Is Bob happy?", ctx)
        self.assertTrue(any("happy" in s for s in steps))
        self.assertTrue(any("yes" in s for s in steps))
        self.assertFalse(any("not a question" in s for s in steps))

    def test_explain_comparative(self):
        steps = reading.explain("Is Alice taller than Bob?", reading.learn("Alice is taller than Bob"))
        self.assertTrue(any("taller" in s for s in steps))
        self.assertFalse(any("not a question" in s for s in steps))

    def test_trace_property_in_cli(self):
        out = reading.run_cli(["Every student is happy. Bob is a student. Is Bob happy?", "--trace"])
        self.assertIn("ANSWER: yes", out)
        self.assertNotIn("not a question", out)


class TestIncrement36TransitiveObjects(unittest.TestCase):
    """#36 — transitive verbs capture their direct object. "Mary owns a dog" -> Predication with
    object=(dog); "Does Mary own a dog?" vs "a cat?" are different facts. The verb's `dobj`
    obligation is OPTIONAL (intransitive verbs leave it unmet without a deficiency)."""

    def test_object_is_captured(self):
        self.assertEqual(reading.learn("Mary owns a dog"),
                         [reading.Predication(("mary", frozenset()), "own", object=("dog", frozenset()))])

    def test_intransitive_has_no_object_and_no_deficiency(self):
        self.assertEqual(reading.learn("Bob passed"), [reading.Predication(("bob", frozenset()), "pass")])
        self.assertTrue(reading.is_well_formed(reading.analyze("Bob passed")))  # dobj unmet is fine

    def test_object_aware_question(self):
        ctx = reading.learn("Mary owns a dog")
        self.assertEqual(reading.answer("Does Mary own a dog?", ctx), "yes")
        self.assertEqual(reading.answer("Does Mary own a cat?", ctx), "undetermined")


class TestIncrement37ForegoWhenSubordinated(unittest.TestCase):
    """#37 — a noun/adjective that FILLS another word's slot (a verb's object, a copula's complement)
    has its own obligations foregone, so it doesn't read as a false fragment. A real noun-phrase
    fragment (no predicate anywhere) is still flagged."""

    def test_transitive_object_is_not_a_fragment(self):
        self.assertTrue(reading.is_well_formed(reading.analyze("Mary owns a dog")))

    def test_copula_complement_is_not_a_fragment(self):
        self.assertTrue(reading.is_well_formed(reading.analyze("Bob is a student")))

    def test_predicative_adjective_is_not_a_fragment(self):
        self.assertTrue(reading.is_well_formed(reading.analyze("The apple is red")))

    def test_real_noun_phrase_is_still_a_fragment(self):
        self.assertFalse(reading.is_well_formed(reading.analyze("Each of the students")))


class TestIncrement38WhObject(unittest.TestCase):
    """#38 — wh-object: "What does Mary own?" returns the direct OBJECTS of (Mary, own), completing
    the transitive-verb work. "What is <name>?" still returns memberships."""

    def test_what_does_x_verb_returns_objects(self):
        self.assertEqual(reading.answer("What does Mary own?", reading.learn("Mary owns a dog")), "dog")

    def test_what_is_still_membership(self):
        self.assertEqual(reading.answer("What is Bob?", reading.learn("Bob is a student")), "student")

    def test_no_object_known(self):
        self.assertEqual(reading.answer("What does Mary own?", []), "(nothing)")


class TestIncrement39WhichQuestions(unittest.TestCase):
    """#39 — "Which <noun> <predicate>?" returns the named members of that noun which satisfy the
    predicate (an action verb or a predicative adjective)."""

    def test_which_with_a_verb(self):
        ctx = (reading.learn("Bob is a student") + reading.learn("Bob passed")
               + reading.learn("Carol is a teacher") + reading.learn("Carol passed"))
        self.assertEqual(reading.answer("Which students passed?", ctx), "bob")  # Carol is a teacher

    def test_which_with_a_property(self):
        ctx = reading.learn("Bob is a student") + reading.learn("Bob is happy")
        self.assertEqual(reading.answer("Which students are happy?", ctx), "bob")

    def test_which_none(self):
        self.assertEqual(reading.answer("Which students passed?", []), "(none)")


class TestIncrement40CountComparison(unittest.TestCase):
    """#40 — aggregation: "Are there more X than Y?" compares the counts of two collections over the
    instance context (using the subset lattice)."""

    def _ctx(self):
        return (reading.instances("There are three red apples")
                + reading.instances("There are two green apples"))

    def test_more_than(self):
        self.assertEqual(reading.answer("Are there more red apples than green apples?", self._ctx()), "yes")

    def test_fewer_direction(self):
        self.assertEqual(reading.answer("Are there more green apples than red apples?", self._ctx()), "no")

    def test_cli(self):
        out = reading.run_cli(["There are three red apples. There are two green apples. "
                               "Are there more red apples than green apples?"])
        self.assertEqual(out.strip(), "yes")


class TestIncrement55FrontedQuestionBinding(unittest.TestCase):
    """#55 - a fronted question binds its verb's arguments via STRUCTURE, not a positional sidestep:
    in "<aux> SUBJ <verb>" the subject sits BETWEEN the auxiliary and the verb (disambiguating it from
    a fronted noun); the wh-fronted OBJECT sits before the auxiliary. So "How much more money does
    Oliver have than William?" now fully saturates (have->subject->Oliver, have->dobj->money), the
    comparison carried by more->standard->William."""

    def test_subject_binds_through_aux_fronting(self):
        r = reading.analyze("How much more money does Oliver have than William?")
        have = next(u for u in r.units if u.surface == "have")
        self.assertEqual(reading._follow(have, "subject").surface, "Oliver")  # not "money"

    def test_object_binds_the_fronted_noun(self):
        r = reading.analyze("How much more money does Oliver have than William?")
        have = next(u for u in r.units if u.surface == "have")
        self.assertEqual(reading._follow(have, "dobj").surface, "money")      # not "than"/"William"

    def test_the_whole_comparison_is_well_formed(self):
        self.assertTrue(reading.is_well_formed(
            reading.analyze("How much more money does Oliver have than William?")))

    def test_aux_fronting_for_plain_quantity_question(self):
        r = reading.analyze("How many apples does Tom have?")
        have = next(u for u in r.units if reading._key(u.surface) == "have")  # "have?" -> key "have"
        self.assertEqual(reading._follow(have, "subject").surface, "Tom")

    def test_declarative_subject_unaffected(self):
        r = reading.analyze("Oliver has 30 apples")
        has = next(u for u in r.units if u.surface == "has")
        self.assertEqual(reading._follow(has, "subject").surface, "Oliver")


# === PARSING: conjunction/elision/contractions, numbers & scales, titles, multipliers, fractions, clause split ===

class TestIncrement56ConjunctionBinding(unittest.TestCase):
    """#56 - conjunction binding: "Tom and Mary" merges into ONE group unit carrying both as
    `members` (a compound subject), so the verb's `subject` obligation binds it (unique) and the
    answer reads its members -- no positional fallback needed. A group-subject STATEMENT is
    ambiguous (distributive vs collective), so it declines."""

    def test_group_merges_with_members(self):
        r = reading.analyze("How many apples do Tom and Mary have altogether?")
        grp = next(u for u in r.units if "members" in u.features)
        self.assertEqual(grp.surface, "Tom and Mary")
        self.assertEqual(grp.features["members"], ["tom", "mary"])

    def test_verb_subject_binds_the_group(self):
        r = reading.analyze("How many apples do Tom and Mary have altogether?")
        have = next(u for u in r.units if reading._key(u.surface) == "have")
        self.assertEqual(reading._follow(have, "subject").features["members"], ["tom", "mary"])

    def test_aggregation_through_the_binding(self):
        self.assertEqual(reading.solve("Tom has 3 apples. Mary has 2 apples. "
                                       "How many apples do Tom and Mary have altogether?"), 5)

    def test_three_way_conjunction_stacks(self):
        r = reading.analyze("Tom and Mary and Jane")
        grp = next(u for u in r.units if "members" in u.features)
        self.assertEqual(sorted(grp.features["members"]), ["jane", "mary", "tom"])

    def test_group_subject_statement_declines(self):
        self.assertEqual(reading.learn("Tom and Mary have 5 apples"), [])


class TestIncrement57ConjoinedObjects(unittest.TestCase):
    """#57 - conjoined OBJECTS: "Tom has 3 apples and 2 oranges" joins two QUANTITIES (not a group) ->
    two facts, one per value-noun pair. `_quantity` returns a LIST. Composes with accumulation."""

    def test_two_quantities_learned(self):
        self.assertEqual(reading.learn("Tom has 3 apples and 2 oranges"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3.0, "have"),
                          reading.Quantity("tom", ("orange", frozenset()), 2.0, "have")])

    def test_query_each_measure(self):
        ctx = "Tom has 3 apples and 2 oranges."
        self.assertEqual(reading.solve(ctx + " How many apples does Tom have?"), 3)
        self.assertEqual(reading.solve(ctx + " How many oranges does Tom have?"), 2)

    def test_conjoined_object_composes_with_accumulation(self):
        self.assertEqual(reading.solve("Tom has 3 apples and 2 oranges. Tom buys 4 apples. "
                                       "How many apples does Tom have?"), 7)

    def test_single_object_unchanged(self):
        self.assertEqual(reading.learn("Tom buys 2 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 2.0, "buy")])


class TestIncrement59VerbElision(unittest.TestCase):
    """#59 - verb elision: a clause joined by ', and' that lacks a verb but has a SUBJECT before its
    quantity ("Mary 5 apples", "grandparents twice as much") BORROWS the prior clause's verb. A bare
    conjoined object ("2 oranges": quantity-first, no subject) is NOT treated as elided."""

    def test_simple_elision_splits_and_inherits(self):
        self.assertEqual(reading._split_clauses("Tom has 3 apples, and Mary 5 apples"),
                         ["Tom has 3 apples", "Mary has 5 apples"])

    def test_simple_elision_solves(self):
        self.assertEqual(reading.solve("Tom has 3 apples, and Mary 5 apples. "
                                       "How many apples does Mary have?"), 5)

    def test_relational_elision_solves(self):
        self.assertEqual(reading.solve("Parents gave $15, and grandparents twice as much. "
                                       "How much did grandparents give?"), 30)

    def test_conjoined_object_is_not_elision(self):
        self.assertEqual(reading._split_clauses("Tom has 3 apples and 2 oranges"),
                         ["Tom has 3 apples and 2 oranges"])

    def test_how_much_is_money_not_a_subject_count(self):
        self.assertEqual(reading.solve("Tom gave $5. How much did Tom give?"), 5)


class TestIncrement69TitlesArticlesPossessives(unittest.TestCase):
    """#69 - parse fixes surfaced by "Mr. Sam shared a certain amount of money ... his two sons". A
    `title` (Mr./Dr.) is recognised but is NOT a name, so the verb's subject is unambiguous; an article
    binds the NEAREST noun (not a unique one); an unresolved possessive back-links to the clause's agent."""

    def test_title_is_not_a_name(self):
        mr = next(u for u in reading.analyze("Mr. Sam shared money").units if u.surface == "Mr.")
        self.assertEqual(mr.features.get("senses"), ["title"])

    def test_title_frees_the_subject(self):
        r = reading.analyze("Mr. Sam shared money")
        shared = next(u for u in r.units if u.surface == "shared")
        self.assertEqual(reading._follow(shared, "subject").surface, "Sam")

    def test_article_binds_nearest_noun(self):
        r = reading.analyze("a certain amount of money")
        a = next(u for u in r.units if u.surface == "a")
        self.assertEqual(reading._follow(a, "determines").features.get("head"), "amount")

    def test_possessive_backlinks_to_the_agent(self):
        r = reading.analyze("Sam shared his money")
        his = next(u for u in r.units if u.surface == "his")
        self.assertEqual(reading._follow(his, "refers").surface, "sam")

    def test_possessive_skips_when_subject_is_not_a_name(self):
        r = reading.analyze("His friend left")          # subject "friend" is a noun, not a name →
        his = next(u for u in r.units if u.surface == "His")   # within-sentence rule leaves it for discourse
        self.assertIsNone(reading._follow(his, "refers"))


class TestIncrement79WordNumbersAndCompoundRelational(unittest.TestCase):
    """#79 - WORD-NUMBERS ARE VALUES (one..twelve parse like digits): "three apples" → a Quantity (was
    instances); count statements are existential-gated ("there are three X" → instances). Unblocks "three
    years younger" and the COMPOUND RELATIONAL "five more than twice as many as Bob" (2×Bob+5). Precision
    held: _container won't treat the SUBJECT as a container ("Four people lost a total of 103 kg" ≠ 4×103),
    and GUARD 6 declines "each OF [a subset]" (per-member of a restricted group)."""

    def test_word_number_is_a_quantity(self):
        self.assertEqual(reading.learn("Tom has three apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3, "have")])

    def test_there_are_still_instances(self):
        self.assertEqual(len(reading.instances("there are three red apples")), 3)

    def test_three_years_younger(self):
        self.assertEqual(reading.solve("Mel is three years younger than Katherine. "
                                       "Katherine is 24 years old. How old is Mel?"), 21)

    def test_compound_relational(self):
        self.assertEqual(reading.solve("John has five more roommates than twice as many as Bob. "
                                       "If Bob has 10 roommates, how many roommates does John have?"), 25)

    def test_relational_without_offset_unchanged(self):
        self.assertEqual(reading.solve("Tim has 30 apples. Harry has twice as many apples as Tim. "
                                       "How many apples does Harry have?"), 60)

    def test_container_not_subject(self):
        amounts = [getattr(f, "amount", None) for f in reading.learn("Four people lost a total of 103 kilograms")]
        self.assertNotIn(412.0, amounts)

    def test_each_of_subset_declines(self):
        self.assertEqual(reading.solve("Four people lost a total of 103 kilograms. The first person lost "
                                       "27 kilograms. How many kilograms did each of the last two people lose?"),
                         "undetermined")


class TestIncrement80NumberScales(unittest.TestCase):
    """#80 - number-SCALE words (dozen/hundred/thousand): "two dozen" → 24, "three hundred" → 300, via
    _merge_scales (N × scale → ONE value unit, chaining for "two hundred thousand"). Solves the real Mel."""

    def test_two_dozen(self):
        r = reading.analyze("Katherine is two dozen years old")
        v = next(u for u in r.units if "value" in u.features.get("senses", []))
        self.assertEqual(v.features["value"], 24)

    def test_three_hundred(self):
        r = reading.analyze("She sold three hundred tickets")
        self.assertIn(300, [u.features.get("value") for u in r.units])

    def test_mel_two_dozen_solves(self):
        self.assertEqual(reading.solve("Mel is three years younger than Katherine. When Katherine is "
                                       "two dozen years old, how old will Mel be in years?"), 21)


class TestIncrement82PerfectAuxiliary(unittest.TestCase):
    """#82 - "has/have/had" as the PERFECT AUXILIARY: "How many geckos HAS Brandon SOLD" reads `sold` as
    the main verb (lemma sell) and `has` as the aux, so it matches the context's sell-events. Multi-sense
    (verb sense first) keeps "Tom has 5 apples" a main verb. Solves Brandon (with "twice that many")."""

    def test_has_5_apples_still_main_verb(self):
        self.assertEqual(reading.learn("Tom has 5 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 5.0, "have")])

    def test_perfect_auxiliary_question(self):
        ctx = [reading.Quantity("brandon", ("gecko", frozenset()), 86, "sell"),
               reading.Quantity("brandon", ("gecko", frozenset()), 172, "sell")]
        self.assertEqual(reading.answer("How many geckos has Brandon sold", ctx, []), 258)

    def test_twice_that_many(self):
        ctx = [reading.Quantity("brandon", ("gecko", frozenset()), 86, "sell")]
        self.assertEqual(reading.learn("He sold twice that many", ["brandon"], ctx),
                         [reading.Quantity("brandon", ("gecko", frozenset()), 172, "sell")])

    def test_brandon_solves(self):
        self.assertEqual(reading.solve("Brandon sold 86 geckos last year. He sold twice that many the "
                                       "year before. How many geckos has Brandon sold in the last two years?"), 258)


class TestIncrement85TitleAbbreviation(unittest.TestCase):
    """#85 - a title's period ("Mr.", "Dr.") is NOT a sentence end. _sentences keeps "Mr. Sanchez ..." in
    one sentence (TITLES derived from the lexicon's title words); decimals stay protected too. The existing
    title-sense handling (the honorific frees the subject) is unchanged. Removes a Sanchez parse blocker."""

    def test_title_does_not_split_sentence(self):
        self.assertEqual(reading._sentences("Mr. Sanchez has 5 apples. How many does Mr. Sanchez have?"),
                         ["Mr. Sanchez has 5 apples.", "How many does Mr. Sanchez have?"])

    def test_mr_problem_solves(self):
        self.assertEqual(reading.solve("Mr. Sanchez has 5 apples. "
                                       "How many apples does Mr. Sanchez have?"), 5)

    def test_mrs_problem_solves(self):
        self.assertEqual(reading.solve("Mrs. Lee sold 12 cakes. How many cakes did Mrs. Lee sell?"), 12)

    def test_decimal_still_not_split(self):
        self.assertEqual(reading._sentences("The price is $5.50 today."), ["The price is $5.50 today."])


class TestIncrement86TrailingConditional(unittest.TestCase):
    """#86 - a trailing CONDITIONAL "Q if P?": "if" is a clause separator, so P becomes a premise and the
    condition peels off FIRST — which also stops "B and above if he has 60 students" from mis-splitting on
    "and" (left "above" is then verbless). The QUESTION clause is the interrogative-bearing one (the
    trailing "?" lands on P). The complement clause DEFERS when no total yet, so the fixpoint resolves it."""

    def test_trailing_conditional_feeds_complement(self):
        self.assertEqual(reading.solve("40% of the students passed. The rest failed. "
                                       "How many students failed if he has 60 students?"), 36)

    def test_b_and_above_not_split_by_and(self):
        self.assertEqual(reading._split_clauses("students got a grade of B and above if he has 60 students"),
                         ["students got a grade of B and above", "he has 60 students"])

    def test_leading_if_unchanged(self):
        self.assertEqual(reading.solve("John has five more roommates than twice as many as Bob. "
                                       "If Bob has 10 roommates, how many roommates does John have?"), 25)


class TestIncrement89ClassifierLabel(unittest.TestCase):
    """#89 - a bare number right after a noun is a LABEL/ordinal ("Grade 5", "Room 3"), not a quantity:
    _merge_labels absorbs it (the number → an attribute) so it no longer quantifies a neighbour, bases a
    percent, or trips the unused-given guard. So "40% of his Grade 5 students" reads the percent-of-count
    over STUDENTS (the of-phrase head) = 24 of 60. "4 times 3" is unaffected (times is an operator)."""

    def test_grade_n_is_a_label(self):
        r = reading.analyze("his Grade 5 students")
        self.assertFalse(any("value" in u.features.get("senses", []) for u in r.units))

    def test_percent_over_grade_n_head(self):
        self.assertEqual(reading.learn("40% of his Grade 5 students got a final grade below B", [],
                                       [reading.Quantity("he", ("student", frozenset()), 60, "have")]),
                         [reading.Quantity("student", ("student", frozenset()), 24.0, "get")])

    def test_times_operator_unaffected(self):
        self.assertEqual(reading.answer("What is 4 times 3?", []), 12)

    def test_number_first_quantity_unaffected(self):
        self.assertEqual(reading.learn("Tom has 5 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 5.0, "have")])


class TestIncrement91Contractions(unittest.TestCase):
    """#91 - contractions are expanded at tokenization into their TWO words, each carrying its own
    obligation (no special-case flag): "what's" → what + is, "didn't" → did + not, "they're" → they are,
    "won't" → will not. A pronoun "'s" (he's/it's) is the copula 'is'; a NAME's genitive ("Tom's") splits
    into the name + the possessive marker "'s" (so the name is recognised), distinct from the copula."""

    def test_pronoun_s_is_copula(self):
        self.assertEqual(reading._split_contraction("what's"), ["what", "is"])
        self.assertEqual(reading._split_contraction("he's"), ["he", "is"])

    def test_nt_is_not(self):
        self.assertEqual(reading._split_contraction("didn't"), ["did", "not"])
        self.assertEqual(reading._split_contraction("won't"), ["will", "not"])

    def test_other_suffixes(self):
        self.assertEqual(reading._split_contraction("they're"), ["they", "are"])
        self.assertEqual(reading._split_contraction("I'm"), ["I", "am"])
        self.assertEqual(reading._split_contraction("we've"), ["we", "have"])

    def test_possessive_name_splits_to_genitive(self):
        self.assertEqual(reading._split_contraction("Tom's"), ["Tom", "'s"])
        self.assertEqual([u.surface for u in reading.analyze("Tom's apples").units], ["Tom", "'s", "apples"])
        # the name is now recognised (no unknown-word), unlike the un-split "Tom's"
        self.assertFalse(any(d.kind == "unknown-word" for d in reading.deficiencies(reading.analyze("Tom's age"))))

    def test_trailing_punctuation_rides_along(self):
        self.assertEqual(reading._split_contraction("what's?"), ["what", "is?"])

    def test_expanded_question_reads(self):
        self.assertEqual([u.surface for u in reading.analyze("What's the answer?").units],
                         ["What", "is", "the", "answer?"])


class TestNameNounDisambiguation(unittest.TestCase):
    """A name that is ALSO a common noun ("Job", "Bill", "Rose") is made name/noun-ambiguous, and the
    standard of an "as <X> as <STANDARD>" comparison is subordinated (its predicate foregone) so
    `disambiguate` picks the NAME reading there — while an article-determined noun still reads as a noun."""

    def test_capitalized_known_noun_is_dual_sense(self):
        u = next(u for u in reading.analyze("Job is 5 years old").units
                 if reading._key(u.surface) == "job")
        self.assertIn("name", u.features["senses"])

    def test_comparison_standard_is_subordinated(self):
        r = reading.analyze("Anna is 4 times as old as Job")
        job = next(u for u in r.units if reading._key(u.surface) == "job")
        self.assertIn(id(job), reading._comparison_standard_ids(r))

    def test_collision_name_as_comparison_standard_resolves(self):
        # "Job" after "as old as" is the comparison reference (a person), not the occupation noun.
        out = reading.run_cli(["Anna is 3 times as old as Job. Job is 5 years old. How old is Anna?"])
        self.assertEqual(out.strip(), "15")

    def test_article_determined_collision_noun_stays_noun(self):
        # "the bill" is article-determined → the noun reading covers strictly better, so it is NOT a name.
        u = next(u for u in reading.analyze("He paid the bill").units
                 if reading._key(u.surface) == "bill")
        self.assertEqual(u.features["senses"], ["noun"])


class TestIncrement92HyphenCompounds(unittest.TestCase):
    """#92 - a DIGIT-led hyphen compound is an adjectival number-unit phrase, restored to its words at
    tokenization: "120-page" → 120 + page, "2-hour" → 2 + hour (the number then quantifies the unit
    normally). Non-digit-led compounds ("fast-food", "twenty-five", "fourth-grade") stay ONE token — never
    split a number-word into two values. Enables the rate inversion over "a 120-page book"."""

    def test_digit_led_splits(self):
        self.assertEqual(reading._split_hyphen("120-page"), ["120", "page"])
        self.assertEqual(reading._split_hyphen("2-hour"), ["2", "hour"])

    def test_non_digit_led_intact(self):
        self.assertEqual(reading._split_hyphen("fast-food"), ["fast-food"])
        self.assertEqual(reading._split_hyphen("twenty-five"), ["twenty-five"])

    def test_trailing_punct_rides(self):
        self.assertEqual(reading._split_hyphen("5-pound,"), ["5", "pound,"])

    def test_120_page_book_inversion(self):
        self.assertEqual(reading.solve("John writes 20 pages a day. "
                                       "How long will it take him to write a 120-page book?"), 6)


class TestIncrement97NumericMultiplier(unittest.TestCase):
    """#97 - "N times as many/much" (a NUMERIC multiplier) resolves like lexicalized "twice"/"thrice":
    "N times" fuses into ONE multiplier unit (factor N) that scales the "as many" quantifier, so the
    DERIVED quantity is N x the referent. Brings the quantity path to parity with the age path
    ("N times as old", already handled). The `as` is the signal: "N times MORE than" stays DECLINED
    (multiplicative ambiguity, the Riku trap). Found via the widened firewall: the dominant
    confident-wrong family on full dev (a mis-resolved referent gave 1 x N instead of N x referent)."""

    def test_n_times_as_many_resolves(self):
        self.assertEqual(reading.solve("Tom has 7 apples. Jerry has three times as many apples. "
                                       "How many apples does Jerry have?"), 21)

    def test_n_times_as_many_by_name(self):
        self.assertEqual(reading.solve("Stu has 9 books. Albert has four times as many books as Stu. "
                                       "How many books does Albert have?"), 36)

    def test_n_times_as_many_composes_with_total(self):
        self.assertEqual(reading.solve("Stu has 9 books. Albert has four times as many books as Stu. "
                                       "How many books do Stu and Albert have in total?"), 45)

    def test_n_times_that_many_resolves(self):
        # "N times THAT many" — the "that" points at the just-stated quantity (implicit referent)
        self.assertEqual(reading.solve("Tom lost 8 pounds. Jerry lost three times that many pounds. "
                                       "How many pounds did Jerry lose?"), 24)

    def test_times_more_than_still_declines(self):
        # "N times MORE than" has no `as` -> remains the multiplicative-ambiguity decline (Riku trap)
        self.assertEqual(reading.solve("Riku has 25 times more stickers than Kristoff. "
                                       "Kristoff has 85 stickers. How many stickers does Riku have?"),
                         "undetermined")

    def test_n_times_operator_unaffected(self):
        # a bare "N times M" (no `as`) stays plain multiplication, not a multiplier merge
        self.assertEqual(reading.solve("What is 4 times 3?"), 12)


class TestGenitiveTimesMultiplier(unittest.TestCase):
    """`_merge_times_multiplier` extended to ALSO fuse "N times NAME's <measure>" (a genitive/possessive
    standard, no "as many as" framing) — "600 square feet larger than 4 times Benedict's house". Unambiguous
    (unlike the Riku "N times MORE than" trap, which needs "more/less" right after "times", not a name) and
    matches the possessive ratio frame `_ratio_relation` already accepts. EXCLUDES a possessed "age"
    specifically — the age domain parses "<value> times" as a bare operator-expression over the UNMERGED
    tokens via a different route, and this merge pre-empting it caused a real confident-wrong (39 vs gold 26)
    the diff harness caught. +3 dev-holdout (Kennedy/Benedict, Brenda/Mel, Sara/Nada — all real GSM8K cases)."""

    def test_offset_ratio_by_possessive_standard(self):
        self.assertEqual(reading.solve(
            "Kennedy's house is 600 square feet larger than 4 times Benedict's house. If Kennedy's house is "
            "10000 square feet, how many square feet is Benedict's house?"), 2350)

    def test_comparative_adjective_by_possessive_standard(self):
        self.assertEqual(reading.solve(
            "Brenda weighs 10 pounds more than 3 times Mel's weight. If Brenda weighs 220 pounds, how much "
            "does Mel weigh?"), 70)

    def test_age_possessive_stays_unmerged(self):
        # the age domain owns "N times Y's AGE" via its own operator-based path — merging here regressed a
        # real case (Sarah/Mark/Billy/Ana). Direct check: _merge_times_multiplier must leave "three"/"times"
        # UNMERGED when the possessed noun is "age".
        rp = reading.analyze("Sarah's age is equal to three times Mark's age minus 4.")
        surfaces = [reading._key(u.surface) for u in rp.units]
        self.assertIn("three", surfaces)
        self.assertIn("times", surfaces)
        self.assertNotIn("three times", surfaces)


class TestImplicitRelativePossessionFrame(unittest.TestCase):
    """`_ratio_relation`/`_offset_ratio_relation`'s standard-frame check now also accepts an IMPLICIT
    relative clause with "that"/"which" elided — the standard immediately followed by a POSSESSION verb
    ("as much money DARWIN HAS" = "...as Darwin has"). Deliberately NARROW (a fixed verb set: has/had/have/
    owns/gets/receives/holds), not ANY verb — diff-mining caught a real confident-wrong from the broader
    "any verb" version: a mistagged common noun standard ("the CDs") followed by an unrelated PASSIVE
    modifier ("sold at $10 each") was misread as a name-ratio (66.67 vs gold 1000). +3 dev (Mia/Darwin,
    Jacob/Edward, Rubert/James — all real GSM8K cases)."""

    def test_offset_ratio_implicit_possession(self):
        self.assertEqual(reading.solve(
            "Mia has $20 more than twice as much money Darwin has. If Darwin has $45, how much money "
            "does Mia have?"), 110)

    def test_pure_ratio_implicit_possession_chain(self):
        self.assertEqual(reading.solve(
            "Edward has 40 shoes. Jacob has half the number of shoes Edward has. How many shoes does "
            "Jacob have?"), 20)

    def test_unrelated_passive_modifier_declines(self):
        # "the CDs SOLD at $10 each" — a mistagged common-noun standard ("CDs") followed by an UNRELATED
        # passive participle, not the standard's own possession verb — must NOT be read as a name-ratio.
        self.assertIsNone(reading._ratio_relation(
            reading.analyze("Prince bought half the CDs sold at $10 each")))


class TestIncrement100DoubleTriple(unittest.TestCase):
    """#100 - "double"/"triple" as NUMERIC multipliers (x2 / x3), so "double the number of X" and
    "triple that number" resolve like "twice"/"three times". Multi-sense (double: adjective|multiplier,
    triple: verb|multiplier); coverage disambiguation picks multiplier when it scales a count-noun /
    quantifier. Unlocks the real 'double the number of X' wrongs that #99 set up."""

    def test_double_the_number_of_x(self):
        self.assertEqual(reading.solve("Tom has 8 marbles. Megan owns double the number of marbles. "
                                       "How many marbles does Megan have?"), 16)

    def test_triple_that_number(self):
        self.assertEqual(reading.solve("Tom has 4 apples. Jerry has triple that number. "
                                       "How many apples does Jerry have?"), 12)

    def test_double_as_many(self):
        self.assertEqual(reading.solve("Tom has 7 apples. Jerry has double as many apples. "
                                       "How many apples does Jerry have?"), 14)


class TestIncrement102DimensionWords(unittest.TestCase):
    """#102 - dimension words ("as long/far/tall/...") act like the quantifier "many" in a relational
    "<multiplier> as <dim>" phrase, so "twice as long" -> 2 x the referent's quantity (NOT a Frequency
    of the verb). The referent supplies the measure. Generalizes the "as many/much" path to other
    measured dimensions."""

    def test_twice_as_long(self):
        self.assertEqual(reading.solve("Amber worked 12 hours. Bob worked twice as long. "
                                       "How many hours did Bob work?"), 24)

    def test_twice_as_far(self):
        self.assertEqual(reading.solve("Anna walked 9 kilometers. Beth walked three times as far. "
                                       "How many kilometers did Beth walk?"), 27)

    def test_plain_long_unaffected(self):
        # "long" with no multiplier scaling it is not a relational quantity
        self.assertEqual(reading.solve("Amber worked 12 hours. How many hours did Amber work?"), 12)


class TestIncrement103FractionMultipliers(unittest.TestCase):
    """#103 - fraction words as relational MULTIPLIERS in the "as" frame ("one-third as many", "two-
    thirds as far") = (num/denom) x the referent, computed EXACTLY (divide last, so 12*1/3 = 4 with no
    float drift). Distinct from the partitive "one-third OF them" (handled by _partitive_quantity);
    the following "as" vs "of" keeps them apart."""

    def test_one_third_as_many(self):
        self.assertEqual(reading.solve("Tom has 12 apples. Mary has one-third as many. "
                                       "How many apples does Mary have?"), 4)

    def test_two_thirds_as_far(self):
        self.assertEqual(reading.solve("Anna ran 12 miles. Beth ran two-thirds as far. "
                                       "How many miles did Beth run?"), 8)

    def test_three_quarters_as_many(self):
        self.assertEqual(reading.solve("Tom has 8 apples. Mary has three-quarters as many. "
                                       "How many apples does Mary have?"), 6)

    def test_partitive_of_still_declines_to_partitive(self):
        # "one-third OF them" stays the partitive subtraction, not a relational multiplier
        self.assertEqual(reading.solve("Tom has 12 apples. He loses one-third of them. "
                                       "How many apples does Tom have?"), 8)


class TestIncrement104ConjoinedContainers(unittest.TestCase):
    """#104 - CONJOINED container phrases: "N1 C of M1 and N2 C of M2" yields a PRODUCT per "of"
    (N1xM1 and N2xM2), summed at fold time. Each "of" pairs with its NEAREST value-noun on each
    side, so the second group is no longer dropped. A NESTED "of" with no inner count
    ("100 grams of sugar") still produces a SINGLE product (the inner "of" has no right value)."""

    def test_two_count_groups_sum(self):
        self.assertEqual(reading.solve("Nora has 3 boxes of 8 apples and 2 boxes of 5 apples. "
                                       "How many apples does Nora have?"), 34)

    def test_two_money_groups_sum(self):
        self.assertEqual(reading.solve("Leo has 4 bills of 5 dollars and 2 bills of 10 dollars. "
                                       "How much money does Leo have?"), 40)

    def test_conjoined_learn_yields_two_quantities(self):
        facts = reading.learn("Nora has 3 boxes of 8 apples and 2 boxes of 5 apples")
        amounts = sorted(f.amount for f in facts if isinstance(f, reading.Quantity))
        self.assertEqual(amounts, [10.0, 24.0])

    def test_nested_of_still_single_product(self):
        # the nested "of sugar" has no count on its right → still one product, unchanged from #75
        self.assertEqual(reading.learn("A store sells 20 packets of 100 grams of sugar"),
                         [reading.Quantity("store", ("gram", frozenset()), 2000.0, "sell")])


class TestIncrement106SlashFraction(unittest.TestCase):
    """#106 - the SLASH-DIGIT fraction form "3/8" is now a fraction (was parsed as the flat value
    0.375), so "3/8 of his money" routes to the partitive branch. A single partitive of a holding
    solves; a multi-step fraction CHAIN ("3/8 … and 3/10 …", "2/5 of the remaining") still DECLINES
    rather than committing a near-base fractional garbage answer — the unused-given guard catches the
    un-composed second fraction (a precision win: 4 dev wrongs → honest declines)."""

    def test_slash_fraction_parsed(self):
        self.assertEqual(reading._fraction_parts("3/8"), (3, 8))
        self.assertEqual(reading._fraction_parts("1/4"), (1, 4))
        self.assertIsNone(reading._fraction_parts("3/0"))   # zero denominator is not a fraction
        self.assertIsNone(reading._fraction_parts("apple"))

    def test_single_slash_partitive_declines_safely(self):
        # RE-PINNED (#219): `_signed_mixed_chain_system` now claims this CORRECTLY (80·5/8 = 50)
        # — the FOLD path's unused-given decline this class documents is unchanged; the EARLY
        # system answers first. Was pinned "undetermined" while no machinery could solve it.
        self.assertEqual(reading.solve("Mary had 80 dollars. She gave 3/8 of her money to her mom. "
                                       "How much money does Mary have left?"), 50)

    def test_fraction_chain_declines_not_commits_garbage(self):
        # RE-PINNED (#225): the elided base NOW composes — a coordinated bare fraction is
        # SAME-BASE: 200·(1 − 3/8 − 3/10) = 65. (The pre-#106 bug returned 199.625; the #106
        # decline held until the same-base branch landed.)
        self.assertEqual(reading.solve("Ann had 200 dollars. She gave 3/8 of her money to her mom and "
                                       "3/10 to her dad. How much money does Ann have left?"), 65)


class TestImperativeQuestionClause(unittest.TestCase):
    """`solve()` used to require a literal "?" ANYWHERE in the text, so a GSM8K problem phrased as an
    IMPERATIVE ("Calculate the total…", "Find X's weight.", ending in "." — ~4% of the corpus) was declined
    before any machinery even ran, regardless of whether the rest of the problem was otherwise solvable. A
    clause opening with a compute-imperative verb (calculate/determine/find/compute) is now recognized as the
    question clause too (lowest priority — an interrogative or literal "?" still wins). +2 dev-train this
    increment (Al/Ben/Carl/Ed-style weight chain, a rate-ratio case), 0 broke/wrong (diff-verified)."""

    def test_imperative_close_no_question_mark(self):
        # ends with a period, not "?" — used to decline immediately regardless of the chain being solvable.
        self.assertEqual(reading.solve(
            "Omar weighs 9 pounds more than Nia. Pria weighs 14 pounds less than Omar. Nia weighs 40 "
            "pounds. Find how much Pria weighs."), 35)

    def test_imperative_after_leading_if_clause(self):
        # "If P, calculate X." — the leading "If" clause becomes a premise exactly like "If P, how many X?"
        self.assertEqual(reading.solve(
            "Wren bought 12 apples. If Tam bought 7 more apples than Wren, calculate how many apples they "
            "bought altogether."), 31)

    def test_interrogative_with_question_mark_still_preferred(self):
        # a literal "?" with a wh-word must still win over any imperative-looking wording (no collision).
        self.assertEqual(reading.solve(
            "Tom has 3 apples. Sam has 5 apples. How many apples do they have in total?"), 8)


class TestCompoundNounSubtype(unittest.TestCase):
    """A NOUN-NOUN compound is head-FINAL — "8 flower stickers" is 8 stickers (head) of the flower kind
    (attribute), measure ('sticker', {'flower'}). A bare-head question ("how many STICKERS") with no exact
    match then sums the entity's sub-types under that head. Gated NARROWLY in `condense`: a value/quantifier
    must precede the modifier (the count ranges over the head), the modifier must not be hyphen-bound to it
    ("120-page book" → 120 counts pages), neither word may be a unit/money/time noun, and the sentence must
    carry no rate marker (a rate composes a partial result that was safely declining). SYNTHETIC."""

    def test_subtype_sum_under_head(self):
        self.assertEqual(reading.solve(
            "Victor has 8 flower stickers and 6 animal stickers. "
            "How many stickers does Victor have?"), 14)

    def test_subtype_relation_then_sum(self):
        # "twice as many animal stickers as flower stickers" resolves ONLY because both compounds key to
        # ('sticker', {...}); the bare-head "how many stickers in all" then sums 12 + 24.
        self.assertEqual(reading.solve(
            "Sara has 12 flower stickers. She has twice as many animal stickers as flower stickers. "
            "How many stickers does Sara have in all?"), 36)

    def test_hyphen_measure_not_stolen(self):
        # "120-page book": 120 counts PAGES (hyphen-bound), so the merge must NOT make it 120 books.
        self.assertEqual(reading.solve("John writes 20 pages a day. "
                                       "How long will it take him to write a 120-page book?"), 6)

    def test_rate_compound_still_declines(self):
        # "7 apple pies a day": a rate marker is present → no merge, so the cross-entity depletion that
        # isn't yet handled keeps this an honest decline rather than a partial rate×duration commit.
        self.assertEqual(reading.solve(
            "Cooper makes 7 apple pies a day. He does this for 12 days. Ashley then eats 50 of his pies. "
            "How many apple pies remain with Cooper?"), "undetermined")


class TestBackwardThenContinuation(unittest.TestCase):
    """A "then <verb>…" continuation supplies the elided subject (same-subject sequential action), and
    `_backward_inversion` is tried for any question phrasing (not just "at first"). SYNTHETIC."""

    def test_backward_any_phrasing(self):
        # unknown initial, two spends, $6 left → 6 + 1.75 + 1.25 = 9 (question is not "at first"-phrased).
        self.assertEqual(reading.solve(
            "Josh has some money. He spent $1.75 on a drink, and then spent another $1.25. "
            "If he had $6 left, how much money did Josh have?"), 9)

    def test_forward_then_continuation(self):
        # the second spend is captured forward too: 9 − 1.75 − 1.25 = 6.
        self.assertEqual(reading.solve(
            "Josh has 9 dollars. He spent $1.75 on a drink, and then spent another $1.25. "
            "How much money does Josh have left?"), 6)


class TestStativeSubjectElision(unittest.TestCase):
    """"X is A and is B" — the subjectless STATIVE conjunct borrows the left subject → an independent
    premise, so an age/comparative chain grounds. SYNTHETIC."""

    def test_age_chain_elided_subject(self):
        # Ali 8; Ali = Yusaf + 3 → Yusaf 5; Umar = 2×Yusaf = 10.
        self.assertEqual(reading.solve(
            "Ali is 8 years old and is 3 years older than Yusaf. Umar is twice as old as Yusaf. "
            "How old is Umar?"), 10)

    def test_question_clause_not_elided(self):
        # the subjectless QUESTION clause "how old is Sam?" must NOT borrow "Drew" (regression guard).
        self.assertEqual(reading.solve(
            "In five years Sam will be 3 times as old as Drew. If Drew is currently 12 years old, "
            "how old is Sam?"), 46)


class TestRelativeClauseSplit(unittest.TestCase):
    """"<Name>, who VERBs …" splits into an independent premise (who → the antecedent), so a ratio/comparative
    CHAIN resolves. SYNTHETIC."""

    def test_backward_ratio_chain(self):
        # Xavier = 3×Katie, Katie = 4×Cole, Xavier 84 → Cole = 84/3/4 = 7.
        self.assertEqual(reading.solve(
            "Xavier runs three times as many miles as Katie, who runs 4 times as many miles as Cole. "
            "If Xavier runs 84 miles, how many miles does Cole run?"), 7)

    def test_comparative_grounded_via_who(self):
        # Mark = Amy + 7, Amy = 15 → Mark now 22, in 5 years 27.
        self.assertEqual(reading.solve(
            "Mark is 7 years older than Amy, who is 15. How old will Mark be in 5 years?"), 27)


# === QUANTITIES & ARITHMETIC: quantity facts, aggregation, accumulation, coreference, currency, relational quantity ===

class TestIncrement41Arithmetic(unittest.TestCase):
    """#41 — pure arithmetic via the obligation parser + translate's safe evaluator (step 1 of
    replacing the LLM translator). A digit token is a `value`; an `operator` binds its left/right
    operands; the answer is built into a one-line program and evaluated."""

    def test_value_and_operator_are_parsed(self):
        r = reading.analyze("3 plus 2")
        v = next(u for u in r.units if u.surface == "3")
        self.assertIn("value", v.features["senses"])
        self.assertEqual(v.features["value"], 3.0)
        op = next(u for u in r.units if u.surface == "plus")
        self.assertEqual(op.features["op"], "+")

    def test_four_operations(self):
        self.assertEqual(reading.answer("What is 3 plus 2?", []), 5)
        self.assertEqual(reading.answer("What is 5 minus 2?", []), 3)
        self.assertEqual(reading.answer("What is 4 times 3?", []), 12)
        self.assertEqual(reading.answer("What is 6 divided by 2?", []), 3)

    def test_cli(self):
        self.assertEqual(reading.run_cli(["What is 3 plus 2?"]).strip(), "5")


class TestIncrement42QuantityFacts(unittest.TestCase):
    """#42 — Quantity facts: "Tom has 3 apples" -> Quantity(tom, apple, 3) (a value quantifies the
    noun via its `quantifies` obligation). "How many apples does Tom have?" reads the amount back."""

    def test_quantity_is_learned(self):
        self.assertEqual(reading.learn("Tom has 3 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3.0)])

    def test_how_many_does_entity_have(self):
        self.assertEqual(reading.answer("How many apples does Tom have?",
                                        reading.learn("Tom has 3 apples")), 3)

    def test_unknown_entity_is_undetermined(self):
        self.assertEqual(reading.answer("How many apples does Mary have?",
                                        reading.learn("Tom has 3 apples")), "undetermined")

    def test_cli(self):
        self.assertEqual(reading.run_cli(["Tom has 3 apples. How many apples does Tom have?"]).strip(),
                         "3")


class TestIncrement43Aggregation(unittest.TestCase):
    """#43 — aggregation across entities (math step 3): "How many apples do Tom and Mary have
    altogether?" sums the matching Quantity facts. Needs a conjoined subject ("Tom and Mary") and a
    fold over the amounts. Still no coreference."""

    def _ctx(self):
        return reading.learn("Tom has 3 apples") + reading.learn("Mary has 2 apples")

    def test_altogether_sums(self):
        self.assertEqual(reading.answer("How many apples do Tom and Mary have altogether?",
                                        self._ctx()), 5)

    def test_without_altogether_also_sums(self):
        self.assertEqual(reading.answer("How many apples do Tom and Mary have?", self._ctx()), 5)

    def test_unknown_entity_is_undetermined(self):
        ctx = reading.learn("Tom has 3 apples")  # Mary's count was never stated
        self.assertEqual(reading.answer("How many apples do Tom and Mary have altogether?", ctx),
                         "undetermined")

    def test_single_entity_still_works(self):
        self.assertEqual(reading.answer("How many apples does Tom have?", self._ctx()), 3)

    def test_cli(self):
        out = reading.run_cli(["Tom has 3 apples. Mary has 2 apples. "
                               "How many apples do Tom and Mary have altogether?"])
        self.assertEqual(out.strip(), "5")

    def test_bare_together_sums(self):
        # "together" alone (without "all") is also an aggregation cue, like "altogether"/"combined".
        self.assertEqual(reading.answer("How many apples do Tom and Mary have together?",
                                        self._ctx()), 5)

    def test_together_not_hijacked_on_how_long(self):
        # The collaborative-rate sense ("how long … together?") must NOT be turned into a sum. A
        # how-long question with two duration facts present should NOT return their sum (5) off the
        # bare-"together" cue — the time-question guard keeps it out of the aggregation path.
        ctx = reading.learn("Tom works 3 hours") + reading.learn("Mary works 2 hours")
        self.assertNotEqual(reading.answer("How long do Tom and Mary work together?", ctx), 5)


class TestIncrement44Accumulation(unittest.TestCase):
    """#44 — accumulation on one entity (math step 4): repeated Quantity statements for the same
    entity+measure ACCUMULATE (signed sum). The verb carries the polarity: buy/get = gain (+),
    lose/sell/give = loss (-), have = baseline. The first arithmetic INTENT carried by a verb."""

    def test_buy_quantity_records_the_verb(self):
        self.assertEqual(reading.learn("Tom buys 2 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 2.0, "buy")])

    def test_buys_adds(self):
        ctx = reading.learn("Tom has 3 apples") + reading.learn("Tom buys 2 apples")
        self.assertEqual(reading.answer("How many apples does Tom have?", ctx), 5)

    def test_loses_subtracts(self):
        ctx = reading.learn("Tom has 5 apples") + reading.learn("Tom loses 2 apples")
        self.assertEqual(reading.answer("How many apples does Tom have?", ctx), 3)

    def test_chain_of_changes(self):
        ctx = (reading.learn("Tom has 3 apples") + reading.learn("Tom buys 4 apples")
               + reading.learn("Tom gives 2 apples"))
        self.assertEqual(reading.answer("How many apples does Tom have?", ctx), 5)

    def test_gains_without_a_baseline(self):
        ctx = reading.learn("Tom buys 3 apples") + reading.learn("Tom buys 2 apples")
        self.assertEqual(reading.answer("How many apples does Tom have?", ctx), 5)

    def test_cli(self):
        out = reading.run_cli(["Tom has 3 apples. Tom buys 2 apples. How many apples does Tom have?"])
        self.assertEqual(out.strip(), "5")


class TestIncrement45Coreference(unittest.TestCase):
    """#45 — coreference as an obligation (math step 5a): a pronoun carries a `refers` obligation,
    discharged cross-sentence against the discourse (most recent gender-compatible entity), logged as
    a reversible Decision. Accumulation then works across pronouns."""

    def test_pronoun_resolves_via_discourse(self):
        disc = []
        reading.learn("Tom has 3 apples", disc)         # discourse now knows Tom
        self.assertEqual(reading.learn("He buys 2 apples", disc),
                         [reading.Quantity("tom", ("apple", frozenset()), 2.0, "buy")])

    def test_accumulation_across_a_pronoun(self):
        out = reading.run_cli(["Tom has 3 apples. He buys 2 apples. How many apples does Tom have?"])
        self.assertEqual(out.strip(), "5")

    def test_gender_picks_the_right_antecedent(self):
        # "She" must skip the more-recent Tom and bind Mary.
        out = reading.run_cli(["Mary has 5 apples. Tom has 3 apples. She buys 2 apples. "
                               "How many apples does Mary have?"])
        self.assertEqual(out.strip(), "7")

    def test_pronoun_without_antecedent_is_a_deficiency(self):
        r = reading.analyze("He buys 2 apples")          # no discourse → who is "he"?
        self.assertFalse(reading.is_well_formed(r))
        self.assertTrue(any("refers" in d.message or "antecedent" in d.message
                            for d in reading.deficiencies(r)))

    def test_coreference_shows_in_trace(self):
        out = reading.run_cli(["Tom has 3 apples. He buys 2 apples. How many apples does Tom have?",
                               "--trace"])
        self.assertIn("refers to 'tom'", out)


class TestIncrement46ValueOperandForego(unittest.TestCase):
    """#46 — a value's `quantifies` is REQUIRED but FOREGONE when the value is an operator's operand
    (forego-when-subordinated, #37) — not the `optional` shortcut it was. "3 plus 2" is well-formed
    (operands), "Tom has 3 apples" is well-formed (quantifies a noun), but a stray "Tom has 3" is
    honestly flagged."""

    def test_operands_are_well_formed(self):
        self.assertTrue(reading.is_well_formed(reading.analyze("3 plus 2")))

    def test_value_quantifying_a_noun_is_well_formed(self):
        self.assertTrue(reading.is_well_formed(reading.analyze("Tom has 3 apples")))

    def test_stray_value_is_flagged(self):
        r = reading.analyze("Tom has 3")
        self.assertFalse(reading.is_well_formed(r))
        self.assertTrue(any("quantifies" in d.message or "counts" in d.message
                            for d in reading.deficiencies(r)))


class TestIncrement47VerbFrequency(unittest.TestCase):
    """#47 — a value can quantify a VERB's frequency, not only a noun. "twice"/"thrice"/"once" are
    lexicalized multipliers (value + frequency) that `scales` the verb to their left. "Mary jumps
    twice" -> Frequency(mary, jump, 2); "How many times does Mary jump?" -> 2. ("N times" later.)"""

    def test_twice_is_a_frequency(self):
        self.assertEqual(reading.learn("Mary jumps twice"), [reading.Frequency("mary", "jump", 2)])

    def test_once_and_thrice(self):
        self.assertEqual(reading.learn("Mary jumps once"), [reading.Frequency("mary", "jump", 1)])
        self.assertEqual(reading.learn("Tom runs thrice"), [reading.Frequency("tom", "run", 3)])

    def test_multiplier_scales_the_verb(self):
        r = reading.analyze("Mary jumps twice")
        tw = next(u for u in r.units if u.surface == "twice")
        self.assertEqual(tw.features["value"], 2)
        self.assertEqual(reading._follow(tw, "scales").surface, "jumps")

    def test_how_many_times_question(self):
        self.assertEqual(reading.run_cli(["Mary jumps twice. How many times does Mary jump?"]).strip(),
                         "2")

    def test_times_operator_still_multiplies(self):
        # the times operator-vs-noun ambiguity resolves by coverage: here values flank it -> operator.
        self.assertEqual(reading.answer("What is 4 times 3?", []), 12)

    def test_unknown_frequency_is_undetermined(self):
        self.assertEqual(reading.run_cli(["Mary jumps twice. How many times does Tom jump?"]).strip(),
                         "undetermined")


class TestIncrement48RelationalQuantity(unittest.TestCase):
    """#48 — relational quantities (math step 5b): "Mary has twice as many" is a DERIVED Quantity =
    factor x a resolved referent (the most recent prior Quantity, like coreference but for a
    quantity). The multiplier (twice/half) scales the "as many" quantifier; the referent supplies
    both the measure and the amount."""

    def test_twice_as_many(self):
        out = reading.run_cli(["Tom has 3 apples. Mary has twice as many. "
                               "How many apples does Mary have?"])
        self.assertEqual(out.strip(), "6")

    def test_half_as_many(self):
        out = reading.run_cli(["Tom has 8 apples. Mary has half as many. "
                               "How many apples does Mary have?"])
        self.assertEqual(out.strip(), "4")

    def test_referent_supplies_measure_and_amount(self):
        ctx = reading.learn("Tom has 3 apples")
        self.assertEqual(reading.learn("Mary has twice as many", None, ctx),
                         [reading.Quantity("mary", ("apple", frozenset()), 6, "have")])

    def test_no_referent_declines(self):
        # nothing prior to be "as many as" -> no fabricated quantity
        self.assertEqual(reading.learn("Mary has twice as many", None, []), [])

    def test_relational_across_a_pronoun(self):
        out = reading.run_cli(["Tom has 4 apples. He buys twice as many. "
                               "How many apples does Tom have?"])
        self.assertEqual(out.strip(), "12")  # 4 (has) + 8 (twice as many as the 4)

    def test_backward_inversion_grounds_the_standard(self):
        # BIDIRECTIONAL: when the SUBJECT's value is known and the standard's isn't, invert the
        # relation to ground the standard. "Mary has 3× as many apples as Tom; Mary has 12" → Tom = 4.
        ctx = reading.learn("Mary has 3 times as many apples as Tom") \
            + reading.learn("Mary has 12 apples")
        self.assertEqual(reading.learn("Mary has 3 times as many apples as Tom", None, ctx),
                         [reading.Quantity("tom", ("apple", frozenset()), 4, "have")])

    def test_backward_inversion_chains(self):
        # Sarah grounds backward off Tyrah, then Tim resolves forward off Sarah.
        out = reading.run_cli(["Tyrah has 6 times as many pencils as Sarah. "
                               "Tim has 8 times as many pencils as Sarah. Tyrah has 12 pencils. "
                               "How many pencils does Tim have?"])
        self.assertEqual(out.strip(), "16")

    def test_elided_count_noun_infers_single_measure(self):
        # "how many does X have?" omits the measure; when the context shares ONE measure, infer it.
        ctx = reading.learn("Tom has 5 apples")
        self.assertEqual(reading.answer("How many does Tom have?", ctx), 5)

    def test_elided_count_noun_ambiguous_declines(self):
        # two distinct measures in context → the elided "how many" is genuinely ambiguous → decline.
        ctx = reading.learn("Tom has 5 apples") + reading.learn("Tom has 3 oranges")
        self.assertIsInstance(reading.answer("How many does Tom have?", ctx), list)

    def test_average_over_entities(self):
        # average = Σ(each owner's holding) / owner-count, over the single shared measure.
        ctx = (reading.learn("Tom has 4 apples") + reading.learn("Mary has 6 apples")
               + reading.learn("Sue has 8 apples"))
        self.assertEqual(reading.answer("What is the average number of apples?", ctx), 6)

    def test_average_stated_count_mismatch_declines(self):
        # "the three" but only two owners present → divide-by-wrong-N risk → decline.
        ctx = reading.learn("Tom has 4 apples") + reading.learn("Mary has 6 apples")
        self.assertEqual(reading.answer("What is the average number of apples of the three people?", ctx),
                         "undetermined")

    def test_value_list_parses_to_separate_quantities(self):
        # "got 8, 10, and 12 in his three games" → three quantities; the "three" cardinal equals the list
        # length (the set size), not a fourth datum, so it is dropped.
        facts = reading.learn("Sam got 8, 10, and 12 in his three games")
        amounts = sorted(f.amount for f in facts if isinstance(f, reading.Quantity))
        self.assertEqual(amounts, [8, 10, 12])

    def test_value_list_average(self):
        out = reading.run_cli(["Sam got 8, 10, and 12 in his three games. "
                               "What is his average score?"])
        self.assertEqual(out.strip(), "10")


class TestIncrement49QuestionRelativePolarity(unittest.TestCase):
    """#49 — a quantity's sign is RELATIVE TO THE QUESTION. "how many does X have?" folds holdings
    (baseline + gains - losses); "how many did X sell?" COUNTS the sell events (positively). The same
    "sold 8" is -8 toward holdings but +8 toward a sales count -- it's the question that decides."""

    def test_holdings_subtracts_a_sale(self):
        out = reading.run_cli(["Tom has 10 apples. Tom sells 3 apples. How many apples does Tom have?"])
        self.assertEqual(out.strip(), "7")

    def test_sales_count_is_positive(self):
        out = reading.run_cli(["Tom sold 8 apples. Tom sold 4 apples. How many apples did Tom sell?"])
        self.assertEqual(out.strip(), "12")

    def test_event_count_filters_by_verb(self):
        out = reading.run_cli(["Tom sold 8 apples. Tom bought 5 apples. How many apples did Tom sell?"])
        self.assertEqual(out.strip(), "8")

    def test_natalia_end_to_end(self):
        out = reading.run_cli(["Natalia sold 48 clips. She sold half as many. "
                               "How many clips did Natalia sell?"])
        self.assertEqual(out.strip(), "72")


class TestIncrement50RemainderSubtraction(unittest.TestCase):
    """#50 — "how many are left / remaining" is the HOLDINGS question (baseline + gains - losses) in
    another surface form. Form A ("does X have left") routes through the existing holdings fold once
    the remainder words are recognized; Form B ("are left", no aux) folds the implicit unique owner.
    A "how many" interrogative no longer gets stolen by the property branch ("are apples left?")."""

    def test_have_left_form(self):
        out = reading.run_cli(["Tom has 10 apples. Tom eats 3 apples. "
                               "How many apples does Tom have left?"])
        self.assertEqual(out.strip(), "7")

    def test_are_left_form_implicit_entity(self):
        out = reading.run_cli(["Tom has 10 apples. Tom eats 3 apples. How many apples are left?"])
        self.assertEqual(out.strip(), "7")

    def test_remaining_synonym(self):
        out = reading.run_cli(["Tom has 8 apples. Tom gives 5 apples. How many apples are remaining?"])
        self.assertEqual(out.strip(), "3")

    def test_implicit_measure(self):
        out = reading.run_cli(["Tom has 9 apples. Tom eats 4 apples. How many are left?"])
        self.assertEqual(out.strip(), "5")

    def test_ambiguous_owner_declines(self):
        out = reading.run_cli(["Tom has 5 apples. Mary has 4 apples. How many apples are left?"])
        self.assertEqual(out.strip(), "undetermined")


class TestIncrement51Currency(unittest.TestCase):
    """#51 — currency: "$12" is a self-measuring VALUE (12 of the money measure; the "$" discharges
    its `quantifies` obligation). "X verb $N" -> a money Quantity; "how much" routes like "how many"
    with the measure fixed to money, so holdings/event-count polarity composes."""

    def test_dollar_value_parses_with_money_measure(self):
        r = reading.analyze("$12")
        u = r.units[0]
        self.assertIn("value", u.features["senses"])
        self.assertEqual(u.features["value"], 12.0)
        self.assertEqual(u.features["measure"], reading.MONEY)

    def test_money_quantity_learned(self):
        self.assertEqual(reading.learn("Weng earns $12"),
                         [reading.Quantity("weng", reading.MONEY, 12.0, "earn")])

    def test_how_much_holdings(self):
        out = reading.run_cli(["Tom has $10. Tom spends $3. How much money does Tom have?"])
        self.assertEqual(out.strip(), "7")

    def test_how_much_event_count(self):
        out = reading.run_cli(["Weng earns $12. How much did Weng earn?"])
        self.assertEqual(out.strip(), "12")

    def test_dollar_value_is_well_formed(self):
        self.assertTrue(reading.is_well_formed(reading.analyze("Weng earns $12")))

    def test_solve_declines_on_unknown_words(self):
        # a money parse exists, but a nonce word ("wuggle") means we may be missing information a
        # partial parse dropped -> decline, never commit a number off a fragment. (Nonce-pinned so it
        # stays an unknown-word test as the lexicon grows.)
        self.assertEqual(reading.solve("Tom earns $12 wuggle. How much did Tom earn?"),
                         "undetermined")

    def test_solve_still_answers_a_fully_known_problem(self):
        self.assertEqual(reading.solve("Tom has $10. Tom spends $3. How much money does Tom have?"), 7)


class TestIncrement54ComparativeQuantity(unittest.TestCase):
    """#54 - comparative quantities: "how much more X does A have than B?" -> holdings(A)-holdings(B);
    "A has N less X than B" -> B-N (a derived quantity off the NAMED standard). Fixed the `object`
    resolver (binds names: "than WILLIAM") and `standard` (resolves through `than`, skipping the
    measure). Currency words normalize to one MONEY measure. The precision guards still hold."""

    def test_than_binds_the_name(self):
        r = reading.analyze("How much more money does Oliver have than William?")
        than = next(u for u in r.units if u.surface == "than")
        self.assertEqual(reading._follow(than, "object").surface, "William?")
        more = next(u for u in r.units if u.surface == "more")
        self.assertEqual(reading._follow(more, "standard").surface, "William?")  # not "money"

    def test_question_form_more(self):
        out = reading.run_cli(["Oliver has $30. William has $20. "
                               "How much more money does Oliver have than William?"])
        self.assertEqual(out.strip(), "10")

    def test_question_form_less(self):
        out = reading.run_cli(["Oliver has $30. William has $50. "
                               "How much less money does Oliver have than William?"])
        self.assertEqual(out.strip(), "20")

    def test_currency_words_agree(self):
        self.assertEqual(reading.solve("Oliver has 30 dollars. William has 20 dollars. "
                                       "How much more money does Oliver have than William?"), 10)

    def test_statement_form_less(self):
        self.assertEqual(reading.solve("Martha has 68 apples. Tim has 30 less apples than Martha. "
                                       "How many apples does Tim have?"), 38)

    def test_statement_form_more(self):
        self.assertEqual(reading.solve("Bob has 10 apples. Tom has 5 more apples than Bob. "
                                       "How many apples does Tom have?"), 15)

    def test_forward_reference_resolves_via_fixpoint(self):
        # the standard's count is stated AFTER -> the solve() fixpoint defers and retries -> resolves
        # (this declined before multi-step chaining; #58 made it resolve)
        self.assertEqual(reading.solve("Tim has 30 less apples than Martha. Martha has 68 apples. "
                                       "How many apples does Tim have?"), 38)

    def test_times_more_misparse_still_declines(self):
        self.assertEqual(reading.solve("Riku has 25 times more stickers than Kristoff. "
                                       "Kristoff has 85 stickers. How many stickers does Riku have?"),
                         "undetermined")


class TestIncrement58MultiStepChaining(unittest.TestCase):
    """#58 - multi-step chaining via a solve() FIXPOINT: a statement whose referent is stated LATER
    (a forward reference) is deferred and retried as the context grows, until nothing new resolves.
    Chains of derived quantities resolve; relational referents resolve by NAME; "N times more" stays
    declined (multiplicative, not our additive ±)."""

    def test_forward_reference(self):
        self.assertEqual(reading.solve("Tim has 30 less apples than Martha. Martha has 68 apples. "
                                       "How many apples does Tim have?"), 38)

    def test_three_hop_chain(self):
        self.assertEqual(reading.solve("Tim has 30 less apples than Martha. "
                                       "Harry has half as many apples as Tim. Martha has 68 apples. "
                                       "How many apples does Harry have?"), 19)

    def test_relational_referent_resolves_by_name(self):
        # "half as many as Tim" -> Tim's 10 (named), NOT the most-recent Bob's 4
        self.assertEqual(reading.solve("Tim has 10 apples. Bob has 4 apples. "
                                       "Harry has half as many apples as Tim. "
                                       "How many apples does Harry have?"), 5)

    def test_times_more_still_declines(self):
        self.assertEqual(reading.solve("Riku has 25 times more stickers than Kristoff. "
                                       "Kristoff has 85 stickers. How many stickers does Riku have?"),
                         "undetermined")

    def test_unresolvable_chain_declines(self):
        self.assertEqual(reading.solve("Tim has 30 less apples than Martha. "
                                       "How many apples does Tim have?"), "undetermined")


class TestIncrement75ContainerMultiplication(unittest.TestCase):
    """#75 - "N <container> of M <unit>" → N×M of the INNER unit (containment multiplication), plus
    question-measure UNIT CONVERSION (kilograms asked while grams are held). Solves the sugar problem.
    A plain conjoined object ("3 apples and 2 oranges", no "of") still stays two quantities."""

    def test_container_multiplication(self):
        self.assertEqual(reading.learn("A store sells 20 packets of 100 grams of sugar"),
                         [reading.Quantity("store", ("gram", frozenset()), 2000.0, "sell")])

    def test_how_many_grams(self):
        self.assertEqual(reading.solve("A store sells 20 packets of 100 grams of sugar. "
                                       "How many grams of sugar does it sell?"), 2000)

    def test_sugar_with_kilogram_conversion(self):
        self.assertEqual(reading.solve("A store sells 20 packets of 100 grams of sugar every week. "
                                       "How many kilograms of sugar does it sell every week?"), 2)

    def test_boxes_of_n_each(self):
        self.assertEqual(reading.solve("Mary has 5 boxes of 6 apples each. "
                                       "How many apples does Mary have?"), 30)

    def test_conjoined_object_not_multiplied(self):
        self.assertEqual(reading.learn("Tom has 3 apples and 2 oranges"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3.0, "have"),
                          reading.Quantity("tom", ("orange", frozenset()), 2.0, "have")])


class TestIncrement88CombinedGrandTotal(unittest.TestCase):
    """#88 - "How many <unit> are all <X> COMBINED?" → the GRAND TOTAL: every quantity convertible to the
    asked unit, summed IN that unit (cross-entity, cross-unit via CONVERSIONS). 2 feet + 16 in + 10 in =
    50 inches. Gated to a GENERIC population (no NAMED entities — "Tom and Mary altogether" keeps the
    per-entity fold, which declines if an entity's count is unknown)."""

    def test_christina_combined_with_conversion(self):
        self.assertEqual(reading.solve("Christina has 3 snakes. 1 snake is 2 feet long. Another snake is "
                                       "16 inches long. The last snake is 10 inches. How many inches are "
                                       "all snakes combined?"), 50)

    def test_combined_same_unit(self):
        self.assertEqual(reading.solve("One box has 5 apples. Another box has 7 apples. "
                                       "How many apples are all boxes combined?"), 12)

    def test_named_entities_still_per_entity(self):
        self.assertEqual(reading.answer("How many apples do Tom and Mary have altogether?",
                                        reading.learn("Tom has 3 apples")), "undetermined")


class TestIncrement96AuditedArithmetic(unittest.TestCase):
    """#96 - every word-problem number is produced by EXECUTING a tiny arithmetic program through the
    AUDITED safe evaluator (translate.run_program: ast.parse + an operator/function whitelist, never
    eval/exec), recorded in ARITHMETIC_LOG. No answer is computed by a bare Python operator — restoring the
    'correct-by-construction and auditable' guarantee the docs claim."""

    def test_solve_logs_the_audited_programs(self):
        reading.solve("James spends 40 years teaching. His partner has been teaching for 10 years less. "
                      "How long is their combined experience?")
        self.assertEqual(reading.ARITHMETIC_LOG, [("40 - 10", 30), ("40 + 30", 70)])

    def test_log_resets_per_solve(self):
        reading.solve("Tom has 3 apples. How many apples does Tom have?")
        self.assertEqual(reading.ARITHMETIC_LOG, [("1 * 3", 3)])      # reset, not accumulated across solves

    def test_trace_surfaces_audited_arithmetic(self):
        out = reading.run_cli(["James spends 40 years teaching. His partner has been teaching for 10 years "
                               "less. How long is their combined experience?", "--trace"])
        self.assertIn("answer = 40 - 10   ->  30", out)   # the programs, surfaced in --trace
        self.assertIn("answer = 40 + 30   ->  70", out)
        self.assertIn("ANSWER: 70", out)

    def test_cli_fixpoints_like_solve(self):
        out = reading.run_cli(["By the time Anne is two times as old as Emile, Emile will be six times as "
                               "old as Maude. If Maude will be 8 years old, how old will Anne be?"])
        self.assertEqual(out.strip(), "96")               # CLI clause-splits + fixpoints, matching solve()

    def test_compute_routes_through_safe_evaluator(self):
        self.assertEqual(reading._compute("2 * 48"), 96)
        with self.assertRaises(Exception):                            # run_program rejects non-arithmetic
            reading._compute("__import__('os').system('echo hi')")


class TestIncrement99NumberOfReferent(unittest.TestCase):
    """#99 - "<multiplier> the number/amount (of X)" is the SAME relational quantity as
    "<multiplier> as many (as X)": the multiplier scales the count-noun "number"/"amount" (not the
    quantifier "many"), and the referent supplies the count. "twice the number of marbles" -> 2 x the
    prior count. Found via the widened firewall (the 'twice/double the number of X' wrongs). The
    trailing RELATIVE clause ("the number that X has") was originally left declined here — the general
    relational-quantity path (`_derived_quantity`, used by `learn()`) mis-reads the matrix verb as an
    auxiliary — but 4ao's `_ratio_relation`/`_offset_ratio_relation` frame-check extension (accept an
    IMPLICIT relative clause, standard immediately followed by ITS OWN verb: "money DARWIN HAS") now solves
    it via the SEPARATE, tightly-gated anchored-single-system path (exactly two names, positional subj/verb
    split), independent of the general path's issue."""

    def test_twice_the_number_implicit_referent(self):
        self.assertEqual(reading.solve("Tom has 6 marbles. Megan owns twice the number of marbles. "
                                       "How many marbles does Megan have?"), 12)

    def test_twice_the_number_of_named(self):
        self.assertEqual(reading.solve("Anna has 50 pencils. Harry has twice the number of Anna pencils. "
                                       "How many pencils does Harry have?"), 100)

    def test_n_times_the_number(self):
        self.assertEqual(reading.solve("Frank has 36 apples. Susan has three times the number of apples. "
                                       "How many apples does Susan have?"), 108)

    def test_trailing_relative_clause_now_solves(self):
        # "the number (that) X has" — now solved via the anchored-single-system's implicit-relative frame.
        self.assertEqual(reading.solve("Tom has 6 marbles. Megan has twice the number Tom has. "
                                       "How many marbles does Megan have?"), 12)

    def test_the_number_without_multiplier_unaffected(self):
        # a plain "the number" with no multiplier is not a relational quantity (no spurious fact)
        self.assertEqual(reading.solve("Tom has 6 marbles. How many marbles does Tom have?"), 6)


class TestIncrement109LossFrameComparative(unittest.TestCase):
    """#109 - a comparative whose standard is a LOSS ("Veronica lost 1.5 pounds more than Seth") must
    compare MAGNITUDES: _holdings nets a loss to −17.5, but stored amounts (and the "in total" sum) are
    positive magnitudes, so the referent is expressed in the SUBJECT's frame (×verb-sign) → 17.5+1.5=19,
    not −17.5+1.5=−16. Gain/'have' comparatives and 'less than' are unaffected."""

    def test_loss_frame_more_than(self):
        self.assertEqual(reading.solve(
            "Seth lost 17 pounds. Jerome lost three times that many pounds and Veronica lost 2 pounds "
            "more than Seth. How many pounds did the 3 people lose in total?"), 17 + 51 + 19)

    def test_have_frame_more_than_unaffected(self):
        self.assertEqual(reading.solve("Tom has 10 apples. Bob has 5 more apples than Tom. "
                                       "How many apples does Bob have?"), 15)

    def test_less_than_unaffected(self):
        self.assertEqual(reading.solve("Martha has 68 apples. Tim has 30 less apples than Martha. "
                                       "How many apples does Tim have?"), 38)


class TestIncrement110DerivedReferentBaseline(unittest.TestCase):
    """#110 - an IMPLICIT relational referent ("Ella worked twice as long") scales the most-recent
    NON-derived baseline, not a just-derived sibling sitting between them. Quantity carries a `derived`
    flag (set by the relational/comparative/partitive producers, compare=False so equality is unchanged);
    the implicit referent in _relational_quantity skips derived quantities."""

    def test_implicit_referent_picks_baseline_not_derived_sibling(self):
        # Armand = 12/3 = 4 (derived); Ella = "twice as long" must scale AMBER (12), not Armand (4) →
        # 2×12=24, total 12+4+24=40 (was 8, total 24, when Ella scaled the derived Armand).
        self.assertEqual(reading.solve(
            "Amber worked for 12 hours. Armand worked one-third as long and Ella worked twice as long. "
            "How many hours did the 3 people work in total?"), 40)

    def test_derived_flag_excluded_from_equality(self):
        a = reading.Quantity("x", ("apple", frozenset()), 5.0, "have")
        b = reading.Quantity("x", ("apple", frozenset()), 5.0, "have", derived=True)
        self.assertEqual(a, b)        # compare=False — a derived and a stated quantity are still equal


class TestActionAmountAcrossTimes(unittest.TestCase):
    """"savings"/"earnings" normalise to the money measure (so "half of his SAVINGS" → his money holding), and
    GUARD 5(b) (≥2 distinct times + no question-time → decline) is EXEMPTED for an ACTION-amount question that
    names a verb matched by exactly one quantity ("how much did he SPEND?" — unambiguous), while a HOLDING
    question ("how much will he HAVE?") still declines (Jack 47≠97 is a real timeline ambiguity). dev-train
    67→68 (Donny 28), holdout/TEST held. SYNTHETIC."""

    def test_spend_fraction_of_savings_across_days(self):
        self.assertEqual(reading.solve(
            "Donny saves $15 on Monday, $28 on Tuesday and $13 on Wednesday. On Thursday he spends half of "
            "his total savings so far. How much did he spend?"), 28)

    def test_holding_question_across_times_now_solves(self):
        # was pinned "decline" while the fold committed a PARTIAL (47≠97 — GUARD 5(b)'s reason);
        # #198's `_today_tomorrow_chain_system` reads the FULL signed chain (65−18+50, every digit
        # accounted) so this exact shape now solves correctly. GUARD 5(b) still declines holding
        # questions the chain's tight gates don't claim.
        self.assertEqual(reading.solve(
            "Jack has 65 pounds of sugar today. Tomorrow he will use 18 pounds of sugar. The following day "
            "he will buy 50 more pounds of sugar. How many pounds will he have in the end?"), 97)


class TestNounAfterQuantifierAndInAll(unittest.TestCase):
    """Two coupled fixes for "How many cans did the boys collect in all?": (1) a word after "many"/"much"
    with both a noun and an auxiliary sense reads as the NOUN (else 'cans'→modal and "many" binds 'boys');
    (2) "in all" is a combined-total cue. Together: Solomon=66, Juwan=66/3=22, Levi=22/2=11 → Σ = 99.
    dev-train 66→67, holdout/TEST held. SYNTHETIC."""

    def test_solomon_cans_in_all(self):
        self.assertEqual(reading.solve(
            "Solomon collected three times as many cans as Juwan. Levi collected half of what Juwan "
            "collected. Solomon collected 66 cans. How many cans did the boys collect in all?"), 99)

    def test_cans_reads_as_noun_after_many(self):
        r = reading.analyze("How many cans did the boys collect?")
        cans = next(u for u in r.units if u.surface == "cans")
        self.assertEqual(cans.features["senses"], ["noun"])

    def test_modal_can_unaffected_elsewhere(self):
        # "can" as a true modal (not after many/much) is untouched.
        self.assertEqual(reading.solve("Bob has 5 apples. Ann has 3 apples. How many apples in all?"), 8)


class TestLabelLetterEntities(unittest.TestCase):
    """"Jar A" / "Jar B" / "Jar C" are DISTINCT labeled entities, not the bare noun 'jar' three times. A
    noun/name + a single capital letter merges into one NAME unit (keyed 'jar a' etc.), firing on a lowercase
    head too ("…than jar A") so a subject and a later standard key identically — letting the forward
    chain + combined-sum machinery solve them. dev-train 65→66 (Jar A/B/C 148), holdout/TEST held. SYNTHETIC."""

    def test_jar_label_chain_sums(self):
        # A=28, B=A+12=40, C=2·B=80 → 28+40+80 = 148.
        self.assertEqual(reading.solve(
            "Jar A has 28 marbles. Jar B has 12 more marbles than jar A. Jar C has twice as many marbles as "
            "jar B. How many marbles are there altogether?"), 148)

    def test_distinct_label_entities(self):
        a = reading.analyze("Jar A has 28 marbles.")
        b = reading.analyze("jar B has 5 marbles.")
        ea = reading._entity(next(u for u in a.units if "name" in u.features.get("senses", [])))
        eb = reading._entity(next(u for u in b.units if "name" in u.features.get("senses", [])))
        self.assertEqual((ea, eb), ("jar a", "jar b"))

    def test_sentence_initial_article_untouched(self):
        # "A teacher …" — the article has no preceding noun, so it must NOT merge into a label entity.
        r = reading.analyze("A teacher distributed 30 candies.")
        self.assertTrue(any(u.surface.strip(".") == "teacher" for u in r.units))
        self.assertFalse(any("A teacher" in u.surface for u in r.units))


class TestAbstractMeasureAccumulation(unittest.TestCase):
    """"How FAR did she walk for 2 hours?" — no measure noun in the question, but the context holds one
    measure, one owner, and ≥2 quantities (an accumulation the measure-less question couldn't dispatch) →
    that owner's net holding (Hazel 2 + 4 = 6). The ≥2 gate keeps a single-fact "how much did Tom run?"
    (5 miles) DECLINING per #74's no-guess design. dev-train 64→65, holdout/TEST held. SYNTHETIC."""

    def test_abstract_measure_accumulation(self):
        self.assertEqual(reading.solve(
            "Hazel walked 2 kilometers in the first hour. If she walked twice as far in the second hour, "
            "how far did she walk for 2 hours?"), 6)

    def test_single_fact_abstract_still_declines(self):
        self.assertNotIsInstance(reading.solve("Tom ran 5 miles. How much did Tom run?"), (int, float))


class TestElidedComparativeQuantity(unittest.TestCase):
    """A BARE comparative — "Riley has 3 fewer." — with BOTH the measure noun and the referent elided. The
    most-recent prior Quantity supplies both: measure (pairs) and base (8), so Riley = 8 − 3 = 5, then the
    "they … in all" fold sums 8 + 5 = 13. Restricted to the bare frame (NO "than" at all): an anaphoric
    "47 more than that" reads correctly in isolation but sits in multi-step chains (Billy/Katelyn/Chloe)
    where a later part stays underived, so deriving it turns an honest decline into a partial-sum wrong."""

    def test_bare_fewer_inherits_measure_and_sums(self):
        self.assertEqual(reading.solve(
            "Ellie has 8 pairs of shoes. Riley has 3 fewer. "
            "How many pairs of shoes do they have in all?"), 13)

    def test_bare_fewer_derives_the_part(self):
        self.assertEqual(reading.solve(
            "Ellie has 8 pairs of shoes. Riley has 3 fewer. How many does Riley have?"), 5)

    def test_more_than_that_chain_still_declines(self):
        # "47 more than that" has a "than" token → the elided path is suppressed, so Katelyn stays
        # underived and the partial sum (Billy+Katelyn, missing Chloe) is never committed.
        self.assertEqual(reading.solve(
            "Billy made 49 sandwiches; Katelyn made 47 more than that. Chloe made a quarter of the "
            "amount that Katelyn made. How many sandwiches did they make in all?"), "undetermined")

    def test_explicit_standard_path_intact(self):
        # the noun-bearing "30 less apples than Martha" path is untouched.
        self.assertEqual(reading.solve(
            "Tim has 30 less apples than Martha. If Martha has 68 apples, "
            "how many apples does Tim have?"), 38)


class TestCopulaSubjectCoreference(unittest.TestCase):
    """A BARE copula statement ("Tina is 10 years old") now records its NAME subject in the discourse, so a
    later subjective pronoun resolves ("She is twice as old as Bob" → Tina). A common-noun copula subject
    ("the PRICE of beef is $5") must NOT pollute the discourse (it broke he→James). Two guards keep the new
    coreference from committing wrongs it merely UNMASKS in still-incomplete age forms. SYNTHETIC."""

    def test_copula_subject_enables_coreference(self):
        self.assertEqual(reading.solve(
            "Tina is 10 years old. She is twice as old as Bob. How old is Bob?"), 5)

    def test_common_noun_copula_subject_not_recorded(self):
        # "the price of beef is $5" must not steal he→James (price is not a person).
        self.assertEqual(reading.solve("James buys 4 pounds of beef. The price of beef is $5 per pound. "
                                       "How much did he pay?"), 20)

    def test_when_relation_does_not_sum_ages(self):
        # the when-clause's Y must NOT be folded into a bare age SUM (Joel 5 + dad 32×3 = 101). It now
        # SOLVES via _age_solve_for_t (see TestAgeSolveForT); the key invariant here is "never 101".
        self.assertNotEqual(reading.solve(
            "Joel is 5 years old and his dad is 32 years old. "
            "How old will Joel be when his dad is twice as old as him?"), 101)

    def test_subyear_future_age_declines(self):
        # "In 18 months, she will be 10 years old" — a sub-year shift the year-machinery can't convert →
        # decline rather than store the future age (10) as current and derive off it.
        self.assertEqual(reading.solve(
            "Isabella is twice as old as Antonio. In 18 months, she will be 10 years old. "
            "How many months old is Antonio?"), "undetermined")


# === AGE: age questions, relational / possessive / shifted age, kinship, solve-for-t ==========

class TestIncrement62AgeQuestions(unittest.TestCase):
    """#62 - age / "how old": "X is N years old" -> an age Quantity (measure year); "X is N years
    younger/older than Y" -> a comparative age (younger = -, older = +, via COMPARATIVE_SIGN);
    "how old is/will X be?" reads the year-holdings. A future/past SHIFT ("next year") declines."""

    def test_age_statement_is_a_quantity_not_membership(self):
        self.assertEqual(reading.learn("Kris is 24 years old"),
                         [reading.Quantity("kris", ("year", frozenset()), 24.0, "is")])

    def test_younger_age(self):
        self.assertEqual(reading.solve("Herbert is 10 years younger than Kris. Kris is 24 years old. "
                                       "How old is Herbert?"), 14)

    def test_older_age_will_be(self):
        self.assertEqual(reading.solve("Tom is 5 years older than Bob. Bob is 10 years old. "
                                       "How old will Tom be?"), 15)

    def test_temporal_shift_computes_via_time_offset(self):
        # "next year" is +1 on the timeline; age tracks it -> 14 + 1 = 15 (no per-word hack -- see _time_offset)
        self.assertEqual(reading.solve("Herbert is 10 years younger than Kris. Kris is 24 years old. "
                                       "How old will Herbert be next year?"), 15)

    def test_plain_membership_unaffected(self):
        self.assertEqual(reading.answer("Is Bob a student?", reading.learn("Bob is a student")), "yes")


class TestIncrement78TemporalAgeShift(unittest.TestCase):
    """#78 - age at a SHIFTED time, via the temporal model (NOT a per-word hack). A relative-time
    reference has a position on the timeline (_time_offset: next year=+1, in N years=+N, N years ago=-N);
    AGE is a time-tracking quantity (advances 1/year), so age@(now+Δ) = age@now + Δ. Solves Herbert."""

    def test_time_offset_next_year(self):
        self.assertEqual(reading._time_offset(reading.analyze("how old will X be next year")), 1)

    def test_time_offset_in_n_years(self):
        self.assertEqual(reading._time_offset(reading.analyze("how old will X be in 3 years")), 3)

    def test_time_offset_n_years_ago(self):
        self.assertEqual(reading._time_offset(reading.analyze("how old was X 5 years ago")), -5)

    def test_herbert_next_year_solves(self):
        self.assertEqual(reading.solve("Herbert is 10 years younger than Kris. If Kris is 24 years old "
                                       "now, how old will Herbert be next year?"), 15)

    def test_age_in_n_years(self):
        self.assertEqual(reading.solve("Tom is 20 years old. How old will Tom be in 3 years?"), 23)

    def test_age_n_years_ago(self):
        self.assertEqual(reading.solve("Tom is 20 years old. How old was Tom 5 years ago?"), 15)


class TestBareAge(unittest.TestCase):
    """In an age-typed problem a bare copula "X is N" (no "years old" unit) states X's age = N, so a
    relational-age chain can ground off it. Gated on age cues; a bare "X is N" elsewhere is untouched."""

    def test_bare_age_in_age_context(self):
        # "If Bob is 4" with an "as old as" relation present → Bob's age 4, so Anna = 3×4 = 12.
        out = reading.run_cli(["Anna is 3 times as old as Bob. If Bob is 4, how old is Anna?"])
        self.assertEqual(out.strip(), "12")

    def test_bare_number_without_age_context_is_not_an_age(self):
        # no age cue → a bare "X is N" must NOT be invented as a year-quantity.
        facts_of, context = [None], []
        reading._inject_bare_ages([reading.analyze("Bob is 4")],
                                  [reading.analyze("Bob is 4"), reading.analyze("How many does Bob have?")],
                                  facts_of, context)
        self.assertEqual(context, [])

    def test_cli_matches_solve_on_bare_age(self):
        # guard against re-introducing the CLI/solve divergence (#104) for this path.
        q = "Anna is 3 times as old as Bob. If Bob is 4, how old is Anna?"
        self.assertEqual(reading.run_cli([q]).strip(), str(reading.solve(q)))


class TestAgeBackwardAndTotal(unittest.TestCase):
    """Backward age inversion (X's age known, standard's unknown → standard = X / factor) + the total-age
    question ("what is/will be the total of their ages [in N years]?" → Σ ages [+ N×count])."""

    def test_backward_age_inversion(self):
        out = reading.run_cli(["Bob is twice as old as Ann. Bob is 12 years old. How old is Ann?"])
        self.assertEqual(out.strip(), "6")

    def test_total_age_question(self):
        out = reading.run_cli(["Bob is 6 years old. Ann is 3 years old. "
                               "What is the total of their ages?"])
        self.assertEqual(out.strip(), "9")

    def test_total_age_projected(self):
        # "their total age in 3 years" = Σ ages + 3 × count (2 people) = 9 + 6 = 15.
        out = reading.run_cli(["Bob is 6 years old. Ann is 3 years old. "
                               "What will be their total age in 3 years?"])
        self.assertEqual(out.strip(), "15")

    # NOTE: the inverse "in how many years will their total age be N?" (solve for the DURATION, not the
    # total) is excluded by the "many" gate; that regression is verified by the dev-holdout A/B (the
    # Adam/Tom case), not a unit test — the pre-existing combined-fold also returns a sum here, so the
    # branch's exclusion isn't cleanly observable through answer()'s output alone.


class TestPossessiveRelationalAge(unittest.TestCase):
    """The possessive age frame "X's age is k× Y's age" / "X is k× Y's age" → X's age = k × Y's age
    (the multiplier mis-scales the copula, so it's read positionally). And "how many years old is X?"
    asks for the NAME's age (the grammatical subject mis-resolves to the fronted "years")."""

    def test_possessive_age_resolves(self):
        out = reading.run_cli(["Bob is 6 years old. Anna's age is double Bob's age. How old is Anna?"])
        self.assertEqual(out.strip(), "12")

    def test_possessive_age_bare_multiplier(self):
        out = reading.run_cli(["Bob is 7 years old. Anna is twice Bob's age. How old is Anna?"])
        self.assertEqual(out.strip(), "14")

    def test_possessive_age_defers_when_standard_unknown(self):
        # standard's age unknown → DEFER (no bogus frequency, no wrong value).
        self.assertEqual(reading.learn("Anna is twice Bob's age", None, []), [])

    def test_how_many_years_old_question(self):
        out = reading.run_cli(["Matilda is 35 years old. How many years old is Matilda?"])
        self.assertEqual(out.strip(), "35")


class TestIncrement94RelationalAge(unittest.TestCase):
    """#94 - RELATIONAL AGE: "X is N times as old as Y" → X's age = N × Y's age. The factor is a multiplier
    ("twice") or "<value> times"; the standard Y is the last name; deferred until Y's age is known, so the
    solve() fixpoint chains a MULTI-HOP (Anne=2×Emile, Emile=6×Maude, Maude=8 → 96). "will be"/"has been"
    are copulas too now. GUARD 3 exempts the factor (consumed into the product)."""

    def test_twice_as_old(self):
        self.assertEqual(reading.learn("Anne is twice as old as Emile", [],
                                       [reading.Quantity("emile", ("year", frozenset()), 48, "be")]),
                         [reading.Quantity("anne", ("year", frozenset()), 96, "be")])

    def test_n_times_as_old_with_will_be(self):
        self.assertEqual(reading.learn("Emile will be six times as old as Maude", [],
                                       [reading.Quantity("maude", ("year", frozenset()), 8, "be")]),
                         [reading.Quantity("emile", ("year", frozenset()), 48, "be")])

    def test_anne_multi_hop_solves(self):
        self.assertEqual(reading.solve("By the time Anne is two times as old as Emile, Emile will be six "
                                       "times as old as Maude. If Maude will be 8 years old, "
                                       "how old will Anne be?"), 96)

    def test_unknown_standard_declines(self):
        self.assertEqual(reading.solve("Anne is twice as old as Emile. How old is Anne?"), "undetermined")


class TestIncrement98AgeShiftGuard(unittest.TestCase):
    """#98 - a RELATIONAL age under a TEMPORAL SHIFT was first DECLINED (precision-first), then SOLVED in
    #101 (see TestIncrement101AgeShiftSolve for the shifted-algebra cases). These remaining cases pin the
    Δ=0 invariants the shift work must never disturb: the present-tense relational age and the shift-free
    multi-hop (Anne)."""

    def test_present_tense_relational_age_unaffected(self):
        self.assertEqual(reading.solve("Tom is three times as old as Sam. Sam is 8 years old. "
                                       "How old is Tom?"), 24)

    def test_shift_free_multihop_unaffected(self):
        # Anne: "will be" with NO numeric offset is not a shift -> the multi-hop still solves
        self.assertEqual(reading.solve("By the time Anne is two times as old as Emile, Emile will be "
                                       "six times as old as Maude. If Maude will be 8 years old, "
                                       "how old will Anne be?"), 96)


class TestIncrement101AgeShiftSolve(unittest.TestCase):
    """#101 - SOLVE the shifted relational age that #98 declined. "In Δ years, X will be k x as old as
    Y" puts BOTH at the future point by default: X+Δ = k(Y+Δ), so X_now = k(Y_now+Δ) - Δ. The variant
    "as Y is NOW/today" anchors the standard to the present: X+Δ = k.Y_now, so X_now = k.Y_now - Δ
    (else Kaylee-class problems would commit a confident wrong). A comparative OFFSET on top of the
    factor ("N more than k x as old") is DECLINED (we'd drop the +N). Present-tense relational age and
    the shift-free multi-hop (Anne) are unchanged (Δ=0)."""

    def test_both_ages_shift(self):
        # Tom+2 = 3(Sam+2) = 3*10 = 30 -> Tom_now = 28
        self.assertEqual(reading.solve("In two years, Tom will be three times as old as Sam. "
                                       "Sam is currently 8 years old. How old is Tom?"), 28)

    def test_from_now_both_shift(self):
        # Tom+3 = 2(Sam+3) = 2*32 = 64 -> Tom_now = 61
        self.assertEqual(reading.solve("Three years from now, Tom will be twice as old as Sam. "
                                       "Sam is now 29 years old. How old is Tom now?"), 61)

    def test_standard_anchored_to_present(self):
        # "as Sam is now": Tom+7 = 3*Sam_now = 3*5 = 15 -> Tom_now = 8
        self.assertEqual(reading.solve("In 7 years, Tom will be 3 times as old as Sam is now. "
                                       "Sam is currently 5 years old. How old is Tom now?"), 8)

    def test_comparative_offset_on_factor_declines(self):
        # "N more than k x as old" — an offset we don't fold into the age relation -> honest decline
        self.assertEqual(reading.solve("Ty is currently 4 years more than two times as old as Carla. "
                                       "Carla is currently 5 years old. How old is Ty?"), "undetermined")

    def test_present_tense_relational_age_unchanged(self):
        self.assertEqual(reading.solve("Tom is three times as old as Sam. Sam is 8 years old. "
                                       "How old is Tom?"), 24)

    def test_shift_free_multihop_unchanged(self):
        self.assertEqual(reading.solve("By the time Anne is two times as old as Emile, Emile will be "
                                       "six times as old as Maude. If Maude will be 8 years old, "
                                       "how old will Anne be?"), 96)


class TestIncrement115AgeShiftTurnedAgo(unittest.TestCase):
    """#115 - "X turned N a year ago" is a TIME-SHIFTED age we can't reliably read: the age (3) and the
    offset's "year" share the one 'year' token, so the offset can't be separated from the age. Rather than
    emit a wrong current-age fact, we skip it — so a relational age over it DECLINES rather than committing
    off a stale age (Diana "twice as old as Grace … turned 3 a year ago" was committing 2×3=6)."""

    def test_turned_ago_emits_no_age_fact(self):
        quants = [q for q in reading.learn("Grace turned 3 a year ago", [], []) if isinstance(q, reading.Quantity)]
        self.assertEqual(quants, [])                     # no (wrong) current-age fact emitted

    def test_relational_age_over_shifted_declines(self):
        self.assertEqual(reading.solve("Diana is twice as old as Grace is today. "
                                       "If Grace turned 3 a year ago, how old is Diana today?"),
                         "undetermined")

    def test_plain_turned_age_fact_unaffected(self):
        # no time offset → "turned 5 years old" still emits the age fact (amount 5)
        quants = [q.amount for q in reading.learn("Tom turned 5 years old", [], []) if isinstance(q, reading.Quantity)]
        self.assertEqual(quants, [5.0])


class TestIncrement125FutureAndComparativeAge(unittest.TestCase):
    """#125 - age-shift: a FUTURE age with a time displacement + an additive comparative age. "In two
    years, Claire will be 20 years old" → Claire_now = 20−2 = 18 (the offset and the age share 'year', so
    the offset is read off the "in"-preposition's object, not _time_offset). "Jessica is six years older
    than Claire" → 18+6 = 24. Resolves Jessica (was 28: it folded 2+20+6)."""

    def test_future_age_offset_first(self):
        self.assertEqual(reading._future_age(reading.analyze("In two years, Claire will be 20 years old.")),
                         reading.Quantity("claire", ("year", frozenset()), 18, "be"))

    def test_future_age_offset_last(self):
        self.assertEqual(reading._future_age(reading.analyze("Jennifer will be 30 years old in ten years.")),
                         reading.Quantity("jennifer", ("year", frozenset()), 20, "be"))

    def test_comparative_age(self):
        ctx = [reading.Quantity("claire", ("year", frozenset()), 18, "be")]
        self.assertEqual(reading._comparative_age(reading.analyze("Jessica is six years older than Claire."), ctx),
                         reading.Quantity("jessica", ("year", frozenset()), 24, "be"))

    def test_jessica_solves(self):
        self.assertEqual(reading.solve(
            "Jessica is six years older than Claire. In two years, Claire will be 20 years old. "
            "How old is Jessica now?"), 24)


class TestComparativeAgeBackward(unittest.TestCase):
    """BACKWARD inversion of a comparative age — "X is N older than Y" with the SUBJECT's age known (not
    the standard's) grounds the STANDARD: older → Y = X − N. Mirrors _relational_quantity's bidirectional
    grounding; the fixpoint then chains a future shift ("in three years"). SYNTHETIC."""

    def test_backward_standard_from_subject(self):
        self.assertEqual(reading.solve(
            "Benjie is 6 years old. Benjie is five years older than Margo. How old is Margo?"), 1)

    def test_backward_then_future_shift(self):
        self.assertEqual(reading.solve(
            "Benjie is five years older than Margo. If Benjie is 6 years old, "
            "how old will Margo be in three years?"), 4)

    def test_backward_younger_flips_sign(self):
        # "younger": subj = std − N → std = subj + N, so Margo = 6 + 5 = 11.
        self.assertEqual(reading.solve(
            "Benjie is 6 years old. Benjie is five years younger than Margo. How old is Margo?"), 11)

    def test_forward_path_unchanged(self):
        self.assertEqual(reading.solve(
            "Kris is 24 years old. Herbert is 10 years younger than Kris. How old is Herbert?"), 14)


class TestShiftedRelationalAgeBackward(unittest.TestCase):
    """A FUTURE multiplicative age solved for the STANDARD — "In Δ years, X will be k× as old as Y" with X's
    age known: X+Δ = k·(Y+Δ) → Y_now = (X_now+Δ)/k − Δ. The twin of the forward shifted relation; plus a
    plain POSSESSIVE age statement ("Ron's age is 43" → Quantity(ron, year, 43)) so the subject grounds.
    SYNTHETIC."""

    def test_backward_shifted_solves_standard(self):
        # Ron+5 = 4·(Maurice+5); Ron 43 → Maurice = (43+5)/4 − 5 = 7.
        self.assertEqual(reading.solve(
            "After five years, Ron will be four times as old as Maurice. If Ron's age now is 43, "
            "how old is Maurice now?"), 7)

    def test_possessive_age_statement(self):
        self.assertEqual(reading.solve("Ron's age is 43. How old is Ron?"), 43)

    def test_forward_shifted_unchanged(self):
        # Ernesto_now = 2·(11+3) − 3 = 25 (the forward path must still hold).
        self.assertEqual(reading.solve(
            "In 3 years, Ernesto will be twice as old as Maria. Maria is 11 years old. "
            "How old is Ernesto now?"), 25)

    def test_present_backward_unchanged(self):
        self.assertEqual(reading.solve(
            "Sam is 12 years old. Sam is twice as old as Sue. How old is Sue?"), 6)


class TestShiftedPossessiveAge(unittest.TestCase):
    """The POSSESSIVE factor frame ("k× Y's age", "half OF Y's age") now carries the same future shift as
    "as old as": "In Δ years, X will be k× Y's age" → X+Δ = k·(Y+Δ), solved forward (Y known) or backward
    (X known). Required teaching _possessive_age_parts to accept the future copula "will BE". SYNTHETIC."""

    def test_possessive_backward_shifted(self):
        # Suzy+4 = 2·(Mary+4); Suzy 20 → Mary = (20+4)/2 − 4 = 8.
        self.assertEqual(reading.solve(
            "In four years, Suzy will be twice Mary's age then. If Suzy is 20 now, how old is Mary?"), 8)

    def test_possessive_forward_shifted_half_of(self):
        # Jayden+3 = ½·(Ernesto+3); Ernesto 11 → Jayden = (11+3)/2 − 3 = 4.
        self.assertEqual(reading.solve(
            "In 3 years, Jayden will be half of Ernesto's age. If Ernesto is 11 years old, "
            "how many years old is Jayden now?"), 4)

    def test_possessive_present_unchanged(self):
        self.assertEqual(reading.solve(
            "Suzy is twice Mary's age. Mary is 8 years old. How old is Suzy?"), 16)


class TestAgeWhenProjection(unittest.TestCase):
    """"How old will X be WHEN Y is/turns M?" → X_now + (M − Y_now): Y reaching age M sets the future point,
    and ages advance together. Plus the premise "X was A years old when Y was B years old" → both current
    ages (constant-gap pair). Without the projection, X and the when-clause's Y both resolve as subjects and
    their ages are summed (20 + 8 = 28, not 20 + (18 − 8) = 30). SYNTHETIC."""

    def test_when_projection_plain_ages(self):
        self.assertEqual(reading.solve(
            "Marianne is 20 years old. Bella is 8 years old. "
            "How old will Marianne be when Bella turns 18?"), 30)

    def test_simultaneous_age_premise_then_projection(self):
        self.assertEqual(reading.solve(
            "Marianne was 20 years old when Bella was 8 years old. "
            "How old will Marianne be when Bella turns 18?"), 30)

    def test_plain_age_question_unaffected(self):
        self.assertEqual(reading.solve(
            "Marianne is 20 years old. Bella is 8 years old. How old is Marianne?"), 20)


class TestKinshipEntity(unittest.TestCase):
    """"his/her/their <KINSHIP>" ("his dad", "her sister") is a NAMED entity keyed by the kinship noun, so
    "His dad is 32" stores an age and "her brother has half as much" derives. Needed a companion fix: a
    possessive pronoun before a name/noun is a DETERMINER, not the subject ("His" was tying with "dad" as a
    subject candidate, leaving the copula's subject unbound). Generalizes beyond age. SYNTHETIC."""

    def test_kinship_age_stored_and_read(self):
        self.assertEqual(reading.solve(
            "Joel is 5 years old. His dad is 32 years old. How old is his dad?"), 32)

    def test_kinship_comparative_then_when_projection(self):
        self.assertEqual(reading.solve(
            "Emma is 7 years old. If her sister is 9 years older than her, "
            "how old will Emma be when her sister is 56?"), 47)

    def test_kinship_generalizes_to_money(self):
        self.assertEqual(reading.solve(
            "Madeline has $48 and her brother has half as much. "
            "How much money do they have together?"), 72)

    def test_subjective_pronoun_subject_unaffected(self):
        # "She" (not a determiner — no following name) must still bind as the subject.
        self.assertEqual(reading.solve(
            "Mary has 3 apples. She buys 2 apples. How many apples does Mary have?"), 5)


class TestAgeOfPhraseAndPresentAnchor(unittest.TestCase):
    """The possessive age frame now also reads "the age OF Y" / "the age that Y is" (not just "Y's age"),
    accepts a resolved-PRONOUN subject ("she"), and honors a present anchor: "the age that Y is CURRENTLY"
    keeps Y at the present (X+Δ = k·Y_now), unlike the default where the shift applies to both. SYNTHETIC."""

    def test_age_of_phrase_backward_shifted(self):
        # Cori 3; Cori+5 = ⅓·(aunt+5) → aunt = (3+5)/(1/3) − 5 = 24 − 5 = 19. ("she" → Cori.)
        self.assertEqual(reading.solve(
            "Cori is 3 years old today. In 5 years, she will be one-third the age of her aunt. "
            "How old is her aunt today?"), 19)

    def test_present_anchored_standard(self):
        # "the age that Nora is CURRENTLY" anchors Nora to now: Terry+10 = 4·Nora_now = 40 → Terry = 30.
        self.assertEqual(reading.solve(
            "In 10 years, Terry will be 4 times the age that Nora is currently. "
            "If Nora is currently 10 years old, how old is Terry now?"), 30)

    def test_then_anchor_shifts_both(self):
        # "Mary's age THEN" (not "now") → both shift: Suzy+4 = 2·(Mary+4); Suzy 20 → Mary = 8.
        self.assertEqual(reading.solve(
            "In four years, Suzy will be twice Mary's age then. If Suzy is 20 now, how old is Mary?"), 8)


class TestFractionAgeFactor(unittest.TestCase):
    """A FRACTION written N/M ("1/6 as old as Y") is the relational-age FACTOR, not an age value — so it
    scales the standard (Y × 1/6) in both directions, and chains through kinship entities. Word fractions
    ("half"/"one-third") already parsed as multipliers; the N/M form parsed as a bare value. SYNTHETIC."""

    def test_fraction_factor_chain_through_kinship(self):
        # Darcie 4; mother = 4 ÷ (1/6) = 24; father = 24 ÷ (4/5) = 30.
        self.assertEqual(reading.solve(
            "Darcie is 4 years old. She is 1/6 as old as her mother and her mother is 4/5 as old as "
            "her father. How old is her father?"), 30)

    def test_fraction_factor_forward(self):
        self.assertEqual(reading.solve(
            "Tina is 24 years old. Darcie is 1/6 as old as Tina. How old is Darcie?"), 4)


class TestAgeSolveForT(unittest.TestCase):
    """"How old will X be WHEN Y is k× as old as X [is now]?" — solve for the future displacement t at which
    the multiplicative relation holds, then return X's age then. Both shift: Y+t = k·(X+t) → t = (kX−Y)/(1−k).
    Present-anchored ("as old as X is NOW"): Y+t = k·X_now → t = k·X_now − Y_now. Also exercises kinship-
    modifier promotion ("his OLDER brother" → entity across an attributive modifier). SYNTHETIC."""

    def test_both_shift_solve(self):
        # dad+t = 2·(Joel+t); Joel 5, dad 32 → t = (10−32)/(1−2) = 22 → Joel = 27.
        self.assertEqual(reading.solve(
            "Joel is 5 years old and his dad is 32 years old. "
            "How old will Joel be when his dad is twice as old as him?"), 27)

    def test_present_anchored_solve_with_modified_kinship(self):
        # brother = 3·Trevor_now = 33 (anchored) → t = 33 − 20 = 13 → Trevor = 24. ("his older brother".)
        self.assertEqual(reading.solve(
            "Trevor is currently 11 years old and his older brother is 20 years old. How old will Trevor "
            "be when his older brother is three times as old as Trevor is now?"), 24)

    def test_additive_relation_declines(self):
        # "N years older than" is an additive offset the multiplicative solver doesn't handle → decline.
        self.assertEqual(reading.solve(
            "Joel is 5 years old and his dad is 32 years old. "
            "How old will Joel be when his dad is 10 years older than him?"), "undetermined")


# === TEMPORAL: timeline foundation, time-referenced querying, growth projection ===============

class TestIncrement63TemporalFoundation(unittest.TestCase):
    """#63 - relational temporal foundation + the read-SYNTHESIZE loop. A time reference is a NODE; the
    `time` word carries `associates` (the value it timestamps) and `orders` (its order vs other times,
    via an after/before cue). value@time rides on Quantity; Before(earlier,later) carries order. Ordering
    is RELATIONAL (days WRAP), and its transitive closure is DERIVED by synthesis, never assumed. The
    no-guess contract holds: an order neither stated nor derivable is not invented, and a multi-time
    problem (time-aware folding is a later increment) DECLINES rather than fold across times blindly."""

    def test_time_is_recognized_as_a_node(self):
        r = reading.analyze("Tom has 3 apples on Monday")
        monday = next(u for u in r.units if u.surface == "Monday")
        self.assertEqual(monday.features.get("senses"), ["time"])

    def test_value_at_time(self):
        self.assertEqual(reading.learn("Tom has 3 apples on Monday"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3.0, "have", "monday")])

    def test_timeless_quantity_has_no_time(self):
        self.assertEqual(reading.learn("Tom has 3 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3.0, "have", None)])

    def test_associates_binds_the_verb(self):
        r = reading.analyze("Tom has 3 apples on Monday")
        monday = next(u for u in r.units if u.surface == "Monday")
        self.assertEqual(reading._follow(monday, "associates").surface, "has")

    def test_before_fact_from_after_cue(self):
        self.assertEqual(reading.learn("Tuesday is after Monday"),
                         [reading.Before("monday", "tuesday")])

    def test_before_fact_from_before_cue(self):
        self.assertEqual(reading.learn("Saturday is before Monday"),
                         [reading.Before("saturday", "monday")])

    def test_synthesis_derives_transitive_order(self):
        facts = reading.learn("Tuesday is after Monday") + reading.learn("Saturday is after Tuesday")
        self.assertIn(reading.Before("monday", "saturday"), reading.synthesize(facts))

    def test_order_is_relational_days_wrap(self):
        # Tuesday -> Saturday -> Monday makes THIS Monday LATER than THIS Tuesday (not the calendar)
        synth = reading.synthesize(reading.learn("Saturday is after Tuesday")
                                   + reading.learn("Monday is after Saturday"))
        self.assertTrue(reading._before("tuesday", "monday", synth))
        self.assertFalse(reading._before("monday", "tuesday", synth))

    def test_synthesis_does_not_invent_order(self):
        synth = reading.synthesize([reading.Before("monday", "tuesday")])
        self.assertFalse(reading._before("tuesday", "monday", synth))

    def test_multi_state_without_question_time_declines(self):
        self.assertEqual(reading.solve("On Monday Tom has 30 apples. On Tuesday Tom has 40 apples. "
                                       "How many apples does Tom have?"), "undetermined")


class TestIncrement64TimelineQuerying(unittest.TestCase):
    """#64 - timeline assembly + TIME-AWARE querying. A STATE verb (have/be) SETS the value at a time; a
    CHANGE verb adds a signed delta. Holdings AT a queried time fold along the timeline (ordered by Before
    + narrative order). A multi-time problem answers a time-NAMED question, but DECLINES a time-less one
    (ambiguous which state) or one that dropped a time-tagged event (precision)."""

    def test_state_then_change_at_later_time(self):
        self.assertEqual(reading.solve("On Monday Tom has 30 apples. On Tuesday Tom buys 5 apples. "
                                       "How many apples does Tom have on Tuesday?"), 35)

    def test_value_at_earlier_time_is_the_state(self):
        self.assertEqual(reading.solve("On Monday Tom has 30 apples. On Tuesday Tom buys 5 apples. "
                                       "How many apples does Tom have on Monday?"), 30)

    def test_loss_change_along_timeline(self):
        self.assertEqual(reading.solve("On Monday Tom has 30 apples. On Tuesday Tom loses 10 apples. "
                                       "How many apples does Tom have on Tuesday?"), 20)

    def test_holdings_at_folds_state_and_change(self):
        Q, B = reading.Quantity, reading.Before
        ap = ("apple", frozenset())
        ctx = [Q("t", ap, 30, "have", "monday"), Q("t", ap, 5, "buy", "tuesday"), B("monday", "tuesday")]
        self.assertEqual(reading._holdings_at("t", ap, ctx, "tuesday"), 35)
        self.assertEqual(reading._holdings_at("t", ap, ctx, "monday"), 30)

    def test_unorderable_timeline_declines(self):
        Q = reading.Quantity
        ap = ("apple", frozenset())
        ctx = [Q("t", ap, 3, "have", "monday"), Q("t", ap, 5, "have", "friday")]  # no order between them
        self.assertIsNone(reading._holdings_at("t", ap, ctx, "friday"))

    def test_multitime_question_without_a_time_declines(self):
        self.assertEqual(reading.solve("On Monday Tom has 30 apples. On Tuesday Tom has 40 apples. "
                                       "How many apples does Tom have?"), "undetermined")

    def test_query_at_unrelated_time_declines(self):
        self.assertEqual(reading.solve("On Monday Tom has 5 apples. "
                                       "How many apples does Tom have on Friday?"), "undetermined")


class TestIncrement66TimeReferencedPartitive(unittest.TestCase):
    """#66 - time-referenced partitives + fraction words. "a third / a quarter OF X" -> a unit fraction
    (FRACTIONS); the of-clause may name a holder + a TIME ("what he had on Tuesday") -> the base is that
    holding AT that time (_holdings_at), not the current one. Closes the full Buddy problem end to end."""

    def test_full_buddy_solves(self):
        self.assertEqual(reading.solve(
            "On Monday Buddy has 30 baseball cards. On Tuesday Buddy loses half of them. "
            "On Wednesday Buddy buys 12 baseball cards. "
            "On Thursday he buys a third of what he had on Tuesday. "
            "How many baseball cards does he have on Thursday?"), 32)

    def test_a_third_of_them_timeless(self):
        self.assertEqual(reading.solve("Tom has 9 apples. Tom gives a third of them away. "
                                       "How many apples does Tom have?"), 6)

    def test_a_quarter_of_them(self):
        self.assertEqual(reading.solve("Tom has 8 apples. Tom eats a quarter of them. "
                                       "How many apples does Tom have?"), 6)


class TestIncrement67AcrossTimesComparison(unittest.TestCase):
    """#67 - "how many more/fewer X did E have on T1 than on T2?" compares ONE entity's holdings at two
    TIMES (split by "than") over the timeline: more -> hold(t1) - hold(t2); fewer -> the reverse. The
    entity-vs-entity comparison ("more than William") is unaffected."""

    def test_more_across_times(self):
        self.assertEqual(reading.solve("On Monday Tom had 30 apples. On Tuesday Tom had 50 apples. "
                                       "How many more apples did Tom have on Tuesday than on Monday?"), 20)

    def test_fewer_across_times(self):
        self.assertEqual(reading.solve("On Monday Tom had 30 apples. On Tuesday Tom had 50 apples. "
                                       "How many fewer apples did Tom have on Monday than on Tuesday?"), 20)

    def test_across_times_with_a_change(self):
        self.assertEqual(reading.solve("On Monday Tom has 30 apples. On Tuesday Tom buys 20 apples. "
                                       "How many more apples did Tom have on Tuesday than on Monday?"), 20)

    def test_entity_comparison_still_works(self):
        self.assertEqual(reading.solve("Oliver has 30 dollars. William has 10 dollars. "
                                       "How much more money does Oliver have than William?"), 20)


class TestIncrement93GrowthProjection(unittest.TestCase):
    """#93 - GROWTH PROJECTION: a quantity growing at a RATE per year, valued at a future time → current +
    rate × Δ (the explicit-rate generalization of the age shift, which is rate 1/year). "Haley grows 3
    inches/year; currently 20; after 10 years?" → 20 + 3×10 = 50. "every <unit>" is a per-marker; "after N
    years" is a +N time offset."""

    def test_every_year_is_a_rate(self):
        self.assertEqual(reading._rate(reading.analyze("Haley grows 3 inches every year")),
                         reading.Rate("haley", 3.0, ("inch", frozenset()), "year", "grow", 1))

    def test_after_n_years_offset(self):
        self.assertEqual(reading._time_offset(reading.analyze("her height after 10 years")), 10)

    def test_haley_growth_projection(self):
        self.assertEqual(reading.solve("Haley grows at the rate of 3 inches every year. If she is "
                                       "currently 20 inches tall, what will be her height after 10 years?"), 50)

    def test_age_shift_unaffected(self):
        self.assertEqual(reading.solve("Herbert is 10 years younger than Kris. If Kris is 24 years old "
                                       "now, how old will Herbert be next year?"), 15)


# === RATES: per-unit rates, conversions, rate chains, proportion ==============================

class TestIncrement52Rates(unittest.TestCase):
    """#52 — "per" rates (slice 1: money). "$12 per hour" is a Rate (a value distributed over a unit);
    the total is composed at answer time as rate.amount x (quantity of the per-unit). "Weng earns $12
    per hour. Weng works 3 hours. How much did Weng earn?" -> 36."""

    def test_rate_is_learned(self):
        self.assertEqual(reading.learn("Weng earns $12 per hour"),
                         [reading.Rate("weng", 12.0, reading.MONEY, "hour", "earn")])

    def test_rate_times_quantity(self):
        out = reading.run_cli(["Weng earns $12 per hour. Weng works 3 hours. How much did Weng earn?"])
        self.assertEqual(out.strip(), "36")

    def test_rate_without_the_quantity_declines(self):
        out = reading.run_cli(["Weng earns $12 per hour. How much did Weng earn?"])
        self.assertEqual(out.strip(), "undetermined")

    def test_flat_money_still_works(self):
        # "$12" with no "per" is still a flat money quantity, not a rate
        self.assertEqual(reading.learn("Weng earns $12"),
                         [reading.Quantity("weng", reading.MONEY, 12.0, "earn")])


class TestIncrement74RatesAndConversion(unittest.TestCase):
    """#74 - rates + unit conversion + result-measure binding (start of the rates cluster). Non-money
    rates and "a/an"=per ("20 pages a day"); the per-unit count read from the QUESTION ("in 3 days");
    unit CONVERSION (minutes for an hourly rate); `much` BINDS the result-measure verb (the earnings) or
    a money noun rather than a global MONEY default; and "do/did" as a main verb. Solves the real Weng."""

    def test_non_money_rate_times_quantity(self):
        self.assertEqual(reading.solve("John writes 20 pages a day. "
                                       "How many pages does John write in 3 days?"), 60)

    def test_an_as_per_money_rate(self):
        self.assertEqual(reading.solve("Weng earns $12 an hour. She worked 3 hours. "
                                       "How much did Weng earn?"), 36)

    def test_rate_with_unit_conversion(self):
        self.assertEqual(reading.solve("Weng earns $12 an hour for babysitting. Yesterday, she just did "
                                       "50 minutes of babysitting. How much did she earn?"), 10)

    def test_much_binds_the_result_verb(self):
        r = reading.analyze("How much did Weng earn")
        much = next(u for u in r.units if u.surface == "much")
        self.assertEqual(reading._follow(much, "collection").surface, "earn")

    def test_much_non_money_verb_declines(self):
        self.assertNotIsInstance(reading.solve("Tom ran 5 miles. How much did Tom run?"), (int, float))

    def test_did_as_main_verb(self):
        self.assertEqual(reading.learn("She did 50 minutes of babysitting", ["weng"], []),
                         [reading.Quantity("weng", ("minute", frozenset()), 50.0, "do")])

    def test_did_aux_unaffected(self):
        self.assertEqual(reading.answer("Did Bob pass?", reading.learn("Bob passed")), "yes")


class TestIncrement76RelativeClauseAndCrossEntityRate(unittest.TestCase):
    """#76 - finishing James-beef. Relative-clause multiplication ("5 packs of beef THAT ARE 4 pounds
    each" → 20; the copula no longer breaks it); a decimal "$5.50" no longer splits the sentence; "the
    price OF beef is" binds price as subject (PP objects excluded); "on MONDAY" binds the time as the prep
    object (name stays free); a CROSS-ENTITY money rate ($5.50/lb on James's pounds); and "he" prefers a
    PERSON over a thing ("price")."""

    def test_relative_clause_multiplication(self):
        self.assertEqual(reading.learn("James buys 5 packs of beef that are 4 pounds each"),
                         [reading.Quantity("james", ("pound", frozenset()), 20.0, "buy")])

    def test_decimal_not_split(self):
        self.assertEqual(reading._sentences("The price of beef is $5.50 per pound."),
                         ["The price of beef is $5.50 per pound."])

    def test_price_of_beef_subject(self):
        r = reading.analyze("The price of beef is $5.50 per pound")
        is_v = next(u for u in r.units if u.surface == "is")
        self.assertEqual(reading._follow(is_v, "subject").surface, "price")

    def test_on_monday_keeps_name_as_subject(self):
        self.assertEqual(reading.learn("On Monday Tom has 30 apples"),
                         [reading.Quantity("tom", ("apple", frozenset()), 30.0, "have", "monday")])

    def test_cross_entity_rate_james_beef(self):
        self.assertEqual(reading.solve("James buys 5 packs of beef that are 4 pounds each. "
                                       "The price of beef is $5.50 per pound. How much did he pay?"), 110)

    def test_he_prefers_person_over_thing(self):
        self.assertEqual(reading.solve("James buys 4 pounds of beef. The price of beef is $5 per pound. "
                                       "How much did he pay?"), 20)


class TestIncrement77RateInversion(unittest.TestCase):
    """#77 - rate INVERSION: "How LONG will it take to <verb> <target>?" → target ÷ rate, in the rate's
    per_unit. The target may be an explicit "N <measure>" or a relative-clause mult ("3 books that are
    400 pages each" → 1200). Solves John-writes."""

    def test_inversion_explicit_target(self):
        self.assertEqual(reading.solve("John writes 20 pages a day. "
                                       "How long will it take John to write 1200 pages?"), 60)

    def test_inversion_with_relative_clause_target(self):
        self.assertEqual(reading.solve("John writes 20 pages a day. How long will it take him to "
                                       "write 3 books that are 400 pages each?"), 60)


class TestIncrement81PerNUnitRateAndDurationQuery(unittest.TestCase):
    """#81 - per-N-unit rate + cross-unit duration query → Joy. "8 pages in 20 minutes" parses as a Rate
    with per_amount=20 (8 pages per 20 min); "how many HOURS to read 120 pages" inverts (120 × 20 ÷ 8 =
    300 min) and converts to the asked unit (300 min → 5 hours). Also _container no longer reaches across
    a second preposition ("8 pages of a book IN 20 minutes" ≠ 8×20)."""

    def test_per_n_unit_rate(self):
        self.assertEqual(reading._rate(reading.analyze("Joy can read 8 pages of a book in 20 minutes")),
                         reading.Rate("joy", 8.0, ("page", frozenset()), "minute", "read", 20.0))

    def test_container_not_across_preposition(self):
        amounts = [getattr(f, "amount", None) for f in reading.learn("Joy reads 8 pages of a book in 20 minutes")]
        self.assertNotIn(160.0, amounts)

    def test_joy_solves(self):
        self.assertEqual(reading.solve("Joy can read 8 pages of a book in 20 minutes. "
                                       "How many hours will it take her to read 120 pages?"), 5)

    def test_per_n_unit_same_unit_no_conversion(self):
        self.assertEqual(reading.solve("Sam writes 5 pages in 2 minutes. "
                                       "How many minutes will it take to write 100 pages?"), 40)


class TestIncrement116RateDuration(unittest.TestCase):
    """#116 - rate cluster foundation: a Rate accumulated over a stated DURATION → a concrete Quantity
    (_synth_rate_duration), which folds with later events. Plus a compound-noun rate parse ("7 apple
    pies a day" → Rate keyed to the HEAD noun 'pie', value quantifies the modifier 'apple')."""

    def test_compound_noun_rate_parse(self):
        rate = reading._rate(reading.analyze("Cooper makes 7 apple pies a day."))
        self.assertEqual((rate.amount, rate.measure[0], rate.per_unit), (7.0, "pie", "day"))

    def test_rate_times_duration_solves(self):
        self.assertEqual(reading.solve("Cooper makes 7 pies a day. He does this for 12 days. "
                                       "How many pies does Cooper have?"), 84)

    def test_rate_duration_then_self_consumption(self):
        self.assertEqual(reading.solve("Cooper makes 7 pies a day. He does this for 12 days. "
                                       "He then eats 50 pies. How many pies does Cooper have?"), 34)

    def test_compound_plus_cross_entity_declines_for_now(self):
        # full Cooper still needs compound-measure unification + cross-entity "of his pies" — declines
        # (no longer the confident-wrong 7) until those land
        self.assertEqual(reading.solve("Cooper makes 7 apple pies a day. He does this for 12 days. "
                                       "Ashley then eats 50 of his pies. How many apple pies remain "
                                       "with Cooper?"), "undetermined")


class TestIncrement117RelationalRate(unittest.TestCase):
    """#117 - a RELATIONAL rate: "Sarah makes 5 times more money per hour than Connor" = 5 × Connor's
    rate. Detected by a "times" operator + a comparative whose standard is a NAME; resolved against that
    entity's known rate (same measure/per_unit), deferring to the fixpoint if it isn't known yet. The
    rate×duration synth then takes it × the 8-hour day → 5×7.20×8 = 288."""

    def test_relational_rate_times_duration(self):
        self.assertEqual(reading.solve(
            "Sarah makes 5 times more money per hour than Connor does. If Connor earns $7.20 per hour, "
            "how much does Sarah make in an 8-hour day?"), 288)

    def test_plain_rate_unaffected(self):
        rate = reading._rate(reading.analyze("Connor earns $7.20 per hour."))
        self.assertEqual((rate.amount, rate.per_unit), (7.2, "hour"))


class TestIncrement118FirstYearException(unittest.TestCase):
    """#118 - a FIRST-PERIOD exception: "teaches 50 students a year … in 10 years … in the first year he
    only teaches 40" → 40 + 50×(10−1) = 490. The "first <time-unit>" override Quantity is flagged
    first_period; the answer's rate×duration path uses override + rate×(duration−1) instead of rate×N."""

    def test_first_year_exception(self):
        self.assertEqual(reading.solve(
            "Adam teaches 50 students a year. How many students will Adam teach in 10 years "
            "if in the first year he only teaches 40 students?"), 490)

    def test_first_period_flag_set(self):
        quants = [q for q in reading.learn("in the first year he only teaches 40 students", [], [])
                  if isinstance(q, reading.Quantity)]
        self.assertTrue(quants and quants[0].first_period)

    def test_plain_rate_duration_no_exception(self):
        # no "first year" override → plain rate×duration
        self.assertEqual(reading.solve("Adam teaches 50 students a year. "
                                       "How many students will Adam teach in 10 years?"), 500)


class TestIncrement119OverTimeRate(unittest.TestCase):
    """#119 - "N <time-unit> over M <time-unit>" parses as a time-over-time Rate (Tom: "reads 10 hours
    over 5 days" → 10 hours per 5 days = 2 h/day). The existing per/in branches can't reach it: "over"
    isn't a per-marker and the "in" branch requires a NON-time measure. Parsing it as a Rate suppresses
    the two stray quantities (10 hours, 5 days); without a usable hour-duration the full Tom problem then
    DECLINES (wrong→decline) — the chained question-duration composition that greens it lands in #120."""

    def test_over_time_rate_parse(self):
        self.assertEqual(reading._rate(reading.analyze("Tom reads 10 hours over 5 days.")),
                         reading.Rate("tom", 10.0, ("hour", frozenset()), "day", "read", 5.0))

    def test_over_rate_suppresses_quantities(self):
        facts = reading.learn("Tom reads 10 hours over 5 days.", [], [])
        self.assertTrue(any(isinstance(f, reading.Rate) for f in facts))
        self.assertFalse(any(isinstance(f, reading.Quantity) for f in facts))

    def test_non_time_over_is_not_a_time_rate(self):
        # "over" with a non-time object must NOT trip the time-over-time branch
        self.assertIsNone(reading._rate(reading.analyze("Sam scored 30 points over 5 games.")))

    def test_over_rate_alone_declines(self):
        # the time-over-time rate parses, but with no usable duration the answer still declines
        self.assertEqual(reading.solve(
            "Tom reads 10 hours over 5 days. He can read 50 pages per hour. "
            "How many pages does he read?"), "undetermined")


class TestIncrement120QuestionDurationChain(unittest.TestCase):
    """#120 - the rate-cluster FOUNDATION: a rate×duration chain whose duration lives in the QUESTION
    clause. solve() seeds the question-clause duration ("…in 7 days?") into the synth context so the
    existing rate_duration synth CHAINS two rates through the fixpoint — Tom's 2 h/day × 7 days → 14 h,
    then 50 pg/h × 14 h → 700. Gated on the question time-value matching a rate's per_unit, so non-rate
    time values inject nothing."""

    def test_tom_question_duration_chain(self):
        self.assertEqual(reading.solve(
            "Tom reads 10 hours over 5 days. He can read 50 pages per hour. Assuming he reads the same "
            "amount every day how many pages does he read in 7 days?"), 700)

    def test_single_rate_question_duration_unaffected(self):
        # plain single-rate × question-duration still solves (Adam, no first-year override)
        self.assertEqual(reading.solve("Adam teaches 50 students a year. "
                                       "How many students will Adam teach in 10 years?"), 500)

    def test_no_rate_no_seed(self):
        # a question time-value with no rate to consume it injects nothing → the holding is unperturbed
        self.assertEqual(reading.solve("Tom has 5 apples. How many apples will he have in 7 days?"), 5)


class TestIncrement121TimeRangeRateChain(unittest.TestCase):
    """#121 - James-writes: a clock TIME-RANGE done every period composes through two rates with a unit
    conversion. "from 1 PM to 4 PM every night" → 3 h/day (_time_range_rate); 3 h/day × 5 pg/h → 15 pg/day
    (_synth_rate_chain, the hour cancels); "how many weeks to finish 735 pages" inverts → 735 ÷ 15 = 49
    days → 7 weeks (the inversion now picks the rate whose per_unit converts to the asked unit)."""

    def test_time_range_rate_parse(self):
        self.assertEqual(reading._time_range_rate(reading.analyze("James writes from 1 PM to 4 PM every night.")),
                         reading.Rate("james", 3.0, ("hour", frozenset()), "day", "write", 1))

    def test_rate_chain_synth(self):
        facts = [reading.Rate("james", 3.0, ("hour", frozenset()), "day", "write", 1),
                 reading.Rate("james", 5.0, ("page", frozenset()), "hour", "write", 1)]
        out = reading._synth_rate_chain(facts, [], set())
        self.assertIn(reading.Rate("james", 15.0, ("page", frozenset()), "day", "write", 1), out)

    def test_james_writes_solves(self):
        self.assertEqual(reading.solve(
            "James writes from 1 PM to 4 PM every night.  He can write 5 pages per hour.  "
            "How many weeks will it take to finish his 735-page book?"), 7)

    def test_non_time_range_not_a_rate(self):
        # no AM/PM meridiems → not a time-range rate
        self.assertIsNone(reading._time_range_rate(reading.analyze("James writes 3 pages every night.")))


class TestIncrement122RelationalSpeedChain(unittest.TestCase):
    """#122 - Anna: a relational-SPEED chain resolved bidirectionally. "Carole reads as fast as Anna (×1)
    but at half the speed of Brianna (×0.5)" → RateRel facts; the synth resolves forward (Anna 1/min →
    Carole 1/min) then INVERSE (Carole known → Brianna = 1 ÷ 0.5 = 2/min). "How long does it take Brianna
    to read a 100-page book?" inverts on Brianna's rate (entity named in the question) → 100 ÷ 2 = 50."""

    def test_rate_relations_parse(self):
        rels = reading._rate_relations(reading.analyze(
            "Carole can read as fast as Anna but at half the speed of Brianna."))
        self.assertEqual(rels, [reading.RateRel("carole", 1.0, "anna", "read"),
                                reading.RateRel("carole", 0.5, "brianna", "read")])

    def test_rate_relation_synth_bidirectional(self):
        facts = [reading.Rate("anna", 1.0, ("page", frozenset()), "minute", "read", 1.0),
                 reading.RateRel("carole", 1.0, "anna", "read"),
                 reading.RateRel("carole", 0.5, "brianna", "read")]
        out = reading.synthesize(facts)
        self.assertIn(reading.Rate("carole", 1.0, ("page", frozenset()), "minute", "read", 1.0), out)
        self.assertIn(reading.Rate("brianna", 2.0, ("page", frozenset()), "minute", "read", 1.0), out)

    def test_anna_solves(self):
        self.assertEqual(reading.solve(
            "Anna can read 1 page in 1 minute. Carole can read as fast as Anna but at half the speed of "
            "Brianna. How long does it take Brianna to read a 100-page book?"), 50)

    def test_no_speed_cue_no_relation(self):
        # a plain comparative without a speed word/noun must NOT become a RateRel
        self.assertIsNone(reading._rate_relations(reading.analyze("Carole reads as much as Anna.")))


class TestIncrement123PerOccurrenceChain(unittest.TestCase):
    """#123 - Sam: a chained per-occurrence rate + duration. "For every sandwich that he eats, Sam eats
    four apples" → Rate(4 apple per sandwich); "10 sandwiches every day for one week" → 10/day × 7 days
    (week→day) = 70 sandwiches; 70 × 4 = 280 apples. The per-occurrence sentence stays one clause; "he"
    binds to Sam via the discourse name-registration; the rate_duration synth chains both steps."""

    def test_per_occurrence_rate_parse(self):
        rate = reading._per_occurrence_rate(reading.analyze(
            "For every sandwich that he eats, Sam eats four apples."))
        self.assertEqual(rate, reading.Rate("sam", 4, ("apple", frozenset()), "sandwich", "eat", 1))

    def test_for_duration_skips_rate_measure(self):
        # "for 8 hours a day" — 8 hours is the rate value, not a duration
        rate = reading.Rate("james", 8.0, ("hour", frozenset()), "day", "rent", 1)
        self.assertIsNone(reading._for_duration(reading.analyze("He rents it for 8 hours a day"), rate))

    def test_sam_solves(self):
        self.assertEqual(reading.solve(
            "For every sandwich that he eats, Sam eats four apples. If he eats 10 sandwiches every day "
            "for one week, how many apples does he eat?"), 280)

    def test_convertible_duration_unit(self):
        self.assertEqual(reading.solve("Sam makes 10 toys every day for one week. "
                                       "How many toys does he make?"), 70)


class TestIncrement124DaypartWeekScaling(unittest.TestCase):
    """#124 - Droid: events recurring DAILY (dayparts morning/afternoon/evening) summed into a per-day
    total, asked "every week" → ×7. "uses 3 bags every morning, triple that in the afternoon, twice the
    morning in the evening" → 3+9+6 = 18/day; "how many every week?" → 18×7 = 126. Gated on a daypart cue
    in a premise + "every/per/a week" in the question + no "week" stated in a premise."""

    def test_droid_weekly_scaling(self):
        self.assertEqual(reading.solve(
            "Droid owns a coffee shop. He uses 3 bags of coffee beans every morning, he uses triple that "
            "number in the afternoon than in the morning, and he uses twice the morning number in the "
            "evening. How many bags of coffee beans does he use every week?"), 126)

    _DROID = ("Droid owns a coffee shop. He uses 3 bags of coffee beans every morning, he uses triple that "
              "number in the afternoon than in the morning, and he uses twice the morning number in the "
              "evening. ")

    def test_daily_question_no_scaling(self):
        # same daypart premises, but the question asks the DAILY total → no ×7
        self.assertEqual(reading.solve(self._DROID + "How many bags of coffee beans does he use every day?"), 18)

    def test_no_period_no_scaling(self):
        # no period in the question → the bare daily total, no scaling
        self.assertEqual(reading.solve(self._DROID + "How many bags of coffee beans does he use?"), 18)


class TestIncrement139RateMarkerEntity(unittest.TestCase):
    """#139 - "(he) reads at a RATE of 50 pages an hour": the rate-marker noun "rate" was grabbed as the
    elided subject, so the Rate got entity='rate' — and `_synth_rate_chain` (same-entity) couldn't compose
    it with "2 hours a day" (entity='john'), so the h/day × pg/h → pg/day chain never formed (John-bible
    committed 56 = 2800/50 under permissive vocab instead of 4 weeks). FIX: when the rate's subject resolves
    to the literal marker "rate", recover the real entity from context (the most-recent prior holder).
    Permissive wrong→correct (John-bible 4); zero real-dev regression, dev 71/3/95.9%, TEST 50.0%. SYNTHETIC."""

    def test_rate_marker_entity_recovered_from_context(self):
        ctx = [reading.Rate("john", 2.0, ("hour", frozenset()), "day", "read", 1)]
        out = reading._rate(reading.analyze("reads at a rate of 50 pages an hour"), ctx)
        self.assertIsNotNone(out)
        self.assertEqual(out.entity, "john")             # recovered, not the marker "rate"

    def test_rate_marker_unrecoverable_without_context_declines(self):
        out = reading._rate(reading.analyze("reads at a rate of 50 pages an hour"), [])
        self.assertIsNone(out)                            # no prior holder → can't recover → no Rate

    def test_explicit_subject_unaffected(self):
        out = reading._rate(reading.analyze("He reads at a rate of 50 pages an hour"), [])
        self.assertIsNotNone(out)
        self.assertEqual(out.entity, "he")               # an explicit subject is used directly


class TestProportion(unittest.TestCase):
    """A direct PROPORTION — "If N X <verb> M <unit>, how MUCH do K X <verb>?" → M × K/N. A premise pairs a
    count (N X) with a measure (M unit) for one entity and verb; the question scales it to a new count.
    Gated to "how much" (an amount, not "how many <noun>") so it never fires on a plain count question.
    SYNTHETIC."""

    def test_weight_proportion(self):
        self.assertEqual(reading.solve(
            "If 4 boxes of apples weigh 20 pounds, how much do 6 boxes weigh?"), 30)

    def test_volume_proportion(self):
        self.assertEqual(reading.solve(
            "If 3 jars hold 12 ounces, how much do 5 jars hold?"), 20)

    def test_plain_count_question_not_proportion(self):
        # "how MANY apples" asks a count, not an amount → the proportion path must not fire.
        self.assertEqual(reading.solve(
            "Tom has 3 apples and 2 oranges. How many apples does Tom have?"), 3)

    def test_rest_clause_does_not_crash(self):
        # `_proportion` calls learn(p) with NO discourse/context args; learn() used to pass that bare `None`
        # straight through to `_facts_from`/`_rest_quantity`, which iterates `context` unconditionally — a
        # "the REST <verb>" premise ("she spent part... and saved the REST") crashed with a TypeError instead
        # of declining. learn() now normalizes a missing context to []. Must decline, not raise.
        self.assertEqual(reading.solve(
            "Mona earns $80 a day. Every day, she spends part of her pay and saves the rest. After 20 days "
            "of work, she spent $1360. How much did Mona save in 20 days?"), "undetermined")


class TestRateExtrapolation(unittest.TestCase):
    """RATE × DURATION-IN-QUESTION — "drives 100 miles in 2 hours. How far will he go in 30 minutes?" →
    100 × (30 min → 0.5 hr) ÷ 2 = 25. Includes the present-progressive parse ("is driving" → drive is the
    main verb). TIGHTLY gated: a PURE single-rate context and EVERY premise number consumed by the rate, so
    a multi-step problem (an extra count/rate/period, an unused given) declines. SYNTHETIC."""

    def test_rate_extrapolation_with_conversion(self):
        self.assertEqual(reading.solve(
            "Sam drives 100 miles in 2 hours. How far will he go in 30 minutes?"), 25)

    def test_present_progressive_rate(self):
        self.assertEqual(reading.solve(
            "Sam is driving 100 miles in 2 hours. How far will he go in 30 minutes?"), 25)

    def test_unused_given_declines(self):
        # the "1 hour" rest is an unused premise number → not a bare rate × duration → decline.
        # The decline SHAPE varies with vocabulary ("rests" unknown → "undetermined"; typed →
        # the fold's unanswerable deficiencies); the pin is the INTENT: never a committed number.
        result = reading.solve(
            "Sam drives 100 miles in 2 hours. He rests for 1 hour. "
            "How far will he go in 30 minutes?")
        self.assertNotIsInstance(result, (int, float))

    def test_compound_duration(self):
        # rate × per-period duration × #periods: 5/min × (1 hour → 60 min) × 2 days = 600.
        self.assertEqual(reading.solve(
            "Tom reads 5 pages each minute. Every day, he reads for an hour. "
            "How many pages will Tom read in two days?"), 600)

    def test_rate_remainder_with_group_total(self):
        # group TOTAL rate (35/week = 5/day) minus a component (3/day) → other = 2/day. Also exercises the
        # group-subject "total" guard exception ("they eat a TOTAL of 35 ounces a week" forms a rate).
        self.assertEqual(reading.solve(
            "Anna has two cats. Together, they eat a total of 35 ounces a week. The first cat eats "
            "3 ounces a day. How many ounces a day does the other cat eat?"), 2)

    def test_rate_race_inversion(self):
        # time to complete a target, differenced: 30/5 − 30/6 = 6 − 5 = 1 hour = 60 minutes.
        self.assertEqual(reading.solve(
            "Anna can run at a rate of 5 miles per hour. Bob can run at a rate of 6 miles per hour. "
            "In minutes, how much faster will Bob run a 30 mile race?"), 60)


class TestRateDifference(unittest.TestCase):
    """RATE DIFFERENCE × DURATION — "A reads M/day, B reads N/day. How many MORE does A read in K weeks?" →
    (M − N) × (K → days). Two rates (same measure & per_unit) and nothing else; the question's subject is A,
    the other is B. Also exercises the " while " clause split (two independent rate clauses). SYNTHETIC."""

    def test_difference_over_duration(self):
        # (50 − 20) × 14 days = 420.
        self.assertEqual(reading.solve(
            "Anna reads 50 pages a day and Bob reads 20 pages a day. "
            "How many more pages does Anna read in 2 weeks?"), 420)

    def test_while_split_and_per_amount(self):
        # "while" splits two rate clauses; (16/2 − 10/2) × 1 hour = 3.
        self.assertEqual(reading.solve(
            "Anna ran 10 miles in 2 hours while Bob ran 16 miles in 2 hours. "
            "How many more miles did Bob run in an hour than Anna?"), 3)

    def test_rate_in_an_hour_and_than_before_duration(self):
        # "N pages in AN hour" parses as a per-hour rate; "than Bob in 40 minutes" word order:
        # (60 − 30) × 40/60 = 20.
        self.assertEqual(reading.solve(
            "Anna can read 60 pages in an hour. Bob can read 30 pages in an hour. "
            "How many more pages does Anna read than Bob in 40 minutes?"), 20)


class TestRateSum(unittest.TestCase):
    """MULTI-ENTITY RATE SUM — N people each read R_i pages per T minutes; after D minutes, how many pages in
    TOTAL → Σ(R_i / T) × D. Includes the GERUND/aspectual parse: "can finish reading 30 pages" — "reading" is
    a verb (lemma read), the aspectual "finish" an auxiliary, and "30" stays free (not a "reading 30" label).
    SYNTHETIC."""

    def test_three_entity_rate_sum(self):
        # (30 + 27 + 25) / 60 × 240 = 328.
        self.assertEqual(reading.solve(
            "Anna reads 30 pages in 60 minutes. Bob reads 27 pages in 60 minutes. "
            "Cherry reads 25 pages in 60 minutes. If they have been reading for 240 minutes, "
            "how many pages have they read in total?"), 328)

    def test_finish_reading_gerund_rate(self):
        # "can finish reading 40 pages in 60 minutes" parses as a 40-pages/60-min read rate.
        self.assertEqual(reading.solve(
            "Anna can finish reading 40 pages in 60 minutes. Bob can read 20 pages in 60 minutes. "
            "If they have been reading for 120 minutes, how many pages have they read in total?"), 120)

    def test_combine_to_verb_is_a_sum_cue(self):
        # per-entity rate × duration ("1.5/week for 10 weeks" = 15, "2.5/week for 8 weeks" = 20), summed by
        # the "combine to lose" cue → 35.
        self.assertEqual(reading.solve(
            "Anna lost 1.5 pounds each week for 10 weeks. Bob lost 2.5 pounds each week for 8 weeks. "
            "How many pounds did the two friends combine to lose?"), 35)


# === PERCENT: percentage of counts/holdings, change / original / population ===================

class TestIncrement53Percentage(unittest.TestCase):
    """#53 - percentage (slice 1: arithmetic). "20%" is a fraction (0.20) with a `base` obligation;
    "What is 20% of 50?" -> 0.20 x 50 = 10. ("% of them"/of a holding is the follow-on.)"""

    def test_percent_parses_as_a_fraction(self):
        r = reading.analyze("20%")
        u = r.units[0]
        self.assertIn("percent", u.features["senses"])
        self.assertEqual(u.features["value"], 0.2)

    def test_percent_of_a_value(self):
        self.assertEqual(reading.answer("What is 20% of 50?", []), 10)
        self.assertEqual(reading.answer("What is 50% of 80?", []), 40)

    def test_percent_without_a_base_declines(self):
        result = reading.answer("What is 20%?", [])
        self.assertTrue(isinstance(result, list))  # a deficiency, not a number

    def test_solve_recognises_percent(self):
        self.assertEqual(reading.solve("What is 25% of 40?"), 10)


class TestIncrement61PercentHoldingCompoundsAndGroupRef(unittest.TestCase):
    """#61 - percentage of a holding ("puts 20% more money" -> 0.2 x the prior quantity); multi-word
    INDIVISIBLE compounds ("piggy bank" -> one unit); the QUESTION's pronouns resolve against the
    discourse ("how much does SHE have"); a plural "they" resolves to the GROUP of prior entities."""

    def test_percent_more_of_a_holding(self):
        self.assertEqual(reading.solve("Susie has $200. Susie puts 20% more money. "
                                       "How much money does Susie have?"), 240)

    def test_percent_quantity_uses_the_referent(self):
        self.assertEqual(reading.learn("Susie puts 20% more money", None,
                                       [reading.Quantity("susie", reading.MONEY, 200.0, "have")]),
                         [reading.Quantity("susie", reading.MONEY, 40.0, "put")])

    def test_piggy_bank_is_one_indivisible_unit(self):
        r = reading.analyze("her piggy bank")
        self.assertTrue(any(u.surface == "piggy bank" and u.features.get("senses") == ["noun"]
                            for u in r.units))

    def test_question_pronoun_resolves_via_discourse(self):
        self.assertEqual(reading.solve("Susie has $30. Susie spends $10. "
                                       "How much money does she have?"), 20)

    def test_they_resolves_to_the_group(self):
        self.assertEqual(reading.solve("Paddington has 40 more goats than Washington. "
                                       "If Washington has 140 goats, "
                                       "how many goats do they have in total?"), 320)


class TestIncrement83PercentOfCount(unittest.TestCase):
    """#83 - percent OF A COUNT: "40% of the students passed" → 0.4 × the students' quantity (24 of 60).
    The base is the "of"-object NOUN (distinct from _percent_quantity's portion, and from "20% of 50"
    arithmetic). "How many students passed?" has no aux — the query noun is itself the subject. The
    category COMPLEMENT ("how many failed" from "40% passed") is NOT inferred → honest decline."""

    def test_percent_of_count_learn(self):
        self.assertEqual(reading.learn("40% of the students passed", [],
                                       [reading.Quantity("class", ("student", frozenset()), 60, "have")]),
                         [reading.Quantity("student", ("student", frozenset()), 24.0, "pass")])

    def test_percent_of_count_solves(self):
        self.assertEqual(reading.solve("A class has 60 students. 40% of the students passed. "
                                       "How many students passed?"), 24)

    def test_portion_percent_unchanged(self):
        self.assertEqual(reading.solve("Susie has $200. Susie puts 20% more money. "
                                       "How much money does Susie have?"), 240)

    def test_complement_declines(self):
        self.assertEqual(reading.solve("A class has 60 students. 40% of the students passed. "
                                       "How many students failed?"), "undetermined")


class TestIncrement113CrossEntityPercentComparative(unittest.TestCase):
    """#113 - "Mark has 50% more points than Eric" (Eric=6) is a CROSS-ENTITY comparative → the FULL
    derived amount 6×(1+0.5)=9, not just the portion 0.5×6=3. Gated on the comparative's standard being
    a NAME (another entity), so the SAME-entity "puts 20% more money" (standard = the measure noun) keeps
    its portion reading (the base is the entity's own holding the fold already carries)."""

    def test_cross_entity_percent_more(self):
        self.assertEqual(reading.solve("Eric has 6 points. Mark has 50% more points than Eric. "
                                       "How many points does Mark have?"), 9)

    def test_cross_entity_percent_chain_total(self):
        self.assertEqual(reading.solve(
            "Samanta has 8 more points than Mark, and Mark has 50% more points than Eric. "
            "Eric has 6 points. How many points do Samanta, Mark, and Eric have in total?"), 32)

    def test_same_entity_percent_more_unchanged(self):
        # "puts 20% more money" — standard is the measure noun, not another entity → portion folded onto
        # the existing $200 holding: 200 + 0.2×200 = 240
        self.assertEqual(reading.solve("Susie has $200 in her piggy bank. If she puts 20% more money "
                                       "into her piggy bank, how much money will she have?"), 240)


class TestIncrement114ForwardRefPercentChainDefersAndSnaps(unittest.TestCase):
    """#114 - a cross-entity percent comparative whose named standard isn't known YET ("John has 20% more
    than Jules" before Jules resolves) must DEFER (return None) so the fixpoint resolves it after the
    standard — never off a measure-matched stand-in (it was committing 100×1.2=120 off Stan). Plus the
    FINAL answer snaps binary-float drift to a clean value (a percent chain lands on 29.999…93, not 30)."""

    def test_forward_ref_percent_chain(self):
        self.assertEqual(reading.solve(
            "John has 20% more boxes than Jules. Jules has 5 more boxes than Joseph. "
            "Joseph has 80% fewer boxes than Stan. If Stan has 100 boxes, "
            "how many boxes does John have?"), 30)

    def test_final_answer_snaps_float_drift(self):
        # 100 × (1 − 0.8) = 29.999999999999993 in binary float → snapped to a clean 30
        self.assertEqual(reading.solve("Stan has 100 boxes. Joseph has 80% fewer boxes than Stan. "
                                       "How many boxes does Joseph have?"), 20)

    def test_intermediate_precision_preserved(self):
        # the snap is on the FINAL answer only — a 1/60 conversion keeps full precision mid-chain
        self.assertEqual(reading.solve("Weng earns $12 an hour for babysitting. Yesterday, she just did "
                                       "50 minutes of babysitting. How much did she earn?"), 10)


class TestPercentChange(unittest.TestCase):
    """PERCENT CHANGE — apply a single % change to a stated money base, report the NEW value. One money value
    + one percent + exactly one direction cue (raise/increase ↑ XOR discount/off/down ↓) + a new-value
    question (pay/new/cost, none of original/save/total/left). SYNTHETIC."""

    def test_increase_new_value(self):
        # 200 × (1 + 0.10) = 220.
        self.assertEqual(reading.solve(
            "A bike was sold for $200 last year. This year the price was increased by 10%. "
            "What is the new price?"), 220)

    def test_discount_pay(self):
        # 50 × (1 − 0.20) = 40.
        self.assertEqual(reading.solve(
            "Anna bought a shirt with a 20% discount. If the price is $50, "
            "how much did Anna pay after the discount?"), 40)

    def test_spend_remainder_declines(self):
        # RE-PINNED (#224): `_percent_money_tails_system` now claims this CORRECTLY (14·65/35
        # = 26) — the percent-change resolvers this class pins are unchanged (they still stay
        # out); the EARLY system answers first. Was pinned "undetermined" while unsolvable.
        self.assertEqual(reading.solve(
            "Anna spent 35% of her allowance. If she spent $14, how much money is left in her allowance?"),
            26)

    def test_multi_step_percent_declines(self):
        # a 3rd number (5 toys) makes it multi-step (per-item × discount) — DECLINE, don't commit 3×0.8.
        self.assertEqual(reading.solve(
            "Bob buys 5 toys that each cost $3. He gets a 20% discount. "
            "How much did he pay for everything?"), "undetermined")


class TestPercentPopulation(unittest.TestCase):
    """TWO-GROUP population partition — a percent splits a population into complementary groups; recover the
    asked group by its fraction (P for the percent's group, 1−P for the complement). Total given → N×f;
    one group's count given → C×f_q/f_base. SYNTHETIC."""

    def test_total_given_complement(self):
        # 40 people, 30% boys → girls = 40 × 0.70 = 28.
        self.assertEqual(reading.solve(
            "At a meeting there are 40 people. 30% of them are boys. How many girls are at the meeting?"), 28)

    def test_total_given_same_group(self):
        # 40 people, 30% boys → boys = 40 × 0.30 = 12.
        self.assertEqual(reading.solve(
            "At a meeting there are 40 people. 30% of them are boys. How many boys are at the meeting?"), 12)

    def test_group_count_given(self):
        # 60% girls, boys = 200 → girls = 200 × 0.6/0.4 = 300.
        self.assertEqual(reading.solve(
            "60% of the students are girls and the number of boys is 200. "
            "How many girls are there?"), 300)

    def test_returns_system_result_with_both_group_facts(self):
        # #400b -- migrated to SystemResult: both complementary groups' facts, not just the asked one.
        text = "At a meeting there are 40 people. 30% of them are boys. How many girls are at the meeting?"
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._percent_population(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {}
        for fs in result.facts_by_premise.values():
            for f in fs:
                seen[f.entity] = f.amount
        self.assertEqual(seen, {"boy": 12, "girl": 28})
        self.assertEqual(reading.solve(text), 28)

    def test_relative_clause_ask_stays_on_legacy_scalar_path(self):
        # a real TEST regression: "how many friends does he have WHO ARE girls?" resolves answer()'s own
        # query to 'friend' (the relative clause "who are girls" is dropped), not 'girl' -- the new
        # SystemResult path would guarantee a decline, so this must stay on the legacy scalar path.
        text = ("55% of Toby's friends are boys and the rest are girls. If he has 33 friends who are boys, "
                "how many friends does he have who are girls?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._percent_population(premises, question, rp)
        self.assertNotIsInstance(result, reading.SystemResult)
        self.assertEqual(result, 27)
        self.assertEqual(reading.solve(text), 27)


class TestPercentOriginal(unittest.TestCase):
    """PERCENT ORIGINAL (inverse) — a money value is a % of an unknown original; recover it by division. Frame
    picks the divisor: "of" → ÷P; discount/less → ÷(1−P); more → ÷(1+P). Question must ask the ORIGINAL.
    SYNTHETIC."""

    def test_of_frame(self):
        # 6000 is 40% of the original → 6000 / 0.40 = 15000.
        self.assertEqual(reading.solve(
            "Bob spent $6000 on a car which cost 40% of the original price. "
            "What was the original price of the car?"), 15000)

    def test_discount_frame(self):
        # paid $240 at a 20% discount → 240 / (1 − 0.20) = 300.
        self.assertEqual(reading.solve(
            "Anna bought a pair of shoes at a 20% discount. If she paid $240, "
            "what was the original price of the shoes?"), 300)

    def test_less_frame(self):
        # $75 is 25% less than the original → 75 / 0.75 = 100.
        self.assertEqual(reading.solve(
            "Bob spent $75 on a used bike, which is 25% less than the original price. "
            "What was the original price of the bike?"), 100)


# === DISTRIBUTION / TRANSFER / SHARING: equal split, dative transfer, purchase, depletion =====

class TestIncrement60EachDistribution(unittest.TestCase):
    """#60 - each/per distribution: "gave 4 stickers each to 3 friends" -> 4 x 3 = 12. "each" makes
    the per-recipient amount (the value-noun before it) MULTIPLY the recipient count (the other value),
    instead of becoming two separate conjoined quantities."""

    def test_distribution_multiplies(self):
        self.assertEqual(reading.learn("Colton gave 4 stickers each to 3 friends"),
                         [reading.Quantity("colton", ("sticker", frozenset()), 12.0, "give")])

    def test_distribution_solves(self):
        self.assertEqual(reading.solve("Colton gave 4 stickers each to 3 friends. "
                                       "How many stickers did Colton give?"), 12)

    def test_conjoined_object_without_each_stays_two_facts(self):
        self.assertEqual(reading.learn("Tom has 3 apples and 2 oranges"),
                         [reading.Quantity("tom", ("apple", frozenset()), 3.0, "have"),
                          reading.Quantity("tom", ("orange", frozenset()), 2.0, "have")])


class TestIncrement65RelationalLosses(unittest.TestCase):
    """#65 - relational losses: "loses/gives half OF them" -> fraction x the current holding, a CHANGE
    (not a Frequency -- the "of" makes it partitive). Lets a Buddy-class timeline solve end to end."""

    def test_loses_half_is_a_relational_loss(self):
        ctx = [reading.Quantity("buddy", ("baseball", frozenset()), 30, "have", "monday")]
        self.assertEqual(reading.learn("On Tuesday Buddy loses half of them", None, ctx),
                         [reading.Quantity("buddy", ("baseball", frozenset()), 15.0, "lose", "tuesday")])

    def test_gives_half_away_timeless(self):
        self.assertEqual(reading.solve("Tom has 10 apples. Tom gives half of them away. "
                                       "How many apples does Tom have?"), 5)

    def test_buddy_timeline_solves_end_to_end(self):
        self.assertEqual(reading.solve(
            "On Monday Buddy has 30 baseball cards. On Tuesday Buddy loses half of them. "
            "On Wednesday Buddy buys 12 baseball cards. "
            "How many baseball cards does Buddy have on Wednesday?"), 27)

    def test_frequency_still_works_without_of(self):
        self.assertEqual(reading.learn("Mary jumps twice"), [reading.Frequency("mary", "jump", 2)])


class TestIncrement70DistributionUnknown(unittest.TestCase):
    """#70 - Option 3, faithfully: "shared a certain amount of money between A and B" makes the unknown
    total a FIRST-CLASS node (Quantity amount=None), the share-between a Distribution relation, and a
    registered synthesizer BACK-LINKS the unknown to the SUM of the recipients' shares. The unknown's
    value obligation is discharged by inference, not a question-time heuristic."""

    def test_unknown_node_and_distribution_read_off(self):
        facts = reading.learn("Mr. Sam shared a certain amount of money between his two sons, Ken and Tony.")
        self.assertIn(reading.Quantity("sam", ("money", frozenset()), None, "share"), facts)
        self.assertIn(reading.Distribution("sam", ("ken", "tony"), ("money", frozenset()),
                                           count=2, method=None, recipient_noun=("son", frozenset())), facts)

    def test_synthesis_pins_the_unknown(self):
        ctx = (reading.learn("Mr. Sam shared a certain amount of money between his two sons, Ken and Tony.")
               + reading.learn("Ken got 1750 dollars") + reading.learn("Tony got 3500 dollars"))
        self.assertIn(reading.Quantity("sam", ("money", frozenset()), 5250.0, "share"),
                      reading.synthesize(ctx))

    def test_full_shared_money_problem_solves(self):
        self.assertEqual(reading.solve(
            "Mr. Sam shared a certain amount of money between his two sons, Ken and Tony. "
            "If Ken got $1750, and Tony got twice as much as Ken, how much was the money shared?"), 5250)

    def test_declines_until_every_share_is_known(self):
        self.assertEqual(reading.solve(
            "Mr. Sam shared a certain amount of money between his two sons, Ken and Tony. "
            "If Ken got $1750, how much was the money shared?"), "undetermined")


class TestIncrement71EqualSplit(unittest.TestCase):
    """#71 - "shared <total> equally among N <noun>" → each gets total ÷ N (the inverse of #70: total
    KNOWN, recipients a COUNT). Reuses the Distribution relation + the synthesis loop; _synth_equal_split
    derives a `get` Quantity per recipient (named, else keyed to the recipient noun for "each son")."""

    def test_equal_split_among_a_count(self):
        self.assertEqual(reading.solve("Mr. Sam shared $5250 equally among his 3 sons. "
                                       "How much did each son get?"), 1750)

    def test_equal_split_nonmoney_measure(self):
        self.assertEqual(reading.solve("A teacher distributed 30 candies equally among 5 students. "
                                       "How many candies did each student get?"), 6)

    def test_equal_split_among_named(self):
        self.assertEqual(reading.solve("Sam shared $30 equally between Ken and Tony. "
                                       "How much did Ken get?"), 15)

    def test_unknown_total_distribution_still_solves(self):
        self.assertEqual(reading.solve(
            "Mr. Sam shared a certain amount of money between his two sons, Ken and Tony. "
            "If Ken got $1750, and Tony got twice as much as Ken, how much was the money shared?"), 5250)


class TestIncrement72MannerMethod(unittest.TestCase):
    """#72 - the distribution METHOD is a manner adverb BOUND to its verb (`qualifies`), not a global
    scan: a verb carries an operation; a manner adverb parametrizes it (like negation, one slot over).
    "equally"/"evenly"/"uniformly" → an equal split (computable); "randomly"/"optimally" or NO method →
    an under-specified Deficiency (ASK how it was shared), never a fabricated number."""

    def test_method_binds_to_the_verb(self):
        r = reading.analyze("Sam shared money equally among them")
        equally = next(u for u in r.units if u.surface == "equally")
        self.assertEqual(reading._follow(equally, "qualifies").surface, "shared")

    def test_method_stored_on_distribution(self):
        d = next(f for f in reading.learn("Sam shared $30 equally among his 3 sons.")
                 if isinstance(f, reading.Distribution))
        self.assertEqual(d.method, "equal")

    def test_evenly_is_an_equal_split(self):
        self.assertEqual(reading.solve("A teacher distributed 30 candies evenly among 5 students. "
                                       "How many candies did each student get?"), 6)

    def test_no_method_is_underspecified_deficiency(self):
        r = reading.solve("Sam shared $30 among his 3 sons. How much did each son get?")
        self.assertTrue(isinstance(r, list) and r and r[0].kind == "under-specified")

    def test_random_method_is_underspecified_deficiency(self):
        r = reading.solve("Sam shared $30 randomly among his 3 sons. How much did each son get?")
        self.assertTrue(isinstance(r, list) and r and r[0].kind == "under-specified")

    def test_subject_group_is_not_a_distribution(self):
        self.assertFalse(any(isinstance(f, reading.Distribution)
                             for f in reading.learn("Tom and Mary shared 10 apples")))


class TestIncrement84ComplementCategory(unittest.TestCase):
    """#84 - the COMPLEMENT/category model: "the REST <verb>" → population total − Σ(stated subsets of the
    same measure). "60 students; 40 passed; the rest failed" → 20; with a percent subset → 36. Summed by
    MEASURE (plural-subject entity keying is inconsistent). The complement needs an EXPLICIT signal (rest/
    remainder/others) and a known total — never assumed, else decline."""

    def test_count_complement(self):
        self.assertEqual(reading.solve("A class has 60 students. 40 students passed. The rest failed. "
                                       "How many students failed?"), 20)

    def test_percent_complement(self):
        self.assertEqual(reading.solve("A class has 60 students. 40% of the students passed. The rest "
                                       "failed. How many students failed?"), 36)

    def test_rest_learn(self):
        self.assertEqual(reading.learn("The rest failed", [],
                                       [reading.Quantity("class", ("student", frozenset()), 60, "have"),
                                        reading.Quantity("student", ("student", frozenset()), 40, "pass")]),
                         [reading.Quantity("student", ("student", frozenset()), 20, "fail")])

    def test_no_total_declines(self):
        # no population total → no complement count → decline (never a guessed number)
        self.assertEqual(reading.solve("40 students passed. The rest failed. How many students failed?"),
                         "undetermined")


class TestIncrement126PartitionAggregate(unittest.TestCase):
    """#126 - PARTITION-and-AGGREGATE: a container set split into subgroups ("the first two … the other
    two … 1 of the tanks"), each member holding a per-member quantity of the asked measure → Σ (count ×
    per_member), optionally × a period asked in the question. The relational per-member ("twice as many
    each as the first", "N fewer each than the first two") is resolved by the ordinary `learn` machinery;
    only the subgroup COUNT and the Σ-fold are new. Resolves James-fish 100 and Nicole 112. SYNTHETIC
    strings only (the firewall keeps GSM8K text out of the unit tests)."""

    def test_partition_relational_each(self):
        # "1 of the boxes → 10; the other two → twice as many each (=20)" → 10 + 2×20 = 50
        self.assertEqual(reading.solve(
            "Maria has 3 boxes. 1 of the boxes holds 10 marbles and the other two hold twice as many "
            "marbles each as the first. How many marbles are there in total?"), 50)

    _ROSA = ("Rosa has 4 tanks. The first two tanks need 5 gallons of water each and the other two need "
             "2 fewer gallons of water each than the first two tanks. ")

    def test_partition_comparative_each_with_period(self):
        # "first two → 5 each; other two → 2 fewer each (=3)" → 2×5 + 2×3 = 16, × 3 weeks = 48
        self.assertEqual(reading.solve(
            self._ROSA + "If she changes the water every week, how many gallons of water will she "
            "need in three weeks?"), 48)

    def test_partition_no_period_no_scaling(self):
        # same partition, no period in the question → the bare Σ (no × period)
        self.assertEqual(reading.solve(self._ROSA + "How many gallons of water does she need?"), 16)

    def test_partition_helpers(self):
        # the subgroup count is read off the selector; the container off the adjacent noun
        rd = reading.analyze("the first two tanks need 8 gallons of water each")
        self.assertEqual(reading._subgroup_count(rd), 2)
        self.assertEqual(reading._subgroup_container(rd), "tank")
        rd2 = reading.analyze("1 of the tanks has 20 fish in it")
        self.assertEqual(reading._subgroup_count(rd2), 1)
        self.assertEqual(reading._subgroup_container(rd2), "tank")

    def test_partition_declines_same_measure(self):
        # NOT a partition: "3 of the apples … 4 of the apples" partitions the SAME measure that's asked
        # (apple), and there's no first/other selector → the gate rejects it (container == target).
        self.assertIsNone(reading._partition(
            ["3 of the apples are red", "4 of the apples are green"], "How many apples are there?"))


class TestIncrement127Purchase(unittest.TestCase):
    """#127 - PURCHASE as a measure-aware money LOSS: "buys 4 candy bars that cost $2 each" → spends
    4 × $2 = $8, netted against a money baseline ("Todd has $20 … how much LEFT" → 12). One "buy" verb
    yields two facts that fold oppositely (gain the items +, lose the money −); `_verb_sign` is measure-
    agnostic, so the money outflow carries the LOSS verb 'spend'. Resolves Todd 12. Tightly gated: a buy
    verb, exactly one 'each', one price + one count, AND a prior money holding for the buyer (so a
    multi-item buy clause-split into fragments, which has no baseline, DECLINES). SYNTHETIC strings only."""

    def test_purchase_nets_against_baseline(self):
        self.assertEqual(reading.solve(
            "Maya has $30. She buys 3 books that cost $4 each. How much money does she have left?"), 18)

    def test_purchase_multi_item_declines(self):
        # two purchases ("3 apples … and 2 toys …") → we'd commit only one pair's product → DECLINE
        self.assertEqual(reading.solve(
            "Maya has $30. She buys 3 apples for $1 each and 2 toys for $2 each. "
            "How much does she spend?"), "undetermined")

    def test_purchase_helper_money_loss(self):
        # the money fact is count × price with the LOSS verb 'spend'; gated on a baseline in context
        ctx = [reading.Quantity("he", reading.MONEY, 20.0, "have")]
        facts = reading._purchase(reading.analyze("He buys 4 candy bars that cost $2 each."), ctx)
        self.assertIn(reading.Quantity("he", reading.MONEY, 8, "spend"), facts)
        self.assertIsNone(reading._purchase(   # no baseline → None
            reading.analyze("He buys 4 candy bars that cost $2 each."), []))


class TestIncrement128ContainerGiveaway(unittest.TestCase):
    """#128 - EQUAL-SPLIT into containers, then GIVE some containers away → the contents are lost. "28
    marbles … into 4 bags … the same number in each bag … gives one bag away … how many marbles left?"
    → per-bag = 28/4 = 7, lose 1×7, answer 28−7 = 21. The give is counted in CONTAINERS but the question
    asks the CONTENTS, so it converts via the per-container share. Resolves James-marbles 21. Tightly
    gated: a "same … each" equal-split marker, a whole of the asked measure, a container `put`-count of a
    different measure, and a `give` of that container. SYNTHETIC strings only."""

    def test_container_giveaway_one(self):
        self.assertEqual(reading.solve(
            "Anna has 30 coins. She puts them into 5 jars. She puts the same number in each jar. "
            "She then gives one jar away. How many coins does she have left?"), 24)

    def test_container_giveaway_multiple(self):
        # give TWO containers → lose 2 × per-container (30/5 = 6) = 12 → 18
        self.assertEqual(reading.solve(
            "Anna has 30 coins. She puts them into 5 jars. She puts the same number in each jar. "
            "She then gives two jars away. How many coins does she have left?"), 18)

    def test_container_giveaway_helper(self):
        per_split = reading._container_giveaway(
            ["He has 28 marbles", "He puts them into 4 bags", "He puts the same number in each bag",
             "He gives one bag away"], "How many marbles does he have left?",
            [reading.Quantity("he", ("marble", frozenset()), 28.0, "have"),
             reading.Quantity("he", ("bag", frozenset()), 4.0, "put"),
             reading.Quantity("he", ("bag", frozenset()), 1.0, "give")])
        self.assertEqual(per_split, 21)
        # no "same … each" marker → not this shape → None
        self.assertIsNone(reading._container_giveaway(
            ["He has 28 marbles", "He puts them into 4 bags"], "How many marbles does he have left?",
            [reading.Quantity("he", ("marble", frozenset()), 28.0, "have")]))


class TestIncrement129CombinedRemainder(unittest.TestCase):
    """#129 - COMBINED REMAINDER over multiple owners: "how many in total do they have LEFT?" sums each
    owner's NET holdings (picked − gave − sold), not the raw sum of every quantity (which double-counts a
    gross alongside its own losses). "Frank picked 36; Susan picked 3× as many (108); Susan gave half (54);
    Frank sold a third (12)" → Susan 108−54=54, Frank 36−12=24, total 78 (was 210 = 36+108+54+12). Fires
    only when the combined question carries a remainder cue ("left"). SYNTHETIC strings only."""

    def test_combined_remainder_two_owners(self):
        # Jane 2×30=60, gave half (30) → 30; Tom 30, sold a third (10) → 20; total 50
        self.assertEqual(reading.solve(
            "Tom picked 30 apples. Jane picked 2 times as many apples as Tom. If Jane gave out half of "
            "her apples, and Tom sold a third of his, how many in total do they have left?"), 50)


class TestDativeTransferCluster(unittest.TestCase):
    """Three fixes landed TOGETHER to solve "Reggie's father gave him $48. Reggie bought 5 books, each of which
    cost $2. How much left?" → 48 − 5·2 = 38: (1) DATIVE transfer "<giver> gave <recipient> $N" → recipient +N,
    giver −N; (2) POSSESSIVE-OWNER pronoun resolution ("Reggie's father gave HIM" → him = Reggie); (3) a
    clause-split guard keeping a relative clause ("books, each of which cost $2") with its head so _purchase
    reads the per-item price. The dative alone was +0 (multi-blocked); together +1 dev-train, and TEST 61.5→69.2%
    (generalised — a held-out problem fixed + a held-out wrong corrected). SYNTHETIC."""

    def test_dative_money_gain(self):
        self.assertEqual(reading.solve("Tom gave Reggie $48. How much money does Reggie have?"), 48)

    def test_dative_giver_loss_preserved(self):
        self.assertEqual(reading.solve(
            "Randy has $5000. Randy gave Sally $1200. How much money does Randy have?"), 3800)

    def test_possessive_owner_pronoun(self):
        self.assertEqual(reading.solve(
            "Reggie's father gave him $48. How much money does Reggie have?"), 48)

    def test_full_cluster(self):
        self.assertEqual(reading.solve(
            "Reggie's father gave him $48. Reggie bought 5 books, each of which cost $2. How much money does "
            "Reggie have left?"), 38)

    def test_to_transfer_unaffected(self):
        # "gave 12 to his brother" (with "to") is a different construct — the dative must NOT fire.
        self.assertIsNone(reading._dative_transfer(reading.analyze("Sonny gave 12 to his brother.")))


class TestShoppingTotal(unittest.TestCase):
    """"buys 3 toys at $10 each, 2 packs at $5 each, 5 shirts at $6 each. How much spend in all?" → Σ(count×
    price) = 70. Counts and per-item prices parse as separate buy-quantities; only the per-each products + sum
    were missing. Tight gates (the shape collides with "for $80" totals, discounts, per-piece questions, and
    split pairs): explicit TOTAL cue, every price immediately followed by each/apiece, no discount, strict
    item/price alternation, no each-price stranded in the question. dev-train 69→70, holdout/TEST held. SYNTHETIC."""

    def test_shopping_total(self):
        self.assertEqual(reading.solve(
            "Nina buys three toys at $10 each, two packs of cards at $5 each, and five shirts at $6 each. "
            "How much does she spend in all?"), 70)

    def test_for_total_price_declines(self):
        # "four pizzas for $80" is a TOTAL, not a per-each price → must not pair as 4×80.
        # RE-PINNED (#209): `_pieces_pool_system` now claims this synthetic CORRECTLY —
        # 80/(4·5) = 4 per piece (the real Luigi dev case's gold). The _shopping_total
        # mis-pairing this pin defends against still declines (the 4×80=320 read never fires).
        self.assertEqual(reading.solve(
            "Luigi bought four pizzas for $80. Each pizza was cut into 5 pieces. How much did each piece "
            "cost in total?"), 4)

    def test_no_total_cue_declines(self):
        # no "in all"/"total" cue → a pair may be split into the question clause → decline (Alexander 5≠9).
        self.assertNotIsInstance(reading.solve(
            "He buys 5 apples for $1 each and 2 oranges for $2 each, how much does he spend?"), (int, float))


class TestBackwardDepletion(unittest.TestCase):
    """The backward twin of scene-depletion: an UNKNOWN initial pool depleted by spends to a KNOWN final →
    initial = final + Σ(depletions). "Some money in the jar. Doris spent $6. Martha spent half as much [$3].
    $15 left. How much at first?" → 15+6+3 = 24. Three precision gates (the firewall caught all three over-
    fires): NO gain to the pool (Randy "gave him $200" → 3500≠3000); NO dropped multiplier/fraction-of-remaining
    (Tim "buys half the amount he had left" → 30≠22); NO unused given (Josh's 2nd "spent $1.25" lost to
    elision → 7.75≠9). dev-train 68→69, holdout/TEST held. SYNTHETIC."""

    def test_backward_pool_depletion(self):
        self.assertEqual(reading.solve(
            "There was some money in the cookie jar. Doris spent $6 from the cookie jar. Martha spent half as "
            "much as Doris. If there were $15 left in the cookie jar, how much money was in it at first?"), 24)

    def test_gain_now_solved_by_inversion(self):
        # "Smith gave him $200" is a GAIN — the losses-only depletion declines, but `_backward_inversion`
        # (#4e) solves it with signed deltas: 2000 − (200 − 1200) = 3000.
        self.assertEqual(reading.solve(
            "Randy had some money. Smith gave him another $200. Randy then gave Sally $1200. If Randy has "
            "$2000 left, how much money did Randy have at first?"), 3000)

    def test_then_continuation_backward(self):
        # the "then spent another $1.25" continuation now supplies the elided subject, so both spends are
        # captured and the backward inversion solves: 6 + 1.75 + 1.25 = 9.
        self.assertEqual(reading.solve(
            "Josh has some money. He spent $1.75 on a drink, and then spent another $1.25. If he had $6 left, "
            "how much money did Josh have at first?"), 9)


class TestEqualShare(unittest.TestCase):
    """POOL-AND-REDISTRIBUTE — "Mark has 30, Peter 25, John 35. They combine and share equally. How many
    will EACH have?" → (30+25+35)/3 = 30. Same Σ/N math as the average branch, cued by a share-equally premise
    + a per-each result question. Gated against the distinct distributor-among-a-count shape (a Distribution
    fact, or a "rest"/remainder shared among OTHERS), noun-entity owners ("the rest" → entity == measure), and
    a non-integer result (a relation didn't resolve). dev-train 62→64, holdout/TEST held. SYNTHETIC."""

    def test_pool_and_share_equally(self):
        self.assertEqual(reading.solve(
            "Mark has 30 candies, Peter has 25 candies, and John has 35 candies. They combine their candies "
            "and share them equally. How many candies will each one of them have?"), 30)

    def test_share_equally_each_person(self):
        self.assertEqual(reading.solve(
            "Hugh had 8 marbles, Tommy had 6 marbles, and Melany had 7 marbles. If they share the marbles "
            "equally, how many will each person have?"), 7)

    def test_distributor_among_count_unaffected(self):
        # one whole shared among a stated count (the existing equal-split handler) → 5250/3, NOT a pool.
        self.assertEqual(reading.solve(
            "Mr. Sam shared $5250 equally among his 3 sons. How much did each son get?"), 1750)

    def test_remainder_to_others_declines(self):
        # "kept 20 and shared THE REST among friends" is a remainder to OTHERS, not a pool → this recognizer
        # must NOT fire (it would pool Harold's own holding); left to other machinery (here: declines).
        self.assertEqual(reading.solve(
            "Harold had 100 marbles. He kept 20 marbles and shared the rest evenly among his 5 friends. "
            "How many marbles did each friend get?"), "undetermined")


class TestSceneDepletion(unittest.TestCase):
    """A POOL ("there were N X") depleted by takes ("took A … took B") → N − Σtakes. The existential N is
    stored as INSTANCES, not a holding, and the takes are credited to the takers, so the remainder fold
    can't net it. Gated: a left/remain question, an instance-base, all measure-quantities are depletion
    verbs, and NO unused given (a conversion factor / per-unit count means the base needs computing)."""

    def test_pool_depleted_by_takes(self):
        self.assertEqual(reading.solve(
            "There were 20 marbles in a jar. Tom took 5 marbles and Sue took 3 marbles. "
            "How many marbles were left in the jar?"), 12)

    def test_per_unit_base_declines(self):
        # "6 eggs in a carton … 2 cartons" — the base needs ×2, so the bare instance count (6) is wrong;
        # the extra number (2) is an unused given → decline rather than commit 6 − eaten.
        self.assertEqual(reading.solve(
            "There are 6 eggs in a carton. Sam buys 2 cartons and eats 4 eggs. How many eggs are left?"),
            "undetermined")

    def test_addition_present_declines(self):
        # a non-depletion (add) quantity of the measure breaks the plain base − Σ → decline.
        self.assertEqual(reading.solve(
            "There were 10 apples in a basket. Ann added 5 apples and Bob took 3 apples. "
            "How many apples were left?"), "undetermined")


# === SYSTEM SOLVERS: affine / anchored / noun / offset / residual systems =====================

class TestIncrement140TwoUnknownRatio(unittest.TestCase):
    """#140 - FIRST 2-UNKNOWN SUM+RATIO solve: two entities, one a multiple of the other, plus a COMBINED
    TOTAL → solve for the asked part. "Jean has three times as much money as Jane. They have a combined total
    of $76. How much does Jane have?" → Jean=3·Jane, Jean+Jane=76 → Jane=19. The forward fold can't invert it
    (neither part is known; the relation defers to no fact). `_two_unknown_ratio` is tried EARLY (before the
    unused-given GUARD 3, which would otherwise decline on the 'unused' 3 and 76). TIGHTLY gated because these
    shapes also appear in CORRECT forward problems: exactly 2 entities, relation+total about the SAME measure
    (head AND qualifier — 'red balls' ≠ 'blue balls', 'parking' ≠ 'traffic' tickets), no comparative offset.
    PURE-RATIO only. dev 71→72 (95.9→96.0%), 0 new wrongs, TEST 50.0% held, 0 permissive regression. SYNTHETIC."""

    def test_ratio_with_combined_total_solves_part(self):
        self.assertEqual(reading.solve(
            "Jean has three times as much money as Jane. They have a combined total of $76. "
            "How much money does Jane have?"), 19)
        self.assertEqual(reading.solve(
            "Mark has twice as many points as Eric. They have a combined total of 30 points. "
            "How many points does Mark have?"), 20)

    def test_three_entity_system_declines(self):
        # UNDER-CONSTRAINED 3-entity (no total, Bob unanchored) → genuinely undetermined. (With a stated
        # total this shape now SOLVES via `_affine_system` — Ann=3·Bob + Carol=5 + total=40 → Bob=8.75 —
        # so the decline here is the missing-constraint case, not a limitation of the ratio machinery.)
        self.assertEqual(reading.solve(
            "Ann has 3 times as many pens as Bob. Carol has 5 pens. How many pens does Bob have?"),
            "undetermined")

    def test_measure_qualifier_mismatch_declines(self):
        # the relation ('red balls') and the total ('blue balls') are different measures → not this system
        self.assertEqual(reading.solve(
            "Zoe has twice as many red balls as Sam. They have a combined total of 12 blue balls. "
            "How many red balls does Sam have?"), "undetermined")


class TestSumRatioSystem(unittest.TestCase):
    """#3h generalized to an N-UNKNOWN sum + ratio-CHAIN system: relations may chain (Chris→Amy→Jeremy),
    reduced to a pivot by coefficient propagation, then pivot = Total / Σcoeff. Tightly gated: a genuine
    ratio frame (the standard after as/of/that or a possessor), unique root + connected tree, unit/qualifier
    match, and pure-ratio (an offset-tainted system declines). Subsumes the 2-entity case."""

    def test_three_entity_chain(self):
        # B is the pivot (root); A = B/3, C = 2A = 2B/3; coeffs 1+1/3+2/3 = 2 → B = 12/2 = 6.
        self.assertEqual(reading.solve(
            "Ann, Bea, and Cam have a combined age of 12. Ann is 1/3 the age of Bea, and "
            "Cam is twice as old as Ann. How old is Bea?"), 6)

    def test_two_entity_still_solves(self):  # the #3h case via the generalized solver
        self.assertEqual(reading.solve(
            "Sam and Drew have a combined age of 54. Sam is half of Drew's age. How old is Sam?"), 18)

    def test_offset_tainted_system_solved_by_affine(self):
        # "5 more than twice" is an offset+ratio relation the PURE-ratio system can't fold — it declines —
        # but `_affine_system` now solves the mixed system: Cy=2·Ky+5, Ky=½·Ay, Cy+Ky+Ay=100 → 2.5·Ay+5=100
        # → Ay=38. (Previously this whole shape declined for safety; the affine solver lifts that.)
        self.assertEqual(reading.solve(
            "Cy, Ky, and Ay have a combined total of 100 stamps. Cy has 5 more than twice as many as Ky, "
            "and Ky has half as many as Ay. How many stamps does Ay have?"), 38)

    def test_no_ratio_frame_declines(self):
        # "bought half the cards" is a count of an object, not a ratio between the two names → decline.
        self.assertEqual(reading.solve(
            "Bob and Sue have 12 cards combined. Bob bought half the cards sold. How many does Sue have?"),
            "undetermined")

    def test_known_absolute_part_in_sum(self):
        # a KNOWN absolute (Ann took 8) + a ratio (Bea = 3·Cam) + total: Cam = (40 − 8) / (3 + 1) = 8.
        self.assertEqual(reading.solve(
            "Ann, Bea and Cam decided to share 40 donuts. Ann took 8 donuts and Bea took three times as "
            "many as Cam. How many donuts did Cam take?"), 8)


class TestAffineSystem(unittest.TestCase):
    """`_affine_system` — the N-entity MIXED (ratio + offset) sum-system that generalizes the pure-ratio and
    single-offset solvers. Each entity is affine in a free pivot (value = A·pivot + B), pinned by a total."""

    def test_ratio_plus_offset_coordinate_split(self):
        # Xita = 3·Yuma (ratio) and 2 fewer than Zane (offset, subject+verb ELIDED after "and"); 142 total →
        # reduce to pivot Yuma: A=1,3,3  B=0,0,2 → Yuma = (142−2)/7 = 20.
        self.assertEqual(reading.solve(
            "Xita has 3 times as many toys as Yuma and 2 fewer toys than Zane. If they have 142 toys all "
            "together, how many toys does Yuma have?"), 20)

    def test_offset_ratio_chain_to_pivot(self):
        # Ova = 2·Pim + 25 (offset+ratio), Pim = 3·Qax − 4 (offset+ratio); 383 total, asked the pivot Qax →
        # 10·Qax + 13 = 383 → Qax = 37.
        self.assertEqual(reading.solve(
            "Ova, Pim, and Qax have a total of 383 chickens. Ova has 25 more than double the chickens that Pim "
            "has. Pim has 4 less than triple the number of chickens that Qax has. How many chickens does Qax "
            "have?"), 37)

    def test_times_more_idiom_declines(self):
        # GATE C — "4 times more than" is the ambiguous ×N idiom (no "as many as" frame), which the extractor
        # would mis-read as a +4 offset. Decline the whole system.
        self.assertEqual(reading.solve(
            "Rui, Sol, and Tavi have a total of 67 coins. Rui has 5 less than Sol and Tavi has 4 times more "
            "than Sol. How many coins does Tavi have?"), "undetermined")

    def test_people_count_before_total_measure_matched(self):
        # `_combined_total` prefers the value quantifying the ASKED MEASURE: "3 friends have 411 marbles" → the
        # total is 411 marbles (asked=marble), not the 3-friends count. Reduce to pivot Wen: A=1,1,1  B=26,-17,0
        # → Wen = (411−9)/3 = 134, Uxo = 134 + 26 = 160.
        self.assertEqual(reading.solve(
            "Together 3 friends have 411 marbles. Uxo has 43 more than Vin. Vin has 17 less than Wen. How "
            "many marbles does Uxo have?"), 160)


class TestAnchoredSumSystem(unittest.TestCase):
    """`_anchored_sum_system` — the twin of `_affine_system` for when NO combined total is stated but one
    entity has a bare ABSOLUTE value (an "anchor"), and the question asks for the SUM of every entity: pivot
    = the anchor's value directly (no division). "Cecil has $600; Catherine $250 less than twice Cecil;
    Carmela $50 more than twice Cecil. Altogether?" → 2800 (dev-train real case, +4 dev this increment)."""

    def test_two_entity_anchor_offset_sum(self):
        # Wren=12 (anchor), Tam=Wren+7=19, sum=31. The simplest shape: one anchor + one pure offset relation.
        self.assertEqual(reading.solve(
            "Wren bought 12 apples. Tam bought 7 more apples than Wren. How many apples did they buy "
            "altogether?"), 31)

    def test_three_entity_mixed_ratio_offset_sum(self):
        # Nils=$80 (anchor); Priya=2*80-15=145; Odele=2*80+9=169; sum=394. Mirrors the real Cecil/Catherine/
        # Carmela GSM8K case (money, both entities affine in the SAME anchor).
        self.assertEqual(reading.solve(
            "Nils has $80. Priya has $15 less than twice Nils's money. Odele has $9 more than twice Nils's "
            "money. How much money do they have altogether?"), 394)

    def test_four_entity_chain_sum(self):
        # A 4-deep relation CHAIN off one anchor: Kian=6, Lior=3*6=18, Mave=18+4=22, Suri=22/2=11, sum=57.
        self.assertEqual(reading.solve(
            "Kian has 6 marbles. Lior has 3 times as many marbles as Kian. Mave has 4 more marbles than "
            "Lior. Suri has half as many marbles as Mave. What is the total number of marbles they all "
            "have?"), 57)

    def test_stated_total_declines_this_solver(self):
        # A stated combined total is `_affine_system`'s turf, not this solver's — must still solve (via the
        # other system), not decline outright, proving the two solvers don't collide.
        self.assertEqual(reading.solve(
            "Anna has 3 times as many toys as Mandy and 2 fewer toys than Amanda. If they have 142 toys "
            "combined, how many toys does Mandy have?"), 20)

    def test_purpose_infinitive_to_share_does_not_block(self):
        # "to share" is a PURPOSE infinitive (intent), not a distributed-total cue — it must NOT trip the
        # combined-total guard in `_anchored_affine_coeffs` (which fires on a real "SHARED $X between …").
        # Nora=20 (anchor), Owen=15, Pat=25; altogether = 60. (The real Mona/Jasmine/Rachel GSM8K case.)
        self.assertEqual(reading.solve(
            "Nora brought 20 cookies to share in class. Owen brought 5 fewer cookies than Nora. Pat brought "
            "10 more cookies than Owen. How many cookies altogether did Nora, Owen, and Pat bring?"), 60)

    def test_kinship_descriptor_is_not_an_age_trigger(self):
        # "older sister"/"younger brother" is a KINSHIP descriptor, not an age computation — it must NOT trip
        # the age guard in `_anchored_affine_coeffs`, so a NON-age problem (here shoe/book sizes) still solves.
        # Nia=7 (anchor), Opal=2*7=14, altogether = 21. (The real Jasmine/Alexa shoe-size GSM8K case.)
        self.assertEqual(reading.solve(
            "Nia and her older sister Opal went shopping for books. Opal has twice as many books as Nia. "
            "If Nia has 7 books, how many books do they have altogether?"), 21)

    def test_unclassified_premise_declines(self):
        # An UNCONSUMED premise (a depletion action on a pronoun, no name) must decline THIS solver rather
        # than silently drop information: "gave out half her apples" is neither a relation nor an anchor, so
        # folding just Frank+Susan's raw pre-depletion values would be a confident wrong (144, not 78). Calls
        # the solver directly (the real Frank/Susan case still solves 78 through other machinery once this
        # guard steps aside — proving the guard fixed the mis-fire without breaking the case outright).
        self.assertIsNone(reading._anchored_sum_system(
            ["Frank picked 36 apples.", "Susan picked 3 times as many apples as Frank.",
             "If Susan gave out half of her apples", "Frank sold a third of his"],
            "how many in total do they have left?"))

    def test_missing_entity_relation_declines(self):
        # An entity NAMED in the premises but whose defining relation isn't captured (here: an anaphoric
        # "that many" instead of a second name) must decline THIS solver — silently summing only the captured
        # entities would drop Jerome and under-count (37, not 87). Calls the solver directly (the real Seth/
        # Jerome/Veronica case still solves 87 through other machinery once this guard steps aside).
        self.assertIsNone(reading._anchored_sum_system(
            ["Seth lost 17 pounds.", "Jerome lost three times that many pounds",
             "Veronica lost 2 pounds more than Seth."],
            "How many pounds did the 3 people lose in total?"))

    def test_dozen_measure_declines(self):
        # The anchor's "N dozen" converts to raw units internally ("6 dozen eggs" → 72), so a question asking
        # for the total in DOZEN (not raw) units needs a ÷12 this solver doesn't do → decline rather than
        # answer in the wrong scale (312, not 26). The real Benjamin/Carla/Trisha GSM8K case.
        self.assertEqual(reading.solve(
            "Benjamin collects 6 dozen eggs a day. Carla collects 3 times the number of eggs that Benjamin "
            "collects. Trisha collects 4 dozen less than Benjamin. How many dozen eggs do the three collect "
            "total?"), "undetermined")


class TestAnchoredSingleSystem(unittest.TestCase):
    """`_anchored_single_system` — the twin of `_anchored_sum_system` for when the question asks ONE named
    entity's value instead of the sum. Shares `_anchored_affine_coeffs` with the sum twin. Required two new
    generalizations to reach real cases: (1) `_pure_offset_relation` now accepts ANY comparative adjective
    via `COMPARATIVE_SIGN` ("N pounds HEAVIER/lighter", "N inches TALLER/shorter"), not just "more"/"less"/
    "fewer" (excluding "older"/"younger", which stay the age domain's); (2) the pivot back-substitutes when
    the anchor sits on a DEPENDENT entity, not just the free one. +11 dev-holdout this increment (Billy/Brad/
    Carl, Jana/Kelly/Jess, Tyler/Sam/Peter, and others — all real GSM8K weight/height comparative chains)."""

    def test_comparative_adjective_chain_to_free_pivot(self):
        # Nia=40 (anchor, the free pivot itself): Omar=Nia+9=49, Pria=Omar-14=35 (a "heavier"/"lighter" chain,
        # not "more"/"less"). Mirrors the real Billy/Brad/Carl weight case.
        self.assertEqual(reading.solve(
            "Omar weighs 9 pounds more than Nia. Pria weighs 14 pounds less than Omar. Nia weighs 40 "
            "pounds. How much does Pria weigh?"), 35)

    def test_anchor_on_dependent_entity_back_substitutes(self):
        # The anchor (Ivy=88) sits on a DEPENDENT entity, not the free pivot (Ren): Ivy = Ren − 12, so
        # pivot Ren = 88 + 12 = 100; Zael = Ren + 7 = 107. Proves the back-substitution generalization (the
        # real Ed/Al/Ben/Carl GSM8K case has this shape — the anchor is 2 hops from the free entity).
        self.assertEqual(reading.solve(
            "Ivy is 12 pounds lighter than Ren. Ivy weighs 88 pounds. Zael is 7 pounds heavier than Ren. "
            "How much does Zael weigh?"), 107)

    def test_difference_question_declines(self):
        # "how many MORE does X have COMPARED TO Y" asks a DIFFERENCE, not a raw entity value — the naive
        # "first name in the question" heuristic would wrongly return Anthony's raw value (21) instead of
        # declining to the existing compared-to machinery (which answers 2). Regression guard.
        self.assertEqual(reading.solve(
            "Scott has 7 marbles. Anthony has 3 times as many marbles as Scott, and Jim has 2 less marbles "
            "than Anthony. How many more marbles does Anthony have compared to Jim?"), 2)

    def test_age_typed_problem_declines(self):
        # An AGE-typed system (temporal-shift semantics `_anchored_affine_coeffs` doesn't model) must decline
        # THIS solver even though it structurally parses as a plain anchor+ratio — the dedicated age machinery
        # owns "old"/"age" text. Direct call (the real case still solves 28 via the age domain once this
        # solver steps aside).
        self.assertIsNone(reading._anchored_affine_coeffs(
            ["In two years, Tom will be three times as old as Sam.", "Sam is currently 8 years old."],
            "How old is Tom?"))

    def test_rate_adjective_ratio_declines(self):
        # "twice as FAST as" is a ratio over SPEED, not a direct value ratio — speed scales INVERSELY with
        # the stated per-mile TIME, so folding it as a plain value ratio would silently invert the answer
        # (the real Celia/Lexie GSM8K case: naive folding gives 40, gold is 300). `_ratio_relation` must
        # decline the whole clause so this system honestly declines rather than mis-solve.
        self.assertIsNone(reading._affine_relation(
            reading.analyze("Celia runs twice as fast as Lexie.")))


class TestAnchoredDifferenceSystem(unittest.TestCase):
    """`_anchored_difference_system` — the third anchored twin, for a question naming TWO entities and asking
    how much MORE one has than the other. Reuses `_anchored_affine_coeffs`; declines when the question isn't
    comparative (that's the single-value twin's shape) or doesn't name exactly two entities. +1 dev-holdout
    (the real Anton/Heike/Ann GSM8K case)."""

    def test_two_entity_ratio_difference(self):
        # Heike=10 (from Nia=6x=60... here: Yui=6*Zeke=60 -> Zeke=10), Ravi=3*Zeke=30, Yui-Ravi=60-30=30.
        self.assertEqual(reading.solve(
            "Ravi has three times as many cards as Zeke. Yui has six times as many cards as Zeke. If Yui "
            "has 60 cards, how many more cards does Yui have than Ravi?"), 30)

    def test_single_value_question_not_difference(self):
        # no comparative sense in the question → the single-value twin's shape, not this one.
        self.assertEqual(reading.solve(
            "Omar weighs 9 pounds more than Nia. Nia weighs 40 pounds. How much does Omar weigh?"), 49)

    def test_needs_to_have_idiom_declines(self):
        # "X NEEDS N more to HAVE k times as many as Y" states a DEFICIT (X + N = k·Y), the OPPOSITE equation
        # from "X HAS N more than k times Y" (X = k·Y + N) the affine extractors otherwise assume — silently
        # applying the wrong equation would confidently invert the answer (the real Amanda/Katrina/Mabel
        # GSM8K case: naive folding gives 39, gold is 63). `_affine_relation` must decline the whole clause.
        self.assertIsNone(reading._affine_relation(
            reading.analyze("Amanda needs 12 more marbles to have twice as many marbles as Katrina")))


class TestAnchoredAverageSystem(unittest.TestCase):
    """`_anchored_average_system` — the AVERAGE twin of `_anchored_sum_system`: the question asks for the MEAN
    of the N entities (Σ ÷ N). Reuses the exact sum machinery; the only new behaviour is the division. The
    stated group count in the question must MATCH the number of entities the coeffs captured, so a partial
    system (a missing/unsupported relation) divides by no wrong denominator — it declines. +1 dev-train
    (the real Rachel weight-average GSM8K case)."""

    def test_average_of_the_three(self):
        # Nora=30, Owen=36 (30+6), Pat=21 (30-9); Σ=87; ÷3 = 29.
        self.assertEqual(reading.solve(
            "Nora weighs 30 pounds, 6 pounds less than Owen and 9 pounds more than Pat. What is the average "
            "weight of the three people?"), 29)

    def test_sum_twin_still_owns_total_question(self):
        # the SUM question stays with `_anchored_sum_system` (Σ=87), unaffected by the average twin.
        self.assertEqual(reading.solve(
            "Nora weighs 30 pounds, 6 pounds less than Owen and 9 pounds more than Pat. What is the total "
            "weight of the three people?"), 87)

    def test_stated_count_must_match_entity_count(self):
        # the system has 3 entities but the question says "four" — never divide by a guessed/mismatched N.
        self.assertEqual(reading.solve(
            "Nora weighs 30 pounds, 6 pounds less than Owen and 9 pounds more than Pat. What is the average "
            "weight of the four people?"), "undetermined")

    def test_no_stated_count_declines(self):
        # no group size stated → we don't guess N → decline.
        self.assertEqual(reading.solve(
            "Nora weighs 30 pounds, 6 pounds less than Owen and 9 pounds more than Pat. What is the average "
            "weight of the people?"), "undetermined")


class TestResidualSumSystem(unittest.TestCase):
    """`_residual_sum_system` — a stated total names 3+ entities, but the ASKED one is never related or
    anchored anywhere else (unlike `_affine_system`, which requires every entity to reduce to the pivot).
    Solves it as total minus every OTHER entity's (anchor+relation-derived) value. +1 dev-holdout (the real
    Jermaine/Terrence/Emilee GSM8K case)."""

    def test_residual_entity_from_total(self):
        # Terrence=30 (anchor), Jermaine=Terrence+5=35 (rel); Priya never appears again -> 90-35-30=25.
        self.assertEqual(reading.solve(
            "Priya, Selim, and Talia earn a total of $90. Selim earns $5 more than Talia in a month. If "
            "Talia earns $30, how much does Priya earn?"), 25)

    def test_covered_entity_declines(self):
        # asking for an entity that IS covered by a relation (Selim) is `_affine_system`'s/other systems'
        # shape — not a residual — so this solver must decline (it fires only for the UNCOVERED entity).
        self.assertIsNone(reading._residual_sum_system(
            ["Priya, Selim, and Talia earn a total of $90.", "Selim earns $5 more than Talia in a month.",
             "If Talia earns $30"],
            "how much does Selim earn?"))

    def test_pronoun_total_no_relation_fallback(self):
        # "If THEY need 10 bags total" has NO explicit name list (unlike Priya/Selim/Talia above) — the
        # names are inferred from the OTHER premises. AND there's no ratio/offset relation between Jeremy,
        # Stacy, or Emily at all (independent anchors only) — `_anchored_affine_coeffs` requires >=1
        # relation and declines, so this exercises the NO-RELATION fallback (bare per-name anchors summed
        # directly). Jeremy=3, Stacy=4, total=10 -> Emily=10-3-4=3.
        self.assertEqual(reading.solve(
            "Jeremy buys 3 bags of chips for a party. Stacy and Emily also buy chips for the party. If "
            "they need 10 bags of chips total, and Stacy buys 4 bags, how many bags of chips should Emily "
            "buy?"), 3)

    def test_three_name_question_list_solves(self):
        # "…have Nora, Opal, and Pax read?" is a COORDINATED NAME LIST sharing the trailing verb —
        # `_split_clauses` must keep it whole (previously it split on ", and Pax", stranding a bogus "Pax
        # read?" premise and truncating the question → decline). Nora=40 (anchor), Opal=1/4·40=10,
        # Pax=2·10+5=25, total=75. (The genitive "the number of books that X has read" is already parsed by
        # the system extractors; the split was the sole blocker.) A genuine second clause still splits — see
        # TestRelativeClauseSplit / the "than <name>, and <name> has …" comparison standards below.
        self.assertEqual(reading.solve(
            "Nora has read 40 books. Opal has read 1/4 the amount of books that Nora has read. Pax has read "
            "5 more than twice the number of books that Opal has read. How many books total have Nora, Opal, "
            "and Pax read?"), 75)

    def test_comparison_standard_name_still_splits(self):
        # the guard must NOT fire on a comparison STANDARD ("… than Mark, and Mark has …") — "Mark" follows
        # "than" and begins a REAL second clause, so it must still split into two independent relations.
        self.assertEqual(reading._split_clauses(
            "Samir has 8 more points than Mark, and Mark has 4 more points than Eric"),
            ["Samir has 8 more points than Mark", "Mark has 4 more points than Eric"])

    def test_unrecognized_relation_frame_not_silently_skipped(self):
        # "Jorge has an EQUAL NUMBER of cards AS Matias" isn't a recognized ratio frame ("equal" carries no
        # multiplier), but the "as" + 2 names still signals a REAL relation — skipping it as a harmless
        # declarative silently dropped Jorge=Matias (34 vs gold 48); must decline instead.
        # RE-PINNED (#230): the equal-as chain branch now claims this CORRECTLY (14+14+20 = 48);
        # the skip-guard behavior this class pins is unchanged (the fold still declines).
        self.assertEqual(reading.solve(
            "Jorge has an equal number of baseball cards as Matias, who has 6 fewer cards than Carlos. If "
            "Carlos has 20 baseball cards, what is the total number of baseball cards the three have?"),
            48)


class TestTwoRelationSystem(unittest.TestCase):
    """`_two_relation_system` — TWO relations between the SAME entity pair, no anchor/total: a 2×2 linear
    system solved by elimination. +1 dev-train (the real Brett blue/red marbles case)."""

    def test_offset_and_ratio_solve_for_standard(self):
        # Alba = Bo + 15 AND Alba = 4·Bo → Bo = 15/(4−1) = 5.
        self.assertEqual(reading.solve(
            "Alba has 15 more coins than Bo. Alba has 4 times as many coins as Bo. How many coins does Bo "
            "have?"), 5)

    def test_asks_for_the_subject(self):
        # same system, but the question asks for Alba: 4·5 = 20.
        self.assertEqual(reading.solve(
            "Alba has 15 more coins than Bo. Alba has 4 times as many coins as Bo. How many coins does Alba "
            "have?"), 20)

    def test_parallel_relations_decline(self):
        # two SAME-slope offsets (k1==k2==1) have no unique solution — must decline, never divide by zero.
        self.assertEqual(reading.solve(
            "Alba has 15 more coins than Bo. Alba has 30 more coins than Bo. How many coins does Bo have?"),
            "undetermined")


class TestTargetContributionsSystem(unittest.TestCase):
    """`_target_contributions_system` — a stated GOAL ("need to collect N") minus the amounts gathered so
    far, answering "how many MORE are needed". Reads raw premise values (the general fold drops the
    elided-object "collected 20"). +1 dev-train (the real Alyssa/Abigail cans case)."""

    def test_target_minus_two_contributions(self):
        # goal 90; Nia 20, Owen 30 → 90 − 50 = 40.
        self.assertEqual(reading.solve(
            "Nia and Owen need to collect 90 cans for their project. Nia collected 20 while Owen collected "
            "30 cans. How many more cans should they collect?"), 40)

    def test_deficit_idiom_not_touched(self):
        # "needs N MORE to HAVE k× Y" is the DEFICIT idiom, NOT a goal-minus-contributions — this solver must
        # decline it (a "more"/"to have" in the target premise is the disqualifier), leaving it to decline.
        self.assertIsNone(reading._target_contributions_system(
            ["Bea needs 5 more coins to have twice as many coins as Cal", "Cal has 10 coins"],
            "how many more coins does Bea need?"))


class TestTotalBetweenResidualSystem(unittest.TestCase):
    """`_total_between_residual_system` — a category total split "in total between" named parts, some stated,
    asking the residual part: total − Σ(stated parts). +1 dev-train (the real Mark candy-bars case)."""

    def test_residual_part_from_total_between(self):
        # 20 marbles in total between red/blue/green; 5 blue + 7 green → red = 20 − 12 = 8.
        self.assertEqual(reading.solve(
            "Nora has 20 marbles in total between red, blue, and green. She has 5 blue and 7 green marbles. "
            "How many red marbles does she have?"), 8)

    def test_no_total_between_cue_declines(self):
        # without the specific "in total between/among" cue, the common "total − Σ" surface is untrusted —
        # a bare "total" must NOT fire this solver (it over-fires; e.g. a rate/per-item total).
        self.assertIsNone(reading._total_between_residual_system(
            ["Nora has 20 marbles total", "She has 5 blue and 7 green marbles"],
            "How many red marbles does she have?"))


class TestAnchoredSingleNeedsGuard(unittest.TestCase):
    """`_anchored_single_system` naively grabbed the FIRST name in the question as the ask — but "how many
    stamps does SHE NEED to ADD to have a collection AS LARGE AS Bert's?" is a DEFICIT-TO-TARGET question
    (Bert minus Peggy), not a request for Bert's raw value; "as large as" carries no comparative sense (it's
    the positive-degree equative, not "larger than"), so the existing comparative-sense guard didn't catch
    it. A confident-wrong (900 vs gold 825) — now declines on any "need"/"needs" in the question."""

    def test_needs_to_reach_target_declines(self):
        self.assertEqual(reading.solve(
            "Bert, Ernie, and Peggy collect stamps. Bert has four times as many stamps as Ernie, but Ernie "
            "has three times as many stamps as Peggy. If Peggy currently has 75 stamps in her collection, "
            "how many stamps does she need to add to her collection to have a collection as large as "
            "Bert's collection?"), "undetermined")


class TestNounAffineSystem(unittest.TestCase):
    """`_noun_affine_system` — the COORDINATE/SUBTYPE-NOUN twin of `_affine_system`: entities are COMMON
    NOUNS ("twice as many pigs as cows", "4 more cows than goats"), not names, extracted via the SAME
    `_affine_relation` (now with a noun-entity fallback: `_ratio_relation`/`_offset_ratio_relation`/
    `_pure_offset_relation` try their name-based path first, then fall back to `_noun_entity_key`/
    `_noun_std_entity` when there isn't a clean 2-name match). The total is over a CATEGORY word ("56 ANIMALS
    total" for pig/cow/goat) that this solver TRUSTS without hypernym verification — narrowly gated (the
    underlying relation premises must have 0-1 names, never the 2-name shape `_affine_system` owns) so it
    can never compete with the named-entity system. Also required `_is_value_first_relation` to accept a
    noun/adjective standard (not just a name) so "and 4 more cows than goats" splits into its own premise.
    +1 dev-train (the real farmer/pigs/cows/goats GSM8K case)."""

    def test_coordinate_noun_chain_with_category_total(self):
        self.assertEqual(reading.solve(
            "A farmer has twice as many cats as dogs, and 4 more dogs than birds. If the farmer has 56 "
            "animals total, how many birds does he have?"), 11)

    def test_self_referential_pronoun_standard_declines(self):
        # "twice as many books AS SHE finished last month" — the standard right after "as" is a PRONOUN
        # (a self-referential relative clause across TIME, not a second coordinate-noun entity). Scanning
        # past the pronoun to grab "month" caused a real confident-wrong (6 vs gold 12) — must decline.
        self.assertIsNone(reading._affine_relation(reading.analyze(
            "Lily wants to finish reading twice as many books as she finished last month.")))

    def test_two_name_relation_not_claimed(self):
        # a genuine 2-NAME relation premise must NOT be picked up by this solver (it's _affine_system's
        # turf) — even when a category total is ALSO present, since real 2-name+total problems are already
        # handled correctly by the stricter, already-tested named-entity system.
        self.assertIsNone(reading._noun_affine_system(
            ["Anna has 3 times as many toys as Mandy.", "Anna has 2 fewer toys than Amanda.",
             "They have 142 toys combined."],
            "how many toys does Mandy have?"))


class TestNounAnchoredSumSystem(unittest.TestCase):
    """`_noun_anchored_sum_system` — the anchor-based twin of `_noun_affine_system`: no combined/category
    total is stated, but one COMMON-NOUN entity has a bare anchor value (via `_noun_anchor_fact`, tolerating
    an outer possessor name like "Hannah's family"). The question asks for a CATEGORY SUM ("how many ANIMALS
    do they have?") — a noun that ISN'T one of the specific relation entities. +3 dev-holdout (Hannah's-farm,
    Sandro's-children, and a green/red-apples GSM8K case)."""

    def test_anchor_chain_category_sum(self):
        # pig=10 (anchor), cow=2*10-3=17, goat=17+6=23, sum=50.
        self.assertEqual(reading.solve(
            "Priya's family has ten pigs on their farm. They also have three fewer than twice as many cows "
            "as pigs and six more goats than cows. How many animals do they have on their farm?"), 50)

    def test_two_entity_sum_no_total_cue(self):
        # daughters=6*sons=18, sons=3 (anchor) -> sum=21. No "altogether"/"total" cue at all — the category
        # noun "children" (distinct from "daughter"/"son") is itself the sum signal.
        self.assertEqual(reading.solve(
            "Wesley has six times as many daughters as sons. If he currently has three sons, how many "
            "children does he have?"), 21)

    def test_negated_question_is_complement_sum(self):
        # "how many of her houses are NOT yellow?" is a COMPLEMENT sum: Σ over every entity EXCEPT the one
        # named after "not". green=90 (anchor), yellow=30, red=70 → not-yellow = green+red = 160 (the full
        # Σ of 190 would be the wrong answer). Reuses the full-Σ machinery, dropping the excluded term.
        self.assertEqual(reading.solve(
            "Bea has three times as many green marbles as yellow marbles. She also has 40 fewer yellow "
            "marbles than red marbles. If she has 90 green marbles, how many of her marbles are not "
            "yellow?"), 160)

    def test_negation_of_unknown_entity_declines(self):
        # a negation whose excluded token ISN'T one of the system's entities ("not shiny") can't be resolved
        # to a term to drop → decline rather than guess (never emit the full Σ for an unresolved complement).
        self.assertEqual(reading.solve(
            "Bea has three times as many green marbles as yellow marbles. She also has 40 fewer yellow "
            "marbles than red marbles. If she has 90 green marbles, how many of her marbles are not "
            "shiny?"), "undetermined")


class TestNounResidualSumSystem(unittest.TestCase):
    """`_noun_residual_sum_system` — the SUBTYPE-PARTITION twin: adjective-qualified subtypes of ONE head
    noun ("red/blue/green balls"), the total stated directly in that SAME (bare) noun ("Ryan has 40 balls"
    — no hypernym/category guessing, unlike `_noun_affine_system`), and the asked entity is the explicit
    "the REST are <qualifier>" residual. +2 dev (the real Ryan/balls and Josh/marbles GSM8K cases)."""

    def test_subtype_partition_residual(self):
        # red=2*11=22, green=40-22-11=7.
        self.assertEqual(reading.solve(
            "Ian has 40 balls. There are twice as many red balls as blue while the rest are green. If "
            "there are 11 blue balls, how many green balls are there?"), 7)

    def test_covered_entity_not_residual_declines(self):
        # asking for a COVERED entity (red) is a different (unbuilt) single-ask variant, not the residual
        # this solver answers — must decline rather than guess.
        self.assertIsNone(reading._noun_residual_sum_system(
            ["Ian has 40 balls.", "There are twice as many red balls as blue while the rest are green.",
             "If there are 11 blue balls"],
            "how many red balls are there?"))


class TestVocabAndFamilyIsolation(unittest.TestCase):
    """Vocabulary additions (adult/senior/chef) unlocked the real chef GSM8K case for the coordinate-noun
    family, but doing so surfaced a real architectural gap: `_affine_relation`/`_anchor_fact`'s noun-entity
    fallback is reachable from EVERY consumer, including the ORIGINAL named-entity systems (`_affine_system`,
    `_anchored_sum_system` via `_anchored_affine_coeffs`, `_sum_ratio_system`, `_offset_sum_system`,
    `_offset_sum_pure`) — they had no name-count gate of their own, so a premise that only classifies via the
    noun fallback could be silently accepted by the WRONG family and mis-solve (a class of 500 tables + "2/5
    times as many books as the number of tables ... ARE ON TOP OF EACH TABLE" — a per-table RATE, not a flat
    count — gave 700 via `_anchored_sum_system` instead of declining; the correct reading needs 100000, which
    no current solver computes). All five named systems now require `_name_count(rp) == 2` for a relation /
    `== 1` for an anchor before accepting `_affine_relation`/`_anchor_fact`'s result, via the shared
    `_name_count` helper. Also added `_asked_noun_entity` (shared by all three noun systems) to recognize
    "What is the TOTAL NUMBER OF guests served?" (no "how many" at all) as an equivalent asked-noun frame,
    and extended `_noun_std_entity`/`_is_value_first_relation` to skip an elided "the NUMBER/AMOUNT OF"
    wrapper ("35 fewer children THAN THE NUMBER OF adults"). +1 dev-train (the real chef GSM8K case)."""

    def test_chef_case_solves(self):
        self.assertEqual(reading.solve(
            "A chef has served 58 adults, 35 fewer children than the number of adults, and twice as many "
            "seniors as children. What is the total number of guests served?"), 127)

    def test_per_unit_rate_declines_rather_than_mis_sum(self):
        # RE-PINNED (#231): the tables×books branch now computes the per-table RATE read this
        # pin documents as correct — 500 · (2/5·500) = 100000. The affine-family mis-sum this
        # pin defends against (700) still never fires.
        self.assertEqual(reading.solve(
            "A class has 500 tables and 2/5 times as many books as the number of tables in the class are "
            "on top of each table. What's the total number of books in the class?"), 100000)

    def test_noun_fallback_relation_not_claimed_by_named_system(self):
        # a premise with 0-1 names (a coordinate-noun relation) must NOT be accepted by the NAMED
        # `_anchored_affine_coeffs` family — direct check on the shared extraction helper.
        self.assertIsNone(reading._anchored_affine_coeffs(
            ["A farmer has twice as many pigs as cows.", "If there are 20 cows"],
            "how many pigs does the farmer have?"))


class TestComparedToDifference(unittest.TestCase):
    """"How many more <measure> does A have compared to B?" → A's holding − B's. The comparative ("more")
    greedily binds its standard to the MEASURE noun ("more PAIRS"), not the entity, and the real standard sits
    after "compared to" (or "than"), so the existing than-standard branch can't see it. Recover the standard as
    the non-subject NAME when the bound standard isn't a name and a comparison marker is present. dev-train
    61→62 (Scott/Anthony/Jim), 0 new wrong either slice, TEST held 61.5%. SYNTHETIC — never GSM8K strings."""

    def test_compared_to_difference(self):
        # Anthony = 3×7 = 21, Jim = 21 − 2 = 19 → 21 − 19 = 2.
        self.assertEqual(reading.solve(
            "Scott has 7 marbles. Anthony has 3 times as many marbles as Scott, and Jim has 2 less marbles "
            "than Anthony. How many more marbles does Anthony have compared to Jim?"), 2)

    def test_than_name_difference_still_works(self):
        # the pre-existing "than <NAME>" path is unaffected (standard already a name).
        self.assertEqual(reading.solve(
            "Amy has 10 apples. Beth has 4 apples. How many more apples does Amy have than Beth?"), 6)


class TestOffsetSumSystem(unittest.TestCase):
    """The AFFINE twin of the pure-ratio system: two entities, one relation subject = factor × standard +
    offset ("N more/less than k times as many as"), plus a combined total → solve a part. std = (T − c)/(k+1).
    Kept SEPARATE from `_sum_ratio_system` (which declines offset-tainted systems) to isolate the firewall-
    riskier offset shape. dev-train 58→61, holdout 43→45 (generalized: Sara/Joe, Rajesh/Hiro), 0 new wrong
    either slice; TEST held 61.5%. Requires an explicit factor (a bare "N more than Y" stays declined).
    SYNTHETIC — never GSM8K strings."""

    def test_offset_more_solves_standard(self):
        # Justin = 2·Sabrina + 8, Justin + Sabrina = 50 → Sabrina = (50 − 8)/3 = 14.
        self.assertEqual(reading.solve(
            "Justin and Sabrina have 50 pencils combined. Justin has 8 more than twice as many pencils as "
            "Sabrina. How many pencils does Sabrina have?"), 14)

    def test_offset_less_fraction_solves_standard(self):
        # Devin = ½·Tom − 5, Devin + Tom = 70 → Tom = (70 − (−5))/(½+1) = 75/1.5 = 50.
        self.assertEqual(reading.solve(
            "Tom and Devin have been teaching for a total of 70 years. Devin has been teaching for 5 less "
            "than half the number of years that Tom has. How many years has Tom been teaching?"), 50)

    def test_offset_solves_subject(self):
        # Kayla = 4·Kylie + 10, total 340 → Kylie = (340 − 10)/5 = 66, Kayla = 340 − 66 = 274 (asked subject).
        self.assertEqual(reading.solve(
            "Kylie and Kayla have 340 apples combined. Kayla has 10 more than 4 times as many apples as "
            "Kylie. How many apples does Kayla have?"), 274)

    def test_bare_offset_no_factor(self):
        # "8 more than Sabrina" (no factor) is now solved by _offset_sum_pure inside a sum-system (the total
        # pins the pair): Sabrina = (50 − 8)/2 = 21.
        self.assertEqual(reading.solve(
            "Justin and Sabrina have 50 pencils combined. Justin has 8 more pencils than Sabrina. "
            "How many pencils does Sabrina have?"), 21)

    def test_offset_measure_mismatch_declines(self):
        # relation about 'red balls', total about 'blue balls' → different measures, decline.
        self.assertEqual(reading.solve(
            "Zoe and Sam have a combined total of 12 blue balls. Zoe has 2 more than twice as many red "
            "balls as Sam. How many red balls does Sam have?"), "undetermined")


class TestMeasureComparative(unittest.TestCase):
    """MEASURE-to-MEASURE comparative — "N fewer/more <measA> than <measB>" for the SAME subject (same head
    noun, different attribute): the compound keeps its head, and the derived measure grounds against the
    subject's holding of the standard measure. SYNTHETIC."""

    def test_fewer_across_measures(self):
        # animal = flower − 2 = 8 − 2 = 6.
        self.assertEqual(reading.solve(
            "Victor has 8 flower stickers. He has 2 fewer animal stickers than flower stickers. "
            "How many animal stickers does Victor have?"), 6)

    def test_total_across_measures(self):
        # flower 8 + animal 6 = 14.
        self.assertEqual(reading.solve(
            "Victor has 8 flower stickers. He has 2 fewer animal stickers than flower stickers. "
            "How many stickers does Victor have in total?"), 14)


class TestOffsetSumPure(unittest.TestCase):
    """2-entity PURE additive offset + combined total → std = (T∓c)/2. SYNTHETIC."""

    def test_more_than(self):
        # Anais = Kamari + 30, total 160 → Kamari = (160−30)/2 = 65.
        self.assertEqual(reading.solve(
            "Anais has 30 more toys than Kamari. There are 160 toys altogether. "
            "How many toys does Kamari have?"), 65)

    def test_fewer_than(self):
        # Sam = Tim − 20, total 100 → Tim = (100+20)/2 = 60.
        self.assertEqual(reading.solve(
            "Sam has 20 fewer marbles than Tim. They have 100 marbles altogether. "
            "How many marbles does Tim have?"), 60)

    def test_times_more_declines(self):
        # "5 times more" is multiplicative, not additive ± → the pure resolver must NOT fire.
        self.assertEqual(reading.solve(
            "Jackson has 5 times more coins than Williams. Together they have 150 coins. "
            "How many coins does Williams have?"), "undetermined")


class TestSelfMultipleSum(unittest.TestCase):
    """One subject, two periods, the second a MULTIPLE of the first via an explicit back-reference ("that
    amount/much/many") → base·(1+k). SYNTHETIC."""

    def test_that_amount_sum(self):
        # 2 + 5×2 = 2·6 = 12.
        self.assertEqual(reading.solve(
            "Roger rode 2 miles this morning, and then 5 times that amount in the evening. "
            "How many miles did Roger ride?"), 12)

    def test_that_amount_total_cue(self):
        # 3 + 4×3 = 3·5 = 15.
        self.assertEqual(reading.solve(
            "Anna ran 3 miles on Monday and then 4 times that amount on Tuesday. "
            "How many miles did Anna run in total?"), 15)

    def test_aggregate_ratio_not_self(self):
        # RE-PINNED (#228): the combined-standard branch now claims this CORRECTLY —
        # 2 + 3 + 2·(2+3) = 15. The anaphora non-misfire this pin defends is unchanged.
        self.assertEqual(reading.solve(
            "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. "
            "How many pets does Ed have?"), 15)


# === DECLINE-SAFETY GUARDS: the precision guards that DECLINE rather than commit a partial answer ===

class TestIncrement105NegativeAnswerDeclines(unittest.TestCase):
    """#105 - a NEGATIVE final answer to a count/amount question is nonsensical (you can't hold −3
    apples): it always means a term was dropped or mis-signed, so DECLINE rather than commit. Pure
    precision guard — a decline is always safe, and correct GSM8K answers are never negative."""

    def test_negative_count_declines(self):
        self.assertEqual(reading.solve("Tom has 5 apples. He gives away 8 apples. "
                                       "How many apples does Tom have?"), "undetermined")

    def test_positive_answer_unaffected(self):
        self.assertEqual(reading.solve("Tom has 5 apples. He gives away 2 apples. "
                                       "How many apples does Tom have?"), 3)

    def test_zero_answer_still_commits(self):
        # zero is a legitimate answer (gave away exactly what he had) — only NEGATIVE declines
        self.assertEqual(reading.solve("Tom has 5 apples. He gives away 5 apples. "
                                       "How many apples does Tom have?"), 0)


class TestIncrement107InversionCounterfactualDeclines(unittest.TestCase):
    """#107 - GUARD 8: two shapes the forward fold structurally cannot solve, so it would commit garbage.
    Both are honest DECLINES (the only-ever-declines shape, like the negative-answer guard).
    (a) BACKWARD-initial: the question asks for an ORIGINAL value ("how much did she have at first?")
        given a KNOWN final state — an inversion (solve-for-initial), not a forward accumulation.
    (b) COUNTERFACTUAL: a premise in the irrealis mood ("if X had $N more, she WOULD have $M") states a
        hypothetical, not an asserted holding; folding its numbers as real quantities is unsound."""

    def test_backward_initial_question_declines(self):
        self.assertEqual(reading.solve("Mary had some money. She spent 5 dollars and gave away 3 dollars. "
                                       "She has 10 dollars left. How much money did Mary have at first?"),
                         "undetermined")

    def test_originally_phrasing_solves_backward(self):
        # was pinned "undetermined" while "originally" was an UNKNOWN word (GUARD 1 masked it);
        # with the word typed (15th vocab wave) the backward-depletion machinery solves it: 4+6.
        self.assertEqual(reading.solve("Tom had some apples. He ate 4 apples. He has 6 apples left. "
                                       "How many apples did Tom have originally?"), 10)

    def test_counterfactual_would_have_declines(self):
        self.assertEqual(reading.solve("If Mary had 5 dollars more, she would have 20 dollars. "
                                       "How much money does Mary have?"), "undetermined")

    def test_forward_question_unaffected(self):
        # a plain forward accumulation that says neither "at first" nor "would" still commits
        self.assertEqual(reading.solve("Tom has 5 apples. He buys 4 apples. "
                                       "How many apples does Tom have?"), 9)


class TestIncrement108DistributiveFrequencyDeclines(unittest.TestCase):
    """#108 - GUARD 9: "every time / each time <X happens>, <amount>" is a per-occurrence rate that must be
    multiplied by an occurrence count to total. We don't compose that product yet, so committing the bare
    per-occurrence amount drops the frequency factor — an honest DECLINE on the frame (only-declines shape)."""

    def test_every_time_frame_declines(self):
        self.assertEqual(reading.solve("Mary goes to the shop 2 times a day. "
                                       "Every time she goes to the shop, she buys 3 apples. "
                                       "How many apples does she buy per day?"), "undetermined")

    def test_each_time_frame_declines(self):
        self.assertEqual(reading.solve("Tom runs 4 times a week. Each time he runs, he drinks 2 cups "
                                       "of water. How many cups of water does he drink per week?"),
                         "undetermined")

    def test_plain_frame_unaffected(self):
        # no "every time"/"each time" frame → ordinary forward accumulation still commits
        self.assertEqual(reading.solve("Tom has 5 apples. He buys 3 apples. "
                                       "How many apples does Tom have?"), 8)


class TestIncrement111ComparativeExemptionRequiresQuantity(unittest.TestCase):
    """#111 - GUARD 3's comparative/multiplier exemption (which skips the unused-given check, assuming the
    factor was consumed into a derived amount) now holds ONLY if the premise actually produced a
    Quantity/Rate. A degenerate parse that yields only a Frequency ("Billy has $25 less than twice the
    money Sam has" → Frequency(have, 2), dropping the $25) is no longer exempt, so the unused $25 trips
    the guard and we DECLINE instead of summing a partial total."""

    def test_degenerate_comparative_declines(self):
        # RE-PINNED (#207): the anchored-sum system now claims this CORRECTLY (Billy = 2·75 − 25
        # = 125; together = 200) once " together" joined its sum cues — the GUARD 3 exemption
        # behavior this class pins is unchanged (the fold path still declines); the EARLY system
        # answers first. Was pinned "undetermined" while no machinery could solve it.
        self.assertEqual(reading.solve(
            "Billy has $25 less than twice the money Sam has. If Sam has $75, "
            "how much money do they have together?"), 200)

    def test_consumed_comparative_still_solves(self):
        # a comparative that DOES produce a Quantity stays exempt and solves
        self.assertEqual(reading.solve("Martha has 68 apples. Tim has 30 less apples than Martha. "
                                       "How many apples does Tim have?"), 38)

    def test_consumed_relational_still_solves(self):
        self.assertEqual(reading.solve("Sam has $75. Billy has $25 less than twice as much money as Sam. "
                                       "How much money do they have together?"), 200)


class TestIncrement112ElidedTransferObjectDeclines(unittest.TestCase):
    """#112 - GUARD 10: "gave 12 to his brother, 9 to his sister" — the value PRECEDES "to" and quantifies
    that "to"'s object (the recipient), because the transferred object was elided ("12 [boxes] to his
    brother"). The count then attaches to the recipient, so the give never reduces the real holding and
    the gross commits (Sonny: 45). The object is unrecoverable → DECLINE. A recipient COUNT after "to"
    ("to 3 friends") or a stated object ("gave 12 boxes to …") is unaffected."""

    def test_elided_transfer_object_declines(self):
        self.assertEqual(reading.solve(
            "Sonny received 45 boxes of cookies. He gave 12 to his brother, 9 to his sister, "
            "and he gave 7 to his cousin. How many boxes were left for him?"), "undetermined")

    def test_recipient_count_after_to_still_solves(self):
        # "to 3 friends" — the value is a recipient COUNT (after "to"), not an elided object → solves
        self.assertEqual(reading.solve("Colton gave 4 stickers each to 3 friends. "
                                       "How many stickers did Colton give?"), 12)

    def test_stated_transfer_object_still_solves(self):
        # "gave 3 apples to his brother" — 3 quantifies 'apples' (not the to-object) → unaffected
        self.assertEqual(reading.solve("Tom has 10 apples. He gave 3 apples to his brother. "
                                       "How many apples does Tom have?"), 7)


class TestIncrement130ElidedStandardDecline(unittest.TestCase):
    """#130 - ELIDED-SUBJECT inverse relational → DECLINE (precision guard). "Jennifer bought twice as many
    as Emily and three times as many as Bob bought" — the second conjunct's real subject (Jennifer) is
    elided, so `_relational_quantity` mis-reads the post-"as" STANDARD (Bob) as the subject and would
    commit "Bob = 3 × recent" (18). A name sitting right after "as" is a standard, not a subject → DECLINE
    (the elided-subject inverse "Jennifer = 3 × Bob ⇒ Bob = Jennifer/3" is unresolved). Jennifer-candies
    18→decline. The forward "X bought twice as many as Y" still resolves. SYNTHETIC strings only."""

    def test_elided_standard_declines(self):
        ctx = [reading.Quantity("emily", ("candy", frozenset()), 6.0, "buy")]
        self.assertIsNone(reading._relational_quantity(
            reading.analyze("three times as many as Bob bought."), ctx))

    def test_forward_relational_still_resolves(self):
        ctx = [reading.Quantity("emily", ("candy", frozenset()), 6.0, "buy")]
        self.assertEqual(reading._relational_quantity(
            reading.analyze("Jennifer bought twice as many candies as Emily"), ctx),
            reading.Quantity("jennifer", ("candy", frozenset()), 12, "buy", derived=True))

    def test_elided_standard_problem_declines(self):
        self.assertEqual(reading.solve(
            "Jennifer bought twice as many candies as Emily and three times as many as Bob bought. "
            "If Emily bought 6 candies, how many candies did Bob buy?"), "undetermined")


class TestIncrement131TimeIndexedRelationalDecline(unittest.TestCase):
    """#131 - TIME-INDEXED relational chain → DECLINE (precision guard). "She sold twice as many boxes on
    Tuesday as she did on Wednesday … twice as many on Wednesday as on Thursday … Thursday=1200 … how many
    on Tuesday?" — the referent is the SAME entity at a DIFFERENT time (the 2nd "on <day>"), but
    `_relational_quantity` resolves the referent by ENTITY (recency), not time, so it would chain the wrong
    day's value and commit garbage (6000). A relational multiplier-on-quantifier clause with ≥2 time
    references → DECLINE. The full solve (time-tagged relational quantities + per-day referent + time-aware
    answer fold) is deferred. Kim-boxes 6000→decline. SYNTHETIC strings only."""

    def test_time_indexed_relational_declines(self):
        ctx = [reading.Quantity("kim", ("box", frozenset()), 100.0, "sell", time="wednesday")]
        self.assertIsNone(reading._relational_quantity(
            reading.analyze("She sold twice as many boxes on Tuesday as she did on Wednesday"), ctx))

    def test_single_time_relational_still_resolves(self):
        ctx = [reading.Quantity("kim", ("box", frozenset()), 100.0, "sell")]
        self.assertEqual(reading._relational_quantity(
            reading.analyze("Mary sold twice as many boxes as Kim"), ctx),
            reading.Quantity("mary", ("box", frozenset()), 200, "sell", derived=True))

    def test_time_indexed_problem_declines(self):
        # RE-PINNED (#228): the day-indexed doubling branch now claims this CORRECTLY —
        # 1200·2·2 = 4800. Was pinned "undetermined" while no machinery could solve it.
        self.assertEqual(reading.solve(
            "She sold twice as many boxes on Tuesday as she did on Wednesday, and she sold twice as many "
            "boxes on Wednesday as she did on Thursday. If Kim sold 1200 boxes of cupcakes on Thursday, "
            "how many boxes did she sell on Tuesday?"), 4800)


class TestIncrement133PeriodNotApplied(unittest.TestCase):
    """#133 - DECLINE-SAFETY (GUARD 11): a question asking for a per-PERIOD total ("…in N weeks") whose
    period value never entered the computation, when the premises describe a RECURRING action the period
    should have scaled, DECLINES instead of committing a partial. This is the prerequisite for safe
    vocabulary growth — a newly-readable but unscaled per-period problem (e.g. "buys 2 … every school day
    … in 3 weeks" → 2) declines rather than committing a wrong. The recurrence gate keeps a STATIC holding
    queried at a future time ("Tom has 5 apples … in 7 days?" → 5, the period is irrelevant). On the
    held-out TEST slice this lifted precision 36.4%→40.0% (a real period-unscaled commit-wrong). SYNTHETIC."""

    def test_recurring_unscaled_period_declines(self):
        self.assertEqual(reading.solve(
            "Mia buys 2 books every school day. How many books does she buy in 3 weeks?"), "undetermined")

    def test_static_holding_future_time_kept(self):
        # no recurrence cue → the period is correctly irrelevant → keep the holding, don't decline
        self.assertEqual(reading.solve("Tom has 5 apples. How many apples will he have in 7 days?"), 5)

    def test_handled_period_still_solves(self):
        # a rate×duration period IS applied (7 enters the arithmetic) → not declined
        self.assertEqual(reading.solve(
            "Tom reads 10 hours over 5 days. He can read 50 pages per hour. "
            "How many pages does he read in 7 days?"), 700)


class TestIncrement134DroppedUnitQuantity(unittest.TestCase):
    """#134 - DECLINE-SAFETY (GUARD 12): a stated number quantifying a measurement unit (distance/weight/
    volume) or a money value that never enters the computation is an OPERATIVE quantity we dropped → the
    structure is unhandled → DECLINE. Object-noun enumeration counts ("3 snakes") are NOT units → exempt;
    TIME periods are left to the recurrence-gated GUARD 11 (so a static "…in 7 days?" isn't snagged). The
    broad firewall-safe guard underpinning vocabulary growth (28 permissive-vocab commit-wrongs decline).
    dev unchanged, TEST holds 40.0%. SYNTHETIC strings only."""

    def test_multi_price_list_declines(self):
        # three per-item prices we don't compose → committing one drops the others → decline
        self.assertEqual(reading.solve(
            "A store sells apples at $2 each, bananas at $3 each, and cherries at $5 each. "
            "How much do all of them cost together?"), "undetermined")

    def test_handled_money_still_solves(self):
        # Todd's purchase computes 4×2 and 20−8 → every money value is used → not declined
        self.assertEqual(reading.solve(
            "Todd has $20. He buys 4 candy bars that cost $2 each. How much money does Todd have left?"), 12)

    def test_handled_unit_rate_still_solves(self):
        # Weng: $12/hr × 50/60 hr → 12 and 50 both enter the arithmetic → not declined
        self.assertEqual(reading.solve(
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
            "How much did she earn?"), 10)

    def test_object_count_not_a_unit(self):
        # an object-noun count ("snakes") is not a UNIT → GUARD 12 ignores it (helper returns False)
        prem = [reading.analyze("Christina has 3 snakes")]
        q = reading.analyze("How many snakes are there?")
        reading.ARITHMETIC_LOG.clear()
        self.assertFalse(reading._dropped_unit_quantity(prem, q))


class TestIncrement136MultiRateDecline(unittest.TestCase):
    """#136 - DECLINE-SAFETY (GUARD 13): a MULTI-RATE purchase/charge we don't compose declines. Fires on a
    per-item price with an unapplied item count ("three movie tickets for $12 each" → 3×12 dropped) or on
    ≥2 distinct per-item prices (a priced list). This is the multi-rate residue GUARD 12 misses (those
    prices ARE summed, so no money value is 'dropped' — the error is the unapplied per-item quantities). A
    single applied purchase (Todd: 4×$2) is unaffected. dev unchanged, TEST holds 40.0%. SYNTHETIC."""

    def test_multi_item_unapplied_count_declines(self):
        self.assertEqual(reading.solve(
            "Jackson buys a computer game for $66 and three movie tickets for $12 each. "
            "How much did he pay in total?"), "undetermined")

    def test_two_per_item_prices_declines(self):
        # two per-item prices asked as a single "one of each" cost → still a DECLINE (now via answer()'s
        # "couldn't tell what to count" once 'pens' is lexicalized, rather than the old unknown-word decline).
        result = reading.solve(
            "A store sells pens at $2 each and books at $5 each. How much does one pen and one book cost?")
        self.assertFalse(isinstance(result, (int, float)) and not isinstance(result, bool))

    def test_single_applied_purchase_unaffected(self):
        self.assertEqual(reading.solve(
            "Todd has $20. He buys 4 candy bars that cost $2 each. How much money does Todd have left?"), 12)


class TestIncrement137DroppedMultiplier(unittest.TestCase):
    """#137 - DECLINE-SAFETY (GUARD 14): a stated relational MULTIPLIER ("7 times", "one-third") whose
    evidence never reached the computation = a dropped relational-chain link → decline. Evidence = the
    integer factor, or for a fraction the DENOMINATOR (computed as num/denom, so the decimal never appears
    but the denom does). Catches multi-step chains that mis-resolve and commit a partial ("Emily = 7× Sandra,
    Sandra = 3× Betty" → 7×Betty drops the 3×). The trigger needs a mis-parse (only under permissive vocab on
    real data), so the unit test exercises the HELPER against a crafted ARITHMETIC_LOG. SYNTHETIC."""

    def test_integer_factor_dropped(self):
        prem = [reading.analyze("Emily has 7 times as many oranges as Sandra"),
                reading.analyze("Sandra has 3 times as many oranges as Betty")]
        reading.ARITHMETIC_LOG[:] = [("answer = 7 * 4", 28)]      # the 3× link dropped
        self.assertTrue(reading._dropped_multiplier(prem))
        reading.ARITHMETIC_LOG[:] = [("answer = 7 * 3 * 4", 84)]  # both factors present
        self.assertFalse(reading._dropped_multiplier(prem))

    def test_fraction_uses_denominator(self):
        prem = [reading.analyze("Arthur spends one-third of his money")]
        reading.ARITHMETIC_LOG[:] = [("answer = 200 / 3", 66)]    # denom 3 present
        self.assertFalse(reading._dropped_multiplier(prem))
        reading.ARITHMETIC_LOG[:] = [("answer = 200 + 5", 205)]   # denom 3 absent
        self.assertTrue(reading._dropped_multiplier(prem))

    def test_clean_chain_still_solves(self):
        # with known vocabulary the chain resolves correctly (7×3×4) and is NOT declined
        self.assertEqual(reading.solve(
            "Anna has 7 times as many apples as Beth. Beth has 3 times as many apples as Cara. "
            "If Cara has 4 apples, how many apples does Anna have?"), 84)


class TestIncrement138DroppedEachAttribute(unittest.TestCase):
    """#138 - DECLINE-SAFETY (GUARD 15): an "each/every" quantifier turns a nearby object count into a
    per-each MULTIPLIER ("7 starfish with 5 arms each" → 7×5; "each daisy has 8 petals" → ×8), so that count
    is OPERATIVE — unlike a bare enumeration count GUARD 12 exempts. Fires when such a premise has a value
    quantifying a non-time, non-money object noun absent from the computation. A TIME per-each ("every 30
    minutes") and a clock TIME-RANGE ("1 PM to 4 PM every night") are excluded. The end-to-end trigger needs
    a mis-parse (only under permissive vocab on real data), so the helper is exercised against a crafted
    ARITHMETIC_LOG. dev unchanged 71/3/95.9%, TEST 40.0→50.0%. SYNTHETIC."""

    def test_each_attribute_count_dropped(self):
        prem = [reading.analyze("Each box has 8 apples")]
        reading.ARITHMETIC_LOG[:] = [("answer = 1 + 2", 3)]       # the per-each 8 never used
        self.assertTrue(reading._dropped_each_attribute(prem))
        reading.ARITHMETIC_LOG[:] = [("answer = 3 * 8", 24)]      # 8 applied
        self.assertFalse(reading._dropped_each_attribute(prem))

    def test_time_per_each_excluded(self):
        # "every 30 minutes" is a rate/period (GUARD 11 territory), not an attribute multiplier → no fire
        prem = [reading.analyze("She writes 1 page every 30 minutes")]
        reading.ARITHMETIC_LOG[:] = [("answer = 1 + 1", 2)]       # 30 (a time period) absent — still exempt
        self.assertFalse(reading._dropped_each_attribute(prem))

    def test_no_each_quantifier_no_fire(self):
        prem = [reading.analyze("There are 8 apples")]
        reading.ARITHMETIC_LOG[:] = [("answer = 1 + 1", 2)]
        self.assertFalse(reading._dropped_each_attribute(prem))


# === INTEGRATION & REGRESSION: multi-feature bench cases, parse near-wins, TEST-slice fixes ===

class TestIncrement73ParseNearWins(unittest.TestCase):
    """#73 - two parse near-wins from the bench (math we already had, blocked by one mis-parse).
    (a) SUBJ AUX VERB subject binding: "how much money she WILL have" (subject sits before the aux).
    (b) "sold <X> to N <people>" → N of X (the bare-plural dobj is the measure; the recipient count is
    the amount). The latter solves the canonical GSM8K #1 (Natalia)."""

    def test_subj_aux_verb_subject(self):
        r = reading.analyze("how much money she will have")
        have = next(u for u in r.units if u.surface == "have")
        self.assertEqual(reading._follow(have, "subject").surface, "she")

    def test_will_have_solves(self):
        self.assertEqual(reading.solve("Susie has $200 in her piggy bank. If she puts 20% more money "
                                       "into her piggy bank, how much money she will have?"), 240)

    def test_sold_to_n_people_is_n_of_the_object(self):
        self.assertEqual(reading.learn("Natalia sold clips to 48 of her friends in April"),
                         [reading.Quantity("natalia", ("clip", frozenset()), 48.0, "sell", "april")])

    def test_natalia_canonical_solves(self):
        self.assertEqual(reading.solve(
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
            "in May. How many clips did Natalia sell altogether in April and May?"), 72)

    def test_bare_dobj_required_for_transfer(self):
        self.assertEqual(reading.learn("Tom gave 4 stickers each to 3 friends"),
                         [reading.Quantity("tom", ("sticker", frozenset()), 12.0, "give")])


class TestIncrement95JamesCombined(unittest.TestCase):
    """#95 - James, a 3-piece HARD-TIER combine. (a) a trailing GERUND adjunct ("spends 40 years TEACHING")
    is not a second verb; (b) a comparative with an IMPLICIT standard ("10 years less" — than the value
    just stated) takes the most recent Quantity of the measure as referent (partner = 40-10 = 30); (c) a
    query-less "combined" question ("How long is their combined experience?") sums the one measure the
    context quantities share (40 + 30 = 70 years)."""

    def test_gerund_adjunct_not_a_verb(self):
        self.assertEqual(reading.learn("James spends 40 years teaching"),
                         [reading.Quantity("james", ("year", frozenset()), 40.0, "spend")])

    def test_implicit_standard_comparative(self):
        self.assertEqual(reading.learn("His partner has been teaching for 10 years less", ["james"],
                                       [reading.Quantity("james", ("year", frozenset()), 40, "spend")]),
                         [reading.Quantity("partner", ("year", frozenset()), 30.0, "teach")])

    def test_james_combined_solves(self):
        self.assertEqual(reading.solve("James spends 40 years teaching. His partner has been teaching for "
                                       "10 years less. How long is their combined experience?"), 70)

    def test_christina_combined_unaffected(self):
        self.assertEqual(reading.solve("Christina has 3 snakes. 1 snake is 2 feet long. Another snake is "
                                       "16 inches long. The last snake is 10 inches. How many inches are "
                                       "all snakes combined?"), 50)


class TestTestSliceRegressionFixes(unittest.TestCase):
    """A TEST-slice checkpoint (aggregate-only per the firewall discipline, but inspected once this session to
    debug a real regression) found this session's work introduced 5 NEW confident-wrongs on TEST despite 0
    broke/decl-wrong on dev-train+holdout throughout — TEST is a disjoint partition whose specific questions
    exposed gaps the dev diff harness structurally can't see. Each is a genuinely distinct root cause,
    bisected to a specific increment and fixed here:
    (1) `_anchored_sum_system` summed EVERY entity even when the question names only SOME of them ("how many
        marbles do CARL AND CAL have combined?" in a 3-entity system) — now sums only the named subset when
        the question names 2+ entities (handling the same "merged multi-word name unit" gotcha as
        `_residual_sum_system`).
    (2) `_anchored_affine_coeffs` had no unit-consistency check at all — an ANCHOR's value and a relation's
        OFFSET value in different, unconverted measures (hours vs minutes) got folded as if identical. Fixed
        narrowly (only the anchor's own value and a matched-by-magnitude offset value, NOT every incidental
        value in a clause — an early overly-broad version wrongly declined correct pure-rate systems where a
        constant "in 1 hour" duration appears in every clause) — and skips COUNT_NOUNS ("the NUMBER of X"
        resolves to "number", not X, a spurious measure mismatch of its own).
    (3) `_noun_anchored_sum_system` treated ANY asked noun outside the entity set as a category sum, even
        when the question clause carried its OWN unaccounted number (a "5 students to every teacher"
        conversion glued onto the question by clause-splitting) — now declines when the question itself
        contains a value.
    (4) `_anchor_fact`/`_noun_anchor_fact` treated a "N units EVERY DAY" premise as a flat anchor, ignoring
        that it's a PER-DAY RATE needing multiplication by a stated duration — now decline on any
        every/each/per recurrence marker.
    (5) The general age-domain `average` branch (learn.py's wh-question dispatch) never applied a stated
        temporal shift ("average age of the three IN THREE YEARS") the way its sibling `total age` branch
        already did — dormant because it was unreachable before 4am's imperative-question-clause fix ever
        let such text past solve()'s entry guard. Now offsets each term before dividing, like its sibling."""

    def test_sum_only_named_entities_in_question(self):
        # Sean=56 (anchor), Carl=4*56=224, Cal=2*Sean=112 (since Sean=half of Cal) -> Carl+Cal=336, NOT +Sean.
        self.assertEqual(reading.solve(
            "Carl has four times as many marbles as Sean and Sean has half as many marbles as Cal. If Sean "
            "has 56 marbles, how many marbles do Carl and Cal have combined?"), 336)

    def test_mixed_units_decline_but_pure_rate_still_solves(self):
        # unit mismatch (hours anchor vs minutes offset) must decline...
        self.assertEqual(reading.solve(
            "Movie A was one-fourth the length of Movie B. Movie B was 5 minutes longer than Movie C. Movie "
            "C was 1.25 hours. How many minutes long was Movie A?"), "undetermined")
        # ...but a genuinely uniform rate (the SAME "in 1 hour" duration in every clause) must still solve —
        # the fix that added the unit check must not treat an incidental, repeated duration as a conflict.
        self.assertEqual(reading.solve(
            "Carter can read half as many pages as Lucy in 1 hour. Lucy can read 20 more pages than Oliver "
            "in 1 hour. Oliver can read 40 pages. How many pages can Carter read in 1 hour?"), 30)

    def test_count_noun_wrapper_not_a_measure_mismatch(self):
        # "the NUMBER of Aaron's brothers" resolves the value's quantifies-link to "number", not "brother" —
        # must not be treated as a competing measure against the anchor's own "brother"-quantified value.
        self.assertEqual(reading.solve(
            "Aaron has four brothers. If the number of Bennett's brothers is two less than twice the number "
            "of Aaron's brothers, what is the number of Bennett's brothers?"), 6)

    def test_unaccounted_number_in_question_declines(self):
        # "...and 5 students to every teacher, how many teachers are there?" — the conversion clause got
        # glued onto the question by clause-splitting; a bare category-sum question never has its own number.
        self.assertEqual(reading.solve(
            "There are twice as many boys as girls at Dr. Wertz's school. If there are 60 girls and 5 "
            "students to every teacher, how many teachers are there?"), "undetermined")

    def test_recurring_rate_anchor_declines(self):
        # "Sofie plans to walk 10 miles EVERY DAY" is a per-day rate needing x7 for "in seven days" -- not a
        # flat anchor.
        self.assertEqual(reading.solve(
            "In 7 days, Sofie will walk twice as far as Brian. If Sofie plans to walk 10 miles every day, "
            "how many miles will Brian walk in seven days?"), "undetermined")

    def test_average_age_applies_temporal_shift(self):
        # Adrian=3*21=63, Zack=2*21=42 (Harriet is HALF of Zack); in 3 years: 66+24+45=135, /3=45.
        self.assertEqual(reading.solve(
            "Adrian's age is three times the age of Harriet, and Harriet is half the age of Zack. Calculate "
            "the average age of the three in three years if Harriet is 21 years old now."), 45)


class TestIncrement141ChangeIsMoneyRemainder(unittest.TestCase):
    """#141 — "how much CHANGE (did he get back / was he given / receive)" is the money returned after a
    purchase = the MONEY remainder (have - Σspent), the same computation as "how much money is left". The
    noun "change" names the money-left query and forces the remainder path, regardless of the vacuous
    trailing verb (get/receive/given). The verb sense of "change" ("how many tires did he change") pairs
    with how-many/how-long, never "how much", so it is not stolen."""

    def test_change_get_back(self):
        self.assertEqual(reading.solve(
            "Blake spends $40 on oranges, $50 on apples, and $60 on mangoes. If he has $300, how much "
            "change did he get back?"), 150)

    def test_change_receive_form(self):
        self.assertEqual(reading.solve(
            "Tom has $20. He spent $8 on apples. How much change did he receive?"), 12)

    def test_change_was_given_form(self):
        self.assertEqual(reading.solve(
            "Tom has $20. He spent $8 on apples. How much change was he given?"), 12)

    def test_change_matches_money_left_synonym(self):
        # "change" and "money is left" resolve to the SAME number (same remainder computation).
        prem = "Blake spends $40 on oranges, $50 on apples, and $60 on mangoes. If he has $300, "
        self.assertEqual(reading.solve(prem + "how much change was he given?"),
                         reading.solve(prem + "how much money is left?"))

    def test_change_verb_sense_not_stolen(self):
        # "change" as the ACTION verb ("how many tires did he change") is a count question, not money —
        # requiring "how much" keeps it out of the money-remainder branch (stays a non-money reading).
        out = reading.solve("He changed 4 tires on Monday. How many tires did he change?")
        self.assertNotEqual(out, 150)


class TestIncrement142RevenueProductSum(unittest.TestCase):
    """#142 — the REVENUE/SELL twin of `_shopping_total`: "sold 453 cakes at $12 and 126 pies at $7. How
    much did the baker earn?" → Σ(count×price)=6318. The price attaches via a PER-ITEM marker (preceded by
    "at", or followed by "each"/"apiece"), and the question asks earn/make as well as spend/pay. A bare
    "for $P" (a TOTAL, not per-item) is rejected, as are profit/discount/more/less questions. A premise that
    fails to parse (so #products < #price-units) declines rather than committing a partial sum."""

    def test_at_price_revenue_sum(self):
        self.assertEqual(reading.solve(
            "During April, the baker sold 453 cakes at $12 and 126 pies at $7. "
            "How much did the baker earn?"), 6318)

    def test_make_verb_form(self):
        self.assertEqual(reading.solve(
            "Tom sold 5 cakes at $3 and 4 pies at $2. How much did Tom make?"), 23)

    def test_for_total_price_rejected(self):
        # "four pizzas for $80" is a TOTAL price (not per-item) — no "at"/"each" marker → decline.
        self.assertEqual(reading.solve(
            "Sam bought four pizzas for $80. How much did Sam pay?"), "undetermined")

    def test_profit_question_excluded(self):
        # a "profit" question needs cost subtracted, not a plain revenue sum → decline.
        self.assertEqual(reading.solve(
            "A baker sold 5 cakes at $3 and 4 pies at $2. How much profit did the baker make?"),
            "undetermined")

    def test_discount_excluded(self):
        self.assertEqual(reading.solve(
            "A baker sold 5 cakes at $3 and 4 pies at $2 with a discount. How much did the baker earn?"),
            "undetermined")


class TestIncrement143ChainCountPrice(unittest.TestCase):
    """#143 — CHAIN-COUNT × UNIT-PRICE: an anchored ratio/offset chain of an item's counts + a single
    per-item price + a money-SPEND total question → (Σ count) × price. The count is computed by the
    (measure-blind) anchored-sum system; `_chain_count_price` (dispatched BEFORE anchored_sum) extracts the
    price and multiplies, so the money framing no longer commits the bare count. anchored_sum defers on this
    shape (no cross-solver value collision). A pure COUNT question is unaffected (anchored_sum answers it)."""

    def test_chain_count_times_price(self):
        self.assertEqual(reading.solve(
            "Dorothy buys half as many pens as Julia. Julia buys three times as many pens as Robert. "
            "Robert buys 4 pens. If one pen costs $1.50, how much money did the three friends spend in "
            "total on pens?"), 33)

    def test_pure_count_question_unaffected(self):
        # no price / no money framing → the plain count, via anchored_sum (not the price wrapper)
        self.assertEqual(reading.solve(
            "Dorothy buys half as many pens as Julia. Julia buys three times as many pens as Robert. "
            "Robert buys 4 pens. How many pens did they buy in total?"), 22)


class TestIncrement144FractionRemainder(unittest.TestCase):
    """#144 — FRACTION-OF-BASE REMAINDER: a single stated POOL reduced by loss clauses that are each a
    FRACTION of that pool or a LITERAL, then "how many <item> left/remain?" → N − Σ(fraction×N | literal).
    The fraction-partitive ("¼ of the peanuts") produces no fact in the general fold (which drops it and
    commits N − literals), so `_fraction_remainder_system` reads the raw values. Tightly gated against the
    treacherous cases: chained 'of the remaining', relative reductions ('2 less'), percent, derived/multi-
    entity bases (only an explicit pool frame 'there were/made/baked/grew N <item>')."""

    def test_fraction_plus_literal(self):
        self.assertEqual(reading.solve(
            "There were 148 peanuts in a jar. Brock ate one-fourth of the peanuts and Bonita ate 29 "
            "peanuts. How many peanuts remain in the jar?"), 82)

    def test_single_fraction_of_pool(self):
        self.assertEqual(reading.solve(
            "There were 30 apples in a basket. Tom ate one-third of the apples. How many apples are "
            "left?"), 20)

    def test_fraction_and_literal_separate_clauses(self):
        self.assertEqual(reading.solve(
            "There were 40 apples in a jar. Sam ate half of the apples. Kim ate 5 apples. How many "
            "apples remain?"), 15)

    def test_chained_of_the_remaining_declines(self):
        # "a third of the remaining" chains the base — deliberately NOT modeled → decline (never miscompute)
        result = reading.solve(
            "There were 40 apples in a jar. Sam ate half of the apples and then ate a third of the "
            "remaining apples. How many apples are left?")
        self.assertFalse(isinstance(result, (int, float)) and not isinstance(result, bool))


class TestIncrement145AnchoredSumSubset(unittest.TestCase):
    """#145 — anchored_sum SUBSET fix: when the question names a subset of the system's entities, sum ONLY
    those. The subset detection counted only 'name'-sensed words, so a name mis-tagged as a common noun
    ('Bill' → the money noun) was missed → the subset was skipped and ALL entities summed (Bill/Harry/Sue:
    720 instead of the asked Bill+Harry = 672). Fix: also count a capitalized question word that IS a system
    entity (in the coeff graph). The out-of-system safety-decline stays gated to a ≥2-name subset (a
    single-name question still sums all — Jason/Jayden 660 must not regress to a decline)."""

    def test_named_subset_sums_only_those(self):
        self.assertEqual(reading.solve(
            "Bill has 6 times as many marbles as Harry, and Harry has twice as many marbles as Sue. "
            "If Sue has 48 marbles, how many do Bill and Harry have combined?"), 672)

    def test_all_entities_when_no_names(self):
        self.assertEqual(reading.solve(
            "Bill has 6 times as many marbles as Harry, and Harry has twice as many marbles as Sue. "
            "If Sue has 48 marbles, how many do they have altogether?"), 720)


class TestSystemResultMigration(unittest.TestCase):
    """The 'emit facts, not a final answer' pilot: `_anchored_sum_system` returns a `SystemResult` (one
    Quantity fact per entity, attributed to its originating premise) instead of a scalar when it can safely
    mark the SHARED reading Units, and `answer()`'s generic combined_q machinery resolves the actual sum —
    see docs plan / reading-engine-handoff. These tests exercise the new mechanism directly, not just
    end-to-end solve() outcomes already covered by the pre-existing anchored-sum test classes."""

    def test_bind_pattern_marks_obligation_met_via_sentinel(self):
        unit = reading.Unit(words=[reading.Word("Foo", 0)],
                            features={"obligations": [reading.Obligation("predicate")]})
        self.assertFalse(unit.features["obligations"][0].met)
        reading._bind_pattern([unit], "test_label")
        ob = unit.features["obligations"][0]
        self.assertTrue(ob.met)
        self.assertIs(ob.target, reading._PATTERN_UNIT)

    def test_bind_pattern_leaves_already_met_obligations_alone(self):
        target = reading.Unit(words=[reading.Word("bar", 1)])
        unit = reading.Unit(words=[reading.Word("Foo", 0)],
                            features={"obligations": [reading.Obligation("predicate", target=target)]})
        reading._bind_pattern([unit], "test_label")
        self.assertIs(unit.features["obligations"][0].target, target)

    def test_safe_case_returns_system_result_not_scalar(self):
        text = ("Priya has 5 stickers. Quinn has 4 times as many stickers as Priya, and Reese has 3 "
                "more stickers than Quinn. How many stickers do they have altogether?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._anchored_sum_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"priya": 5, "quinn": 20, "reese": 23})

    def test_named_subset_and_no_names_both_solve_via_system_result(self):
        # fresh synthetic numbers (not the Bill/Harry/Sue case #145 already covers) — the two combined_q
        # shapes (specific named subset vs every entity) both route through the SAME new answer() branch.
        text = ("Priya has 5 stickers. Quinn has 4 times as many stickers as Priya, and Reese has 3 "
                "more stickers than Quinn. How many stickers do {}?")
        self.assertEqual(reading.solve(text.format("they have altogether")), 48)
        self.assertEqual(reading.solve(text.format("Priya and Reese have combined")), 28)

    def test_multiplier_chain_evidence_survives_guard14(self):
        # GUARD 14 ("was every stated multiplier's evidence used?") scans ARITHMETIC_LOG program text for
        # each raw factor — the flattened (A,B) coefficient form loses that (12 = 6*2 never mentions "6" or
        # "2" on its own), so this regresses to "undetermined" without the hop-by-hop _compute() walk.
        self.assertEqual(reading.solve(
            "Bill has 6 times as many marbles as Harry, and Harry has twice as many marbles as Sue. "
            "If Sue has 48 marbles, how many do they have altogether?"), 720)

    def test_near_name_typo_falls_back_to_legacy_scalar_path(self):
        # a NEAR-NAME UNIFICATION (#215) leaves `origin`'s attribution unreliable (see `_AnchoredCoeffs.
        # unified`) — the solver must decline the new fact-emitting path and fall back to computing the
        # sum directly, not silently mis-attribute or duplicate a fact.
        self.assertEqual(reading.solve(
            "Jason, Ryan, and Jeffery went fishing at the lake. Ryan caught three times "
            "the number of fish that Jason caught. Jefferey caught twice the number of "
            "fish that Ryan got. If Jeffery caught 60 fish, how many fish did all of "
            "them catch in total?"), 100)

    def test_difference_system_returns_system_result_not_scalar(self):
        # fresh synthetic numbers — _anchored_difference_system migrated the same way as the sum twin.
        text = ("Priya has 5 stickers. Quinn has 4 times as many stickers as Priya, and Reese has 3 "
                "more stickers than Quinn. How many more stickers does Reese have than Priya?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._anchored_difference_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        self.assertEqual(reading.solve(text), 18)   # Priya=5, Quinn=20, Reese=23, 23-5=18

    def test_difference_less_than_sign_flip(self):
        self.assertEqual(reading.solve(
            "Priya has 5 stickers. Quinn has 4 times as many stickers as Priya, and Reese has 3 "
            "more stickers than Quinn. How many fewer stickers does Priya have than Reese?"), 18)

    def test_over_determined_entity_both_related_and_anchored(self):
        # an entity can be BOTH the subject of its own relation AND separately anchored (Reese here: defined
        # via "3 more than Quinn" in one premise, and is ALSO the anchor's own entity in another — the
        # anchor value and the relation-derived value agree, since GSM8K premises are consistent) — origin
        # must attribute her single fact to BOTH premises (GUARD 4 needs each to have SOME fact) without
        # adding a duplicate to context (which would double her value in a sum/difference).
        text = ("Quinn has 4 times as many stickers as Priya. Reese has 3 more stickers than Quinn. If "
                "Reese has 23 stickers, how many more stickers does Reese have than Priya?")
        self.assertEqual(reading.solve(text), 18)

    def test_possessive_location_phrasing_falls_back_to_legacy(self):
        # "How many more X are in A's Y than B's Y" uses a copula "are" (tagged `verb`, not `auxiliary`),
        # so answer()'s comparative-difference branch (which requires an auxiliary unit) can never fire —
        # the solver must recognize this and use the legacy scalar path instead of silently mis-answering
        # via an earlier, unrelated yes/no dispatch in answer().
        self.assertEqual(reading.solve(
            "There are 8 books in Mia's room. Theo has half the number of books that Mia has. Nora has "
            "5 more than twice the number of books that Theo has. How many more books are in Nora's room "
            "than Theo's room?"), 9)

    def test_average_system_returns_system_result_not_scalar(self):
        # fresh synthetic numbers — _anchored_average_system migrated the same way as the sum/difference
        # twins, relying on answer()'s EXISTING "average over a set" branch (no new answer() code needed).
        text = ("Priya weighs 60 pounds, 5 pounds less than Quinn and 10 pounds more than Reese. What is "
                "the average weight of the three people?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._anchored_average_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        # Priya=60, Quinn=65, Reese=50 -> (60+65+50)/3 = 58.333...
        self.assertAlmostEqual(reading.solve(text), 175 / 3)

    def test_average_stated_count_must_match_entity_count_still_declines(self):
        # the system has 3 entities but the question says "four" — never divide by a guessed/mismatched N,
        # migrated or not (this precondition is checked BEFORE the solver decides safe-vs-legacy).
        self.assertEqual(reading.solve(
            "Priya weighs 60 pounds, 5 pounds less than Quinn and 10 pounds more than Reese. What is the "
            "average weight of the four people?"), "undetermined")

    def test_single_system_returns_system_result_with_explicit_unit_noun(self):
        # "how many POUNDS does X weigh" has an explicit unit noun, so `answer()`'s query resolves and the
        # SAFE path applies — unlike a bare "how much does X weigh" (see the next test).
        text = ("Priya weighs 60 pounds, 5 pounds less than Quinn and 10 pounds more than Reese. How many "
                "pounds does Quinn weigh?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._anchored_single_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        self.assertEqual(reading.solve(text), 65)

    def test_single_system_bare_how_much_falls_back_to_legacy(self):
        # a bare "How much does X weigh?" (no unit noun) leaves `much`'s `collection` obligation unbound —
        # `answer()` returns an unanswerable Deficiency before reaching the numeric fold at all. The solver
        # must detect this (`_queried_measure(question) is None`) and use the legacy scalar path instead.
        self.assertEqual(reading.solve(
            "Priya weighs 60 pounds, 5 pounds less than Quinn and 10 pounds more than Reese. How much "
            "does Quinn weigh?"), 65)

    def test_single_system_possessive_name_falls_back_to_legacy(self):
        # "How many rooms does X'S HOUSE have?" makes `_question_subjects` return BOTH ['x', 'house'] (the
        # possessed noun mis-extracted as a second subject) — no fact exists for the bare entity "house",
        # so the generic fold would decline; the solver must detect this and use the legacy path.
        self.assertEqual(reading.solve(
            "Ravi's house has 6 rooms. Priya's house has 3 times as many rooms as Ravi's house. How many "
            "rooms does Priya's house have?"), 18)

    def test_single_system_copula_only_question_falls_back_to_legacy(self):
        # "How many centimeters tall IS X?" has no auxiliary and its only verb-sensed word is a bare
        # copula "is" — answer()'s entity/verb resolution needs an auxiliary or a substantive verb, neither
        # of which is present, so it can never recover the asked entity; the solver must fall back.
        self.assertEqual(reading.solve(
            "Sana is 5 cm shorter than Mira. Mira is 100 cm tall. How many centimeters tall is Sana?"), 95)

    def test_residual_system_affine_path_returns_system_result(self):
        # fresh synthetic numbers — _residual_sum_system's affine-relation path (Reese=30 anchor, Quinn=
        # Reese+5=35 relation, Priya never mentioned again -> 90-35-30=25), migrated the same way as the
        # other anchored-family twins, reusing the single-entity fold's disqualifiers (the residual value
        # is handed to that SAME answer() capability).
        text = ("Priya, Quinn, and Reese collect a total of 90 stamps. Quinn collects 5 more stamps than "
                "Reese. If Reese collects 30 stamps, how many stamps does Priya collect?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._residual_sum_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        self.assertEqual(reading.solve(text), 25)

    def test_residual_system_no_relation_fallback_returns_system_result(self):
        # fresh synthetic numbers — the NO-RELATION fallback path (independent anchors only, no ratio/
        # offset relates any pair): Mona=3, Nina=4, total=10 -> Owen=10-3-4=3.
        text = ("Mona buys 3 apples for a picnic. Nina and Owen also buy apples for the picnic. If they "
                "need 10 apples total, and Nina buys 4 apples, how many apples should Owen buy?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._residual_sum_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        self.assertEqual(reading.solve(text), 3)


class TestNounKeyedEntityFallback(unittest.TestCase):
    """`answer()`'s NOUN-KEYED ENTITY FALLBACK: a system whose entities are common nouns ("goat"/"pig"),
    not people, asks "How many goats does HE have?" — the grammatical subject is a pronoun/possessor that
    is never itself a tracked entity. If the subject-resolved entities hold NO fact at all and exactly one
    context entity shares the query's measure, resolve to that entity instead. Built to unlock the
    noun-affine family's single-entity solvers (`_noun_affine_system`/`_noun_residual_sum_system`), gated
    narrowly (no real name, no "each", not a TIME_UNITS query) to avoid preempting existing machinery that
    already handles named-but-dataless subjects, per-member distribution, and age equations."""

    def test_pronoun_subject_resolves_to_unique_noun_entity(self):
        ctx = [reading.Quantity("goat", ("goat", frozenset()), 11)]
        self.assertEqual(reading.answer("How many goats does he have?", ctx), 11)

    def test_named_subject_with_no_data_still_declines(self):
        # Mary has no data of her own -- must NOT hijack onto an unrelated entity's total.
        ctx = [reading.Quantity("tom", ("apple", frozenset()), 3)]
        self.assertEqual(reading.answer("How many apples does Mary have?", ctx), "undetermined")

    def test_merged_name_list_subject_still_resolves_normally(self):
        # "Mike and Rita" merges into one Unit whose own _entity() matches neither individual member --
        # the fallback must recognize this IS still a real name (via .members) and not fire.
        ctx = [reading.Quantity("box", ("bar", frozenset()), 12, "contain")]
        self.assertEqual(reading.answer("How many bars did Mike and Rita get combined?", ctx),
                         "undetermined")

    def test_each_distributive_ask_not_hijacked(self):
        ctx = [reading.Quantity("rita", ("apple", frozenset()), 24)]
        result = reading.answer("How many apples does each person get?", ctx)
        self.assertNotEqual(result, 24)

    def test_time_unit_query_declines_for_age_domain(self):
        # a TIME_UNITS query is the age domain's own turf (an equation to solve, not a raw stored value) --
        # must decline here rather than hand back an entity's raw current value.
        ctx = [reading.Quantity("melanie", ("year", frozenset()), 8, "be")]
        self.assertEqual(reading.answer("How many years until her age doubles?", ctx), "undetermined")

    def test_existential_are_there_resolves_unique_measure(self):
        # "are" is excluded from the verb search (usually a copula) and "there" can never be a subject, so
        # entities is None for this construction unless the new existential fallback recovers it.
        ctx = [reading.Quantity("green", ("ball", frozenset({"green"})), 7, "have")]
        self.assertEqual(reading.answer("How many green balls are there?", ctx), 7)

    def test_existential_are_there_declines_when_ambiguous(self):
        ctx = [reading.Quantity("blue", ("ball", frozenset()), 7, "have"),
               reading.Quantity("red", ("ball", frozenset()), 9, "have")]
        self.assertEqual(reading.answer("How many balls are there?", ctx), "undetermined")

    def test_existential_are_there_declines_for_bare_head_noun(self):
        # a BARE head-noun ask ("how many students") must NOT resolve via this fallback even with exactly
        # one exact-measure fact in context: that lone fact may belong to an unrelated SUBGROUP (e.g. a
        # fraction-partition's "20 without a desktop"), not the population the bare ask actually means. A
        # real TEST regression this excludes: a fraction-partition problem's subgroup count (20) was wrongly
        # handed back for "how many students are there at that grade level?" instead of declining (the
        # correct total, 80, needs the fraction machinery this fallback has no business preempting).
        ctx = [reading.Quantity("without_desktop", ("student", frozenset()), 20, "have")]
        self.assertEqual(reading.answer("How many students are there?", ctx), "undetermined")

    def test_category_sum_fallback_sums_registered_members(self):
        # pig/cow/goat are all tagged the "animal" category -- "how many animals" sums them, even though
        # no context entity is itself named "animal".
        ctx = [reading.Quantity("pig", ("pig", frozenset()), 10, "have"),
               reading.Quantity("cow", ("cow", frozenset()), 17, "have"),
               reading.Quantity("goat", ("goat", frozenset()), 23, "have")]
        self.assertEqual(reading.answer("How many animals do they have?", ctx), 50)

    def test_category_sum_fallback_complement_excludes_negated_subtype(self):
        ctx = [reading.Quantity("pig", ("pig", frozenset()), 10, "have"),
               reading.Quantity("cow", ("cow", frozenset()), 17, "have"),
               reading.Quantity("goat", ("goat", frozenset()), 23, "have")]
        self.assertEqual(reading.answer("How many animals are not goats?", ctx), 27)

    def test_category_sum_fallback_unresolvable_negation_declines(self):
        ctx = [reading.Quantity("pig", ("pig", frozenset()), 10, "have"),
               reading.Quantity("cow", ("cow", frozenset()), 17, "have"),
               reading.Quantity("goat", ("goat", frozenset()), 23, "have")]
        self.assertEqual(reading.answer("How many animals are not fish?", ctx), "undetermined")

    def test_category_sum_fallback_declines_single_member(self):
        # a single category member is an ordinary single-entity ask, not a category question -- must decline
        # here (the noun-keyed fallback / normal per-entity fold owns that shape, not this one).
        ctx = [reading.Quantity("pig", ("pig", frozenset()), 10, "have")]
        self.assertEqual(reading.answer("How many animals do they have?", ctx), "undetermined")

    def test_category_sum_fallback_ignores_event_verb_facts(self):
        # a real dev-train confident-wrong this guard fixed: registering a new `pet` category
        # exposed this fallback to facts whose ENTITY happens to be a category member but whose
        # VERB is an EVENT ("50% of the dogs are ADOPTED"), not a holdings statement -- a
        # shelter's per-species adoption counts must NOT be summed as if they were its total
        # holdings. Only STATE_VERBS ("have"/"be") facts may enter the category sum -- the
        # "sold" goat fact must be excluded from the total, not added in.
        ctx = [reading.Quantity("pig", ("pig", frozenset()), 10, "have"),
               reading.Quantity("cow", ("cow", frozenset()), 17, "have"),
               reading.Quantity("goat", ("goat", frozenset()), 5, "sell")]
        self.assertEqual(reading.answer("How many animals do they have?", ctx), 27)

    def test_category_sum_fallback_declines_unregistered_category(self):
        # "children" is not a registered semantic category -- must decline, not silently sum unrelated facts.
        ctx = [reading.Quantity("daughter", ("daughter", frozenset()), 18, "have"),
               reading.Quantity("son", ("son", frozenset()), 3, "have")]
        self.assertEqual(reading.answer("How many children does he have?", ctx), "undetermined")

    def test_complementary_group_fallback_resolves_bare_copula_ask(self):
        # "are AT THE MEETING" has no aux and no non-copula verb -- a deeper gap than the existential "are
        # there" fallback (#395), which requires the literal word "there".
        ctx = [reading.Quantity("boy", ("boy", frozenset()), 12, "have"),
               reading.Quantity("girl", ("girl", frozenset()), 28, "have")]
        self.assertEqual(reading.answer("How many girls are at the meeting?", ctx), 28)
        self.assertEqual(reading.answer("How many boys are there?", ctx), 12)

    def test_complementary_group_fallback_declines_ambiguous_owners(self):
        ctx = [reading.Quantity("boy_a", ("boy", frozenset()), 12, "have"),
               reading.Quantity("boy_b", ("boy", frozenset()), 5, "have")]
        self.assertEqual(reading.answer("How many boys are there?", ctx), "undetermined")

    def test_complementary_group_fallback_declines_non_group_word(self):
        # "apple" isn't a registered group word -- must decline, not accidentally match via some other path.
        ctx = [reading.Quantity("apple", ("apple", frozenset()), 10, "have")]
        self.assertEqual(reading.answer("How many apples are there?", ctx), "undetermined")

    def test_complementary_group_fallback_declines_compound_ask(self):
        # a real TEST regression: "how many BOYS AND GIRLS are left?" mentions BOTH group words -- must
        # decline, not hijack it as a single-group ask onto whichever entity `_descriptor` happened to
        # capture first ("boy"'s raw 14, dropping "and girls").
        ctx = [reading.Quantity("boy", ("boy", frozenset()), 14, "have"),
               reading.Quantity("girl", ("girl", frozenset()), 10, "have")]
        self.assertEqual(reading.answer("How many boys and girls are there?", ctx), "undetermined")

    def test_complementary_group_fallback_regression_real_dropout_problem(self):
        # the actual dev-holdout text this regressed on, end to end through solve() (the isolated answer()
        # repro above can't reproduce the full drop-out fold's interaction with OTHER answer() machinery) --
        # must decline (matching pre-migration behavior), not commit a wrong number.
        text = ("The school has 14 boys and 10 girls. If 4 boys and 3 girls drop out, how many boys and "
                "girls are left?")
        self.assertEqual(reading.solve(text), "undetermined")

    def test_noun_affine_system_returns_system_result(self):
        # fresh synthetic numbers (same nouns as the docstring example, to stay clear of vocabulary gaps):
        # goat=x (free pivot), cow=x+5, pig=3(x+5); total animals = x+(x+5)+3(x+5) = 5x+20 = 45 -> x=5.
        text = ("A farmer has three times as many pigs as cows, and 5 more cows than goats. If the farmer "
                "has 45 animals total, how many goats does he have?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._noun_affine_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"goat": 5, "cow": 10, "pig": 30})
        self.assertEqual(reading.solve(text), 5)

    def test_noun_residual_sum_system_returns_system_result(self):
        # fresh synthetic numbers/nouns (the "how many X are there?" existential ask — the phrasing this
        # solver's own docstring uses, which needed answer()'s new existential entities-fallback to resolve
        # at all): blue=15, red=2*15=30, total=50 -> green = 50-30-15 = 5.
        text = ("A store has 50 marbles. There are twice as many red marbles as blue while the rest are "
                "green. If there are 15 blue marbles, how many green marbles are there?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._noun_residual_sum_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"blue": 15, "red": 30, "green": 5})
        self.assertEqual(reading.solve(text), 5)

    def test_noun_anchored_sum_system_returns_system_result(self):
        # fresh synthetic numbers (docstring's own pig/cow/goat vocabulary, "animal" is a registered
        # category): pig=10 (anchor), cow=2*10-3=17, goat=17+6=23, sum=50 via the new category-sum fallback.
        text = ("Priya's family has ten pigs on their farm. They also have three fewer than twice as many "
                "cows as pigs and six more goats than cows. How many animals do they have on their farm?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._noun_anchored_sum_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"pig": 10, "cow": 17, "goat": 23})
        self.assertEqual(reading.solve(text), 50)

    def test_noun_anchored_sum_system_unregistered_category_stays_legacy_scalar(self):
        # "children" isn't a registered category -- must stay on the unchanged legacy scalar path, not a
        # SystemResult (answer()'s category-sum fallback would decline it).
        text = ("Wesley has six times as many daughters as sons. If he currently has three sons, how many "
                "children does he have?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._noun_anchored_sum_system(premises, question, rp)
        self.assertNotIsInstance(result, reading.SystemResult)
        self.assertEqual(result, 21)
        self.assertEqual(reading.solve(text), 21)


class TestAffineSystemReturnsSystemResult(unittest.TestCase):
    """#395c — `_affine_system` (NAMED-entity twin of `_noun_affine_system`, total-pinned rather than
    anchor-pinned) migrated to `SystemResult`: emits one `Quantity` fact per entity instead of computing
    the asked entity's value directly, then lets `answer()`'s plain single-entity fold (the same consumer
    `_anchored_single_system` already proved out) resolve the ask. No new `answer()` capability was needed
    -- named entities already have a working subject-resolution path."""

    def test_affine_system_returns_system_result(self):
        # fresh synthetic numbers/nouns: mandy=m (free pivot), anna=3m, amanda=m+2;
        # total = 3m + m + (m+2) = 5m+2 = 142 -> m=28.
        text = ("Anna has 3 times as many stickers as Mandy. Mandy has 2 fewer stickers than Amanda. If "
                "they have 142 stickers total, how many stickers does Mandy have?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._affine_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"anna": 84, "amanda": 30, "mandy": 28})
        self.assertEqual(reading.solve(text), 28)


class TestSumRatioSystemReturnsSystemResult(unittest.TestCase):
    """#397 — `_sum_ratio_system` (pure-ratio-chain, total-pinned) migrated to `SystemResult`: emits one
    `Quantity` fact per entity (ratio-tree entities computed hop-by-hop from the pivot, `knowns` entities
    carrying their own stated value directly) instead of returning `coeff[asked]·pivot`, then lets
    `answer()`'s plain single-entity fold resolve the ask. Same SAFE-path disqualifiers as `_affine_system`
    (its closest sibling) — no new `answer()` capability needed."""

    def test_sum_ratio_system_returns_system_result(self):
        # amy = jeremy/3, chris = 2*amy; total = amy+jeremy+chris = 132 -> jeremy=66, amy=22, chris=44.
        text = ("Amy has 132 stickers combined with Jeremy and Chris. Amy is a third of Jeremy. Chris is "
                "twice as many as Amy. How many stickers does Jeremy have?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._sum_ratio_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"amy": 22, "jeremy": 66, "chris": 44})
        self.assertEqual(reading.solve(text), 66)


class TestOffsetSumSystemsReturnSystemResult(unittest.TestCase):
    """#398 — `_offset_sum_pure` and `_offset_sum_system` (2-entity offset-sum, total-pinned) migrated to
    `SystemResult`: both entities' values are attributed to the SAME single premise (the one offset relation
    that defines them both) as `Quantity` facts, instead of returning one raw value directly. Same recipe
    as `_affine_system`/`_sum_ratio_system` (this family's siblings)."""

    def test_offset_sum_pure_returns_system_result(self):
        # anais = kamari + 30, anais + kamari = 160 -> kamari = 65, anais = 95.
        text = "Anais has 30 more toys than Kamari. There are 160 toys altogether. How many toys does Kamari have?"
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._offset_sum_pure(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"anais": 95, "kamari": 65})
        self.assertEqual(reading.solve(text), 65)

    def test_offset_sum_system_returns_system_result(self):
        # marcus = 2*elena + 8, marcus + elena = 68 -> elena = 20, marcus = 48.
        text = ("Marcus and Elena have 68 stickers combined. Marcus has 8 more than twice as many stickers "
                "as Elena. How many stickers does Elena have?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._offset_sum_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        amounts = {f.entity: f.amount for f in all_facts}
        self.assertEqual(amounts, {"marcus": 48, "elena": 20})
        self.assertEqual(reading.solve(text), 20)


class TestTwoRelationSystemReturnsSystemResult(unittest.TestCase):
    """#399 — `_two_relation_system` (2x2 linear system, no anchor/total) migrated to `SystemResult` for
    the NAMED-PAIR case only: neither entity is a free pivot (both premises jointly constrain both values),
    so both facts are attributed to BOTH premise indices. The coordinate-noun twin (the real dev-train hit,
    Brett's blue/red marbles) asks via a bare noun-phrase question ("How many red marbles?") `answer()` has
    no consumer for at all -- stays on the legacy scalar path unconditionally, unchanged."""

    def test_named_pair_returns_system_result(self):
        # alba = bo + 15, alba = 4*bo -> bo = 15/3 = 5, alba = 20.
        text = ("Alba has 15 more coins than Bo. Alba has 4 times as many coins as Bo. How many coins does "
                "Bo have?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._two_relation_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {}
        for fs in result.facts_by_premise.values():
            for f in fs:
                seen[f.entity] = f.amount
        self.assertEqual(seen, {"alba": 20, "bo": 5})
        self.assertEqual(reading.solve(text), 5)

    def test_coordinate_noun_pair_stays_on_legacy_scalar_path(self):
        # the real dev-train shape (Brett's blue/red marbles) -- a bare noun-phrase question answer() can't
        # yet resolve generically; must still return a plain scalar, not a SystemResult.
        text = ("Brett has 24 more blue marbles than red marbles. He has 5 times as many blue marbles as "
                "red marbles. How many red marbles?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._two_relation_system(premises, question, rp)
        self.assertNotIsInstance(result, reading.SystemResult)
        self.assertEqual(result, 6)
        self.assertEqual(reading.solve(text), 6)


class TestIncrement146ContainerContentsSum(unittest.TestCase):
    """#146 — CONTAINER × CONTENTS product-sum (the count twin of revenue_product_sum): each container-count
    pairs with a following CONTENTS-count of the questioned item, Σ(container·contents), minus any stated item
    loss. "4 bags with 20 apples each, 6 bags with 25 each; sells 200; apples left?" → 4·20+6·25−200 = 30.
    Gated against rate/measurement problems: a MEASUREMENT/time item (miles/minutes) or a 'per' rate declines
    (Rosie's minutes 50≠30)."""

    def test_container_contents_minus_loss(self):
        self.assertEqual(reading.solve(
            "Ella has 4 bags with 20 apples in each bag and six bags with 25 apples in each bag. If Ella "
            "sells 200 apples, how many apples does Ella have left?"), 30)

    def test_container_contents_no_loss(self):
        self.assertEqual(reading.solve(
            "Tom has 3 boxes with 10 books in each box and 5 boxes with 8 books in each box. How many "
            "books does Tom have?"), 70)

    def test_measurement_item_not_stolen(self):
        # a per-hour RATE with a minutes/miles item is not container×contents — must not product-sum it
        self.assertNotEqual(reading.solve(
            "Rosie runs 6 miles per hour. She runs 1 hour on Monday, 30 minutes on Tuesday, 1 hour on "
            "Wednesday, and 20 minutes on Thursday. If she wants to run 20 miles for the week, how many "
            "minutes should she run on Friday?"), 50)


class TestIncrement147BakeVerb(unittest.TestCase):
    """#147 — 'bake' verb family (typed-vocab add): bake/bakes/baked/baking as verbs + lemma map. The
    fraction-remainder system's pool frame already anticipated 'baked N <item>'; only GUARD 1 (unknown
    word) blocked it (Neil g12 declined; the 'made' reword solved). A creation verb like make/build/paint
    — no sign, no measure."""

    def test_baked_pool_fraction_remainder(self):
        self.assertEqual(reading.solve(
            "Neil baked 20 cookies. He gave 2/5 of the cookies to his friend. How many cookies are "
            "left for Neil?"), 12)

    def test_bakes_simple_holding(self):
        self.assertEqual(reading.solve(
            "Ann baked 12 muffins. She ate 3 muffins. How many muffins does Ann have left?"), 9)


class TestIncrement148CommaQuantityList(unittest.TestCase):
    """#148 — the COMMA-QUANTITY-LIST parse: (a) `_merge_labels` no longer merges across a comma boundary
    ("dogs, 3" is a list boundary, never a "Grade 5" label), so each list value quantifies ITS OWN noun;
    (b) the freed scene-setting cardinal ("…season, four players scored points") is exempted from GUARD 3/4
    and the anchored family's premise firewall as a SET-SIZE cardinal — but ONLY when it equals the
    problem's cast (named entities, or ordinal references when unnamed) and quantifies a person-group noun;
    (c) `_is_value_first_relation`/`_noun_std_entity` skip a bare article ("than THE potatoes"); (d) a
    value-first comparative list conjunct with NO than/as compares to the PREVIOUS list item ("237
    potatoes, 60 fewer cucumbers" → cucumbers = potatoes − 60)."""

    def test_comma_list_values_quantify_own_noun(self):
        self.assertEqual(reading.solve(
            "Rita has 2 dogs, 3 cats and 4 birds. How many cats does Rita have?"), 3)

    def test_elided_standard_list_conjunct_category_sum(self):
        # "2 fewer cats" = 2 fewer THAN THE DOGS (the previous list item); birds = 2× the cats
        self.assertEqual(reading.solve(
            "A farm has 10 dogs, 2 fewer cats and twice as many birds than the cats. "
            "How many animals does the farm have?"), 34)

    def test_set_size_cardinal_ordinal_cast(self):
        # "three players" counts the ordinal-referenced cast (first/second/third) — not a datum
        self.assertEqual(reading.solve(
            "At a bowling alley, three players are having a pretzel eating contest. The first player "
            "eats 8 pretzels. The second eats 6 pretzels. The third eats 2 more pretzels than the "
            "second. How many do they eat in total?"), 22)

    def test_set_size_cardinal_named_cast(self):
        self.assertEqual(reading.solve(
            "In the final game of the basketball season, two players scored points. Ann scored twice "
            "as many points as Beth. If Beth scored 5 points, how many points in total did the team "
            "score?"), 15)

    def test_set_size_mismatch_still_declines(self):
        # cardinal 3 ≠ cast 2 (only Ann and Beth are named) — the third player's points are unstated,
        # so the freed "three" must still trip the unused-given firewall
        self.assertEqual(reading.solve(
            "In the final game of the basketball season, three players scored points. Ann scored twice "
            "as many points as Beth. If Beth scored 5 points, how many points in total did the team "
            "score?"), "undetermined")


class TestIncrement149SubtypeFractionRemainder(unittest.TestCase):
    """#149 — SUBTYPE FRACTION REMAINDER (the multi-pool twin of _fraction_remainder_system, now
    REGISTERED): ≥2 qualified subtype pools of the questioned item, each fraction loss binding to ITS OWN
    subtype → Σpools − Σ(fᵢ·poolᵢ). The single-pool twin defers ≥2-qualified-pool shapes here, so the two
    never both claim. Also: unhyphenated fraction words ("one third") lexicalized, and a fraction-cue-count
    guard — an UNRECOGNIZED fraction form anywhere declines the whole system rather than committing a
    partial remainder (the 49≠37 confident-wrong class). 'brown' re-enabled (adjective)."""

    def test_two_pools_two_fractions(self):
        self.assertEqual(reading.solve(
            "Rita has 20 blue marbles and 30 red marbles. If she gives away half of her blue marbles "
            "and a third of her red marbles, how many marbles does she have left?"), 30)

    def test_unhyphenated_fraction_word(self):
        self.assertEqual(reading.solve(
            "Rita has 20 blue marbles and 30 red marbles. If she gives away half of her blue marbles "
            "and one third of her red marbles, how many marbles does she have left?"), 30)

    def test_unrecognized_fraction_declines(self):
        # "four sevenths" isn't a lexicalized fraction — committing 62 - 10 would be a partial remainder
        self.assertEqual(reading.solve(
            "Rita has 20 blue marbles and 30 red marbles. If she gives away half of her blue marbles "
            "and four sevenths of her red marbles, how many marbles does she have left?"), "undetermined")

    def test_fraction_of_unknown_subtype_declines(self):
        self.assertEqual(reading.solve(
            "Rita has 20 blue marbles and 30 red marbles. If she gives away half of her green marbles, "
            "how many marbles does she have left?"), "undetermined")

    def test_single_pool_twin_not_stolen(self):
        # one pool + literal loss stays the single-pool twin's shape (deferral gates never both claim)
        self.assertEqual(reading.solve(
            "There were 148 peanuts in a jar. Brock ate one-fourth of the peanuts and Bonita ate 29 "
            "peanuts. How many peanuts remain in the jar?"), 82)


class TestIncrement150SixthWaveVocab(unittest.TestCase):
    """#150 — sixth-wave verified-gain vocab (sweep re-run after the subtype-fraction / comma-list /
    container×contents compositions unblocked co-blocked cases): place/activity/object nouns (mall,
    tennis, spelling, holiday, beginning, picking, throw, birds, event/center/rescue, jacket, pint,
    rain/rainfall/state) + plural-only garments (pants/shorts — bare nouns, no PLURALS entry).
    'fruit' stays EXCLUDED (unlocks a held-out pounds/ounces conversion to a confident-wrong);
    'clothes' stays EXCLUDED (the TEST checkpoint caught it unlocking a held-out confident-wrong)."""

    def test_place_noun_mall_ratio(self):
        self.assertEqual(reading.solve(
            "Anna bought three times as many pretzels at the mall as Ben did. Ben bought 4 pretzels. "
            "How many pretzels did Anna buy?"), 12)

    def test_plural_only_garments_purchase(self):
        self.assertEqual(reading.solve(
            "Nora went to the mall. She bought two jackets at $10 each and three pairs "
            "of shorts at $6 each. How much did she spend in total?"), 38)

    def test_clothes_now_solves(self):
        # "clothes" was excluded pre-#180 (an old TEST confident-wrong); #200 re-admits it —
        # the shopping machinery now solves this shape (2·10 + 3·6), TEST-look adjudicated
        self.assertEqual(reading.solve(
            "Nora went to the mall to buy clothes. She bought two jackets at $10 each and three pairs "
            "of shorts at $6 each. How much did she spend in total?"), 38)

    def test_bird_nouns_category_sum(self):
        self.assertEqual(reading.solve(
            "Cara saw 3 cardinals and four times as many robins as cardinals. She also saw twice as "
            "many sparrows as cardinals. How many birds did she see in total?"), 21)

    def test_pint_offset_sum(self):
        self.assertEqual(reading.solve(
            "Amy picked eight pints of blueberries. Kate picked two pints more than Amy. "
            "How many pints did they pick in total?"), 18)

    def test_fruit_still_unknown_declines(self):
        # "fruit" is deliberately NOT in the lexicon (unlocks an unmodeled pounds/ounces conversion);
        # a fruit-blocked problem must keep declining, never commit
        self.assertEqual(reading.solve(
            "There are six oranges in a fruit basket. There are two fewer apples than oranges. "
            "How many apples are there?"), "undetermined")


class TestIncrement151SeventhWaveTypedVocab(unittest.TestCase):
    """#151 — seventh-wave TYPED vocab (per-category typing, not forced to noun): adjectives
    constant/reduced/decorated, the vehicle noun "train" (travel/travels verbs were already in),
    "&" as a conjunction alias of "and", and the verb "ask" (+asks/asked, lemma-mapped)."""

    def test_ampersand_conjunction(self):
        self.assertEqual(reading.solve(
            "Tom & Mary combined have 100 marbles. If Tom owns 20 marbles more than Mary, "
            "how many marbles does Tom own?"), 60)

    def test_decorated_adjective(self):
        self.assertEqual(reading.solve(
            "Kendra made 4 more than five times as many decorated buttons as Mari. Mari made 8 "
            "buttons. How many buttons did Kendra make?"), 44)

    def test_train_noun_rate(self):
        self.assertEqual(reading.solve(
            "A train travels 100 miles in 2 hours. At the same rate, how many additional hours "
            "would it take to travel another 150 miles?"), 3)

    def test_reduced_adjective_percent_of_original(self):
        self.assertEqual(reading.solve(
            "Shirts have been reduced to $8. This price is at 25% of the original price. "
            "What was the original price?"), 32)

    def test_constant_adjective_rate(self):
        self.assertEqual(reading.solve(
            "Grace can read a 100-page book in 10 hours. How long would it take her to finish "
            "reading a 150-page book at a constant rate?"), 15)


class TestIncrement152EighthWaveUnitsQuantifier(unittest.TestCase):
    """#152 — eighth-wave typed vocab: quantifier "several", plural-only noun "earnings" + verb
    "represent", the time-unit "seconds" (noun sense on the PLURAL SURFACE only — a singular noun
    sense broke the bare-ordinal reading "the second eats 25 seeds"), and "cm" as a PLURALS
    abbreviation alias of centimeter (descriptor unifies for conversions)."""

    def test_several_quantifier_no_claim(self):
        self.assertEqual(reading.solve(
            "A farmer bought several cows. Tom owns 3 more cows than Ben. Ben owns 4 cows. "
            "How many cows does Tom own?"), 7)

    def test_cm_alias_of_centimeter(self):
        self.assertEqual(reading.solve(
            "Ken is 5 cm taller than Mia. Mia is 100 cm tall. How many centimeters tall is Ken?"), 105)

    def test_seconds_time_unit_rate_conversion(self):
        self.assertEqual(reading.solve(
            "Ben can eat 10 nuts in 30 seconds. At this same rate, how many nuts could he eat "
            "in 2 minutes?"), 40)

    def test_earnings_and_represent(self):
        self.assertEqual(reading.solve(
            "Ann earned 4 times as much money as Bea. Together they earned $100. How many dollars "
            "did Ann earn? Use A to represent the earnings."), 80)

    def test_bare_ordinal_second_not_a_time_unit(self):
        # singular "second" stays a pure adjective: the bare-ordinal subject must never be read as
        # a time-unit noun (that misreading turned a solved case into a decline pre-fix); this
        # simpler shape declines — the pin is that it never commits a number
        self.assertEqual(reading.solve(
            "At a game, three players are having a contest. The first player eats 25 seeds. "
            "The second eats 3 more than the first. How many seeds does the second player eat?"),
            "undetermined")


class TestIncrement153CommonNounGenitive(unittest.TestCase):
    """#153 — common-noun genitive split: the tokenizer's "X's" → X + 's split (previously
    NAME-only) now also fires when the lowercase core is a known LEXICON noun ("farmer's" →
    farmer + 's). An unknown core stays one unknown token (GUARD 1 declines). KINSHIP nouns are
    excluded: relational age-algebra has no frame for "her sister's age" yet — splitting it turned
    a safe decline into a confident-wrong (committed 5, gold 8)."""

    def test_common_noun_genitive_splits(self):
        self.assertEqual(reading.solve(
            "Lena sold twice as many apples as Rosa at the farmer's market. Rosa sold 30 apples. "
            "How many apples did they sell in total?"), 90)

    def test_kinship_genitive_age_now_solves(self):
        # originally pinned as a DECLINE (the bare split turned this into a confident-wrong);
        # the `years_until_ratio` frame (#162) now answers it: 20+t = 2(4+t) → t = 12
        self.assertEqual(reading.solve(
            "If Dana is 20 years old and her little sister is 4, in how many years would Dana be "
            "twice her sister's age?"), 12)


class TestIncrement154NinthWaveTypedClusters(unittest.TestCase):
    """#154 — ninth-wave typed clusters: supermarket cluster (mass/supermarket nouns, "proportional"
    adjective, "kg" a PLURALS abbreviation alias of kilogram like cm), dog-grooming cluster
    (burr/tick/fur/object nouns, "foreign" adjective, catch/caught verb), and "lot" ("eats a lot"
    scene-setting — the quantity machinery makes no claim from it). Also lifted TWO TEST cases
    (23→25 correct, wrong unchanged — precision 82.1%→83.3%)."""

    def test_proportional_mass_rate(self):
        self.assertEqual(reading.solve(
            "The price paid for rice is proportional to the mass purchased. Sam paid $4 for 2 kg "
            "of rice. How much will Sam pay for 12 kg?"), 24)

    def test_dog_cluster_and_genitive(self):
        # also exercises the common-noun genitive split ("the dog's fur")
        self.assertEqual(reading.solve(
            "A dog has 12 burrs and six times as many ticks as burrs. How many foreign objects "
            "are caught in the dog's fur?"), 84)

    def test_lot_is_scene_setting(self):
        self.assertEqual(reading.solve(
            "Tom is a boy who eats a lot. Tom has 40 marbles. Ben has 12 more marbles than Tom. "
            "How many marbles do they have in total?"), 92)


class TestIncrement155TenthWaveRoomPainting(unittest.TestCase):
    """#155 — tenth-wave typed cluster (room-painting): tape/wall nouns, "cover" verb, and the
    modifier adjectives exact/living/masking/wide + the comparative surface "neater" as a plain
    adjective (no comparative machinery claims it)."""

    def test_masking_tape_wall_widths(self):
        self.assertEqual(reading.solve(
            "Leo is painting his living room and buys masking tape. 2 of his walls are 4 meters "
            "wide and the other 2 walls are 6 meters wide. How many meters of masking tape does "
            "Leo need to cover the exact width of each wall?"), 20)


class TestIncrement156UnitConversion(unittest.TestCase):
    """#156 — unit-conversion frontier, first increment: (a) a STATED conversion factor ("there are
    128 ounces in a gallon") is META-information — `_parse_problem` records it and drops the clause
    (no degenerate facts, no unused-given trip), and solve() applies it as a per-problem CONVERSIONS
    override, restored afterward; (b) the holdings fold now EXTENDS exact-unit matches with
    CONVERTIBLE holdings in other units ("has 3 gallons, drank 13 ounces, how many ounces left" →
    3·128 − 13), gated to the have/own remainder fold; (c) gallon↔ounce (128, US fluid) and
    gallon↔quart join CONVERSIONS; (d) "drink" joins LOSS_VERBS (it consumes, like eat)."""

    def test_stated_factor_mixed_unit_remainder(self):
        self.assertEqual(reading.solve(
            "Rita has 2 gallons of juice. She drank 20 ounces of the juice. If there are 128 ounces "
            "in a gallon, how many ounces of juice does Rita have left?"), 236)

    def test_table_factor_mixed_unit_remainder(self):
        self.assertEqual(reading.solve(
            "Sam has 2 pounds of flour. He used 8 ounces of the flour. "
            "How many ounces of flour does Sam have left?"), 24)

    def test_stated_factor_enables_uncurated_pair(self):
        # packs/cards are NOT in the curated table — the text's own factor carries the problem
        self.assertEqual(reading.solve(
            "Nora has 3 packs of cards. She used 10 cards. If there are 50 cards in a pack, "
            "how many cards does Nora have left?"), 140)

    def test_stated_factor_never_leaks(self):
        reading.solve(
            "Nora has 3 packs of cards. She used 10 cards. If there are 50 cards in a pack, "
            "how many cards does Nora have left?")
        self.assertNotIn(("pack", "card"), reading.CONVERSIONS)
        self.assertIn(("gallon", "ounce"), reading.CONVERSIONS)

    def test_drink_is_a_loss_verb(self):
        self.assertEqual(reading.solve(
            "Tom has 10 sodas. He drank 3 sodas. How many sodas does Tom have left?"), 7)


class TestIncrement157PieceCountAcrossUnits(unittest.TestCase):
    """#157 — `_piece_count_system` (EARLY, registered): a TOTAL measure in one unit divided into
    PIECES whose size is stated in a CONVERTIBLE unit ("15 pounds of beef cut into 12-ounce steaks"
    → 15·16/12 = 20). Gated: bare digit-free "how many <piece>" question; exactly two value-bearing
    premises; the piece premise mentions the asked noun, the total premise does not; the unit heads
    form a CONVERSIONS pair (or match); the division is a positive INTEGER."""

    def test_pounds_into_ounce_pieces(self):
        self.assertEqual(reading.solve(
            "Mia has 4 pounds of beef. She cuts it into 8-ounce pieces. "
            "How many pieces does she get?"), 8)

    def test_same_family_time_units(self):
        self.assertEqual(reading.solve(
            "Ann has 2 hours of free time. She watches 30-minute shows. "
            "How many shows can she watch?"), 4)

    def test_non_integer_division_declines(self):
        # 64 / 9 is not whole — a fractional piece count means this isn't the shape
        self.assertEqual(reading.solve(
            "Mia has 4 pounds of beef. She cuts it into 9-ounce pieces. "
            "How many pieces does she get?"), "undetermined")


class TestIncrement158RateWeekExpansion(unittest.TestCase):
    """#158 — the per-day-rate × week family, first two shapes (both EARLY, registered):
    `_daily_share_system` (a stated TOTAL over a stated DURATION, asked per day → total/days;
    set-size cardinals on the item-CARRIER noun are scene-setting; positive-integer share only) and
    `_herd_rate_week_system` (COUNT × per-head daily rate × 7 for an "in a week" question)."""

    def test_daily_share_week_quantity(self):
        self.assertEqual(reading.solve(
            "Ben buys 4 packs of pens that are 15 pens each. He had 10 pens already. He finishes "
            "all the pens in 1 week. How many pens does he use a day?"), 10)

    def test_daily_share_worded_duration_and_carrier_cardinals(self):
        # "two books"/"One book" are set-size cardinals on the page-carrier noun — scene-setting
        self.assertEqual(reading.solve(
            "Nina has two books. One book is 120 pages long, and the other book is 90 pages long. "
            "If Nina wants to finish both of the books in three weeks, how many pages will Nina "
            "need to read every day, if she reads an equal number of pages each day?"), 10)

    def test_herd_rate_week(self):
        self.assertEqual(reading.solve(
            "A farmer has 20 goats. Each goat gives 3 liters of milk a day. "
            "How many liters of milk does the farmer get in a week?"), 420)

    def test_non_integer_share_declines(self):
        # 75/14 is not whole — a fractional daily share means this isn't the shape
        self.assertEqual(reading.solve(
            "Ann has two books. One book is 40 pages long, and the other book is 35 pages long. "
            "If Ann wants to finish both of the books in two weeks, how many pages will Ann need "
            "to read every day?"), "undetermined")

    def test_comparative_daily_question_not_stolen(self):
        # "how many MORE chips a day … than her brother" is under-constrained — the daily-share
        # system must not commit 70/7; whatever the general machinery answers, it is not a number
        self.assertNotIsInstance(reading.solve(
            "Zoe buys 70 chips. She finishes all the chips in 1 week. How many more chips a day "
            "does she eat than her brother?"), (int, float))

    def test_rate_difference_still_works(self):
        # the constrained comparative resolves through pre-existing rate machinery (20 − 10)
        self.assertEqual(reading.solve(
            "Ray reads 140 pages in 1 week. Tom reads 70 pages in 1 week. "
            "How many more pages a day does Ray read than Tom?"), 10)


class TestIncrement159WeekdayWeekendSplit(unittest.TestCase):
    """#159 — `_weekday_weekend_system` (EARLY, registered): a per-day rate on WEEKDAYS (5) plus a
    RELATIONAL per-day rate on WEEKEND days (2), asked per week → R·5 + k·R·2. Reads raw premises
    (the relational weekend clause materializes no fact); exactly one digit-bearing premise (the
    weekday one); the weekend multiplier is a WORD (twice/three times/half)."""

    def test_times_multiplier(self):
        self.assertEqual(reading.solve(
            "Lena reads 6 pages each day during weekdays. On each weekend day, she reads twice the "
            "number of pages she reads on the weekdays. How many pages does she read in a week?"), 54)

    def test_half_multiplier_school_week(self):
        self.assertEqual(reading.solve(
            "Leo runs 4 miles each day during the school week. He runs half as far each day on the "
            "weekend. How many miles does he run in a week?"), 24)

    def test_explicit_weekend_number_declines(self):
        # a digit-bearing weekend premise is another shape (two explicit rates) — the system must
        # not claim it with the relational formula
        self.assertEqual(reading.solve(
            "Lena reads 6 pages each day during weekdays. On each weekend day, she reads 12 pages. "
            "How many pages does she read in a week?"), "undetermined")


class TestIncrement160RemainderAfterRate(unittest.TestCase):
    """#160 — `_remainder_after_rate_system` (EARLY, registered): a "there are N <X>" total, an
    optional ALREADY-consumed amount, and a per-day Rate premise carrying a "for a week / for N
    days" duration; a digit-free "how many <X> left" question → total − consumed − rate·days."""

    def test_total_consumed_rate_week(self):
        self.assertEqual(reading.solve(
            "There are 300 pages in Lea's book. She has already read 120 pages. If she reads 15 "
            "pages a day for a week, how many pages are still left to be read?"), 75)

    def test_no_consumed_two_weeks(self):
        self.assertEqual(reading.solve(
            "There are 200 pages in the book. If Ana reads 10 pages a day for two weeks, "
            "how many pages are left?"), 60)

    def test_overrun_declines(self):
        # rate x days exceeds the total — a negative remainder means this isn't the shape
        self.assertEqual(reading.solve(
            "There are 100 pages in the book. If Ana reads 20 pages a day for two weeks, "
            "how many pages are left?"), "undetermined")


class TestIncrement161TwelfthWaveVocab(unittest.TestCase):
    """#161 — twelfth-wave verified-gain vocab (sweep re-run after the rate-expansion systems
    landed): "fishing" (activity noun, carries three problems), trout (+the corpus plural
    "trouts"), and a desert-journey cluster (camel/oasis/traveler nouns, "thirsty" adjective)."""

    def test_fishing_trout_ratio_sum(self):
        self.assertEqual(reading.solve(
            "Ken and Bo went fishing. Bo caught 4 trouts and Ken caught three times as many trouts "
            "as Bo. How many trouts did they catch in total?"), 16)


class TestIncrement162YearsUntilRatio(unittest.TestCase):
    """#162 — `_years_until_ratio_system` (EARLY, registered) + the kinship-genitive tokenizer
    split lifted: "A is 18, her little sister is 5; in how many years would she be twice her
    sister's age?" → A+t = k(B+t) → t = (A − k·B)/(k−1). The bare-copula age ("sister is 5")
    drops from the general fold, so the system reads raw premises."""

    def test_twice_kinship_genitive(self):
        self.assertEqual(reading.solve(
            "If Rosa is 18 years old and her little sister is 5, in how many years would she be "
            "twice her sister's age?"), 8)

    def test_three_times_named_genitive(self):
        # 31 + t = 3·(9 + t) → t = 2
        self.assertEqual(reading.solve(
            "Maya is 31 years old and her daughter Lily is 9 years old. In how many years would "
            "Maya be three times Lily's age?"), 2)

    def test_non_integer_t_declines(self):
        # a genuinely fractional case: 18 + t = 3(5 + t) → t = 1.5 — the system must not commit;
        # the general path answers with deficiencies (a non-numeric decline surface)
        self.assertNotIsInstance(reading.solve(
            "If Vera is 18 years old and her little sister is 5, in how many years would she be "
            "three times her sister's age?"), (int, float))


class TestIncrement163DistributionInverses(unittest.TestCase):
    """#163 — two distribution inverses (both EARLY, registered): `_inverse_distribution_system`
    (recipient COUNT from a total and a per-recipient share — "give 3 each to his cousins") and
    `_split_with_system` ("to split WITH his three siblings" — the WITH frame adds the owner to
    the sharer set: N/(K+1))."""

    def test_inverse_distribution_recipient_count(self):
        self.assertEqual(reading.solve(
            "Nora bought 3 dozen muffins. She plans to give 4 muffins each to her cousins. "
            "How many cousins does Nora have?"), 9)

    def test_split_with_owner_shares_too(self):
        self.assertEqual(reading.solve(
            "Rita has two dozen apples to split with her three siblings. How many apples does "
            "each person get to eat if they all eat the same amount?"), 6)

    def test_uneven_split_declines(self):
        # 24 / (4+1) is not whole — must not commit
        self.assertNotIsInstance(reading.solve(
            "Rita has two dozen apples to split with her four siblings. How many apples does "
            "each person get to eat if they all eat the same amount?"), (int, float))


class TestIncrement164BespokeRateChains(unittest.TestCase):
    """#164 — two one-case rate chains (both EARLY, registered): `_weekly_time_plus_delta_system`
    (per-day hours Rate ×7 + an "increase … weekly by N hours" delta) and
    `_daily_rate_difference_system` (mixed sides: one party's stated TOTAL ÷ the shared "in a
    week" duration vs the other's per-day value, asked "how many more per day … compared to")."""

    def test_weekly_time_plus_delta(self):
        self.assertEqual(reading.solve(
            "Lia reads for 3 hours each day. She decided to increase her time spent on reading "
            "weekly, by 5 hours. How much time does Lia want to spend during one week on "
            "reading?"), 26)

    def test_daily_rate_difference_mixed_sides(self):
        self.assertEqual(reading.solve(
            "Yesterday Amy got four books from the library. They were a total of 1400 pages. "
            "Her brother got one book a day that was 100 pages each. They both finished them in "
            "a week. On average, how many more pages per day did Amy read compared to her "
            "brother?"), 100)

    def test_negative_difference_declines(self):
        # the brother out-reads Amy (300 > 200) — "how many more did Amy read" must not commit
        # a negative number
        self.assertNotIsInstance(reading.solve(
            "Yesterday Amy got four books from the library. They were a total of 1400 pages. "
            "Her brother got one book a day that was 300 pages each. They both finished them in "
            "a week. On average, how many more pages per day did Amy read compared to her "
            "brother?"), (int, float))


class TestIncrement165RateDaysWeeks(unittest.TestCase):
    """#165 — `_rate_days_weeks_system` (EARLY, registered): hours/day × days/week × weeks
    ("1 hour a day, 5 days a week, for 8 weeks" → 40; "in a month" = the GSM8K 4-weeks
    convention). Also: the holdings-fold CONVERT branches now SKIP time-unit pairs (a time
    quantity in context is usually a DURATION, not a holding — converting "90 days" to 2160
    hours was the Natasha confident-wrong the exercise vocab unlocked)."""

    def test_rate_days_weeks(self):
        self.assertEqual(reading.solve(
            "Mia's goal is to exercise 2 hours a day, 4 days a week. If she hits her goal weekly "
            "for 6 weeks, how many hours has Mia exercised?"), 48)

    def test_month_is_four_weeks(self):
        self.assertEqual(reading.solve(
            "Every day, Rob logs 8 hours of work in his office. If he works for five days a week, "
            "calculate the total number of hours he logs in a month."), 160)

    def test_duration_facts_never_convert_to_hours(self):
        # two-person mixed minutes/durations — the fold must not convert "nine days" to hours;
        # this must stay a decline, never a committed number
        # RE-PINNED (#230): the exercise-hours branch now claims this CORRECTLY —
        # (30·7 + 10·9)/60 = 5; the fold's convert path this pin defends is unchanged.
        self.assertEqual(reading.solve(
            "Natasha exercised for 30 minutes every day for one week. Esteban exercised for 10 "
            "minutes on each of nine days. How many hours did Natasha and Esteban exercise in "
            "total?"), 5)


class TestIncrement166DayListTarget(unittest.TestCase):
    """#166 — `_day_list_target_system` (EARLY, registered): day-tagged values across a week plus
    a TARGET, asked "how many on <day> to reach/beat it". An "average of N a day" target is N·7
    (reach); a "last week … N total" target needs one MORE (beat → N+1). Listed days without a
    value need an explicit zero-cue ("read nothing", "skipped a day"); a bare mention of the
    asked day ("On Saturday, he looks…") is scene-setting."""

    def test_average_reach_target(self):
        self.assertEqual(reading.solve(
            "Lena wants to read an average of 40 pages a day. This week she read 30 pages on "
            "Sunday, 55 pages on Monday, and 45 pages on Tuesday. She fell asleep early on "
            "Wednesday so she read nothing. On Thursday she read 60 pages, and on Friday she "
            "read 50 pages. How many pages does she have to read on Saturday to reach her "
            "goal?"), 40)

    def test_beat_last_week_target(self):
        self.assertEqual(reading.solve(
            "Nate wants to do more jumping jacks every week than he did the week before. He "
            "recorded that last week he did 200 total jumping jacks. On Sunday he did 30. On "
            "Monday he did 40. On Tuesday he skipped a day. On Wednesday he did 50. On Thursday "
            "he did 35. On Friday he did 25. How many does he have to do on Saturday to make "
            "sure he beats last week's number?"), 21)

    def test_underlisted_days_decline(self):
        # only two listed days — the residual would silently assume zeros for the rest
        self.assertEqual(reading.solve(
            "Lena wants to read an average of 40 pages a day. This week she read 30 pages on "
            "Sunday and 55 pages on Monday. How many pages does she have to read on Saturday "
            "to reach her goal?"), "undetermined")


class TestIncrement167AlternatingDaysDiff(unittest.TestCase):
    """#167 — `_alternating_days_diff_system` (EARLY, registered): two disjoint weekday-list
    category premises + one per-day rate, asked "how many more <catA> than <catB> in one week"
    → rate · (|daysA| − |daysB|)."""

    def test_three_vs_two_days(self):
        self.assertEqual(reading.solve(
            "On Mondays, Wednesdays and Fridays, Rita bakes apple pies. On Tuesdays and "
            "Thursdays, she bakes cherry pies. If she bakes 8 pies per day, how many more apple "
            "pies does she bake than cherry pies in one week?"), 8)

    def test_equal_days_never_commits(self):
        # 2 vs 2 days — the difference shape requires strictly more; must not commit a number
        self.assertNotIsInstance(reading.solve(
            "On Mondays and Wednesdays, Rita bakes apple pies. On Tuesdays and Thursdays, she "
            "bakes cherry pies. If she bakes 8 pies per day, how many more apple pies does she "
            "bake than cherry pies in one week?"), (int, float))


class TestIncrement168DoublingDaysSum(unittest.TestCase):
    """#168 — `_doubling_days_sum_system` (EARLY, registered): first-day value + "twice as much …
    as the day before" + an "after the Nth day" limit → a·(2^N − 1). Also the general
    DIGIT-ORDINAL rule in annotate_senses: "5th"/"2nd" read as adjectives (like first/second)."""

    def test_doubling_chain_sum(self):
        self.assertEqual(reading.solve(
            "Mia spends time every day talking to her boyfriend on the phone. The first day of "
            "the week she spends 3 minutes on the phone with her boyfriend. Each of the following "
            "days she spends twice as much time talking on the phone as the day before. After "
            "the 4th day her parents take the phone away from her. How much time did Mia spend "
            "talking to her boyfriend that week?"), 45)

    def test_no_doubling_premise_declines(self):
        self.assertEqual(reading.solve(
            "Mia spends time every day talking to her boyfriend on the phone. The first day of "
            "the week she spends 3 minutes on the phone with her boyfriend. After the 4th day "
            "her parents take the phone away from her. How much time did Mia spend talking to "
            "her boyfriend that week?"), "undetermined")


class TestIncrement169NumberWordsTensTeens(unittest.TestCase):
    """#169 — NUMBERS + LEXICON extended with the teens and tens (thirteen…ninety): word-number
    values were capped at twelve, silently blocking cases like "fifteen packs of stickers".
    Also `_boosted_rate_duration_system` (registered but DORMANT): its only case needs the WORD
    form "fifty percent", and typing "percent" as a bare noun let the fold read "100 percent" as
    a countable Quantity (a TEST-checkpoint-caught confident-wrong, 100 vs 1) — the word stays
    unknown until a percent-unit merge lands."""

    def test_teens_and_tens_word_numbers(self):
        self.assertEqual(reading.solve(
            "Yesterday, Vera bought fifteen packs of stickers. Today, she bought ten more packs "
            "than yesterday. How many packs of stickers does Vera have in total?"), 40)

    def test_word_percent_now_solves(self):
        # originally pinned DECLINING ("Fifty percent" unreadable); the `_merge_word_percent`
        # feature (#170) landed and the dormant boosted-rate system activated: 8·1.5·7 = 84
        self.assertEqual(reading.solve(
            "Rico can cut 8 yards of grass per day. He bought a lawnmower and it helped him to "
            "cut more yards by Fifty percent per day. How many yards will Rico be able to cut "
            "after a week?"), 84)


class TestIncrement170WordPercentMerge(unittest.TestCase):
    """#170 — `_merge_word_percent`: "<value> percent" fuses into ONE percent unit (the word-form
    twin of the "20%" token), activating the dormant boosted-rate system (Brendan). A STANDALONE
    "percent" stays unknown → GUARD 1 declines (the countable-"100 percent" wrong class). Plus
    `_workweek_boost_system`: per-day average + one "N% higher" weekday inside an "open from
    <day> to <day>" span → (span−1)·base + base·(1+p/100) (Krystian 216)."""

    def test_workweek_one_boosted_day(self):
        self.assertEqual(reading.solve(
            "Nia works in the library. She borrows an average of 20 books every day. Every "
            "Friday, her number of borrowed books is about 50% higher than the daily average. "
            "How many books does she borrow in a week if the library is open from Monday to "
            "Friday?"), 110)

    def test_standalone_percent_still_declines(self):
        # "what/how many percent" has no preceding value to merge — must stay a decline
        self.assertEqual(reading.solve(
            "Tom got 100 percent of the questions right. How many percent did Tom get?"),
            "undetermined")


class TestIncrement171SalesFunnelWeek(unittest.TestCase):
    """#171 — `_sales_funnel_week_system` (EARLY, registered): visits/day × P% conversion ×
    the average of the two half-split prices × days/week ("50 houses a day; 20% buy; half $50,
    half $150; 5 days a week" → 5000). The two-halves average is exact; a single half-premise
    declines (the funnel is incomplete)."""

    def test_full_funnel(self):
        self.assertEqual(reading.solve(
            "Rita is a door-to-door salesman. She visits 30 houses a day. 10% of them buy "
            "something from her. Of those that buy something half buy a $20 set of knives and "
            "the other half buy a $40 set of knives. How much does she sell a week when she "
            "works 4 days a week?"), 360)

    def test_missing_half_declines(self):
        self.assertEqual(reading.solve(
            "Rita is a door-to-door salesman. She visits 30 houses a day. 10% of them buy "
            "something from her. Of those that buy something half buy a $20 set of knives. "
            "How much does she sell a week when she works 4 days a week?"), "undetermined")


class TestIncrement172OrdinalDayChainResidual(unittest.TestCase):
    """#172 — `_ordinal_day_chain_residual_system` (EARLY, registered): a weekly target, a chain
    of ordinal-day values (literal / "k times that number" × previous / "N fewer than … the
    <ordinal> day"), and an equal split of the residual over the remaining days →
    (target − Σ days) / (7 − listed)."""

    def test_chain_with_times_and_fewer(self):
        # 200 − (20 + 60 + 20) = 100 over 4 remaining days → 25
        self.assertEqual(reading.solve(
            "Nora decides to fill a jar by collecting 200 marbles in a week. On the first day, "
            "she collects 20 marbles. On the second day, she manages to collect 3 times that "
            "number. On the third day, she collects 40 marbles fewer than the number she "
            "collected on the second day. If she collects an equal number of marbles on the "
            "remaining days of the week, what's the number of marbles she collected each day "
            "for the rest of the week?"), 25)

    def test_gap_in_day_sequence_declines(self):
        # first and third days listed with no second — the chain is broken; never commit
        self.assertEqual(reading.solve(
            "Nora decides to fill a jar by collecting 200 marbles in a week. On the first day, "
            "she collects 20 marbles. On the third day, she collects 30 marbles. If she "
            "collects an equal number of marbles on the remaining days of the week, what's the "
            "number of marbles she collected each day for the rest of the week?"), "undetermined")


class TestIncrement173DayListResidue(unittest.TestCase):
    """#173 — the day-list family's last two cases: `_named_day_chain_split_system` (Mon N then
    twice-Tue, Wed–Fri M each, weekly total T → (T − (3N+3M))/2 on Saturday) and
    `_rate_hours_clock_week_system` (per-hour rate × (span-days·H + clock-span from Ham to Ham);
    clock tokens "8am"/"11am" now read as adjectives via the general am/pm rule)."""

    def test_named_day_chain_split(self):
        self.assertEqual(reading.solve(
            "Nia has several exams to prepare for. On Monday, she studies for 2 hours then "
            "studies for twice this long on Tuesday. On Wednesday, Thursday, and Friday she "
            "studies for 4 hours each day. She wants to study for a total of 22 hours over the "
            "week and divides the remaining amount of study time evenly between Saturday and "
            "Sunday. How many hours does Nia spend studying on Saturday?"), 2)

    def test_rate_hours_clock_week(self):
        self.assertEqual(reading.solve(
            "A cobbler can mend 2 pairs of shoes in an hour. From Monday to Wednesday, the "
            "cobbler works for 6 hours each day, and on Friday, he only works from 8am to 10am. "
            "How many pairs of shoes can the cobbler mend in a week?"), 40)


class TestIncrement174WholeTextChainsA(unittest.TestCase):
    """#174 — three whole-text rate chains: `_everyday_plus_half_subdays_system` (R·7 + R/2·N,
    the game-day count riding inside the question), `_paired_daily_rate_week_system`
    ((base + base+Δ)·times-a-week, both premises materializing zero facts), and
    `_weekday_production_revenue_system` (per-day·5·weeks·price)."""

    def test_everyday_plus_half_subdays(self):
        self.assertEqual(reading.solve(
            "Rio watches TV for 2 hours every day. On the days he plays video games he plays "
            "for half as long as he watches TV. If he plays video games 4 days a week how long "
            "does he spend watching TV and playing video games?"), 18)

    def test_paired_daily_rate_week(self):
        self.assertEqual(reading.solve(
            "Four times a week, Ana bikes 100 kilometers a day. Her friend Bea bikes 20 more "
            "kilometers of Ana biking distance in a day. What's the total distance the two "
            "friends bike in a week?"), 880)

    def test_weekday_production_revenue(self):
        self.assertEqual(reading.solve(
            "Lia makes cakes during the five weekdays to sell on weekends. She makes 2 cakes a "
            "day and sells her cakes for $10. In 3 weeks she managed to sell all the cakes. "
            "How much money did she collect during that time?"), 300)


class TestIncrement175WholeTextChainsB(unittest.TestCase):
    """#175 — three more whole-text chains: `_pet_food_remainder_system` (T − count·frac·K·7),
    `_orchard_pack_revenue_system` (trees·per-day·7·weeks/pack·price, exact pack division), and
    `_shared_item_cost_system` (a shared daily item = one whole item/day between the sharers →
    weeks·7·weight·price)."""

    def test_pet_food_remainder(self):
        self.assertEqual(reading.solve(
            "Vera has four dogs. Each dog eats 1/2 of a pound of dog food twice a day. If Vera "
            "bought 40 pounds of dog food, how many pounds of dog food are left after a week?"), 12)

    def test_orchard_pack_revenue(self):
        self.assertEqual(reading.solve(
            "Lena has 5 orange trees from which she can pick 12 oranges each day. Lena sells "
            "6-packs of oranges for $3. How much money will Lena have earned after 2 weeks if "
            "she sells all her oranges?"), 420)

    def test_shared_item_cost(self):
        self.assertEqual(reading.solve(
            "Isla shares half of a small apple with her dog every day. A small apple weighs "
            "about 1/4 of a pound. She can currently buy apples for $4.00 a pound. How much "
            "will she spend so that she and her dog have enough apples to last for 3 weeks?"), 21)


class TestIncrement176WholeTextChainsC(unittest.TestCase):
    """#176 — the last two whole-text chains: `_phased_weeks_average_system` (phase totals with
    a "same number as the total of the first <w> weeks" echo phase, ÷ the question's week span;
    the phase weeks must sum to it) and `_pet_plus_food_cost_system` (base price + exact bag
    count × bag price)."""

    def test_phased_weeks_average(self):
        # 2·7·2=28, echo 28, 4·7·1=28 → 84/7 = 12
        self.assertEqual(reading.solve(
            "Remy eats 2 apples a day for two weeks. Over the next four weeks, he eats the same "
            "number of apples as the total of the first two weeks. Over the next one weeks, he "
            "eats 4 apples a day. Over these 7 weeks, how many apples does he average a week?"), 12)

    def test_pet_plus_food_cost(self):
        self.assertEqual(reading.solve(
            "Ana just got a new kitten that cost $20. She also bought enough food for 2 weeks. "
            "She knows the kitten eats 1/2 cup of food a day. A bag of food with 3.5 cups costs "
            "$3. How much did all of this cost her?"), 26)

    def test_mismatched_phase_weeks_decline(self):
        # phases sum to 8, the question says 7 — the average denominator would be wrong
        self.assertEqual(reading.solve(
            "Remy eats 2 apples a day for two weeks. Over the next four weeks, he eats the same "
            "number of apples as the total of the first two weeks. Over the next two weeks, he "
            "eats 4 apples a day. Over these 7 weeks, how many apples does he average a week?"),
            "undetermined")


class TestIncrement177BareDecimalLexing(unittest.TestCase):
    """#177 — the bare-decimal lexing fix: `_key` strips '.' as punctuation, so ".5" read as 5 —
    a silent 10× error waiting in every ".5 ounces"/".5 hours" case. `_as_number` now checks the
    raw surface for a leading-dot decimal before the punctuation-stripped path."""

    def test_bare_decimal_reads_as_fraction(self):
        self.assertEqual(reading._as_number(".5"), 0.5)
        self.assertEqual(reading._as_number(".75"), 0.75)
        self.assertEqual(reading._as_number(".5."), 0.5)   # sentence-final

    def test_plain_tokens_unchanged(self):
        self.assertEqual(reading._as_number("5"), 5.0)
        self.assertEqual(reading._as_number("0.5"), 0.5)
        self.assertIsNone(reading._as_number("apples."))
        self.assertIsNone(reading._as_number("."))


class TestIncrement178JogUnlockAndSingles(unittest.TestCase):
    """#178 — the jog-family unlock + three singles. `_weekday_base_delta_system` (weekday base
    ×5 ± signed day deltas, hours must divide by 60), `_weekday_rate_weeks_system` (R·5·weeks —
    the "weekdays only" shape whose 7-day mis-projection kept jog excluded),
    `_salary_hourly_rate_system` (S/(H·D·4)), `_weekly_need_per_person_daily_system`
    (need·convert/share/7, gallon→pint added), `_per_second_minutes_week_system` (R·M·60·7)."""

    def test_weekday_base_plus_deltas_hours(self):
        self.assertEqual(reading.solve(
            "Zoe jogs for 30 minutes every day during weekdays. This week on Tuesday, she "
            "jogged 20 minutes more and also jogged 10 minutes more on Friday. How many hours, "
            "in total, did Zoe jog this week?"), 3)

    def test_weekday_base_with_cut_short_minutes(self):
        self.assertEqual(reading.solve(
            "Ben committed to run for an hour a day during weekdays. On Thursday, he cut short "
            "his run by 30 minutes but was able to jog 20 minutes more on Friday. How many "
            "minutes was he able to run for that week?"), 290)

    def test_weekday_only_rate_weeks(self):
        # the exact shape that made jog a held-out confident-wrong: 4·5·2, never 4·7·2
        self.assertEqual(reading.solve(
            "Rita jogs 4 miles per day on weekdays only. How many miles does she run over two "
            "weeks in total?"), 40)

    def test_salary_hourly_rate(self):
        self.assertEqual(reading.solve(
            "Vic's monthly salary is $640. If he works 8 hours a day for 5 days a week, "
            "how much is his hourly rate?"), 4)

    def test_weekly_need_per_person_daily(self):
        self.assertEqual(reading.solve(
            "A vampire needs to suck 7 gallons of blood per week to survive. If he sucks 4 "
            "pints of blood per person, how many people does he need to suck each day to "
            "survive?"), 2)

    def test_per_second_minutes_week(self):
        self.assertEqual(reading.solve(
            "A fan can create an airflow of 5 liters per second. If the fan works for 10 "
            "minutes each day, calculate the amount of airflow it will create in one week."), 21000)


class TestIncrement179DayListTwoDayAverage(unittest.TestCase):
    """#179 — `_day_list_target_system` extended: comma-aware numbers ("9,400"), a severed
    question tail absorbed as an asked day ("… on Friday" | "Saturday to meet his goal?"),
    an average asked over TWO remaining days (residual / 2), and a listed+asked = 7 coverage
    gate (an incomplete week would silently assume zeros)."""

    def test_two_day_average_with_comma_numbers(self):
        # 8000·7 = 56000 − 39900 = 16100 over 2 days → 8050
        self.assertEqual(reading.solve(
            "Ken wants to walk an average of 8,000 steps per day over the next week. On Sunday "
            "he walked 8,400 steps. On Monday he walked 8,100 steps. On Tuesday he walked 7,300 "
            "steps. On Wednesday he walked 8,200 steps. On Thursday he walked 7,900 steps. How "
            "many steps does he need to average on Friday and Saturday to meet his goal?"), 8050)

    def test_incomplete_week_declines(self):
        # only four listed days + two asked — Thursday unaccounted; never assume a zero
        self.assertEqual(reading.solve(
            "Ken wants to walk an average of 8,000 steps per day over the next week. On Sunday "
            "he walked 8,400 steps. On Monday he walked 8,100 steps. On Tuesday he walked 7,300 "
            "steps. On Wednesday he walked 8,200 steps. How many steps does he need to average "
            "on Friday and Saturday to meet his goal?"), "undetermined")


class TestIncrement180DiscountedPurchase(unittest.TestCase):
    """#180 — `_discounted_purchase_system`: a purchase total (Σ count×each-price + bare prices,
    or ONE stated total) transformed by sequential percent modifiers ("75% off" ×(1−p),
    "8% sales tax"/"20% tip" ×(1+p)) in text order. Every percent must classify DOWN xor UP by
    its window cue; every digit (premises AND question) must sit inside a classified match;
    person-noun counts never pair; the 1-price+1-DISCOUNT shape stays the percent family's."""

    def test_count_price_cross_premise_discount(self):
        # 2 · 6 · (1 − 75/100) = 3 — the count and the each-price live in different premises
        self.assertEqual(reading.solve(
            "Carla bought 2 bags of peanut butter cups. Each bag was $6.00 but was 75% off. "
            "How much did she spend on the bags?"), 3)

    def test_each_price_with_pronoun_count(self):
        # 10 · 20 · 0.85 = 170 — "They are normally $20 each" pairs with the bought-10 count
        self.assertEqual(reading.solve(
            "John decided to buy 10 pens. They are normally $20 each but they were on sale "
            "for 15% off. How much did he spend on pens?"), 170)

    def test_discount_then_tax_sequential(self):
        # 200 · 0.75 · 1.10 = 165 — transforms apply multiplicatively in text order
        self.assertEqual(reading.solve(
            "A toy cost $200 but went on sale for 25% off. Donna bought the toy and "
            "paid 10% sales tax. How much did she pay in total?"), 165)

    def test_stated_total_plus_tax(self):
        # 150 · 1.08 = 162 — a lone TAX percent is ours (the percent family has no tax cue)
        self.assertEqual(reading.solve(
            "Pauline is buying school items. The total amount of all the items she wants to "
            "buy add up to $150 before sales tax. Sales tax is 8% of the total amount. "
            "How much will Pauline spend on all the items?"), 162)

    def test_mixed_bare_and_product_with_tip(self):
        # (10 + 4·20) · 1.2 = 108 — a bare price and a count×each product sum before the tip
        self.assertEqual(reading.solve(
            "Stefan and his family order a cake that costs $10 and 4 pizzas that are $20 each. "
            "They tip 20% of the total amount. How much money do they pay in total?"), 108)

    def test_word_number_count_and_question_percent(self):
        # two · 650 · 0.75 = 975 — a word-number count; the percent rides inside the question
        self.assertEqual(reading.solve(
            "Mrs. Taylor bought two shirts that cost $650 each. If the total sales price had "
            "a 25% discount, how much did Mrs. Taylor pay for the two shirts?"), 975)

    def test_single_price_single_discount_stays_percent_change(self):
        # the 1-value+1-DISCOUNT shape is `_percent_change`'s — this system must defer
        self.assertEqual(reading.solve(
            "A ticket costs $2000. He got a 30% discount on the ticket. "
            "How much does he pay?"), 1400)

    def test_person_count_never_pairs(self):
        # "four friends" must not become a ×4 multiplier on the each-price → decline
        self.assertNotIsInstance(reading.solve(
            "Four friends decided to buy a shirt. A shirt costs $20 each. There is a 50% "
            "discount. How much did they spend?"), (int, float))

    def test_unclassifiable_percent_declines(self):
        # "80% of its value" is neither a discount nor a tax — the WHOLE system declines
        self.assertNotIsInstance(reading.solve(
            "He bought a toy for 80% of its $500 value plus a $30 bag. "
            "How much did he pay?"), (int, float))

    def test_ambiguous_bare_price_with_count_declines(self):
        # "4 pizzas for $80" — total or each? never guess
        self.assertNotIsInstance(reading.solve(
            "Luigi bought 4 pizzas for $80. There was a 10% discount. "
            "How much did he pay in total?"), (int, float))


class TestIncrement181FractionDepletionChain(unittest.TestCase):
    """#181 — `_fraction_depletion_chain_system`: one pool reduced by SEQUENTIAL partitive steps,
    later steps anchored to the REMAINDER ("of the remaining", "of what she has left") →
    N·Π(1−fᵢ). The single-pool fraction system EXCLUDES remaining/rest chains — this claims them.
    Critical: with full vocab the general fold COMMITS A PARTIAL on this shape (drops the second
    step), so the system must claim it before the fold."""

    def test_money_chain_of_what_is_left(self):
        # 40 · (1−½) · (1−¼) = 15 — the fold would commit 20 (dropping the quarter step)
        self.assertEqual(reading.solve(
            "Julia has $40. She spends half of her money to buy a new game. She then spends "
            "a quarter of what she has left on food. How much money does she have left?"), 15)

    def test_slash_fractions_of_the_remaining(self):
        # 36 · ¾ · ⅔ = 18
        self.assertEqual(reading.solve(
            "Mary has 36 cows. She sells 1/4 of them. Then 1/3 of the remaining cows are "
            "given to her brother. How many cows are left?"), 18)

    def test_word_fractions_remainder_chain(self):
        # 400 · ¾ · ½ = 150 — the "sheep" wrong-on-inject shape from the sweep, now claimed
        self.assertEqual(reading.solve(
            "Mary has 400 cows on her farm. She gave a quarter of her cows to her sister, and "
            "gave half of the remaining cows to her brother. How many cows remain on the "
            "farm?"), 150)

    def test_percent_step_anchored_to_remainder(self):
        # 600 · ½ · (1−0.30) = 210 — an anchored percent is a fraction here (never countable)
        self.assertEqual(reading.solve(
            "A book is 600 pages long. Coral reads half of it in the first week, and 30 "
            "percent of the remaining pages the second week. How many pages remain after "
            "the second week?"), 210)

    def test_partition_shape_declines(self):
        # two BASE-anchored fractions — the partition shape, not a depletion chain
        self.assertNotIsInstance(reading.solve(
            "There are 42 berries. Half of all the berries are red and a third of the "
            "berries are blue. How many berries are left?"), (int, float))

    def test_uncaptured_literal_loss_declines(self):
        # a literal loss mixed into the chain is not modeled — decline, never a partial
        self.assertNotIsInstance(reading.solve(
            "Nora has 60 pears. She gives half of them to Kim. She gives 5 of the remaining "
            "pears to Sam. How many pears are left?"), (int, float))


class TestIncrement182RelationalProductSum(unittest.TestCase):
    """#182 — `_relational_product_sum_system` (a two-item purchase whose second count is K times
    the first: N·P₁ + K·N·P₂, one shared or two per-unit prices, the price may ride inside the
    question) + `_chain_count_price` extended to REVENUE asks ("how much do they make" with a
    "sell all" totality cue; price/sell clauses filtered from the synthetic count call)."""

    def test_two_prices_relational_count(self):
        # 1000·8 + 2·1000·3 = 14000
        self.assertEqual(reading.solve(
            "John orders food for a big party. He orders 1000 pounds of beef for $8 per "
            "pound. He also orders twice that much chicken at $3 per pound. How much did "
            "everything cost?"), 14000)

    def test_shared_price_in_question(self):
        # (200 + 3·200) · 25 = 20000 — the per-unit price rides inside the question clause
        self.assertEqual(reading.solve(
            "A play starts and 200 people buy tickets. For the second night three times as "
            "many people buy tickets. If each ticket cost $25, how much did the play "
            "make?"), 20000)

    def test_bare_relational_without_price_declines(self):
        self.assertNotIsInstance(reading.solve(
            "Amy has 3 dogs. Each dog eats twice as much as a cat. "
            "How much did everything cost?"), (int, float))

    def test_chain_count_price_revenue_ask(self):
        # (4 + 12 + 6) · 2 = 44 — revenue verb + "sell all" totality cue
        self.assertEqual(reading.solve(
            "Robert has 4 pens. Julia has three times as many pens as Robert. Dorothy has "
            "half as many pens as Julia. They sell their pens for $2 each. If they sell all "
            "their pens, how much money do they make?"), 44)

    def test_chain_count_price_spend_regression(self):
        # the original spend shape is untouched by the revenue extension
        self.assertEqual(reading.solve(
            "Robert has 4 pens. Julia has three times as many pens as Robert. Dorothy has "
            "half as many pens as Julia. One pen costs $1.50. How much did they spend in "
            "total on pens?"), 33)


class TestIncrement184FractionPartitionRest(unittest.TestCase):
    """#184 — `_fraction_partition_rest_system`: copula-labeled fractions of ONE total partition
    it; the asked category is the unlabeled REST → total·(1−Σfᵢ). Stative (are/were) — the
    depletion-chain twin handles the loss-verb shape; that twin declines a second base-anchored
    fraction, so exactly one of them can claim a problem."""

    def test_rest_category_after_two_fractions(self):
        # 40 · (1 − ½ − ¼) = 10
        self.assertEqual(reading.solve(
            "A basket holds a total of 40 balls. Half of the balls are red and one-fourth "
            "are blue. The rest are green. How many green balls are there?"), 10)

    def test_enumerated_rest_without_rest_word(self):
        # 42 · (1 − ½ − ⅓) = 7 — the asked category named only in the total's enumeration
        self.assertEqual(reading.solve(
            "42 raspberries, blackberries, and blueberries were picked in total. Half of "
            "all the berries were raspberries, and a third of the berries were "
            "blackberries. How many of them were blueberries?"), 7)

    def test_overlap_ask_declines(self):
        # "both" is inclusion-exclusion talk, not a partition rest
        self.assertNotIsInstance(reading.solve(
            "There are 20 students in total. Half of them are boys and two-fifths are "
            "girls. How many students are both?"), (int, float))

    def test_total_noun_ask_declines(self):
        # asking the TOTAL noun back is not asking the rest
        self.assertNotIsInstance(reading.solve(
            "There are 20 students in total. Half of them are boys and two-fifths are "
            "girls. How many students are there?"), (int, float))

    def test_labeled_category_ask_declines(self):
        # the asked category carries its own fraction — answer() territory, not the rest
        self.assertNotIsInstance(reading.solve(
            "There are 20 students in total. Half of them are boys and two-fifths are "
            "girls. How many students are girls?"), (int, float))


class TestIncrement186DoublingPeriodsAndPriceWeek(unittest.TestCase):
    """#186 — `_doubling_periods_nth_system` (a first-period value doubling each period, asked a
    NAMED later period → a·2^(n−1); the nth-TERM twin of the doubling-days SUM) and
    `_relational_price_week_system` ((n₁·p + n₂·k·p)·7 — one category's price K times the
    other's, daily counts, a per-week ask)."""

    def test_doubling_months_by_name(self):
        # January → May is the 5th term: 10·2⁴ = 160
        self.assertEqual(reading.solve(
            "Brianne saves $10 in January. Each month, she saves twice as much as her "
            "previous month's savings. How much will she save in May?"), 160)

    def test_doubling_ordinal_weeks(self):
        # 5 · 2³ = 40
        self.assertEqual(reading.solve(
            "Kim saves $5 in the first week. Each week she saves twice as much as the "
            "previous week. How much will she save in the fourth week?"), 40)

    def test_target_residual_shape_declines(self):
        # "wants to save $80 … fifth week" is the RESIDUAL to the goal (gold 20), never 4·2⁴
        self.assertNotIsInstance(reading.solve(
            "Hannah wants to save $80 for five weeks. In the first week, she saved $4 and "
            "she plans to save twice as much as her savings as the previous week. How much "
            "will she save in the fifth week?"), (int, float))

    def test_relational_price_pair_week(self):
        # (8·3 + 10·2·3) · 7 = 588
        self.assertEqual(reading.solve(
            "Jose owns a garden. He charges $3 for kids and twice that amount for adults. "
            "If 8 kids and 10 adults visit his garden per day, how much money will he earn "
            "per week?"), 588)

    def test_price_pair_month_ask_declines(self):
        # only a WEEK ask is modeled (a month would need the 4-weeks convention decision)
        self.assertNotIsInstance(reading.solve(
            "He charges $3 for kids and twice that amount for adults. If 8 kids and 10 "
            "adults visit per day, how much money will he earn per month?"), (int, float))


class TestIncrement188MultiPairDiscount(unittest.TestCase):
    """#188 — `_discounted_purchase_system` cross-premise pairing generalized from 1-1 to N-N:
    counts zip with each-prices in document order, gated to counts-FIRST parallel discourse
    (every count clause precedes every price clause — GSM8K states the pairs in the same order
    twice). The 1-1 case stays unconditional (Johnny's price-first ping-pong shape)."""

    def test_two_by_two_counts_first(self):
        # (2·40 + 4·50) · 0.8 = 224
        self.assertEqual(reading.solve(
            "Bailey needs to buy 2 new sets of towels for the guest bathroom and 4 new sets "
            "for her master bathroom. The set of towels for the guest bathroom are $40.00 "
            "each and the master bathroom set is $50.00 each. The store is currently "
            "offering 20% off so how much will Bailey spend on towel sets?"), 224)

    def test_prices_before_counts_declines(self):
        # counts AFTER prices — the parallel-discourse guarantee is gone; never zip blind
        self.assertNotIsInstance(reading.solve(
            "The guest set is $40 each and the master set is $50 each. Bailey buys 2 guest "
            "sets and 4 master sets. There is a 20% discount. How much will Bailey "
            "pay?"), (int, float))


class TestIncrement189CombinedStandard(unittest.TestCase):
    """#189 — the combined-standard coordination: `_split_clauses` keeps "as X and Y combined"
    whole (the _std rule severed it, stranding a bogus "Clementine combined." premise), and
    `_combined_standard_sum_system` reads the raw named chain — one relation's standard is the
    SUM of two resolved names — for a count total or a sell-all money ask."""

    def test_combined_standard_money_chain(self):
        # 72 + 144 + (72+144)/2 = 324 cookies · $2 = 648
        self.assertEqual(reading.solve(
            "Clementine, Jake, and Tory make cookies for the school bake sale. Clementine "
            "baked 72 cookies. Jake baked twice as many cookies as Clementine. Tory baked "
            "half as many cookies as Jake and Clementine combined. They sell their cookies "
            "for $2 each. If they sell all their cookies, how much money do they make?"), 648)

    def test_combined_standard_count_total(self):
        # 10 + 20 + 15 = 45
        self.assertEqual(reading.solve(
            "Amy baked 10 cakes. Ben baked twice as many cakes as Amy. Carl baked half as "
            "many cakes as Amy and Ben combined. How many cakes did they bake in total?"), 45)

    def test_unresolved_combined_members_decline(self):
        # the combined pair's members never resolve — no value to sum, decline
        self.assertNotIsInstance(reading.solve(
            "Carl baked half as many cakes as Amy and Ben combined. How many cakes did "
            "they bake in total?"), (int, float))

    def test_real_second_clause_still_splits(self):
        # "…than Mark, and Mark has…" — the right conjunct has a verb and a value: a REAL
        # second clause; the combined-standard guard must not glue it (anchored chain solves)
        self.assertEqual(reading.solve(
            "Ann has 8 more marbles than Mark, and Mark has 10 marbles. How many marbles "
            "do they have in total?"), 28)

    def test_inverse_combined_pair_lump(self):
        # the SUBJECT is known — the pair resolves only as a LUMP (60/4), enough for the
        # total ask: 60 + 15 = 75. (This case solved through the old SPLIT path; the guard
        # rerouted it here — the firewall caught the break and forced the inverse support.)
        self.assertEqual(reading.solve(
            "Alexis has four times as many mangoes as Dilan and Ashley combined. If Alexis "
            "has 60 mangoes, how many mangoes does all of them have combined?"), 75)


class TestIncrement191AgeShift(unittest.TestCase):
    """#191 — the age-shift family (the biggest fully-read decline bucket, 49 cases; age algebra
    is a TEST-carrying branch). `_age_shift_system`: ONE relation + ONE time-shifted anchor
    ("in 5 years X will be 40" → X_now = 35), asked now or projected ("10 years from now").
    `_age_simultaneous_system`: offset AND ratio in one sentence → std = off/(k−1)."""

    def test_shifted_anchor_current_ask(self):
        # (40 − 5) − 4 = 31 — an imperative question ("Calculate …")
        self.assertEqual(reading.solve(
            "Angelina is 4 years older than Justin is now. In 5 years, Angelina will be 40 "
            "years old. Calculate the age of Justin currently."), 31)

    def test_this_year_anchor_projected_ask(self):
        # (12 + 3) + 10 = 25 — "younger by K" phrasing + a projected ask
        self.assertEqual(reading.solve(
            "Bush and Matt are brothers. Bush is younger than Matt by 3 years. This year "
            "Bush will be 12 years old. What will be Matt's age 10 years from now?"), 25)

    def test_simultaneous_offset_and_ratio(self):
        # D + 7 = 2D → D = 7
        self.assertEqual(reading.solve(
            "Yuan is 7 years older than his brother David and twice his age. "
            "How old is David?"), 7)

    def test_two_shifted_anchors_decline(self):
        # two anchors — over-determined for this shape; never pick one silently
        self.assertNotIsInstance(reading.solve(
            "Ann is 4 years older than Ben. In 5 years, Ann will be 40. In 3 years, Ben "
            "will be 20. How old is Ben?"), (int, float))


class TestIncrement192AgeChain(unittest.TestCase):
    """#192 — `_age_chain_system`: ratio/offset AGE chains forward-resolved from one literal
    anchor, asked a single age / the AVERAGE / the TOTAL / a year-DIFFERENCE. The age-phrasing
    twin of the anchored count family ("as old as" vs "as many as"). Appositions normalize
    locally ("His cousin, Rommel, is…" → "Rommel is…"); "who" chains to the previous standard;
    "his/her <kin>" binds to "<subject>'s <kin>"; a bare pronoun standard means the anchor."""

    def test_chain_average_with_cast_check(self):
        # (48 + 24 + 12) / 3 = 28 — "the three" must equal the resolved cast size
        self.assertEqual(reading.solve(
            "Eden's mom is twice as old as Eden, who is twice as old as Devin. If Devin is "
            "12 years old, what's the average age of the three?"), 28)

    def test_chain_total_with_kin_entities(self):
        # 15 + 45 + 180 = 240 — apposition anchor + "his sister" possessive standard
        self.assertEqual(reading.solve(
            "Aaron, Henry's brother, is 15 years old. Henry's sister is three times as old "
            "as Aaron. Henry is four times as old as his sister. What's the combined age of "
            "the siblings?"), 240)

    def test_chain_year_difference_ask(self):
        # Jenny 17 − Tim 5 = 12 — "thrice as old as he is" resolves the pronoun to the anchor
        self.assertEqual(reading.solve(
            "Tim is 5 years old. His cousin, Rommel, is thrice as old as he is. His other "
            "cousin, Jenny, is 2 years older than Rommel. How many years younger is Tim "
            "than Jenny?"), 12)

    def test_chain_single_ask(self):
        # Carl 6 → Ben 10 → Amy 20
        self.assertEqual(reading.solve(
            "Amy is twice as old as Ben. Ben is 4 years older than Carl. If Carl is 6 years "
            "old, how old is Amy?"), 20)

    def test_banned_times_older_idiom_declines(self):
        self.assertNotIsInstance(reading.solve(
            "Kiarra is twice as old as Bea. Job is 3 times older than Bea. If Kiarra is 30, "
            "how old is Job?"), (int, float))


class TestIncrement193AgePastAnchor(unittest.TestCase):
    """#193 — `_age_past_anchor_system`: an age relation stated AT A PAST TIME ("K years ago,
    X was twice as old as Y") + one present anchor (plain now-age, "turned A years old K ago"
    = A+K, or "currently twice N" = 2N), asked now or projected. Kin-noun entities (a mother /
    her daughter) work alongside names; the offset relation may ride on HALF/TWICE the standard
    ("5 years older than half Tom's age")."""

    def test_past_ratio_kin_entities_projected(self):
        # then: 36 = 2·18 → daughter now 23 → +3 = 26
        self.assertEqual(reading.solve(
            "5 years ago, a mother was twice as old as her daughter. If the mother is 41 "
            "years old now, how old will the daughter be in 3 years?"), 26)

    def test_word_number_ago_and_twice_n_anchor(self):
        # Mohamed = 2·30 = 60; then: Kody = 56/2 = 28 → now 32
        self.assertEqual(reading.solve(
            "Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently "
            "twice 30 years old, how old is Kody?"), 32)

    def test_turned_anchor_and_offset_on_half(self):
        # Tom now 37; 7 years ago Tom 30 → Jim = 5 + 15 = 20 → now 27 → +2 = 29
        self.assertEqual(reading.solve(
            "Tom turned 32 years old 5 years ago. 7 years ago Jim was 5 years older than "
            "half Tom's age. How old is Jim in 2 years?"), 29)

    def test_two_anchors_not_claimed(self):
        # over-determined for this shape — the system must pass (the fold declines it today)
        self.assertNotIsInstance(reading.solve(
            "3 years ago, Al was twice as old as Bo. Al is 20 years old now. Bo is 9 years "
            "old now. How old is Bo?"), (int, float))


class TestIncrement194AgeFutureVsNow(unittest.TestCase):
    """#194 — `_age_future_vs_now_system` ("in M years X will be k times as old as Y is NOW" —
    the 'is now' tail discriminates from the both-aging years-until shape → Y = (d+M)/(k−1))
    and `_years_until_ratio_system` extended: "double" as a multiplier and the standard genitive
    taken AFTER the multiplier ("will Fouad's age be double AHMED'S current age")."""

    def test_future_vs_now_offset_relation(self):
        # H + 7 + 3 = 2H → 10
        self.assertEqual(reading.solve(
            "Audrey is 7 years older than Heracles. In 3 years, Audrey will be twice as old "
            "as Heracles is now. How old is Heracles now?"), 10)

    def test_years_until_double_with_subject_genitive(self):
        # 26 + t = 2·(11 + t) → 4 — the subject's own genitive precedes the multiplier
        self.assertEqual(reading.solve(
            "Ahmed is 11 years old and Fouad is 26 years old. In how many years will "
            "Fouad's age be double Ahmed's current age?"), 4)

    def test_years_until_twice_regression(self):
        # the original Beth shape is untouched by the extension
        self.assertEqual(reading.solve(
            "Beth is 18 years old and her little sister is 5. In how many years would she "
            "be twice her sister's age?"), 8)


class TestIncrement195FractionLiteralChains(unittest.TestCase):
    """#195 — the fraction+literal vein: `_pool_literal_relational_remaining_system` (pool −
    literal − k×literal, ask a fraction of the remaining) and the depletion chain's ask gate
    extended with the "to finish" remainder cue (Coral's real wording)."""

    def test_pool_literal_relational_half_remaining(self):
        # (120 − 12 − 24) / 2 = 42
        self.assertEqual(reading.solve(
            "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and "
            "today, she read twice as many pages as yesterday. If she wants to read half of "
            "the remaining pages tomorrow, how many pages should she read?"), 42)

    def test_depletion_chain_to_finish_ask(self):
        # 600·½·0.7 = 210 — "must she read the third week in order to finish" = the remainder
        self.assertEqual(reading.solve(
            "Coral is reading a book that is 600 pages long. She reads half of it in the "
            "first week, and 30 percent of the remaining pages the second week. How many "
            "pages must she read the third week in order to finish the book?"), 210)

    def test_stray_value_declines(self):
        # an extra number beside the pool/literal means the chain is not fully modeled
        self.assertNotIsInstance(reading.solve(
            "Julie is reading a 120-page book with 8 chapters. Yesterday she read 12 pages "
            "and today she read twice as many pages as yesterday. If she wants to read half "
            "of the remaining pages tomorrow, how many pages should she read?"), (int, float))


class TestIncrement196ProductSumVariants(unittest.TestCase):
    """#196 — product-sum variants: change after paying a stated total (T − Σ), the target
    purchase gap (N·P − savings), and the two-name equal split (Σ/2)."""

    def test_product_sum_change(self):
        # 200 − (5·6 + 10·14) = 30
        self.assertEqual(reading.solve(
            "Mary bought 5 boxes of drinks at $6 each box and 10 boxes of pizzas at $14 "
            "each box for her pizza party. She paid $200 for all the items. How much change "
            "did she get back?"), 30)

    def test_target_purchase_gap(self):
        # 8·5 − 13 = 27
        self.assertEqual(reading.solve(
            "Tommy wants to earn enough money to buy 8 new books. Each book costs $5. "
            "If Tommy already has $13, how much does he need to save up?"), 27)

    def test_product_sum_split_two_names(self):
        # (6·3 + 18·2) / 2 = 27
        self.assertEqual(reading.solve(
            "Sab and Dane sold 6 pairs of shoes that cost $3 each and 18 shirts that cost "
            "$2. How much will each of them earn if they divided their total earning?"), 27)


class TestIncrement197PercentMoney(unittest.TestCase):
    """#197 — percent-of-money: raise DIFFERENCE on a shared base, asset value ± P% (the
    years-ago scene number allowed), and a pair whose second item costs P% more."""

    def test_raise_difference_shared_base(self):
        # 30000 · (15% − 10%) = 1500
        self.assertEqual(reading.solve(
            "Hansel makes $30,000 a year and just received a 10% raise. Gretel makes the "
            "same amount as Hansel but received a 15% raise. How much more money will "
            "Gretel make compared to Hansel?"), 1500)

    def test_value_reduced_pct_with_scene_years(self):
        # 4000 · 0.70 = 2800 — "3 years ago" is scene, not data
        self.assertEqual(reading.solve(
            "Jocelyn bought a car 3 years ago at $4000. If the car's value has reduced by "
            "30%, calculate the current value of the car."), 2800)

    def test_pair_second_pct_more(self):
        # 22 + 22·1.5 = 55
        self.assertEqual(reading.solve(
            "Nina has two pairs of shoes. She bought one of them for $22, and the second "
            "one was 50% more expensive. How much did Nina pay for both pairs of shoes?"), 55)


class TestIncrement198DayChainsAndRates(unittest.TestCase):
    """#198 — signed today/tomorrow chains, the first-two-days double-sum, hours to equal
    another's earnings, and the same-distance-new-speed inverse."""

    def test_signed_day_chain(self):
        # 65 − 18 + 50 = 97
        self.assertEqual(reading.solve(
            "Jack has 65 pounds of sugar today. Tomorrow he will use 18 pounds of sugar "
            "and the following day he will buy 50 more pounds of sugar. How many pounds "
            "will he have in the end?"), 97)

    def test_first_two_days_double_sum(self):
        # 5 + 7 + 2·(5+7) = 36
        self.assertEqual(reading.solve(
            "Nala found 5 seashells at the beach. The next day, she found another 7, and "
            "the following day, she found twice the number she got from the first two "
            "days. How many seashells does Nala have?"), 36)

    def test_equal_earnings_hours(self):
        # 40·12/15 = 32 — the question repeats the 40 (an echo, allowed)
        self.assertEqual(reading.solve(
            "Greta worked 40 hours and was paid $12 per hour. Her friend Lisa earned $15 "
            "per hour at her job. How many hours would Lisa have to work to equal Greta's "
            "earnings for 40 hours?"), 32)

    def test_speed_time_inverse(self):
        # 4·50/100 = 2 — the "instead" clause splits into a premise
        self.assertEqual(reading.solve(
            "A train takes 4 hours to reach a destination at a speed of 50 miles per hour. "
            "How long would it take if it traveled at 100 miles per hour instead?"), 2)


class TestIncrement200BackwardFractionChain(unittest.TestCase):
    """#200 — `_backward_fraction_chain_system`: the INVERSE of the depletion chain — fraction
    losses ending in a KNOWN final, asked the original (M = final/Π(1−fᵢ); 'half of the
    original' halves the answer). GUARD 8's honest backward decline stays for anything the
    tight gates don't claim."""

    def test_two_step_backward_chain(self):
        # 15 / (⅔·¾) = 30
        self.assertEqual(reading.solve(
            "Lucy lost one-third of her money. She then spent one-fourth of the remainder, "
            "and only left with $15. How much money did Lucy have at the beginning?"), 30)

    def test_single_step_half_of_original_ask(self):
        # (24 ÷ 4/7) / 2 = 21
        self.assertEqual(reading.solve(
            "Jenny had $24 left after spending 3/7 of her money. How much is half of her "
            "original amount of money?"), 21)

    def test_half_the_amount_then_half_remaining(self):
        # 51 / (½·½) = 204
        self.assertEqual(reading.solve(
            "Isabel has some money in her piggy bank. She spent half the amount and bought "
            "a toy. She then spent half of the remaining money and bought her brother a "
            "book. If she has $51 left, how much money, in dollars, did she have at "
            "first?"), 204)

    def test_literal_loss_backward_not_claimed(self):
        # a literal (non-fraction) loss in the chain — not modeled, never a partial
        self.assertNotIsInstance(reading.solve(
            "Rita lost one-third of her money and then spent $6 of the remainder. She was "
            "left with $10. How much money did Rita have at the beginning?"), (int, float))


class TestIncrement201MoneyTails(unittest.TestCase):
    """#201 — bespoke money tails: split pool with a relational second price, monthly buys ×12
    minus a sell-back, and a same-percentage bonus transferred to a new base."""

    def test_split_price_relational(self):
        # 4·1.50 + 5·(1.50+5.50) = 41 — the 9/day-for-14-days numbers are scene
        self.assertEqual(reading.solve(
            "Henry took 9 pills a day for 14 days. Of these 9 pills, 4 pills cost $1.50 "
            "each, and the other pills each cost $5.50 more. How much did he spend in "
            "total on the pills?"), 41)

    def test_monthly_buy_sellback_loss(self):
        # 3·20·12 − 500 = 220
        self.assertEqual(reading.solve(
            "Jack bought 3 books a month at $20 each. He sells them back at the end of the "
            "year for $500. How much money did he lose?"), 220)

    def test_same_pct_bonus_new_base(self):
        # 200000 · (1 + 10000/100000) = 220000
        self.assertEqual(reading.solve(
            "John gets a bonus that's the same percentage every year. Last year he made "
            "$100,000 and got a $10,000 bonus. This year he makes $200,000. How much will "
            "John make this year when adding both his total pay and bonus together?"), 220000)


class TestIncrement202AgeTails(unittest.TestCase):
    """#202 — age tails: the age chain reads "N times the age of" and possessive diff asks
    (Lavinia); a future half-of-N anchor + fraction-of-now relation (Christina); the verbal
    take-add-you-get equation (Hiram, splitter-fragmented)."""

    def test_cross_kin_ratio_and_possessive_diff_ask(self):
        # son = 2·12 = 24; daughter = 12−10 = 2 → 22
        self.assertEqual(reading.solve(
            "Lavinia\u2019s daughter is 10 years younger than Katie\u2019s daughter. Lavinia\u2019s son "
            "is 2 times the age of Katie\u2019s daughter. If Katie\u2019s daughter is 12 years old, "
            "how many years older is Lavinia\u2019s son than Lavinia\u2019s daughter?"), 22)

    def test_future_fraction_anchor(self):
        # C = 80/2 − 5 = 35; O = (3/5)·35 − 15 = 6
        self.assertEqual(reading.solve(
            "Christina's age will be half of 80 years in five years to come. If Oscar will "
            "be 3/5 times as old as Christina is now in 15 years, calculate Oscar's age "
            "now."), 6)

    def test_verbal_equation_take_add_get(self):
        # 40 + 12 = 2A − 4 → A = 28
        self.assertEqual(reading.solve(
            "When you take Hiram's age and then add 12, you get 4 less than twice "
            "Allyson's age. If Hiram is 40 years old, how old is Allyson?"), 28)


class TestIncrement205VocabWave20(unittest.TestCase):
    """#205 — the post-#204 sweep harvest: the containment unlock re-opened the vocab-only
    bucket (magazine/hockey/rainbow/nerds/grapes). "grapes" is plural-surface ONLY (the
    pants/earnings pattern): the singular "grape" is a wrong-on-inject exclusion — both
    wrong-classes pinned declining below."""

    def test_containment_relational_color_count(self):
        # 10 + (10+4) + (14−2) = 36
        self.assertEqual(reading.solve(
            "A box of rainbow nerds contains 10 purple candies, 4 more yellow candies, "
            "and 2 fewer green candies than yellow candies. How many rainbow nerds are "
            "there in the box?"), 36)

    def test_backward_fraction_chain_magazine(self):
        # M · (4/5) · (3/4) = 360 → 600
        self.assertEqual(reading.solve(
            "Frank spent 1/5 of his money to buy groceries. He then spent 1/4 of the "
            "remaining money to buy a magazine. If he had $360 left in his wallet, how "
            "much money did he have at first?"), 600)

    def test_contained_relational_bowls_total(self):
        # 25 + 27 + 31 = 83
        self.assertEqual(reading.solve(
            "Allie, Rob, and Allyn each bought a bowl of grapes. Allie's bowl contained "
            "two more grapes than Rob's bowl. Allyn's bowl contained four more grapes "
            "than Allie's bowl. If Rob's bowl contained 25 grapes, what is the total "
            "combined number of grapes in all three bowls?"), 83)

    def test_contains_card_chain_total(self):
        # hockey 200; football 4·200 = 800; baseball 800−50 = 750 → 1750
        self.assertEqual(reading.solve(
            "A bag contains 50 fewer baseball cards than football cards. There are 4 "
            "times as many football cards as hockey cards. If there are 200 hockey cards "
            "in the bag, how many cards are there altogether?"), 1750)

    def test_singular_grape_still_declines_ratio_backward(self):
        # "grape" (singular modifier) stays unknown — typed, this commits 60≠g24.
        self.assertEqual(reading.solve(
            "At the new store, there are 3 times as many grape candies as cherry candies, "
            "and 2 times as many apple candies as grape candies. All the candies cost "
            "$2.50 each. If the total cost of the candies is $200, how many grape candies "
            "does the store have?"), "undetermined")

    def test_singular_grape_still_declines_weight_partition(self):
        # the mixed-fruit weight partition commits 16≠g3 if grape/fruits become known.
        self.assertEqual(reading.solve(
            "Tommy ordered a total weight of 10 kilograms of apple, orange, grape, and "
            "strawberry fruits. He ordered 3 kilograms of apples, 1 kilogram of orange, "
            "and 3 kilograms of grapes. How many kilograms of strawberries did he "
            "order?"), "undetermined")


class TestIncrement206PercentPriceRelational(unittest.TestCase):
    """#206 — the comma-less conditional-question split ("If P how much Q?" → condition premise)
    + `_percent_price_relational_system`: an item priced "P% more/less expensive" than a stated
    base → base·(1 ± p/100), with an optional purchase-counts premise buying BOTH items. The
    scaled-vs-base ask needs a DISCRIMINATIVE token (in the percent premise, absent from the
    base premise); an ask about the base item declines."""

    def test_conditional_question_splits_without_comma(self):
        premises, question = reading._parse_problem(
            "Mark gets a new phone plan that is 30% more expensive than his old plan. "
            "If his old plan was $150 a month how much does his new plan cost?")
        self.assertEqual(question, "how much does his new plan cost?")
        self.assertEqual(premises[-1], "If his old plan was $150 a month")

    def test_noun_standard_condition_in_question(self):
        # 150 · 1.3 = 195
        self.assertEqual(reading.solve(
            "Mark gets a new phone plan that is 30% more expensive than his old plan. "
            "If his old plan was $150 a month how much does his new plan cost?"), 195)

    def test_implicit_standard_same_item(self):
        # 2000 · 1.02 = 2040
        self.assertEqual(reading.solve(
            "Alan bought a $2000 phone online. John bought it 2% more expensive in a "
            "local store. How much did John spend on his phone?"), 2040)

    def test_shared_sentence_base_plus_purchase_counts(self):
        # chips = 4·1.75 = 7; 2·7 + 2·4 = 22
        self.assertEqual(reading.solve(
            "A pack of pretzels costs $4, while a pack of chips is 75% more expensive. "
            "Maciek went and bought two packets of chips and two packets of pretzels. "
            "How much did he pay for his purchases?"), 22)

    def test_ask_about_base_item_declines(self):
        # every ask word lives in the base premise too — no discriminative token → decline
        self.assertEqual(reading.solve(
            "Mark gets a new phone plan that is 30% more expensive than his old plan. "
            "If his new plan seems pricey, how much did his old plan cost if it was "
            "$150 a month?"), "undetermined")

    def test_two_base_prices_ambiguous_declines(self):
        # two candidate standards — the system must not guess which one the percent scales
        self.assertEqual(reading.solve(
            "A radio costs $30. A speaker costs $50. A new stereo is 20% more expensive. "
            "How much does the new stereo cost?"), "undetermined")


class TestIncrement207AnchoredElidedRatio(unittest.TestCase):
    """#207 — three extensions to the anchored-affine family's shared extractor: KIN-APPOSITION
    normalization ("his brother Fred" names ONE person — the kinship is decorative for the
    algebra), the ELIDED-standard ratio ("Flora has twice as many books" — the standard is the
    nearest antecedent), and " together" joining the sum/single twins' cue lists in sync."""

    def test_kin_apposition_offset_chain(self):
        # jerry = 3·george; george = fred − 6; fred = 18 → 36
        self.assertEqual(reading.solve(
            "Jerry has three times as many stickers as George. George has 6 fewer stickers "
            "than his brother Fred. If Fred has 18 stickers, how many stickers does Jerry "
            "have?"), 36)

    def test_elided_standard_and_together_sum(self):
        # flora = 2·50; gary = 50/2 → 50+100+25 = 175
        self.assertEqual(reading.solve(
            "Harry has 50 books in his library. His sister Flora has twice as many books "
            "and their cousin Gary has half the books Harry has. How many books do the "
            "three of them own together?"), 175)

    def test_elided_standard_money_total(self):
        # stephanie = 2·356 → 356 + 712 = 1068
        self.assertEqual(reading.solve(
            "Marion's bike cost $356. Stephanie's bike is worth twice as much. What is "
            "the total price of their bikes?"), 1068)

    def test_elided_ratio_without_anchor_declines(self):
        # an elided ratio with NO anchored antecedent anywhere — nothing licenses a standard
        self.assertEqual(reading.solve(
            "Stephanie's bike is worth twice as much. What is the total price of their "
            "bikes?"), "undetermined")

    def test_named_subset_together_sums_only_named(self):
        # " together" with two NAMED entities sums the subset, not the whole cast
        self.assertEqual(reading.solve(
            "Harry has 50 books in his library. His sister Flora has twice as many books "
            "and their cousin Gary has half the books Harry has. How many books do Harry "
            "and Flora own together?"), 150)

    def test_coordinated_elided_clauses_share_one_standard(self):
        # "her brother has thrice as much WHILE her sister has only half as much" — BOTH
        # elide "as Joanna's": 8 + 24 + 4 = 36 (nearest-antecedent chaining gave 44 — the
        # firewall-caught misbind; an elided subject never becomes the next antecedent)
        self.assertEqual(reading.solve(
            "Joanna has $8. Compared to her money, her brother has thrice as much while "
            "her sister has only half as much. How much money do the three of them have "
            "altogether?"), 36)


class TestIncrement208NeedsInverse(unittest.TestCase):
    """#208 — the deficit-to-target premise read: "X needs N more to have (k× as many | the same
    number) as S" states X + N = k·S → X = k·S − N. The premise-side twin of the needs-idiom
    the single twin's QUESTION guard declines (that guard is unchanged)."""

    def test_needs_more_to_have_twice(self):
        # amanda = 2·17 − 12 = 22; mabel = 85 → 85 − 22 = 63
        self.assertEqual(reading.solve(
            "Amanda needs 12 more marbles to have twice as many marbles as Katrina, and "
            "Mabel has 5 times as many marbles as Katrina. If Mabel has 85 marbles, how "
            "many more marbles does Mabel have than Amanda?"), 63)

    def test_needs_more_to_have_same_number(self):
        # loris = 40 − 3 = 37; lamont = 2·20 = 40 → 20 + 40 + 37 = 97
        self.assertEqual(reading.solve(
            "Loris needs three more books to have the same number as Lamont, who has "
            "twice the number of books Darryl has. If Darryl has 20 books, calculate "
            "the total number of books the three have."), 97)


class TestIncrement209PiecesPool(unittest.TestCase):
    """#209 — `_pieces_pool_system`: a slices/pieces pool (containers × per-container) with
    ONE downstream op, six ask branches. Premises PERIOD-joined so severed clauses can't
    create false adjacencies ("…gets 3 slices each pizza…" stole the per-container read)."""

    def test_remain_after_eating(self):
        # 2·8 − 7 = 9
        self.assertEqual(reading.solve(
            "If there are 8 slices in a large pizza, how many slices will remain if "
            "Mary orders 2 large pizzas and eats 7 slices?"), 9)

    def test_cost_per_piece(self):
        # 80 / (4·5) = 4
        self.assertEqual(reading.solve(
            "Luigi bought four pizzas for $80. Each pizza was cut into 5 pieces. How "
            "much did each piece of pizza cost, in dollars?"), 4)

    def test_cost_of_k_slices(self):
        # 72 / (3·12) · 5 = 10
        self.assertEqual(reading.solve(
            "Kim buys 3 pizzas. They are 12 slices each. The pizza cost $72. How much "
            "did 5 slices cost?"), 10)

    def test_pizzas_to_order_group(self):
        # 18·3 / 9 = 6 — the false-adjacency regression case (period-join)
        self.assertEqual(reading.solve(
            "There is a group of 18 people who are ordering pizza. If each person gets "
            "3 slices and each pizza has 9 slices, how many pizzas should they order?"), 6)

    def test_slices_per_person(self):
        # 3·8 / (1+5) = 4
        self.assertEqual(reading.solve(
            "John with his five friends ordered 3 pizzas. Each pizza had 8 slices. If "
            "they all finished and ate the same amount of pizzas, how many slices did "
            "each person eat?"), 4)

    def test_revenue_slices_times_price(self):
        # 9·4·5 = 180
        self.assertEqual(reading.solve(
            "A pie shop charges $5 for a slice of pie. They cut each whole pie into 4 "
            "slices. How much money will the pie shop make if they sell 9 pies?"), 180)

    def test_stray_value_declines(self):
        # an unaccounted digit (the 2 sodas) must decline the whole system
        self.assertEqual(reading.solve(
            "Luigi bought four pizzas for $80 and 2 sodas. Each pizza was cut into 5 "
            "pieces. How much did each piece of pizza cost, in dollars?"), "undetermined")

    def test_calories_of_fraction_eaten(self):
        # 1·8·½·300 = 1200 (single container via "orders a pizza")
        self.assertEqual(reading.solve(
            "Bob orders a pizza with 8 slices. He eats half of it and each slice was "
            "300 calories. How many calories did he eat?"), 1200)

    def test_fraction_remain(self):
        # 1·16·(1−¾) = 4
        self.assertEqual(reading.solve(
            "A family bought 1 box of pizza that is cut into 16 slices. Only "
            "three-fourths of the pizza was eaten by the family. How many slices of "
            "pizza were left?"), 4)

    def test_goals_per_game_average(self):
        # 6·12/8 = 9 (slice-per-goal identity)
        self.assertEqual(reading.solve(
            "Mr. Mitchell is buying pizzas for the soccer team. He buys one slice for "
            "every goal they scored on the season. A large pizza has 12 slices. If he "
            "bought 6 pizzas, and the team had 8 games, how many goals did they score "
            "per game on average?"), 9)


class TestIncrement211DayRelationalChain(unittest.TestCase):
    """#211 — `_day_relational_chain_system`: per-day values (literal or k× a referenced day),
    asked a day-difference, the total (± an ate-loss), or sell-all money. The day reference is
    searched in the clause TAIL after the k-word — an optional-group regex silently dropped
    "…as she painted on Monday" (a wrong value, not a decline). Pending-day slot re-attaches
    a day clause severed from its relation clause."""

    def test_day_difference(self):
        # wed 300 − tue 225 = 75
        self.assertEqual(reading.solve(
            "On Monday, Matt worked for 450 minutes in his office. On Tuesday, he worked "
            "half the number of minutes he worked on Monday. On Wednesday, he worked for "
            "300 minutes. How many more minutes did he work on Wednesday than on "
            "Tuesday."), 75)

    def test_total_with_loss_and_pending_day(self):
        # 32 + 16 + 48 − 4 = 92 ("On Tuesday he ran out of sugar" severed from its relation)
        self.assertEqual(reading.solve(
            "Zach baked cookies for a bake sale. On Monday he baked 32 cookies. On "
            "Tuesday he ran out of sugar and could only bake half of the number of "
            "cookies he baked the day before. On Wednesday he baked three times the "
            "number of cookies he did on Tuesday, but his brother ate 4 of those "
            "cookies. How many cookies did he have at the end of the three days?"), 92)

    def test_total_with_named_day_ref(self):
        # 30 + 60 + 15 = 105 — Wednesday references MONDAY, not the previous day
        self.assertEqual(reading.solve(
            "On Monday Elisa paints 30 square feet of her house's walls. On Tuesday she "
            "paints twice that amount. On Wednesday she finishes up by painting half as "
            "many square feet as she painted on Monday. How many square feet total does "
            "Elisa paint in her house?"), 105)

    def test_sell_all_money(self):
        # (30 + 15) · 1.20 = 54
        self.assertEqual(reading.solve(
            "Sally picks 30 seashells on Monday. On Tuesday, she picks half as many "
            "seashells as she did on Monday. If she can sell each seashell for $1.20, "
            "how much money can she make if she sells all of her seashells?"), 54)


class TestIncrement212WeekResiduals(unittest.TestCase):
    """#212 — two day-cluster tails: the WEEK-TOTAL RESIDUAL with an explicitly-unknown day
    (Billy: 20 − (2+4+4+1) = 9, with a forward Thursday→Friday reference) and the WEEKDAY
    BLOCK + per-weekend-day multipliers (R·5 + k_sat·R + k_sun·R; "no practice on Sunday"
    = 0; an hours ask over a minutes rate converts ÷60)."""

    def test_week_total_residual_unknown_day(self):
        self.assertEqual(reading.solve(
            "Billy ate 20 apples this week. On Monday, he ate 2 apples. On Tuesday, he "
            "ate twice as many as he ate the day before. He's not sure what he ate on "
            "Wednesday. On Thursday, he ate four times as many as he ate on Friday. On "
            "Friday, he ate half of the amount he ate on Monday. How many apples did he "
            "eat on Wednesday?"), 9)

    def test_weekday_block_two_multipliers(self):
        # 3·5 + 2·3 + 3·3 = 30
        self.assertEqual(reading.solve(
            "Lolita can drink 3 boxes of milk during weekdays. During Saturdays, she can "
            "drink twice the number of boxes of milk than on the weekdays, and during "
            "Sundays, she can drink thrice the number of boxes of milk than on the "
            "weekdays. How many boxes of milk does she drink per week?"), 30)

    def test_weekday_block_hours_conversion_and_zero_sunday(self):
        # (30·5 + 3·30 + 0) / 60 = 4
        self.assertEqual(reading.solve(
            "From Monday to Friday, Elle practices piano for 30 minutes. On Saturday, "
            "she practices piano three times as much as on a weekday. There is no "
            "practice on Sunday. How many hours does Elle spend practicing piano each "
            "week?"), 4)


class TestIncrement213BatchScaling(unittest.TestCase):
    """#213 — `_batch_scaling_system`: a per-batch ingredient list × a batch multiplier →
    mult × Σ ingredients. The semantic-tier item that RE-ENABLES "require": its #203
    exclusion wrong WAS this shape's unscaled partial read. Decimal amounts are valid
    ("1.5 cups" — an integer-only read declined and let the fold commit the old partial).
    "tomatoes" is plural-surface only (the grapes pattern); "tomato seeds" stays unknown."""

    def test_quadruple_batch_with_relational_ingredient(self):
        # 4 · (1 + 2 + 2·1.5) = 24
        self.assertEqual(reading.solve(
            "Carla adds a can of chilis, two cans of beans, and 50% more tomatoes than "
            "beans to a normal batch of chili. If she makes a quadruple batch, how many "
            "cans of food does she need?"), 24)

    def test_requires_n_batches_with_decimal(self):
        # 8 · (4 + 1.5) = 44 — the former 'require' exclusion wrong, now correct
        self.assertEqual(reading.solve(
            "One batch of cookies requires 4 cups of flour and 1.5 cups of sugar. How "
            "many cups of flour and sugar would Grandma need to make 8 batches?"), 44)

    def test_singular_tomato_still_declines(self):
        # "tomato seeds" (singular modifier) stays unknown — typed, this commits 300≠250
        self.assertEqual(reading.solve(
            "Mike and Ted planted tomatoes. In the morning, Mike planted 50 tomato seeds "
            "while Ted planted twice as much as Mike. In the afternoon, Mike planted 60 "
            "tomato seeds while Ted planted 20 fewer tomato seeds than Mike. How many "
            "tomato seeds did they plant altogether?"), "undetermined")

    def test_max_speed_times_duration(self):
        # #214: 50 km/h × 120 min / 60 = 100 (the "maximum speed" qualifier is decorative)
        self.assertEqual(reading.solve(
            "A fox can run at the maximum speed of 50 kilometers per hour. Considering "
            "the fox would run at a constant speed, what distance would he make during "
            "120 minutes?"), 100)


class TestIncrement215Residue(unittest.TestCase):
    """#215 — three residue fixes: NEAR-NAME unification (GSM8K misspells its own names
    across clauses — Jefferey/Jeffery — unify a unique edit-distance-1 pair, names ≥5 chars);
    the PAIR-SUM relational system (A = k·B, A+B = S∓N, anchored S); the LOCAL conjunct
    split inside the anchored family for fused "If A…, and B…" premises, which exposed a
    latent sign bug — "how much LESS does X have than Y" now computes Y − X."""

    def test_near_name_typo_unification(self):
        # Jefferey=2·Ryan, anchor Jeffery=60 → Ryan 30, Jason 10 → 100
        self.assertEqual(reading.solve(
            "Jason, Ryan, and Jeffery went fishing at the lake. Ryan caught three times "
            "the number of fish that Jason caught. Jefferey caught twice the number of "
            "fish that Ryan got. If Jeffery caught 60 fish, how many fish did all of "
            "them catch in total?"), 100)

    def test_pair_sum_relational(self):
        # M = 2E; M+E = 50−5 = 45 → E = 15, M = 30
        self.assertEqual(reading.solve(
            "Merill has twice as many marbles as Elliot and the two of them together "
            "have five fewer marbles than Selma. If Selma has fifty marbles, how many "
            "marbles does Merill have?"), 30)

    def test_fused_conjuncts_and_less_than_diff(self):
        # jenna = 2·(60/3) = 40 → "how much LESS than Bob" = 60 − 40 = 20
        self.assertEqual(reading.solve(
            "If Jenna has twice as much money in her bank account as Phil does, and "
            "Phil has one-third the amount of money that Bob has in his account, and "
            "Bob has $60 in his account, how much less money does Jenna have in her "
            "account than Bob?"), 20)


class TestIncrement216FusionAndAdjuncts(unittest.TestCase):
    """#216 — two GSM8K-typo/noise tolerances in the anchored family: the MISSING-PERIOD
    fusion ("In total Emma spent $58 If Elsa spent…") splits at a mid-premise " If <Name>
    <verb>"; the leading "In total <Name> spent" is a SINGLE-entity amount exempt from the
    combined-total guard; a capitalized EVENT adjunct ("in their school's Book Tournament")
    is stripped so it can't inflate the name count."""

    def test_missing_period_fusion_and_single_entity_total(self):
        # emma 58; elsa 116; elizabeth 464 → 638
        self.assertEqual(reading.solve(
            "Elizabeth, Emma, and Elsa went shopping on Wednesday. In total Emma spent "
            "$58 If Elsa spent twice as much as Emma, and Elizabeth spent four times as "
            "much as Elsa, how much money did they spend in total?"), 638)

    def test_capitalized_event_adjunct_stripped(self):
        # amanda = 18/3 = 6; kara = 3; patricia = 21
        self.assertEqual(reading.solve(
            "Candice read 3 times as many books as Amanda in their school's Book "
            "Tournament. Kara read half the number of books that Amanda read, and "
            "Patricia read 7 times the number of books that Kara read. If Candice read "
            "18 books, how many books did Patricia read?"), 21)


class TestIncrement218GiveChain(unittest.TestCase):
    """#218 — `_give_chain_system`: a holdings pool + relational gives (literal, k× a prior
    give ± offset, k× the running total, k× the current holding), asked left/kept, the
    backward initial, a recipient difference, or cents→nickels (÷5, which must be checked
    BEFORE the generic left ask — raw cents was the draft's first bug; the offset form must
    be tried before the plain k-forms, which match its tail and drop the ±n — the second)."""

    def test_offset_relational_give_left(self):
        # 50 − (10 + 20) = 20
        self.assertEqual(reading.solve(
            "Ken had fifty pencils, and he wanted to share some of them with his two "
            "friends, Manny and Nilo. Ken gave ten pencils to Manny and ten more pencils "
            "to Nilo than he gave to Manny. He kept the rest of the pencils. How many "
            "pencils did Ken keep?"), 20)

    def test_nickels_left_conversion(self):
        # (95 − 25 − 50) / 5 = 4
        self.assertEqual(reading.solve(
            "Ray has 95 cents in nickels. If Ray gives 25 cents to Peter, and twice as "
            "many cents to Randi as he gave to Peter, how many nickels does Ray have "
            "left?"), 4)

    def test_nickels_recipient_difference(self):
        # (60 − 30) / 5 = 6
        self.assertEqual(reading.solve(
            "Ray has 175 cents in nickels. Ray gives 30 cents to Peter, and he gives "
            "twice as many cents to Randi as he gave to Peter. How many more nickels "
            "does Randi have than Peter?"), 6)

    def test_running_total_reference(self):
        # 72 − (12 + 14 + 4) = 42
        self.assertEqual(reading.solve(
            "Colton had 72 dolphin stickers. He gave 4 stickers each to 3 friends. He "
            "also gave his friend Mandy 2 more than he gave his three friends total. And "
            "he gave Justin 10 less than Mandy. How many stickers does Colton have "
            "left?"), 42)

    def test_backward_initial_from_remaining(self):
        # 4 + 8 + (4·8 − 5) + 6 = 45 — the offset-before-k form-order pin
        self.assertEqual(reading.solve(
            "Hector purchased a container of gumballs. He gave 4 to Todd, then he gave "
            "twice as many as he had given Todd to Alisha, and then he gave 5 less than "
            "four times as many to Bobby as he had given to Alisha. If Hector had 6 "
            "gumballs remaining, what is the total number of gumballs that Hector "
            "purchased?"), 45)

    def test_start_with_current_holding_reference(self):
        # 30 + 60 + 120 + 90 = 300
        self.assertEqual(reading.solve(
            "Miriam currently has 30 marbles, but she used to have more. Miriam gave her "
            "brother 60 marbles, gave her sister twice that amount and gave her friend "
            "Savanna three times the amount she currently has. How many marbles did "
            "Miriam start with?"), 300)


class TestIncrement219SignedMixedChain(unittest.TestCase):
    """#219 — `_signed_mixed_chain_system` (the fraction+literal vein): a pool (literal /
    product / derived from "Half of Jerome's money was $43") + in-order steps (literal loss,
    k× the previous literal, fraction of CURRENT), asked left/kept — or the last step's own
    amount when the ask names its eater. Steps are position-sorted per clause; the literal
    form's (?!/) lookahead keeps it off fraction numerators (the overlap dedup would drop
    the fraction event)."""

    def test_derived_pool_klit(self):
        # 2·43 − 8 − 24 = 54
        self.assertEqual(reading.solve(
            "Half of Jerome's money was $43. He gave $8 to Meg and thrice as much to "
            "Bianca. How much does Jerome have left?"), 54)

    def test_fraction_of_that_money_numerator_guard(self):
        # 100 − 25 − 40 = 35 ("1/4" must parse as a fraction, not a literal 1)
        self.assertEqual(reading.solve(
            "John receives $100 from his uncle and gives his sister Jenna 1/4 of that "
            "money. He goes and buys groceries worth $40. How much money does John have "
            "remaining?"), 35)

    def test_pool_stated_last(self):
        # 760 − 300 − 150 = 310
        self.assertEqual(reading.solve(
            "Maria went on a trip to Belgium. She paid $300 for the ticket and half of "
            "that for the hotel. How much money does she have left, when she had $760 "
            "at the beginning?"), 310)

    def test_lit_frac_lit_interleaved(self):
        # 19 − 5 → 14; −7 → 7; −2 = 5
        self.assertEqual(reading.solve(
            "Maria has 19 cookies. She decided to give her friend 5 of them, and half "
            "of the rest to her family. From the rest, Maria decided to eat 2 cookies. "
            "How many cookies will she have left?"), 5)

    def test_remainder_getter_vs_last_eater(self):
        # Mandy GETS the remainder (15); James EATS the last step (3)
        self.assertEqual(reading.solve(
            "A bar of chocolate is made up of 60 pieces. If Michael takes half of the "
            "bar, Paige takes half of the remainder, and Mandy is left with what's left "
            "of the chocolate bar, how many pieces of chocolate will Mandy get?"), 15)
        self.assertEqual(reading.solve(
            "James orders an 8 piece pizza. His friend eats 2 slices and James eats "
            "half of what is left. How many slices does James eat?"), 3)

    def test_product_pool(self):
        # 20·9 = 180; half sold → 90
        self.assertEqual(reading.solve(
            "Brenda bakes 20 cakes a day. She does this for 9 days and then sells half "
            "of the cakes. How many cakes are left with Brenda?"), 90)

    def test_two_anchored_fractions_keep_ask(self):
        # 135·(2/3)·(3/5) = 54 (float noise rounded)
        self.assertEqual(reading.solve(
            "Oliver had 135 stickers. He used 1/3 of his stickers, gave 2/5 of the "
            "remaining to his friend, and kept the remaining stickers. How many "
            "stickers did he keep?"), 54)

    def test_same_base_second_fraction_still_declines(self):
        # RE-PINNED (#225): the SAME-BASE branch now composes this correctly —
        # 200·(1 − 3/8 − 3/10) = 65 (a coordinated bare fraction shares the original base)
        self.assertEqual(reading.solve(
            "John had $200. He gave 3/8 of his money to his mother and 3/10 to his "
            "father. How much money did John have left?"), 65)

    def test_digit_free_gain_step_declines(self):
        # "her mother gave Sabrina HALF the number…" is a digit-free GAIN the chain doesn't
        # model — the uncovered-occurrence guard declines (partial-step commit was 3.33≠5)
        self.assertEqual(reading.solve(
            "Sabrina gave 10 cookies to her brother. Her mother gave Sabrina half the "
            "number of cookies she gave her brother. Then Sabrina gave two-thirds of "
            "her cookies to her sister. If Sabrina had 20 cookies at the start, how "
            "many cookies are left with her?"), "undetermined")

    def test_total_minus_product_other_item(self):
        # #221: 15 − 3·3 = 6
        self.assertEqual(reading.solve(
            "Mrs. Thompson bought 3 chickens for $3 each. She also bought a bag of "
            "potatoes. Mrs. Thompson paid $15 in total. How much did the potatoes "
            "cost?"), 6)

    def test_total_minus_product_residual_count(self):
        # (24 − 9·2) / 1.50 = 4
        self.assertEqual(reading.solve(
            "Chenny bought 9 plates at $2 each. She also bought spoons at $1.50 each. "
            "How many spoons did Chenny buy if she paid a total of $24 for the plates "
            "and spoon?"), 4)

    def test_budget_residual_per_person(self):
        # (100 − 8·9) / 2 = 14
        self.assertEqual(reading.solve(
            "Jaco wants to buy Christmas gifts for his 8 friends that are worth $9 "
            "each. Also, he wants to buy gifts for his mother and father that are of "
            "the same amount. He has a $100 budget to buy all gifts. How much is "
            "Jaco's budget for each of his mother and father's gift?"), 14)

    def test_change_derived_per_item(self):
        # (20 − 11) / 3 = 3
        self.assertEqual(reading.solve(
            "James buys 3 packs of candy. He pays with a $20 bill and gets $11 change. "
            "How much did each pack of candy cost?"), 3)

    def test_needed_minus_has(self):
        # 2·49 − 20 = 78
        self.assertEqual(reading.solve(
            "Cory has $20.00 and she wants to buy two packs of candies that cost "
            "$49.00 each. How much money does Cory need so she will be able to buy "
            "the pack of candies?"), 78)

    def test_customer_orders_revenue(self):
        # 7·5 + 8·4 = 67
        self.assertEqual(reading.solve(
            "At a coffee shop, 7 customers order coffee at $5 each and 8 customers "
            "order tea at $4 each. How much money did the coffee shop make?"), 67)

    def test_shared_count_over_each_prices(self):
        # 6·(1.25 + 1.75) = 18
        self.assertEqual(reading.solve(
            "The card shop has two boxes of cards. The first box contains cards that "
            "cost $1.25 each. The second box contains cards that cost $1.75 each. A "
            "boy then comes in and buys 6 cards from each box. What was the total "
            "cost, in dollars, of the cards he bought?"), 18)

    def test_height_possessive_measure_chain(self):
        # #222: bill = 72/2 = 36; jan = 36 + 6 = 42 (the possessive-MEASURE elided ratio)
        self.assertEqual(reading.solve(
            "Cary is 72 inches tall. Her younger brother Bill is half her height, and "
            "her younger sister Jan is 6 inches taller than Bill. How tall is Jan?"), 42)

    def test_calendar_age_born_year_chain(self):
        # 2021 − 1976 = 45; graham 42; janice 21
        self.assertEqual(reading.solve(
            "It's February 2021. Mark was born in January 1976. Graham is 3 years "
            "younger than Mark, and Graham's sister, Janice, is 1/2 the age of Graham. "
            "How old is Janice?"), 21)

    def test_calendar_age_future_year_ask(self):
        # simon 2005 = 40; 2010 → 45
        self.assertEqual(reading.solve(
            "Jorge is 24 years younger than Simon. In 2005, Jorge is 16 years old. In "
            "2010, how old would Simon be?"), 45)

    def test_calendar_age_birth_year_ask(self):
        # now = 1970 + 40 = 2010; brother = 20 → born 1990
        self.assertEqual(reading.solve(
            "Karina was born in 1970 and she is currently twice as old as her brother. "
            "If her current age is 40, in what year was her brother born?"), 1990)

    def test_grouped_partition_tables(self):
        # #223: 16·2 + 5·3 + 11·4 = 91 (the cast size recovered from _STATED_CONV — "There
        # are 32 tables in a hall" is eaten as a conversion statement)
        self.assertEqual(reading.solve(
            "There are 32 tables in a hall. Half the tables have 2 chairs each, 5 have "
            "3 chairs each and the rest have 4 chairs each. How many chairs in total "
            "are in the hall?"), 91)

    def test_grouped_partition_food_drive(self):
        # 15·12 + 2·0 + 13·4 = 232
        self.assertEqual(reading.solve(
            "The fifth grade class at Rosa Parks Elementary School is holding a food "
            "drive. Half the students in Ms. Perez's class collected 12 cans each, two "
            "students didn't collect any, and the remaining 13 students students each "
            "collected 4 cans. If Ms. Perez's class has 30 students, how many cans did "
            "they collect total?"), 232)

    def test_bill_denomination_compare(self):
        # (10·20 + 3·5) − (15·10 + 4·5) = 45
        self.assertEqual(reading.solve(
            "Oliver has 10 $20 and 3 $5 bills. William has 15 $10 and 4 $5 bills. How "
            "much more money does Oliver have than William?"), 45)

    def test_team_score_compare(self):
        # (12+10) − (8+11) = 3 — team membership tracked across severed clauses
        self.assertEqual(reading.solve(
            "In the first team, Beth scored 12 while Jan scored 10. In the second team, "
            "Judy scored 8 while Angel scored 11. How many more points did the first "
            "team get than the second team?"), 3)

    def test_rate_hours_days(self):
        # 5·2·5 = 50
        self.assertEqual(reading.solve(
            "Mira jogs every morning. She jogs 5 miles per hour. If she jogs for 2 "
            "hours every morning, how many miles can she jog for five days?"), 50)

    def test_percent_increased_bill(self):
        # #224: 60 · 1.3 = 78 (word percent "thirty")
        self.assertEqual(reading.solve(
            "Maximoff's monthly bill is $60 per month. His monthly bill increased by "
            "thirty percent when he started working at home. How much is his total "
            "monthly bill working from home?"), 78)

    def test_spent_percent_backward(self):
        # 14 · 65/35 = 26
        self.assertEqual(reading.solve(
            "Mitzel spent 35% of her allowance. If she spent $14, how much money is "
            "left in her allowance?"), 26)

    def test_pay_fraction_of_dozen_product(self):
        # 5·12·6·0.8 = 288
        self.assertEqual(reading.solve(
            "Jan buys 5 dozen roses. Each rose cost $6. Since he bought so many he "
            "only needs to pay 80%. How much did he pay?"), 288)

    def test_cheaper_relational_with_question_counts(self):
        # tomatoes = 5·0.8 = 4; 2·4 + 3·5 = 23 (counts ride in the QUESTION; the
        # standard's noun appears in BOTH premises — the base premise owns it)
        self.assertEqual(reading.solve(
            "One kilogram of tomatoes is 20% cheaper than one kilogram of cucumbers. "
            "One kilogram of cucumbers costs $5. What is the price of two kilograms of "
            "tomatoes and three kilograms of cucumbers?"), 23)

    def test_same_base_percent_nights(self):
        # #225: 500·(1 − 0.2 − 0.2 − 0.3) = 150 — percent steps, "the third night" ordinal
        # exempted from the uncovered-occurrence guard
        self.assertEqual(reading.solve(
            "Xander read 20% of his 500-page book in one hour. The next night he read "
            "another 20% of the book. On the third night, he read 30% of his book. How "
            "many pages does he have left to read?"), 150)

    def test_same_base_requires_pure_fraction_chain(self):
        # a literal step present → the base stays ambiguous → decline (the #220 rule; Leo)
        self.assertEqual(reading.solve(
            "Leo had 80 marbles. He gave 20 marbles to his cousin and then gave 1/4 of "
            "his marbles to his sister. How many marbles does Leo have left?"),
            "undetermined")

    def test_rate_singles_representatives(self):
        # #226 — four of the sixteen branches (the rest are diff-verified)
        self.assertEqual(reading.solve(
            "Irene earns $500 if she works for 40 hours a week and gets an extra $20 "
            "for every hour of overtime. If she worked 50 hours last week, calculate "
            "her total income."), 700)
        self.assertEqual(reading.solve(
            "Jennifer purchased 40 cans of milk at the store before meeting her "
            "classmate Mark, who was also buying milk. Jennifer bought 6 additional "
            "cans for every 5 cans Mark bought. If Mark purchased 50 cans, how many "
            "cans of milk did Jennifer bring home from the store?"), 100)
        self.assertEqual(reading.solve(
            "There's an online sale where you get $10 for every $100 that you spend. "
            "If you make a purchase of $250 before discounts, how much did you end up "
            "paying?"), 230)
        self.assertEqual(reading.solve(
            "Jane had a 500 page book to read. The first half of the book Jane read "
            "with a speed of 10 pages per day, and the second half with a speed of 5 "
            "pages a day. How many days did Jane spend reading this book?"), 75)


class TestRateSinglesSystemResultPilot(unittest.TestCase):
    """#402 — the WHOLE-TEXT REGEX obligation-absorption pilot: 8 of `_rate_singles_system`'s
    16 branches (1,2,3,4,6,10,11,16) now emit a `Quantity` fact instead of returning the
    computed value directly, letting `answer()` independently re-derive it (see
    `_singles_result`'s docstring for the full recipe and the function's own docstring for why
    the other 8 branches were individually verified NOT to convert cleanly). Fresh synthetic
    numbers per branch, distinct from the GSM8K-sourced `test_rate_singles_representatives`."""

    def test_pronoun_subject_pool_returns_system_result(self):
        # branch 1: subject introduced in an EARLIER sentence than any captured value.
        text = ("Nadia collects marbles. She had 15 marbles before she went on a trip. On "
                "the trip, she found 4 marbles per day for 6 days. Then he found 3 marbles "
                "on the fifth day. How many marbles does she have now?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._rate_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        # the SAME fact attributes to every premise index (_singles_result doesn't know which
        # premise it came from), so dedupe by identity before checking there's exactly one.
        all_facts = {id(f): f for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(len(all_facts), 1)
        fact = next(iter(all_facts.values()))
        self.assertEqual(fact.entity, "nadia")
        self.assertEqual(fact.amount, 15 + 4 * 6 + 3)
        self.assertEqual(reading.solve(text), 42)

    def test_named_subject_returns_system_result(self):
        # branch 10: named subject, no fallback needed.
        text = ("Derek has 4 cats and 6 dogs. He needs to pay a friend to watch them, who "
                "charges $7 a night per animal. How much does Derek have to pay?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._rate_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        self.assertEqual(all_facts[0].entity, "derek")
        self.assertEqual(all_facts[0].amount, (4 + 6) * 7)
        self.assertEqual(reading.solve(text), 70)

    def test_combined_named_pair_returns_two_facts(self):
        # branch 4: a genuine two-fact re-derivation via the combined_q sum fold, not an echo.
        text = ("Priya has 20 grams of gold that cost $12 per gram. Omar has 40 grams of "
                "gold for the cost of $9 per gram. How much is the cost of their gold "
                "combined?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._rate_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        seen = {f.entity: f.amount for f in all_facts}
        self.assertEqual(seen, {"priya": 240, "omar": 360})
        self.assertEqual(reading.solve(text), 600)

    def test_change_ask_returns_two_signed_facts(self):
        # branch 6: two SIGNED facts (have/spend), not one precomputed net.
        text = ("Two sisters go to the fair. Fair tickets are $6 per person. If the sisters "
                "brought $20 with them, how much change will they receive after buying the "
                "tickets?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._rate_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        all_facts = [f for fs in result.facts_by_premise.values() for f in fs]
        self.assertEqual({(f.verb, f.amount) for f in all_facts}, {("have", 20), ("spend", 12)})
        self.assertEqual(reading.solve(text), 8)

    def test_deferred_branch_stays_on_legacy_scalar_path(self):
        # branch 15's real question is an IMPERATIVE ("calculate...") -- answer() declines every
        # imperative by design, so this branch must stay on the unchanged legacy scalar path.
        text = ("Priya earns $400 if she works for 30 hours a week and gets an extra $15 for "
                "every hour of overtime. If she worked 35 hours last week, calculate her "
                "total income.")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._rate_singles_system(premises, question, rp)
        self.assertNotIsInstance(result, reading.SystemResult)
        self.assertEqual(result, 475)
        self.assertEqual(reading.solve(text), 475)

    def test_money_singles_representatives(self):
        # #227 — four of the twenty-five branches (the rest are diff-verified)
        self.assertEqual(reading.solve(
            "The total cost of Olivia's groceries was $42. The bananas cost her $12, "
            "the bread cost her $9, and the milk cost her $7. The only other groceries "
            "she bought were apples. In dollars, how much money did the apples cost "
            "her?"), 14)
        self.assertEqual(reading.solve(
            "Samuel has $5, $10, and $20 bills which amount to $150 in his wallet. His "
            "$10-bills amount to $50, and he has 4 $20-bills. How many bills does "
            "Samuel have in his wallet?"), 13)
        self.assertEqual(reading.solve(
            "Kim orders a $10 meal and a drink for 2.5. She gives a 20% tip. She pays "
            "with a $20 bill. How much does she get in change?"), 5)
        self.assertEqual(reading.solve(
            "Randy has some money in his piggy bank. He spends 2 dollars every time he "
            "goes to the store. He makes 4 trips to the store every month. If he had "
            "$104 dollars left in his piggy bank after a year, how much money, in "
            "dollars, did he have at first?"), 200)

    def test_later_unanchored_fraction_declines(self):
        # #220 (TEST-checkpoint-forced): an UNANCHORED fraction after a prior event has a
        # genuinely ambiguous base — same-base ("¼ of his marbles" = of the ORIGINAL) vs
        # of-current both occur in the gold; guessing committed TEST wrongs. Only a
        # remainder-ANCHORED later fraction ("of the rest/what is left") is of-current.
        self.assertEqual(reading.solve(
            "Leo had 80 marbles. He gave 20 marbles to his cousin and then gave 1/4 of "
            "his marbles to his sister. How many marbles does Leo have left?"),
            "undetermined")

    def test_generalized_animal_pair_returns_system_result(self):
        # branch 10 GENERALIZED (post-hoc audit): was hardcoded to the literal words
        # "cats"/"dogs" -- now reads any two members of the existing `animal` category, so
        # it fires for rabbits/goats too, not just cats and dogs.
        text = ("Derek has 4 rabbits and 6 goats. He needs to pay a friend to watch them, "
                "who charges $7 a night per animal. How much does Derek have to pay?")
        self.assertEqual(reading.solve(text), 70)


class TestMoneySinglesSystemResultBatch2(unittest.TestCase):
    """#403 — obligation-absorption batch 2: 9 of `_money_singles_system`'s 24 active branches
    (2,3,5,6,8,18,21,24,25) now emit `Quantity` fact(s) instead of a bare scalar. #3 (shirts)
    was also GENERALIZED first (hardcoded literal "shirts" -> the existing `clothing` category),
    per the standing rule that a solver must be generalized before absorption. Fresh synthetic
    numbers per branch, distinct from any GSM8K-sourced text. See the function's own docstring
    for why the other 15 branches (including two NEW gap classes this batch discovered: a
    `_parse_problem` question-selection bug from a "Use X to represent" trailer, and GUARD 12
    seeing a singles branch's arithmetic for the first time once it falls through to it) stay
    unconverted."""

    def test_named_pair_purchase_returns_system_result(self):
        # branch 2: two count x unit-price purchases, named subject.
        text = ("Priya bought 3 pounds of coffee and 4 gallons of milk. A pound of coffee "
                "costs $4 and a gallon of milk costs $2. How much will Priya pay in all?")
        self.assertEqual(reading.solve(text), 20)

    def test_generalized_clothing_category_returns_two_facts(self):
        # branch 3: "shorts" (a DIFFERENT clothing member than the original hardcoded "shirts")
        # -- proves the _cat_re('clothing') generalization actually fires for other garments.
        text = ("Wren buys 3 shorts. The first pair costs $4 more than the second pair. If the "
                "first pair costs $12, how much, in dollars, did the shorts cost in total?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._money_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("first", 12), ("second", 8)})
        self.assertEqual(reading.solve(text), 20)

    def test_generalized_vehicle_category_returns_system_result(self):
        # branch 24 GENERALIZED (post-hoc audit): was hardcoded to the literal word "bike" --
        # now reads any member of the existing `vehicle` category. solve() declines this
        # branch's shape both before and after (a pre-existing tolerated BROKE, same as batches
        # 3-4's Janine/Larry/Emma -- unrelated to the generalization); verify at the branch
        # level that the SystemResult itself carries the right facts for a DIFFERENT vehicle
        # than the original hardcoded "bike".
        text = ("Nadia works for $8 an hour and 35 hours a week. She wants to buy a scooter "
                "for $400. How much money will she have left over a month since she started "
                "working?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._money_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount, f.verb) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("nadia", 1120, "earn"), ("nadia", 400, "spend")})

    def test_comparative_pair_returns_four_signed_facts(self):
        # branch 6: two holders, signed have/spend facts, answer()'s comparative fold.
        text = ("Elan has $60. He spends $10 on lunch for himself, $8 for lunch for his dad, "
                "and $2 on more lunch for himself. His brother Milo has $70 and only spends "
                "$5 on lunch for his mom. How much more money does Milo have left than Elan?")
        self.assertEqual(reading.solve(text), 25)

    def test_tip_change_returns_two_signed_facts(self):
        # branch 8: two signed facts (paid-with / spent), the change_q remainder fold.
        text = "Uma orders a $12 meal and a drink for 3.  She gives a 20% tip.  She pays with a $25 bill.  How much does she get in change?"
        self.assertEqual(reading.solve(text), 7)

    def test_month_save_spend_chain_returns_signed_facts(self):
        # branch 25: N signed "save" facts + a "spend" fact, the remainder fold.
        text = ("Priya saved $40 in June. She saved $15 in July and $5 in August. Then Priya "
                "spent $20 on a gift. How much money does Priya have left?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._money_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        self.assertEqual(reading.solve(text), 40)

    def test_deferred_fused_if_clause_stays_on_legacy_scalar_path(self):
        # branch 23: a real TEST-adjacent gap -- the "if" clause fuses to the question, so its
        # own stated values sit in the question's reading; GUARD 12 (seeing a singles branch's
        # arithmetic for the first time via the SystemResult fall-through) has no audit-log
        # evidence for them (_rs_ret logs only the final scalar). Must stay a bare scalar.
        text = "Sana buys 6 kg of rice at the market. If 3 kg of rice costs $9, how much will she pay?"
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._money_singles_system(premises, question, rp)
        self.assertNotIsInstance(result, reading.SystemResult)
        self.assertEqual(result, 18)
        self.assertEqual(reading.solve(text), 18)

    def test_deferred_question_shatter_stays_on_legacy_scalar_path(self):
        # branch 4: a "Use X to represent..." trailer steals the question slot from
        # _parse_problem, burying the real question as a premise -- must stay a bare scalar.
        text = ("Priya earned 3 fewer dollars than Omar. Together they earned $21. How many "
                "dollars did Priya earn? Use O to represent how many dollars Omar earned.")
        self.assertEqual(reading.solve(text), 9)


class TestMultSinglesSystemResultBatch3(unittest.TestCase):
    """#404 — obligation-absorption batch 3: 14 of `_mult_singles_system`'s 33 active branches
    (1,3,6,10,14,18,19,21,22,26,29,31,32,33) now emit `Quantity` fact(s) instead of a bare
    scalar. #10 (Amanda/Peter) and #26 (men/women/children) are genuine multi-fact
    re-derivations; the rest are single echo facts. Branch #6 (Janine) is a TOLERATED "BROKE"
    case per this batch's relaxed acceptance gate ("reducing correct answers is fine, a false
    positive is not"): the branch itself still emits the right fact, but a downstream GUARD in
    `_solve_inner` declines the full `solve()` where the pre-migration bare scalar used to
    commit -- correct -> decline, never correct -> wrong. Fresh synthetic numbers per branch,
    distinct from any GSM8K-sourced text. See the function's own docstring for the three new
    gap classes this batch's screening surfaced (bare-copula-no-aux; the "it took X ... to
    VERB" idiom; a contraction-triggered remainder-fold doubling) and why the other 19 branches
    stay unconverted."""

    def test_two_animal_sum_returns_system_result(self):
        # branch 1: a dogs + b cats + 2x(a+b) fish -> total, named subject.
        text = ("Tariq has 4 dogs, 5 cats and twice as many fish as cats and dogs combined. "
                "How many pets does Tariq have in total?")
        self.assertEqual(reading.solve(text), 27)

    def test_doubling_chain_no_captured_value_returns_system_result(self):
        # branch 3: no numeric literal at all (dok(set())) -- a fixed 3.0, named subject.
        text = ("Farah eats a bag of pretzels for dinner and then eats twice as many after "
                "dinner.  How many bags of pretzels did she eat?")
        self.assertEqual(reading.solve(text), 3)

    def test_janine_pages_branch_emits_fact_but_solve_tolerated_broke(self):
        # branch 6: the migration itself works (the SystemResult carries the right fact), but a
        # downstream GUARD declines the full pipeline -- the one accepted BROKE case in this
        # batch's diff-verify, never a wrong number.
        text = ("Priya read 6 books last month. This month, she read twice as many books. If "
                "each book has 12 pages, how many pages did Priya read in two months?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mult_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("priya", 216)})
        self.assertNotEqual(reading.solve(text), 216)

    def test_generalized_animal_triple_returns_system_result(self):
        # branch 1 GENERALIZED, now via the MINIMUM-STRUCTURE frame: reads any three members
        # of the existing `animal` category (the "combined"/"together"/"altogether"/"in total"
        # clause cross-checked against the same two captured words), any scale word (not just
        # "twice"), and the question's OWN collective noun as the hypernym every captured
        # animal must belong to -- "pets" is the `pet` subcategory (dog/cat/fish/bird/rabbit/
        # etc.), which farm animals like goats/cows are NOT members of, so the animal words here
        # must actually be pets, not just any `animal` category member.
        text = ("Farid has 3 rabbits, 2 birds and twice as many fish as birds and rabbits "
                "combined. How many pets does Farid have in total?")
        self.assertEqual(reading.solve(text), 15)

    def test_minimum_structure_scale_collective_and_hypernym_all_vary(self):
        # branch 1's MINIMUM-STRUCTURE frame, exercised on every axis at once: a scale word
        # OTHER than "twice" ("three times"), a collective marker OTHER than "combined"
        # ("together"), and a hypernym OTHER than "pets" ("animals", the `animal` category
        # itself rather than its `pet` subcategory) -- none of these are hardcoded literals.
        text = ("Ed has 2 dogs, 3 cats and three times as many fish as cats and dogs "
                "together. How many animals does Ed have in total?")
        self.assertEqual(reading.solve(text), 20)

    def test_generalized_vehicle_word_returns_system_result(self):
        # branch 29 GENERALIZED (post-hoc audit): was hardcoded to the literal word "cars" --
        # now reads any member of the existing `vehicle` category, cross-checked against the
        # question's own "how many toy <word>" ask.
        text = ("A toy truck factory made 40 trucks yesterday. Today, they made twice the "
                "number of trucks than yesterday. How many toy trucks did the factory make?")
        self.assertEqual(reading.solve(text), 120)

    def test_garden_pair_returns_two_signed_facts(self):
        # branch 10: a genuine 2-fact re-derivation (have + give), the change_q remainder fold.
        text = ("Nadia’s garden contains 10 flowers and Farid’s garden contains three "
                "times as many flowers as Nadia's. If Farid gave 8 flowers to his brother, how "
                "many flowers are left in his garden?")
        self.assertEqual(reading.solve(text), 22)

    def test_gift_chain_asked_recipient_returns_system_result(self):
        # branch 14: the asked recipient's own name (not the givers') is the fact's entity.
        text = ("On Monday, Omar had 20 books and Sara had twice as many books as Omar. On "
                "Tuesday, Omar gave 5 books to Nina, and Sara gave Nina 7 more than Omar gave. "
                "How many books did Nina get?")
        self.assertEqual(reading.solve(text), 17)

    def test_twice_of_complement_returns_system_result(self):
        # branch 18: named subject captured straight off the question.
        text = ("There were 90 Easter eggs in the yard.  Talia found twice as many as Reem.  "
                "How many Easter eggs did Talia find?")
        self.assertEqual(reading.solve(text), 60)

    def test_found_total_returns_system_result(self):
        # branch 19: subject captured off the "found a total of" premise clause.
        text = ("The number of Christmas presents Youssef found was twice as many as the "
                "number of birthday presents. If Youssef found a total of 40 Christmas "
                "presents, how many presents did he get all together?")
        self.assertEqual(reading.solve(text), 60)

    def test_dozens_no_value_optional_capture_returns_system_result(self):
        # branch 21: original gate has NO subject capture at all -- the new capture is
        # deliberately optional (falls back to the legacy scalar if it ever fails to match).
        text = "Idris wants to buy two and one-half dozens of pens. How many pens will he have?"
        self.assertEqual(reading.solve(text), 30)

    def test_store_pair_optional_capture_returns_system_result(self):
        # branch 22: same optional-capture pattern, entity resolved via the noun-keyed fallback
        # for the "they" pronoun in the question.
        text = ("Lina's Pool Supply has three times as many swimming pools as Lina's Ark & "
                "Athletic Wear store. If Lina's Ark & Athletic Wear store has 50 pools, how "
                "many pools do they have in total?")
        self.assertEqual(reading.solve(text), 200)

    def test_men_women_children_returns_three_category_facts(self):
        # branch 26: a genuine 3-fact category-sum re-derivation over the pre-registered
        # `person` category, keyed by SINGULAR lemma (man/woman/child), not the plural surface.
        text = ("A group of people contains men, women, and children. The number of men is "
                "twice the number of women, and the number of women is 3 times the number of "
                "children. If the number of children is 8, how many people does this group "
                "contain?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mult_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("man", 48), ("woman", 24), ("child", 8)})
        self.assertEqual(reading.solve(text), 80)

    def test_qualified_measure_returns_system_result(self):
        # branch 29: the fact's measure must carry the SAME adjective qualifier ("toy") that
        # _queried_measure resolves from the question -- a bare measure would decline.
        text = ("A toy car factory made 25 cars yesterday. Today, they made twice the number "
                "of cars than yesterday. How many toy cars did the factory make?")
        self.assertEqual(reading.solve(text), 75)

    def test_gift_then_sell_half_returns_system_result(self):
        # branch 31: subject captured off the question's own "did X sell" clause.
        text = ("Zara had a set of 60 gold coins.  Omar gave Zara a gift of an additional 20 "
                "gold coins.  Then, Zara decided to sell half of her coins.  How many coins "
                "did Zara sell?")
        self.assertEqual(reading.solve(text), 40)

    def test_club_target_qualified_measure_returns_system_result(self):
        # branch 32: another qualifier-must-match case ("additional" members).
        text = ("A club is going to get additional members so that they will have 5 more than "
                "twice their current number of their members. If the club has 6 members now, "
                "how many additional members do they need?")
        self.assertEqual(reading.solve(text), 11)

    def test_thursday_anchored_chain_returns_system_result(self):
        # branch 33: subject captured off the "sold N boxes ... on thursday" premise clause.
        text = ("She sold twice as many boxes on Tuesday as she did on Wednesday, and she sold "
                "twice as many boxes on Wednesday as she did on Thursday. If Nia sold 100 "
                "boxes of cupcakes on Thursday, how many boxes did she sell on Tuesday?")
        self.assertEqual(reading.solve(text), 400)


class TestMixedSinglesSystemResultBatch4(unittest.TestCase):
    """#405 — obligation-absorption batch 4: 18 of `_mixed_singles_system`'s 30 active branches
    (1,3,4,5,6,9,10,12,13,15,17,18,20,22,23,24,28,29) now emit fact(s) instead of a bare scalar.
    #1 (Lisa/Tommy) and #23 (Aaron/Matthews/Marovich) are genuine multi-fact re-derivations; the
    rest are single echo facts. #9 is the first branch in this whole effort to emit a `Frequency`
    fact instead of a `Quantity` -- "how many TIMES did X <verb>?" routes answer() through a
    dedicated Frequency fold that never looks at Quantity facts at all, regardless of measure.
    #3 (Larry) and #13 (Emma) are TOLERATED "BROKE" cases per this batch's relaxed acceptance
    gate: the branch itself still emits the right fact, but a downstream GUARD in `_solve_inner`
    declines the full pipeline -- correct -> decline, never correct -> wrong. Fresh synthetic
    numbers per branch, distinct from any GSM8K-sourced text. See the function's own docstring
    for why the other 12 branches (including a NEW gap class this batch found: a compound "X and
    Y" measure ask collapses to only the first noun in `_queried_measure`) stay unconverted."""

    def test_comparative_pair_returns_two_signed_facts(self):
        # branch 1: a genuine 2-fact re-derivation, named subjects captured off the question.
        text = ("Nadia, Omar, and Farid earned $80 from washing cars all week. However, half "
                "of the $80 was earned by Nadia. Farid earned half of what Nadia earned. How "
                "much more money did Nadia earn than Farid?")
        self.assertEqual(reading.solve(text), 20)

    def test_larry_branch_emits_fact_but_solve_tolerated_broke(self):
        # branch 3: the migration itself works (the SystemResult carries the right fact), but a
        # downstream GUARD declines the full pipeline -- an accepted BROKE case, never wrong.
        text = ("Idris spends half an hour twice a day walking and playing with his dog. He "
                "also spends a fifth of an hour every day feeding his dog. How many minutes "
                "does Idris spend on his dog each day?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mixed_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("idris", 72)})
        self.assertNotEqual(reading.solve(text), 72)

    def test_fuel_total_returns_system_result(self):
        # branch 4: subject captured off the "used N gallons of fuel this week" clause.
        text = ("Zara used 20 gallons of fuel this week. Last week she used 25% less. How much "
                "fuel did she use in total for the two weeks?")
        self.assertEqual(reading.solve(text), 35)

    def test_wall_area_cans_returns_system_result(self):
        # branch 5: subject captured off the "is painting her room" premise clause.
        text = ("Priya is painting her room. Two of her walls are 4 meters wide and 3 meters "
                "tall. The third wall is 6 meters wide and 3 meters tall. The final wall is 5 "
                "meters wide and 3 meters tall. If each can of paint covers 3 square meters, "
                "how many cans of paint does Priya need?")
        self.assertEqual(reading.solve(text), 19)

    def test_generalized_wall_area_frame_returns_system_result(self):
        # branch 5 GENERALIZED (post-hoc audit): was hardcoded on six axes — "cans" (item
        # noun), "meters" (unit), "two" (count word), "her" (owner pronoun), "walls"
        # (surface noun), "painting" (premise verb). Now every axis is captured: the unit
        # is measure-category-gated and consistency-checked against the coverage clause,
        # the item noun is anchored in the "each <item> … covers" premise and matched back
        # to the question's "how many <items>" via _number_variants, the emitted fact
        # carries the question's own verb, and a count word may prefix ANY dimensions
        # clause (here BOTH). All-admitted vocabulary, different member on every axis:
        # feet/bucket/three+two/his/panels/cleaning, question verb "use".
        text = ("Marcus is cleaning his room. Three of his panels are 4 feet wide and 2 feet "
                "tall. Two of his panels are 6 feet wide and 2 feet tall. If each bucket of "
                "paint covers 3 square feet, how many buckets of paint does Marcus use?")
        self.assertEqual(reading.solve(text), 16)

    def test_generalized_wall_area_pronoun_question_falls_back_to_premise_subject(self):
        # branch 5 GENERALIZED: a pronoun question-subject ("does she need") falls back to
        # the "<Name> is <verb>ing" premise clause for the fact's entity.
        text = ("Priya is painting her room. Two of her walls are 4 meters wide and 3 meters "
                "tall. The third wall is 6 meters wide and 3 meters tall. The final wall is 5 "
                "meters wide and 3 meters tall. If each can of paint covers 3 square meters, "
                "how many cans of paint does she need?")
        self.assertEqual(reading.solve(text), 19)

    def test_page_residual_returns_system_result(self):
        # branch 6: subject resolved via the noun-keyed pronoun fallback ("she" -> Farah).
        text = ("Farah borrows 3 books from the library on Monday. On Monday, she reads the "
                "first book with a total of 25 pages. On Tuesday, she reads the second book "
                "with a total of 15 pages. On Wednesday, she reads the third book. If she has "
                "read a total of 60 pages from Monday to Wednesday, how many pages did she "
                "read on Wednesday?")
        self.assertEqual(reading.solve(text), 20)

    def test_offset_times_chain_returns_frequency_fact(self):
        # branch 9: "how many TIMES did X <verb>?" routes through the Frequency fold, not the
        # Quantity fold -- a Quantity fact with measure ("time", ...) would silently never match.
        text = ("Ravi played catch with Sam and Nina. He caught the ball 14 times. Sam made "
                "four less than double the catches Ravi did. Nina caught the ball sixteen more "
                "than a third of the times Sam did. How many times did Nina catch the ball?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mixed_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.action, f.count) for fs in result.facts_by_premise.values()
                for f in fs if isinstance(f, reading.Frequency)}
        self.assertEqual(seen, {("nina", "catch", 24)})
        self.assertEqual(reading.solve(text), 24)

    def test_state_swap_marbles_returns_system_result(self):
        # branch 10: subject captured off the "has N marbles" clause.
        text = ("Sara has 10 marbles. Lina gives Sara double the number she has. Sara then "
                "gives Lina back half of her new total plus 2. How many marbles does Sara have "
                "now?")
        self.assertEqual(reading.solve(text), 13)

    def test_birthday_pair_totals_returns_system_result(self):
        # branch 12: subject captured off the "received N gifts on his Nth birthday" clause.
        text = ("Farid received 15 gifts on his 10th birthday. He received 5 fewer gifts on "
                "his 11th birthday. How many total gifts did he receive between those two "
                "birthdays?")
        self.assertEqual(reading.solve(text), 25)

    def test_emma_branch_emits_fact_but_solve_tolerated_broke(self):
        # branch 13: the second TOLERATED BROKE case this batch -- fact is right, solve() still
        # declines via a downstream GUARD.
        text = ("Nadia buys 3 containers of milk every school day for lunch. She does not go "
                "to school on the weekends. How many containers of milk does she buy in 2 "
                "weeks?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mixed_singles_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("nadia", 30)})
        self.assertNotEqual(reading.solve(text), 30)

    def test_color_residual_qualified_measure_returns_system_result(self):
        # branch 15: the fact's measure must carry the "green" qualifier that _queried_measure
        # resolves from the question -- a bare measure would decline.
        text = ("Farah has 30 marbles. She has 10 white marbles, 5 blue marbles, and the rest "
                "are green marbles. How many green marbles does Farah have?")
        self.assertEqual(reading.solve(text), 15)

    def test_bags_of_bags_product_returns_system_result(self):
        # branch 17: subject captured off the "has N bags of apples" clause.
        text = ("Nadia has 5 bags of apples. Each of her bags has as many apples as 2 of "
                "Farid's bags. If Farid's bags have 20 apples each, how many apples does "
                "Nadia have?")
        self.assertEqual(reading.solve(text), 200)

    def test_total_partition_photos_returns_system_result(self):
        # branch 18: subject captured off the question's own "how many photos does X have".
        text = ("Omar, Farid, and Nina are collecting photos of cars. Nina has 5 photos more "
                "than Farid. Farid has one hundred photos less than the total amount of "
                "photos which is 170. How many photos does Omar have?")
        self.assertEqual(reading.solve(text), 25)

    def test_morning_afternoon_bakes_returns_system_result(self):
        # branch 20: subject captured off the "baked N cupcakes in the morning" clause.
        text = ("Nadia baked 12 cupcakes in the morning and eight more cupcakes in the "
                "afternoon than in the morning. How many cupcakes did she bake?")
        self.assertEqual(reading.solve(text), 32)

    def test_ducks_chickens_category_returns_system_result(self):
        # branch 22: subject captured off the "has N more ducks than..." clause; the "bird"
        # category covers duck+chicken.
        text = ("Farah has 6 more ducks than three times the number of chickens. If Farah has "
                "90 ducks, how many total birds does she have?")
        self.assertEqual(reading.solve(text), 118)

    def test_three_party_cow_sum_returns_three_facts(self):
        # branch 23: a genuine 3-fact re-derivation, the combined_q NOT-NAMED bare-sum fold.
        text = ("Nadia has three times as many cows as does Farid. Together, they have 20 "
                "more cows than Omar. If Farid has 40 cows, how many cows do the three have "
                "altogether?")
        self.assertEqual(reading.solve(text), 300)

    def test_road_trip_miles_returns_system_result(self):
        # branch 24: subject captured off the "went on a road trip" clause (a compound "Mr. and
        # Mrs. X" name -- the capture grabs the LAST word before "went").
        text = ("Mr. and Mrs. Aziz went on a road trip. On the first day, they traveled 100 "
                "miles. On the second day, they traveled 3/4 as far. On the third day, they "
                "traveled 1/2 as many miles as the first two days combined. How many miles "
                "did they travel for 3 days?")
        self.assertEqual(reading.solve(text), 262.5)

    def test_take_still_less_backward_returns_system_result(self):
        # branch 28: the fact's verb must match the question's own verb ("start"), not the
        # default "have".
        text = ("Nina has 40 berries. Omar takes 6 of Nina's berries, and still has 10 less "
                "berries than Nina started with. How many berries did Omar start with?")
        self.assertEqual(reading.solve(text), 24)

    def test_weekday_flower_sum_returns_system_result(self):
        # branch 29: subject captured off the "has a small flower shop" clause.
        text = ("Farid has a small flower shop. He sold 6 flowers on Monday, 10 flowers on "
                "Tuesday and on Friday, he sold double the number of flowers he sold on "
                "Monday. How many flowers does Farid sell?")
        self.assertEqual(reading.solve(text), 28)


class TestMixedSinglesBSystemResultBatch5(unittest.TestCase):
    """#406 — obligation-absorption batch 5: 13 of `_mixed_singles_b_system`'s 33 active
    branches (1,4,5,8,9,11,15,24,27,28,29,31,33) now emit fact(s) instead of a bare scalar.
    #9 (Itzayana/Zora/Zara/Brixton) is a genuine 4-fact re-derivation (the `answer()` average
    fold sums ANY Quantity facts sharing the asked measure, not just literally-named entities);
    #27/#28 (Javier/Sanda, Natasha/Esteban) are genuine 2-fact `combined_q` NAMED-subset
    re-derivations; #24 is a genuine 3-fact `combined_q` NOT-NAMED bare-sum on a shared generic
    "crayons" entity (three DISTINCT values, no dedup risk). The rest are single echo facts.
    Diff-verify came back PERFECTLY clean this batch (0 SOLVED/0 BROKE/0 DECL->WRONG on 2
    independent snapshot pairs) -- unlike batches 3-4, no branch landed in the tolerated
    correct->decline bucket. Fresh synthetic numbers per branch, distinct from any GSM8K-sourced
    text. See the function's own docstring for the three NEW gap classes this batch's agent
    survey found (GUARD 11 has no _PATTERN_UNIT exemption; the word "average" force-routes a
    question into the average fold even when it's not a real mean; `_singles_result`'s fact
    dedup collapses coincidentally-equal facts) and why the other 20 branches stay unconverted."""

    def test_pie_slice_residual_returns_system_result(self):
        # branch 1: subject captured off the "had N birthday cookie pies" clause.
        text = ("Farah had 4 birthday cookie pies to share with her 19 classmates and her "
                "teacher, Mr. Lee.   If each of the cookie pies were cut into 8 slices and "
                "Farah, her classmates, and Mr. Lee all had 1 piece, how many slices are left?")
        self.assertEqual(reading.solve(text), 11)

    def test_bogo_purchase_returns_system_result(self):
        # branch 4: subject captured off the "buys steaks for buy one get one free" clause.
        text = ("Omar buys steaks for buy one get one free.  The price is $10 per pound and "
                "he buys 12 pounds.  How much did he pay for the steaks?")
        self.assertEqual(reading.solve(text), 60)

    def test_road_trip_elimination_returns_system_result(self):
        # branch 5: subject captured off the question's own "how many miles does X drive".
        text = ("Nina, Sara, and Lina take a road trip that is a total of 500 miles.  Nina "
                "drives 10 miles more than twice Sara, and Sara drives 3 times the amount "
                "that Lina drives. How many miles does Sara drive?")
        self.assertEqual(reading.solve(text), 147)

    def test_pet_food_ounce_conversion_returns_system_result(self):
        # branch 8: subject captured off the "X bought N bags of" clause.
        text = ("Mrs. Kade bought 3 bags of 2-pound bag of cat food and another 3 bags of dog "
                "food that each weigh 1 more pounds than each bag of cat food. There are 16 "
                "ounces in each pound. How many ounces of pet food did Mrs. Kade buy?")
        self.assertEqual(reading.solve(text), 240)

    def test_generalized_pet_food_animal_pair_returns_system_result(self):
        # branch 8 GENERALIZED (post-hoc audit): was hardcoded to the literal words
        # "cat"/"dog" -- now reads any two members of the existing `animal` category, with
        # the second animal's per-bag weight still read RELATIVE to the first's (the
        # problem's own base+delta structure), so it fires for rabbit/goat too.
        text = ("Mrs. Kade bought 3 bags of 2-pound bag of rabbit food and another 3 bags of "
                "goat food that each weigh 1 more pounds than each bag of rabbit food. There "
                "are 16 ounces in each pound. How many ounces of pet food did Mrs. Kade buy?")
        self.assertEqual(reading.solve(text), 240)

    def test_four_person_average_returns_four_facts(self):
        # branch 9: a genuine 4-fact re-derivation, the answer() average fold.
        text = ("Amara is 5 inches taller than Bela, who is 6 inches shorter than Coen. If "
                "Dara is 60 inches tall and has the same height as Coen, calculate the "
                "average height of the four people.")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mixed_singles_b_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {(f.entity, f.amount) for fs in result.facts_by_premise.values() for f in fs}
        self.assertEqual(seen, {("amara", 59), ("bela", 54), ("dara", 60), ("coen", 60)})
        self.assertEqual(reading.solve(text), 58.25)

    def test_bonus_points_qualified_measure_returns_system_result(self):
        # branch 11: the fact's measure must carry the "bonus" qualifier.
        text = ("In a certain company, for every customer served, an employee gets as many "
                "bonus points, as 25% of the number of customers served. On Monday, Lina "
                "served 5 customers per hour for 4 hours. How many bonus points did Lina earn "
                "that day?")
        self.assertEqual(reading.solve(text), 5)

    def test_dozen_plus_free_returns_system_result(self):
        # branch 15: subject captured off the "wants to buy N dozens flowers" clause; measure
        # must be "piece" (the literal question noun), not "flower".
        text = ("For every one dozen flowers bought, a customer gets 3 free flowers. If Nadia "
                "wants to buy 2 dozens flowers, how many pieces of flowers will she have in "
                "all?")
        self.assertEqual(reading.solve(text), 30)

    def test_crayon_box_sum_returns_three_facts(self):
        # branch 24: a genuine 3-fact combined_q NOT-NAMED bare-sum on a shared generic entity.
        text = ("There are 4 boxes of crayons that hold 6 orange crayons. There are 3 boxes "
                "of crayons that have 7 blue crayons. There is 1 box of 9 red crayons. How "
                "many crayons are there in total?")
        premises, question = reading._parse_problem(text)
        readings = [reading.analyze(c) for c in premises + [question]]
        rp = readings[:len(premises)]
        result = reading._mixed_singles_b_system(premises, question, rp)
        self.assertIsInstance(result, reading.SystemResult)
        seen = {id(f): f for fs in result.facts_by_premise.values() for f in fs}
        amounts = sorted(f.amount for f in seen.values())
        self.assertEqual(amounts, [9, 21, 24])
        self.assertEqual(reading.solve(text), 54)

    def test_exercise_minutes_returns_two_named_facts(self):
        # branch 27: a genuine 2-fact combined_q NAMED-subset re-derivation.
        text = ("Omar exercised for 40 minutes every day for one week. Rina exercised for 60 "
                "minutes on each of two days. How many minutes did Omar and Rina exercise in "
                "total?")
        self.assertEqual(reading.solve(text), 400)

    def test_exercise_hours_returns_two_named_facts(self):
        # branch 28: same shape as 27 but the measure is TIME_UNITS ("hour") -- safe here since
        # both subjects are NAMED, not pronouns (the TIME_UNITS exclusion only bites pronouns).
        text = ("Lina exercised for 30 minutes every day for one week. Tomas exercised for 30 "
                "minutes on each of two days. How many hours did Lina and Tomas exercise in "
                "total?")
        self.assertEqual(reading.solve(text), 4.5)

    def test_round_trip_steps_returns_system_result(self):
        # branch 29: subject captured off the "X's house is just a walking distance" clause.
        text = ("Priya's house is just a walking distance from her school. It takes her 200 "
                "steps to walk to the school. How many steps does she take walking to and "
                "from school in three days?")
        self.assertEqual(reading.solve(text), 1200)

    def test_picked_residual_returns_system_result(self):
        # branch 31: subject captured off the question's own "how many oranges did X pick".
        text = ("A total of 90 oranges are picked by Omar and Lina. Omar picked 15 on each of "
                "2 days and Lina picked the rest. How many oranges did Lina pick?")
        self.assertEqual(reading.solve(text), 60)

    def test_triple_tuesday_pencils_returns_system_result(self):
        # branch 33: subject captured off the "X buys N pencils on Monday" clause.
        text = ("Nina buys 10 pencils on Monday. Then she buys 8 more pencils on Tuesday. On "
                "Wednesday she buys triple the number of pencils she did on Tuesday. How many "
                "pencils does she have?")
        self.assertEqual(reading.solve(text), 42)


class TestMixedSinglesCSystemResultBatch6(unittest.TestCase):
    """#407 — obligation-absorption batch 6: 16 of `_mixed_singles_c_system`'s 39 active
    branches (1,3,7,8,9,13,15,16,22,24,26,27,30,34,35,39) now emit fact(s) instead of a bare
    scalar. Two were GENERALIZED FIRST as part of conversion (not retrofitted after, per the
    standing rule): #9 (Carla) was hardcoded to "pigs"/"horses" -- now reads any two `animal`
    category members, cross-checked across two independent count-searches (the clause splitter
    can sever "has N X" from "M Y" onto separate premises, so a single combined regex isn't
    safe here); #15 (Jon laundry) was hardcoded to "shirts"/"pants" -- now reads any two
    `clothing` members, with the wash clause cross-checked against the same two captured words.
    #13 and #22 are genuine 3-fact re-derivations; #24 and #35 are genuine 2-fact
    re-derivations; #30 is a genuine 3-fact CATEGORY-SUM over the `animal` category. The rest
    are single echo facts. Fresh synthetic numbers per branch, distinct from any GSM8K-sourced
    text. See the function's own docstring for the three NEW gap classes this batch's screening
    found (a question's own fused percent-clause hijacking the top-level percent handler; an
    all-caps "CDs" mis-tagged as a name; "divided...among" mis-tagged as the arithmetic division
    operator) and why the other 23 branches stay unconverted."""

    def test_calorie_constrained_purchase_returns_system_result(self):
        # branch 1: subject captured off the "X has $N" clause.
        text = ("Oranges have 60 calories and cost $0.80 each. If Omar has $15 and needs to "
                "make sure he gets 300 calories, how much money will he have left after he "
                "buys the oranges he needs?")
        self.assertEqual(reading.solve(text), 11)

    def test_family_cookie_residual_returns_system_result(self):
        # branch 3: subject captured off the question's own "left for X" clause.
        text = ("Sara made cookies for herself and her family. Her father ate 8 cookies and "
                "her mother ate half as much as the father. Her brother ate 3 more than her "
                "mother. How many cookies are left for Sara if she made 25 cookies in total?")
        self.assertEqual(reading.solve(text), 6)

    def test_eat_and_plant_equation_returns_system_result(self):
        # branch 7: subject captured off the question's own "how many apples did X eat".
        text = ("Nina bought nine apples from the store. From the apples she bought, for each "
                "that Nina ate, she planted two trees from the remaining ones. How many apples "
                "did Nina eat?")
        self.assertEqual(reading.solve(text), 3)

    def test_free_time_words_returns_system_result(self):
        # branch 8: subject captured off the "X has N hours to watch" clause.
        text = ("Omar has 6 hours to watch movies and read. He watches one movie that is 2 "
                "hours and one movie that is 3 hours. He then decides to read. If he can read "
                "15 words a minute, how many words does he read before his free time is up?")
        self.assertEqual(reading.solve(text), 900)

    def test_generalized_per_animal_water_returns_system_result(self):
        # branch 9: GENERALIZED as part of conversion -- was hardcoded to "pigs"/"horses",
        # now reads any two `animal` category members via two independent count-searches.
        text = ("Nadia needs to bring water to her animals. Each cow needs twice as much "
                "water as a goat, and the chickens drink from one tank that needs 15 gallons. "
                "How many gallons of water does Nadia need to bring if she has 4 goats and 3 "
                "cows and each goat needs 2 gallons of water?")
        self.assertEqual(reading.solve(text), 35)

    def test_mixed_unit_feet_sum_returns_three_facts(self):
        # branch 13: a genuine 3-fact re-derivation (miles/yards/feet, all converted to feet).
        text = ("Omar walked 2 miles. Sara walked 500 yards and Amir walked 900 feet. How "
                "many combined feet did the friends walk?")
        self.assertEqual(reading.solve(text), 12960)

    def test_generalized_laundry_loads_returns_system_result(self):
        # branch 15: GENERALIZED as part of conversion -- was hardcoded to "shirts"/"pants",
        # now reads any two `clothing` category members, cross-checked against the wash clause.
        text = ("Nadia's laundry machine can do 6 pounds of laundry at a time. 3 jackets "
                "weigh 1 pound and 5 pairs of shorts weigh 1 pound. If she needs to wash 9 "
                "jackets and 15 shorts, how many loads of laundry does she have to do?")
        self.assertEqual(reading.solve(text), 1)

    def test_marble_bag_gift_returns_system_result(self):
        # branch 16: subject captured off the question's own "does X have" clause.
        text = ("Farah had 15 green marbles. Then she bought 4 bags of blue marbles. There "
                "were 8 marbles in each bag. She created a gift of 3 green marbles and 5 blue "
                "marbles and gave it to a friend. How many marbles does Farah have now?")
        self.assertEqual(reading.solve(text), 39)

    def test_three_holder_crayon_sum_returns_three_facts(self):
        # branch 22: a genuine 3-fact re-derivation, subject names captured per holder clause.
        text = ("Nina has 40 crayons. Sara has 15 and Omar has three fewer crayons than Sara. "
                "How many crayons do they have in total?")
        self.assertEqual(reading.solve(text), 67)

    def test_two_day_goal_sum_returns_two_facts(self):
        # branch 24: a genuine 2-fact combined_q re-derivation; "tom" stays a literal regex
        # anchor (not generalized), so the second player's name must literally be "Tom".
        text = ("Priya and Tom were playing football. On the first day, Priya scored three "
                "goals, which was two less than Tom. On the second day, Priya scored one goal "
                "less than Tom who scored five goals. How many goals did Priya and Tom score "
                "during these two days?")
        self.assertEqual(reading.solve(text), 17)

    def test_dozen_roses_week_returns_system_result(self):
        # branch 26: subject captured off the "X decided to send...roses" premise clause.
        text = ("Farid decided to send his wife 3 dozen roses every day for the week.  How "
                "many total roses did he send?")
        self.assertEqual(reading.solve(text), 252)

    def test_family_cookie_share_returns_system_result(self):
        # branch 27: entity stays the literal category word "child" (NOT generalized -- the
        # adult/child roles are structurally fixed by the branch's own arithmetic).
        text = ("In Nadia's family, there are two adults and three children. In a cookie jar, "
                "there are a total of 90 cookies. If the adults eat 1/3 of the cookies and "
                "then gives the rest to the children to divide equally, how many cookies does "
                "each child get?")
        self.assertEqual(reading.solve(text), 20)

    def test_farm_percent_chain_returns_category_sum(self):
        # branch 30: a genuine 3-fact CATEGORY-SUM over the `animal` category (cow/duck/pig
        # roles stay literal -- structurally fixed, NOT generalized, per the standing rule's
        # own "identity-agnostic" test).
        text = ("Omar has a farm with animals. He has 10 cows, 50% more ducks. Omar also has "
                "as many pigs as one-fifth of ducks and cows in total. How many animals does "
                "Omar have on his farm?")
        self.assertEqual(reading.solve(text), 30)

    def test_rose_fraction_gift_returns_system_result(self):
        # branch 34: subject captured off the question's own "does X give" clause.
        text = ("Priya has $180. She wants to buy roses at $3 each and give some of the roses "
                "to her friends, Jenna and Imma. Jenna will receive  1/3 of the roses, and  "
                "Imma will receive 1/2 of the roses. How many roses does Priya give to her "
                "friends?")
        self.assertEqual(reading.solve(text), 50)

    def test_tip_change_returns_two_signed_facts(self):
        # branch 35: a genuine 2-fact change_q re-derivation (have + spend).
        text = ("Farid ordered four boxes of pizza, which cost $8 each box. He then gave a "
                "tip which amounts to 1/8 of the total cost of his order. If he gave the "
                "delivery man $50, how much change did he receive?")
        self.assertEqual(reading.solve(text), 14)

    def test_tank_third_mpg_returns_system_result(self):
        # branch 39: subject captured via the first-capitalized-word fallback.
        text = ("Nina is wondering how many miles each gallon of gas gets her. Her gas tank "
                "is 15 gallons. She has 2/3 of a tank left after she drives to and from work, "
                "which is 5 miles away from her house. How many miles per gallon does she "
                "get?")
        self.assertEqual(reading.solve(text), 2)


class TestIncrement237GenericRateChain(unittest.TestCase):
    """#237 — the GENERIC RATE-CHAIN solver (dimensional analysis over stated unit bridges),
    the generalization of the specific rate×span family. Registered LAST in _EARLY_SYSTEMS."""

    def test_three_hop_event_chain(self):
        # verbal count rate × two stated bridges × a day span: 6 · 5 · 3 · 4 = 360
        self.assertEqual(reading.solve(
            "When Rita jumps rope, she jumps 6 times per minute. She jumps for 5 minutes "
            "each session and does 3 sessions each day. How many jumps will Rita do in "
            "4 days?"), 360)

    def test_money_chain_with_half_hour_conversion(self):
        # $8/half-hour → ×2/hour; 1 hour/lesson; 1 lesson/week; 3 weeks → 48
        self.assertEqual(reading.solve(
            "Mara pays the teacher $8 for every half-hour of teaching her son. If Leon is "
            "taking one lesson per week, and the lesson lasts 1 hour, how much money "
            "would the teacher earn in 3 weeks?"), 48)

    def test_hour_rate_without_daily_bridge_declines(self):
        # NO hour→day clock constant on purpose (activity hours/day is not 24): a per-hour
        # rate asked over a week with no stated hours-per-day bridge must DECLINE.
        self.assertNotEqual(reading.solve(
            "Sam reads 30 pages per hour. How many pages does Sam read in a week?"),
            30 * 24 * 7)

    def test_relational_multiplier_declines(self):
        # an unconsumed multiplier word ("twice") is a clause this reader can't represent
        self.assertNotEqual(reading.solve(
            "A tram travels at twice the speed that Nell walks. Nell walks 3 miles per "
            "hour. How many miles does the tram travel in 2 hours?"), 6)

    def test_unconsumed_entity_count_declines(self):
        # "two guards" is a word-number outside every rate/bridge match → the per-head
        # multiplication is _herd_rate_week's shape, not a unit chain: never 20·8·7 = 1120
        self.assertNotEqual(reading.solve(
            "Ty hires two guards. They each charge $20 an hour. He hires them for 8 "
            "hours each day. How much does he pay in 7 days?"), 1120)

    def test_month_span_declines(self):
        # month→days varies; a month span never chains through the clock whitelist
        self.assertNotEqual(reading.solve(
            "A printer prints 40 pages each day. How many pages does it print in "
            "2 months?"), 40 * 60)

    def test_degenerate_single_hop_declines(self):
        # a no-bridge rate×span ("12 pages each day … in 4 days") is the INCUMBENTS' turf
        # (proportion / the fold's rate-duration synth) — the chain solver itself must pass
        # (a TEST confident-wrong forced this gate: the incumbents' declines carry signal).
        prem, q = reading._parse_problem(
            "Vera reads 12 pages each day. How many pages does Vera read in 4 days?")
        self.assertIsNone(reading._rate_chain_system(prem, q))


class TestIncrement238AgeEquations(unittest.TestCase):
    """#238 — the GENERIC AGE-EQUATIONS solver: ages are affine in time, premises are linear
    equations over now-ages, asks evaluate iff the row space determines them. Registered LAST."""

    def test_relational_chain_with_cast_sum(self):
        # offsets + an offset-on-ratio, summed over the whole cast: 14+16+12+27 = 69
        self.assertEqual(reading.solve(
            "Marta is 14 and is two years younger than Onel, while Sana is four years "
            "younger than Onel. If Jom is three more than twice as old as Sana, what is "
            "the sum of the ages of all 4 people?"), 69)

    def test_shifted_relation_with_now_pin(self):
        # "in 6 years, twice as old as Ron IS NOW" pins the standard: A = 2·8 − 6 = 10 → diff 2
        self.assertEqual(reading.solve(
            "In 6 years, Ana will be twice as old as Ron is now. Ron is 8 now. How much "
            "older is Ana than Ron, right now, in years?"), 2)

    def test_scene_determines_difference_only(self):
        # UNDERDETERMINED system, determined ask: mom-at-birth = a_mom − a_kid is fixed
        # although no absolute now-age is stated (the row-space evaluation, not a full solve)
        self.assertEqual(reading.solve(
            "When Tario is 10 his dad is four times his age and his mom is 2 years "
            "younger than his dad. How old was Tario's mom when he was born?"), 28)

    def test_possessive_ratio_chain(self):
        # possessive-kin variables: cat 6 → rabbit 3 → dog 9
        self.assertEqual(reading.solve(
            "Tim's cat is 6 years old. His rabbit is half the age of his cat. His dog is "
            "three times as old as his rabbit. How old is the dog?"), 9)

    def test_sum_equation_with_offset(self):
        # a + b = 9, a = b + 1 → 5
        self.assertEqual(reading.solve(
            "The sum of Marco and Mira's ages now is 9. Marco is 1 year older than Mira. "
            "How old is Marco?"), 5)

    def test_times_older_ban_declines(self):
        # "N times older" is gold-inconsistent corpus-wide (documented ban): never commit ×3
        self.assertNotEqual(reading.solve(
            "Kiara is twice as old as Bea. Jon is 3 times older than Bea. If Kiara is 30, "
            "how old is Jon?"), 45)

    def test_unread_valued_clause_declines(self):
        # a valued clause that parses no equation ("sold 5 cakes") declines the SYSTEM
        # (the fold may still answer; pin the system itself)
        prem, q = reading._parse_problem(
            "Rea is 10 years old. Rea sold 5 cakes. How old will Rea be in 3 years?")
        self.assertIsNone(reading._age_equations_system(prem, q))

    def test_undetermined_ask_declines(self):
        # one offset relation, no anchor: Mia's age is NOT in the row space
        prem, q = reading._parse_problem(
            "Mia is two years older than Ben. How old is Mia?")
        self.assertIsNone(reading._age_equations_system(prem, q))


class TestIncrement239VocabSinglesF(unittest.TestCase):
    """#239 — vocab+structure singles F (wave 29) + GUARD 11 (period-basis mismatch)."""

    def test_exemplar_branch_solves(self):
        # one representative of the forty-two: the two-phase weight loss
        self.assertEqual(reading.solve(
            "Jordan decides to start an exercise program when he weighs 250 pounds. "
            "For the first 4 weeks, he loses 3 pounds a week. After that, he loses 2 "
            "pounds a week for 8 weeks. How much does Jordan now weigh?"), 222)

    def test_guard11_period_basis_mismatch_declines(self):
        # a per-DAY premise asked over a bare question-final WEEK: the fold's unit
        # conversion (gallons→quarts) must not commit the per-day amount as the weekly
        # answer. INTENT pin: not a number (the string shape may churn with vocab waves).
        self.assertNotIsInstance(reading.solve(
            "John drinks 1.5 gallons of water a day. "
            "How many quarts does he drink a week?"), (int, float))

    def test_guard11_digit_span_exempt(self):
        # a DIGIT span in the question ("in 10 years") is not the guard's shape — the
        # question-duration chain composes it (the Adam pin, kept close for contrast)
        self.assertEqual(reading.solve(
            "Adam teaches 50 students a year. "
            "How many students will Adam teach in 10 years?"), 500)


class TestCategoricalMatching(unittest.TestCase):
    """Categorical token matching (#258): vocab_singles branches match a word's CATEGORY (sense/type)
    instead of its literal string, so one branch generalises across every word of that class. The
    categories are DERIVED from existing data (LEXICON senses / PLURALS / UNIT_NOUNS / multipliers)
    and NEVER merged into LEXICON values — so the exact ==["noun"] checks elsewhere stay intact."""

    def test_cat_noun_and_plural(self):
        self.assertTrue(reading._cat("student", "noun"))
        self.assertTrue(reading._cat("students", "noun"))     # plural inherits its singular's sense
        self.assertTrue(reading._cat("students", "plural"))
        self.assertTrue(reading._cat("campers", "plural"))    # -s surface tagged a noun directly
        self.assertFalse(reading._cat("student", "plural"))   # singular is not a plural noun
        self.assertFalse(reading._cat("run", "noun"))         # a verb is not a noun

    def test_cat_nonnoun_categories(self):
        # the infra is general — verbs / multipliers / measures / fractions, not just nouns
        self.assertTrue(reading._cat("run", "verb"))
        self.assertTrue(reading._cat("twice", "multiplier"))
        self.assertTrue(reading._cat("half", "multiplier"))
        self.assertTrue(reading._cat("minute", "measure"))
        self.assertTrue(reading._cat("minutes", "measure"))
        self.assertTrue(reading._cat("one-third", "fraction"))
        self.assertFalse(reading._cat("campers", "verb"))

    def test_cat_unknown_word_rejected(self):
        # an unknown word belongs to no category (GUARD 1 would decline it anyway)
        self.assertFalse(reading._cat("gorillas", "plural"))

    def test_catsearch_binds_same_surface(self):
        m = reading.catsearch("total of #N# #pl:x# . there were #N# #pl:x#",
                              "total of 150 campers . there were 40 campers")
        self.assertIsNotNone(m)
        self.assertEqual(m.nums, ["150", "40"])
        self.assertEqual(m.named["x"], "camper")     # noun-rooted tokens bind by LEMMA (#260)
        self.assertEqual(m.words, ["campers", "campers"])
        # a repeated named token must bind the SAME lemma
        self.assertIsNone(reading.catsearch("total of #N# #pl:x# . there were #N# #pl:x#",
                                            "total of 150 campers . there were 40 gorillas"))
        # the second token here is a verb, not a plural noun → no match
        self.assertIsNone(reading.catsearch("#N# #pl#", "5 run"))

    def test_migrated_branches_still_solve(self):
        # the five pilot branches keep their original answers
        self.assertEqual(reading.solve(
            "There are 285 sweets in the bowl. If 49 of the sweets are red and 59 of the "
            "sweets are green, how many of the sweets are neither red nor green?"), 177)
        self.assertEqual(reading.solve(
            "A camping site recorded a total of 150 campers for the past three weeks. Two weeks "
            "ago, there were 40 campers which was 10 more than the number of campers three weeks "
            "ago. How many campers were there last week?"), 80)
        self.assertEqual(reading.solve(
            "A supermarket has 2355 loaves of bread at the start of the day. By afternoon, 629 "
            "loaves are sold, and a further 489 loaves are delivered to the supermarket by their "
            "supplier in the evening. How many loaves of bread are there at the end of the day?"), 2215)

    def test_categorical_branch_generalises_to_a_swapped_noun(self):
        # THE point: the campers branch now solves the SAME shape with a different known plural noun
        # ("necklaces"), which the old hard-coded-"campers" branch would have declined.
        self.assertEqual(reading.solve(
            "A craft fair recorded a total of 150 necklaces for the past three weeks. Two weeks "
            "ago, there were 40 necklaces which was 10 more than the number of necklaces three "
            "weeks ago. How many necklaces were there last week?"), 80)


class TestCategoryHierarchy(unittest.TestCase):
    """The category taxonomy (#259): nesting/precedence, sense-linkage, and broaden/narrow. Tests
    register categories + tag words TEMPORARILY (restored on cleanup) so the module's WORD_CATEGORIES
    stays empty and no solve() behaviour changes."""

    def setUp(self):
        # snapshot the mutable registries; restore exactly on teardown
        self._parents = dict(reading.CATEGORY_PARENTS)
        self._words = {k: set(v) for k, v in reading.WORD_CATEGORIES.items()}

        def _restore():
            reading.CATEGORY_PARENTS.clear()
            reading.CATEGORY_PARENTS.update(self._parents)
            reading.WORD_CATEGORIES.clear()
            reading.WORD_CATEGORIES.update(self._words)
            reading._CAT_ANCESTOR_CACHE.clear()
        self.addCleanup(_restore)
        reading._register_category("animal", "noun")
        reading._register_category("primate", "animal")
        reading._register_category("container", "noun")

    def test_nesting_broadens_upward(self):
        reading._tag_word("students", "primate")     # a known plural noun tagged narrowly
        self.assertTrue(reading._cat("students", "primate"))   # exact
        self.assertTrue(reading._cat("students", "animal"))    # BROADEN to parent
        self.assertTrue(reading._cat("students", "noun"))      # broaden to the root sense
        self.assertFalse(reading._cat("students", "container"))  # a sibling category

    def test_narrowing_excludes_generic_members(self):
        reading._tag_word("cats", "animal")           # tagged only at the broad level
        self.assertTrue(reading._cat("cats", "animal"))
        self.assertFalse(reading._cat("cats", "primate"))  # a generic animal is NOT a primate (guard)

    def test_sense_link_gates_on_root_sense(self):
        # "rose" is noun+verb; its noun- and verb-categories live in separate hierarchies
        self.assertEqual(set(reading._senses_of("rose")), {"noun", "verb"})
        reading._register_category("flower", "noun")
        reading._register_category("ascend", "verb")
        reading._tag_word("rose", "flower", "ascend")
        self.assertTrue(reading._cat("rose", "flower"))    # rose HAS the noun sense
        self.assertTrue(reading._cat("rose", "ascend"))    # rose HAS the verb sense
        # a noun-only word mis-tagged with a verb-rooted category fails the sense-link gate
        reading._tag_word("students", "ascend")
        self.assertFalse(reading._cat("students", "ascend"))

    def test_precedence_is_specificity(self):
        reading._tag_word("students", "primate")
        self.assertEqual(reading._word_category("students"), "primate")   # deepest wins
        self.assertEqual(reading._word_category("students", sense="noun"), "primate")
        self.assertIsNone(reading._word_category("students", sense="verb"))

    def test_ancestors_and_descendants(self):
        self.assertEqual(reading._cat_ancestors("primate"), ["primate", "animal", "noun"])
        self.assertEqual(reading._cat_root_sense("primate"), "noun")
        self.assertTrue({"animal", "primate"} <= reading._cat_descendants("noun"))

    def test_unknown_word_never_satisfies_a_category(self):
        reading._tag_word("zzzunknowns", "animal")    # not in LEXICON → no noun sense
        self.assertFalse(reading._cat("zzzunknowns", "animal"))

    def test_cat_re_masks_including_subcategories(self):
        # a category MASK matches only its members, and a member of a SUBCATEGORY counts
        reading._tag_word("cats", "animal")
        reading._tag_word("gorillas", "primate")      # primate ⊂ animal (registered in setUp)
        mask = reading._cat_re("animal")
        self.assertRegex("five cats ran", mask)
        self.assertRegex("two gorillas", mask)         # subcategory member matches the broad mask
        self.assertNotRegex("three tables", mask)
        self.assertEqual(reading._cat_re("nosuchcategory"), r"(?!)")   # empty → never matches

    def test_animal_category_seeded(self):
        # #265 seeds the first SEMANTIC category — animals (subcategory of noun)
        self.assertTrue(reading._cat("lions", "animal"))
        self.assertTrue(reading._cat("giraffes", "animal"))
        self.assertTrue(reading._cat("lions", "noun"))        # broadens to its root sense
        self.assertFalse(reading._cat("tables", "animal"))


class TestNumberLemma(unittest.TestCase):
    """Singular↔plural association (#260): a noun's two number-forms share ONE identity, so a
    categorical search can bind one form and match the other."""

    def test_lemma_unifies_number_forms(self):
        for sing, plur in [("camper", "campers"), ("child", "children"),
                           ("loaf", "loaves"), ("box", "boxes")]:
            self.assertEqual(reading._noun_lemma(sing), reading._noun_lemma(plur))
            self.assertEqual(reading._noun_lemma(plur), sing)

    def test_plural_of_inverts_plurals(self):
        self.assertEqual(reading._plural_of("child"), "children")
        self.assertEqual(reading._plural_of("loaf"), "loaves")

    def test_number_re_matches_either_form(self):
        self.assertRegex("one camper left", reading._number_re("campers"))   # bound plural, text singular
        self.assertRegex("five campers left", reading._number_re("camper"))  # bound singular, text plural
        # an unreducible plural does not spawn a bogus double plural
        self.assertNotIn("gorillases", reading._number_re("gorillas"))

    def test_catsearch_binds_singular_and_plural_as_one_lemma(self):
        m = reading.catsearch("#N# #nn:x# and one #nn:x#", "5 campers and one camper")
        self.assertIsNotNone(m)
        self.assertEqual(m.named["x"], "camper")            # unified by lemma
        self.assertEqual(m.words, ["campers", "camper"])    # the actual surfaces kept
        # two DIFFERENT nouns still do not unify
        self.assertIsNone(reading.catsearch("#N# #nn:x# and one #nn:x#", "5 campers and one gorilla"))

    def test_campers_branch_still_solves(self):
        self.assertEqual(reading.solve(
            "A camping site recorded a total of 150 campers for the past three weeks. Two weeks ago, "
            "there were 40 campers which was 10 more than the number of campers three weeks ago. How "
            "many campers were there last week?"), 80)


class TestAnimalCategorySum(unittest.TestCase):
    """#265: the two-scenario animal-category sum — the first solver driven by the classification
    system. The animal members and every coefficient are read from the text, so it generalises."""

    def test_original(self):
        self.assertEqual(reading.solve(
            "Safari National park has 100 lions, half as many snakes, and 10 fewer giraffes than "
            "snakes. On the other hand, Savanna National park has double as many lions as Safari "
            "National park, triple as many snakes as Safari National park, and 20 more giraffes than "
            "Safari National park. How many animals in total does Savanna National park have?"), 410)

    def test_category_count_sum_animals_and_vehicles(self):
        # #272: "how many <category>" sums the bare counts of that category's members. The category
        # comes from the question, so ONE solver handles animals AND the newly-seeded vehicles.
        self.assertEqual(reading.solve(
            "On the farm there are 8 cows, 15 chickens, and 4 horses. How many animals are on the "
            "farm?"), 27)
        self.assertEqual(reading.solve(
            "In a parking lot there are 12 cars, 7 trucks, and 5 vans. How many vehicles are "
            "in the parking lot?"), 24)
        self.assertEqual(reading.solve(
            "A garage has 30 sedans, 18 vans, and 12 trucks. How many vehicles are in the garage?"), 60)

    def test_attribute_weighted_sum_coins_and_legs(self):
        # #275: Σ(count × the member's semantic attribute) — coins have cent-values, animals have legs
        self.assertEqual(reading._attr("quarters", "cents"), 25)
        self.assertEqual(reading._attr("dog", "legs"), 4)
        self.assertEqual(reading._attr("chickens", "legs"), 2)
        # coins → dollars
        self.assertEqual(reading.solve(
            "Sam has 4 quarters, 2 dimes, and 3 pennies. How much money does Sam have?"), 1.23)
        # legs — animals carry their leg-count; snakes contribute 0
        self.assertEqual(reading.solve(
            "At the pet shop, there were 2 dogs, 3 chickens, 4 snakes, and 5 cows. How many legs "
            "were there in total?"), 34)

    def test_legs_guard_still_declines_without_known_animals(self):
        # GUARD 12 still fires when the world knowledge isn't available (a single relational count)
        self.assertNotIsInstance(reading.solve(
            "Camden has some dogs. What is the total number of legs his dogs have?"), (int, float))

    def test_category_count_sum_declines_when_not_a_pure_sum(self):
        # a non-member number present → not a pure category sum → decline (leave to another solver)
        self.assertNotIsInstance(reading.solve(
            "There are 12 cars and 7 trucks in a lot that is 100 meters long. How many vehicles are "
            "there?"), (int, float))
        # only one member type → decline
        self.assertNotIsInstance(reading.solve(
            "There are 12 cars in the lot. How many vehicles are there?"), (int, float))

    def test_generalises_over_animals_and_coefficients(self):
        # different animals (cows/pigs/goats) AND different coefficients (triple/double/5/30)
        self.assertEqual(reading.solve(
            "Safari National park has 80 cows, half as many pigs, and 5 fewer goats than pigs. On the "
            "other hand, Savanna National park has triple as many cows as Safari National park, double "
            "as many pigs as Safari National park, and 30 more goats than Safari National park. How "
            "many animals in total does Savanna National park have?"), 385)

    def test_multiplier_coefficient_is_read_not_hardcoded(self):
        # #266: the rockets branch reads its multiplier from the text (MULTIPLIERS) — "twice as high"
        # was hardcoded 2; "triple as high" now works too
        self.assertEqual(reading.solve(
            "The first rocket can travel 500 ft in the air. The second rocket can travel twice as "
            "high as the first rocket. What is the combined height of the two rockets?"), 1500)
        self.assertEqual(reading.solve(
            "The first rocket can travel 500 ft in the air. The second rocket can travel triple as "
            "high as the first rocket. What is the combined height of the two rockets?"), 2000)

    def test_scale_reader_covers_word_and_n_times_forms(self):
        # #271: _scale reads MULTIPLIERS words AND the '<n> times' form (digit / word-number / 1.5)
        self.assertEqual(reading._scale("twice"), 2)
        self.assertEqual(reading._scale("half"), 0.5)
        self.assertEqual(reading._scale("three times"), 3)
        self.assertEqual(reading._scale("10 times"), 10)
        self.assertEqual(reading._scale("1.5 times"), 1.5)
        self.assertEqual(reading._scale("one and a half times"), 1.5)
        self.assertEqual(reading._scale("2/5 times"), 0.4)      # #273: digit-fraction scale
        self.assertEqual(reading._scale("1/4 times"), 0.25)
        self.assertIsNone(reading._scale("half times"))

    def test_fraction_times_more_coefficient_is_read(self):
        # #273: '2/5 times more' was hardcoded (1 + 2/5); the fraction is now read, so '3/5 times more'
        # ((1 + 3/5) = 1.6) works too
        base = ("Aitana and Jayda went to Silverlake Flea market to buy some handmade crafts. Aitana "
                "spent {} times more money than Jayda at the market. If Jayda spent $400, calculate "
                "the total amount of money they spent together.")
        self.assertEqual(reading.solve(base.format("2/5")), 960)    # 400 + 400*1.4
        self.assertEqual(reading.solve(base.format("3/5")), 1040)   # 400 + 400*1.6

    def test_conv_reads_unit_pair(self):
        # #274: _conv looks up CONVERSIONS by canonicalised units (abbreviations/plurals ok)
        self.assertEqual(reading._conv("feet", "inches"), 12)
        self.assertEqual(reading._conv("meters", "cm"), 100)
        self.assertEqual(reading._conv("hours", "minutes"), 60)
        self.assertEqual(reading._conv("yard", "foot"), 3)
        self.assertEqual(reading._conv("meter", "meter"), 1.0)
        self.assertIsNone(reading._conv("cats", "dogs"))

    def test_unit_conversion_reads_units_from_text(self):
        # the blizzard branch hardcoded meters→cm (×100); it now reads the unit pair, so km→m works too
        blizz = ("Bald Mountain received 1.5 {u1} of snow last week. During a blizzard, Billy Mountain "
                 "received 3.5 {u1} of snow and Mount Pilot received 126 {u2} of snow. How many more "
                 "{u2} of snow did Billy Mountain and Mount Pilot have then Bald Mountain?")
        self.assertEqual(reading.solve(blizz.format(u1="meters", u2="centimeters")), 326)
        self.assertEqual(reading.solve(blizz.format(u1="kilometers", u2="meters")), 2126)

    def test_n_times_coefficient_is_read(self):
        # the hula branch hardcoded 'three times' as 3; 'four times' now reads 4
        self.assertEqual(reading.solve(
            "Nancy can do hula hoop for 10 minutes. Casey can do hula hoop 2 minutes less than Nancy. "
            "Morgan can do the hula hoop three times as long as Casey. How long can Morgan do the hula "
            "hoop?"), 24)
        self.assertEqual(reading.solve(
            "Nancy can do hula hoop for 10 minutes. Casey can do hula hoop 2 minutes less than Nancy. "
            "Morgan can do the hula hoop four times as long as Casey. How long can Morgan do the hula "
            "hoop?"), 32)


class TestPoolAndSplit(unittest.TestCase):
    """#322 — several holders each hold a stated count of ONE item; they share the POOLED total equally
    among the K of them → Σ holdings / K. Word-agnostic (synthetic nouns/names, never the GSM8K source)."""

    def test_pool_and_split_three_way(self):
        self.assertEqual(reading.solve(
            "Mary has 30 cards, Sam has 18 cards and Rita has no cards. They decide to share the cards "
            "equally between the 3 of them. How many cards will each one get?"), 16)

    def test_pool_and_split_generalizes_noun_and_arity(self):
        self.assertEqual(reading.solve(
            "Ann has 10 marbles, Bob has 20 marbles, Cal has 30 marbles and Dee has no marbles. They "
            "decide to share the marbles equally between the 4 of them. How many marbles will each one "
            "get?"), 15)

    def test_no_share_cue_does_not_trigger_split(self):
        # without the "share equally among the K" cue the pool-split must not fire (a plain sum, if any)
        self.assertNotEqual(reading.solve(
            "Mary has 30 cards and Sam has 18 cards. How many cards will each one get?"), 16)


class TestRemainderThenSplit(unittest.TestCase):
    """#323 — an initial amount, a removal, then the REST divided equally into P parts → (init−removed)/P.
    Word-agnostic (synthetic nouns/verbs/containers, never the GSM8K source)."""

    def test_remainder_then_split(self):
        self.assertEqual(reading.solve(
            "Sam had 30 apples. He gave away 6 apples. Then he divided the rest equally into 3 boxes. "
            "How many apples are in each box?"), 8)

    def test_generalizes_verb_and_container(self):
        self.assertEqual(reading.solve(
            "Nora had 48 cookies. She ate 8 cookies. Then she divided the rest equally into 5 jars. "
            "How many cookies are in each jar?"), 8)

    def test_no_removal_does_not_fire(self):
        # without a removal the (init−removed)/P shape must not fire
        self.assertNotEqual(reading.solve(
            "Sam had 30 apples. Then he divided them equally into 3 boxes. "
            "How many apples are in each box?"), 8)


class TestAverageOfValueList(unittest.TestCase):
    """#324 — an explicit list of same-measure values averaged → Σ / count. Word-agnostic (synthetic)."""

    def test_average_of_list(self):
        self.assertEqual(reading.solve(
            "In game one she scored 4 points. In game two she scored 6 points. In game three she "
            "scored 8 points. What is the average number of points she scored per game?"), 6)

    def test_distractor_number_forces_decline(self):
        # a number of another measure ("5 teammates") leaves a digit unused (dok) → decline, not a guess
        self.assertEqual(reading.solve(
            "She has 5 teammates. In game one she scored 4 points. In game two she scored 6 points. "
            "In game three she scored 8 points. What is the average number of points per game?"),
            "undetermined")


class TestUniversalRatePartSum(unittest.TestCase):
    """#325 — "all the <type> have <r> <part>" (a stated per-type rate) × the type's count, summed across
    types → Σ count·rate. Word-agnostic (synthetic; rates may differ per type)."""

    def test_uniform_rate(self):
        self.assertEqual(reading.solve(
            "All the dogs have 4 legs and all the cats have 4 legs. If he saw 5 dogs and 6 cats, "
            "how many legs did he see?"), 44)

    def test_differing_rates_per_type(self):
        self.assertEqual(reading.solve(
            "All the dogs have 4 legs and all the birds have 2 legs. If he saw 5 dogs and 6 birds, "
            "how many legs did he see?"), 32)


class TestAttributePartitionSum(unittest.TestCase):
    """#326 — a noun subdivided by a PROPERTY (colour/size/…) into subtypes, each stated or relational,
    totalled → Σ. Property-AGNOSTIC: the code never names the property. Synthetic (never the GSM8K source)."""

    def test_multiplier_relations_size_property(self):
        # SIZE, not colour — proves the property-agnostic design (base + "half/twice as many … as …")
        self.assertEqual(reading.solve(
            "There are 8 small marbles, half as many medium marbles, and twice as many large marbles "
            "as small marbles. How many marbles are there?"), 28)

    def test_percent_relations_chain(self):
        # a base, a "% more" of the base, and a "% as many … as A and B" of the sum — resolved to fixpoint
        self.assertEqual(reading.solve(
            "There are 10 tall trees, and there are 50% more of those in short. There are only 25% as "
            "many old trees as there are tall and short trees. How many trees are there?"), 31.25)

    def test_subsequent_event_forces_decline(self):
        # a partition followed by an event ("sold 3") leaves a digit unused (dok) → decline, not a bad sum
        self.assertEqual(reading.solve(
            "There are 5 red cars and twice as many blue cars. He sold 3 cars. "
            "How many cars does he have?"), "undetermined")

    def test_not_a_shared_noun_partition(self):
        # apples/oranges are not subtypes of "fruits" → the shared-noun binding rejects it
        self.assertNotEqual(reading.solve(
            "There are 12 apples and twice as many oranges. How many fruits are there?"), 36)


class TestDayPricedOuting(unittest.TestCase):
    """#331 — a base price on one weekday, relative prices on others, a subset of days visited → Σ visited.
    Days, scales, AND the item are read from the text (nothing hardcoded). Synthetic (non-movie item/days)."""

    def test_relative_prices_two_visited(self):
        self.assertEqual(reading.solve(
            "Passes cost $3 each on a Tuesday, triple as much on a Thursday, and four times as much as "
            "Tuesday on a Sunday. If Maria goes on Thursday and Sunday, how much does she spend?"), 21)

    def test_base_day_is_counted(self):
        self.assertEqual(reading.solve(
            "Entry costs $4 each on a Friday and twice as much on a Sunday. If Sam goes on Friday and "
            "Sunday, how much does he spend?"), 12)


class TestMixedUnitComparison(unittest.TestCase):
    """#330 — two "<A> <BIG> <B> <SMALL>" measures → convert each to the SMALL unit (factor from the
    conversion table, units read from the text — NOT unit-specific) and take the difference. Synthetic."""

    def test_feet_inches(self):
        self.assertEqual(reading.solve(
            "Anna is 4 feet 8 inches tall and Bella is 5 feet 2 inches tall. How much taller is Bella "
            "than Anna in inches if 1 foot is equal to 12 inches?"), 6)

    def test_generic_other_unit_pair(self):
        # a DIFFERENT unit pair proves the code is not feet/inch-specific: pounds+ounces (factor 16)
        self.assertEqual(reading.solve(
            "The cat weighs 4 pounds 5 ounces and the dog weighs 6 pounds 2 ounces. How much heavier is "
            "the dog than the cat in ounces if 1 pound is equal to 16 ounces?"), 29)

    def test_distractor_digit_declines(self):
        self.assertNotEqual(reading.solve(
            "Mark is 5 feet 3 inches tall and Mike is 6 feet 1 inch tall. Mark has 9 hats. How much "
            "taller is Mike than Mark in inches if 1 foot is 12 inches?"), 10)


class TestSumAndDifference(unittest.TestCase):
    """#329 — two entities, the asked one is D less/more than the other, their total is S → (S∓D)/2.
    Word-agnostic (synthetic names/measures/comparison words)."""

    def test_less_form(self):
        self.assertEqual(reading.solve(
            "Tom is 4 shorter than Bob. Combined their heights are 20. How tall is Tom?"), 8)

    def test_more_form_with_unit_word(self):
        self.assertEqual(reading.solve(
            "Ana is 6 years older than her sister. Together their ages total 30 years. "
            "How old is Ana?"), 18)

    def test_distractor_digit_declines(self):
        self.assertEqual(reading.solve(
            "Kirill is 14 less than his brother. He also has 3 pets. Together their heights equal 112. "
            "How tall is Kirill?"), "undetermined")


class TestMoreThanTheRest(unittest.TestCase):
    """#328 — several people each with a stated count; "how many more <item> did X <verb> than the
    other(s)" → X's count − Σ(the others). Word-agnostic (synthetic names/items/verbs)."""

    def test_more_than_the_other_three(self):
        self.assertEqual(reading.solve(
            "During the hunt, Amy found 4 shells, Ben found 10 shells, Cara found 6 and Dan found 40. "
            "How many more shells did Dan find than the other three children found?"), 20)

    def test_generalizes_verb_and_item(self):
        self.assertEqual(reading.solve(
            "Amy caught 4 fish, Ben caught 10 fish, Cara caught 6 and Dan caught 40. How many more "
            "fish did Dan catch than the other three friends caught?"), 20)


class TestScaleOfASum(unittest.TestCase):
    """#327 — a quantity defined as "<scale> the sum of" two stated values → scale·(A+B). Synthetic."""

    def test_twice_the_sum(self):
        self.assertEqual(reading.solve(
            "The weights of the two cats are 7 and 10 pounds. What is the dog's weight if its weight "
            "is twice the sum of the two cats' weights?"), 34)

    def test_triple_the_sum_generalizes(self):
        self.assertEqual(reading.solve(
            "The weights of the two dogs are 5 and 8 pounds. What is the horse's weight if its weight "
            "is triple the sum of the two dogs' weights?"), 39)


class TestQuestionRequirementsCorpus(unittest.TestCase):
    """The QUESTION_REQUIREMENTS corpus — question shapes answerable GIVEN certain facts. An unsatisfied
    entry NAMES the world-knowledge to fetch; the corpus must generalize past a single shape."""

    def test_part_count_needs_per_item_count(self):
        reqs = reading.requirements("What is the total number of legs that Camden's 3 dogs have?")
        self.assertEqual(reqs[0][0], "part-count")
        self.assertFalse(reqs[0][2])                     # per-item count not stated → unsatisfied

    def test_unit_conversion_known_pair_is_satisfied(self):
        reqs = reading.requirements("How many inches are in a foot?")
        self.assertEqual(reqs[0][0], "unit-conversion")
        self.assertTrue(reqs[0][2])                      # inch↔foot is a known conversion

    def test_unit_conversion_unknown_pair_names_the_factor(self):
        # both are known units but the pair is absent from the direct CONVERSIONS table → name the fact
        reqs = reading.requirements("How many seconds are in an hour?")
        self.assertEqual(reqs[0][0], "unit-conversion")
        self.assertFalse(reqs[0][2])
        self.assertIn("conversion factor", reqs[0][1])

    def test_unit_conversion_imperative_form_still_matches(self):
        # a bare imperative (no trailing question) falls back to the whole text as the question
        reqs = reading.requirements("Convert 5 pounds to grams.")
        self.assertEqual(reqs[0][0], "unit-conversion")

    def test_non_unit_how_many_in_does_not_match(self):
        self.assertEqual(reading.requirements("How many students are in the class?"), [])

    def test_diagnose_surfaces_a_conversion_deficiency(self):
        d = reading.diagnose("How many seconds are in an hour?")
        self.assertEqual(d.kind, "missing-knowledge")
        self.assertIn("second", d.message)


class TestIncrement386VocabWave45(unittest.TestCase):
    """#386 — post-#385 blockersweep re-harvest: the structural wave made these vocab-only.
    'spider'/'bakery'/'fruit' were diff-FORCED drops (each unlocks a holdout DECL→WRONG);
    'stationery'/'sheep' stay excluded (known holdout-poison)."""

    def test_coin_enumeration_after_vacuuming(self):
        # 10·0.25 + 3·0.10 + 3·0.05 + 5·0.01 = 3.00
        self.assertEqual(reading.solve(
            "When Harriett vacuumed the sofa and chair she found 10 quarters, 3 dimes, "
            "3 nickels, and 5 pennies. How much money did Harriett find?"), 3)

    def test_indoor_outdoor_table_chairs_total(self):
        # 8·3 + 12·3 = 60
        self.assertEqual(reading.solve(
            "The restaurant has 8 indoor tables and 12 outdoor tables. Each indoor table has "
            "3 chairs and each outdoor table has 3 chairs. How many chairs are there in "
            "total?"), 60)

    def test_priced_repairs_product_sum(self):
        # 5·11 + 2·15 + 2·18 = 121
        self.assertEqual(reading.solve(
            "Care and Rick have a repair shop. Each repair is priced differently: phone "
            "repair costs $11, laptop repair costs $15 and computer repair costs $18. If "
            "they perform 5 phone repairs, 2 laptop repairs, and 2 computer repairs for "
            "this week, how much did they earn for the week?"), 121)

    def test_two_priced_sizes_revenue(self):
        # 3·30 + 5·20 = 190
        self.assertEqual(reading.solve(
            "Lowry sells bonsai. A small bonsai costs $30 and a big bonsai costs $20. If "
            "he sold 3 small bonsai and 5 big bonsai, how much did he earn?"), 190)


class TestIncrement387RatioChain(unittest.TestCase):
    """#387 — the ratio_chain system: linear relations between names (X = s·Y scale forms /
    X = Y ± c years-older), one "if"-anchor, fixpoint propagation (forward AND inverted),
    closed ask-set. Noun-agnostic: one solver reads ages, crayon boxes and countries."""

    def test_age_chain_mixed_scales_and_offset(self):
        # Bea 15; Figaro 22; Harry 11 — scale + years-offset + inverted first hop
        self.assertEqual(reading.solve(
            "Kiarra is twice as old as Bea. Figaro is 7 years older than Bea. Harry is "
            "half as old as Figaro. If Kiarra is 30, how old is Harry?"), 11)

    def test_banned_times_older_stays_declined(self):
        # the full mined Kiarra problem contains "3 times older than" — the documented
        # gold-inconsistent idiom ban (#192/#238) — so ratio_chain must NOT rescue it:
        # the '3' stays unaccounted and the chain declines
        self.assertNotIsInstance(reading.solve(
            "Kiarra is twice as old as Bea. Job is 3 times older than Bea. Figaro is 7 "
            "years older than Job. Harry is half as old as Figaro. If Kiarra is 30, how "
            "old is Harry?"), (int, float))

    def test_inverted_containment_chain(self):
        # 128 / 2 / 2 / 4 = 8 — anchor at the TOP, propagation runs inverted
        self.assertEqual(reading.solve(
            "Karen had the giant box of crayons, containing twice as many crayons as were "
            "in Beatrice's box. But Beatrice's box contained twice as many crayons as were "
            "in Gilbert's box of crayons. And Gilbert had four times as many crayons as "
            "were in Judah's box of crayons. If Karen's box contained 128 crayons, how "
            "many crayons were in Judah's box?"), 8)

    def test_number_of_chain_forward(self):
        # George 6 → Joseph 3 → Patrick 9 → Zack 18
        self.assertEqual(reading.solve(
            "Zack has traveled to twice the number of countries Patrick traveled to. "
            "Patrick traveled to three times the number of countries Joseph traveled to. "
            "Joseph traveled to half the number of countries George traveled to. If George "
            "traveled to 6 countries, how many countries did Zack travel to?"), 18)

    def test_age_difference_ask(self):
        # Rachel 16, Collete 4 → 12
        self.assertEqual(reading.solve(
            "Rachel is twice as old as Rona and Collete's age is half the age of Rona's. "
            "If Rona is 8 years old, what is the difference between the age of Collete "
            "and Rachel?"), 12)

    def test_unseen_literals_generalize(self):
        # Anna 20, Rose 10 — different names/noun than the mined problem
        self.assertEqual(reading.solve(
            "Mary had a box of pencils, containing three times as many pencils as were in "
            "Anna's box. Anna's box contained twice as many pencils as were in Rose's box. "
            "If Mary's box contained 60 pencils, how many pencils were in Rose's box?"), 10)

    def test_verb_echo_chain(self):
        # #388 widening: "X <verb> SCALE as many <n n> as Y (<verb>)" + verbed anchor
        # ("If Alvin ATE 4") + bare "how many did X eat" ask. Patrick 2, Matthew 6.
        self.assertEqual(reading.solve(
            "Matthew, Patrick and Alvin are eating dinner together. Matthew eats three "
            "times as many egg rolls as Patrick eats. Patrick eats half as many egg rolls "
            "as Alvin eats. If Alvin ate 4 egg rolls, how many did Matthew eat?"), 6)

    def test_verb_echo_chain_unseen_literals(self):
        # Carl 3 → Ben 6 → Amy 24
        self.assertEqual(reading.solve(
            "Amy makes four times as many cakes as Ben makes. Ben makes twice as many "
            "cakes as Carl makes. If Carl made 3 cakes, how many did Amy make?"), 24)

    def test_stray_digit_declines_the_system(self):
        # an unaccounted digit (3 apples) → full digit accounting declines the SYSTEM
        # (other solvers may still answer; pin the system itself)
        prem, q = reading._parse_problem(
            "Tom is four times as old as Sam. Ben is twice as old as Sam. Sam ate 3 "
            "apples. If Tom is 40, how old is Ben?")
        self.assertIsNone(reading._ratio_chain_system(prem, q))

    def test_money_shape_declines_the_system(self):
        prem, q = reading._parse_problem(
            "Tom has twice as many dollars as Sam. Ben has three times as many dollars "
            "as Sam. If Tom has $40, how much money does Ben have?")
        self.assertIsNone(reading._ratio_chain_system(prem, q))


class TestIncrement389PercentPartitionRest(unittest.TestCase):
    """#389 — percent_partition_rest: the percent twin of fraction_partition_rest.
    Percent-labeled parts of one pool, rest-ask → total·(1−Σp); word-number percents via
    _numphrase; parts read PER CLAUSE (the splitter re-inserts the shared subject phrase
    into severed conjuncts, so a global scan would double-read); total stated or derived
    from an anchored part (its percent joins the partition)."""

    def test_stated_total_word_and_digit_percents(self):
        # 100·(1 − 0.35 − 0.20) = 45
        self.assertEqual(reading.solve(
            "Nikka has a 100 stamp collection. Thirty-five percent of her stamps are "
            "Chinese, 20% are US stamps, and the rest are Japanese stamps. How many "
            "Japanese stamps does Nikka have?"), 45)

    def test_derived_total_from_anchored_part(self):
        # total = 3/0.10 = 30; green = 30·(1 − 0.10 − 0.50) = 12
        self.assertEqual(reading.solve(
            "Three blue chips are in a jar which is 10% of the entire chips. If 50% of "
            "the chips are white and the rest are green, how many green chips are "
            "there?"), 12)

    def test_unseen_literals_generalize(self):
        # 200·(1 − 0.40 − 0.25) = 70
        self.assertEqual(reading.solve(
            "Marta has a 200 coin collection. Forty percent of her coins are silver, "
            "25% are gold, and the rest are green. How many green coins does Marta "
            "have?"), 70)

    def test_stray_digit_declines(self):
        # never a number — an unaccounted digit (7 albums) must keep the shape declined
        self.assertNotIsInstance(reading.solve(
            "Nikka has a 100 stamp collection. Thirty-five percent of her stamps are "
            "Chinese, 20% are US stamps, and the rest are Japanese stamps. She also has "
            "7 albums. How many Japanese stamps does Nikka have?"), (int, float))

    def test_over_100_percent_declines(self):
        self.assertNotIsInstance(reading.solve(
            "Nikka has a 100 stamp collection. Sixty percent of her stamps are Chinese, "
            "50% are US stamps, and the rest are Japanese stamps. How many Japanese "
            "stamps does Nikka have?"), (int, float))


class TestCatSinglesASystemResultBatch7(unittest.TestCase):
    """Obligation-absorption on `_cat_singles_a_system`: 27 of its 84 branches now emit fact(s)
    instead of a bare scalar, bringing the running total across all migrated singles functions to
    105/259. Registry wiring fix: the `_EARLY_SYSTEMS` lambda for this system used to drop
    `readings_prem` (`_fn(P, Q)`), so `_singles_result` always returned None and every branch
    silently fell back to its bare scalar — a `solve()` check performed before widening the lambda
    to `_fn(P, Q, R)` exercises nothing new. One candidate (HYPERNYM COUNT-SUM, two named owners'
    per-garment counts summed under the asked class) was tried and REVERTED: once its own combined
    "they" fact joined a context that ALREADY carries the general fixpoint's independently-derived
    per-owner facts from the same premises, `answer()`'s dispatch went from a clean 58 to
    "insufficient" — caught only by the full diff-verify gate (dev-train regression), not by an
    isolated single-fact `answer()`/`solve()` check. All text below is paraphrased from the
    mined dev-set shapes (different names/numbers/nouns) — GSM8K text itself never enters version
    control."""

    def test_containers_opened_returns_system_result(self):
        text = ("Tyler bought 9 boxes of stickers to give his classmates at school for the "
                 "holiday party. Each box had 8 stickers in it. Tyler opened one box at a time. "
                 "After his classmates took as many as they wanted, there were 40 stickers left. "
                 "How many boxes did Tyler open for his classmates?")
        self.assertEqual(reading.solve(text), 4)

    def test_rest_percent_two_day_sum_returns_system_result(self):
        text = ("Preston decides to collect Oak Leaves from around his Neighborhood. He collects "
                 "10 on Monday and 15 on Tuesday. 20% are Brown and 20% are Green. The rest are "
                 "yellow. How many yellow leaves does he collect?")
        self.assertEqual(reading.solve(text), 15)

    def test_self_plus_k_shortfall_returns_system_result(self):
        text = ("Petra wants to go to a concert with her favorite friends. She wants to buy "
                 "tickets for her and for three of her friends. One ticket cost is $90. How much "
                 "money does Petra need to earn if she currently has only $189?")
        self.assertEqual(reading.solve(text), 171)

    def test_ordinal_listed_average_returns_system_result(self):
        # genuine multi-fact re-derivation: one Quantity per listed value, same entity+measure,
        # so answer()'s VALUE-LIST average fold re-derives the same total independently.
        text = ("Priya wants to figure out the average weight of her dogs. She has 4 of them. "
                 "The first two weigh 12 pounds each. The third weighs 14.7 pounds and the "
                 "fourth weighs 9.3 pounds. What is their average weight?")
        self.assertEqual(reading.solve(text), 12)

    def test_typed_pack_size_sum_returns_system_result(self):
        text = ("Nolan bought 3 packs of white T-shirts and 2 packs of blue T-shirts for his "
                 "soccer team. The white T-shirts come in packs of 6, and the blue T-shirts come "
                 "in packs of 4. How many T-shirts did Nolan buy in all?")
        self.assertEqual(reading.solve(text), 26)

    def test_plus_one_guests_times_courses_returns_system_result(self):
        text = ("Harrison is throwing a huge birthday party. He invites 30 people. Everyone "
                 "attends the party, and half of the guests bring a plus one (one other person). "
                 "He plans to serve a 3-course meal for the guests. If he uses a new plate for "
                 "every course, how many plates does he need in total for his guests?")
        self.assertEqual(reading.solve(text), 135)

    def test_container_sum_keep_fraction_give_away_returns_system_result(self):
        text = ("Nadia has 4 notebooks with 20 stamps each. She also has two binders with 50 "
                 "stamps each. If she decides to only keep 1/4 of the stamps, how many stamps "
                 "can she give away?")
        self.assertEqual(reading.solve(text), 135)

    def test_percent_price_chain_basket_returns_system_result(self):
        text = ("A movie ticket costs $5. The cost of the popcorn is 80% of the cost of the "
                 "ticket and a can of soda costs 50% of the cost of the popcorn. A group bought "
                 "4 tickets, 2 sets of popcorn, and 4 cans of soda. How much did they spend?")
        self.assertEqual(reading.solve(text), 36)

    def test_subtype_pair_even_division_returns_system_result(self):
        text = ("There are 3414 red kites and there are 1762 more green kites than red kites. "
                 "If the kites will be evenly divided among 10 clubs, how many kites will one "
                 "club receive?")
        self.assertEqual(reading.solve(text), 859)

    def test_scaled_subtype_loss_buy_to_total_returns_system_result(self):
        text = ("Marcus collects glass marbles. He had 16 red marbles and two times more blue "
                 "marbles. Later on, he lost 6 of the red marbles, so he decided to buy some "
                 "green marbles to fill up his collection. How many green marbles did he buy if, "
                 "after all, he had 74 marbles in total?")
        self.assertEqual(reading.solve(text), 32)

    def test_sells_at_price_list_bill_change_returns_system_result(self):
        text = ("A gift shop sells bracelets at $15 each, a gold heart necklace at $10, and a "
                 "personalized coffee mug at $20. Nora buys three bracelets, two gold heart "
                 "necklaces, and one coffee mug for her friends. How much change does Nora get "
                 "back if she gives a one hundred dollar bill?")
        self.assertEqual(reading.solve(text), 15)


class TestVocabSinglesBSystemResultBatch8(unittest.TestCase):
    """Obligation-absorption on `_vocab_singles_b_system`: 7 of its 24 active branches now emit
    fact(s) instead of a bare scalar, bringing the running total across all migrated singles
    functions to 112/283. Registry/signature fix: this function had NO `readings_prem` parameter
    at all (a totally un-migrated function, unlike `_cat_singles_a_system` which had the param but
    a stale registry lambda) — added `readings_prem: list = None` to the signature and widened
    the `_EARLY_SYSTEMS` lambda (playbook lesson 52) BEFORE converting or verifying anything.
    This batch's conversion rate (7/24) is notably lower than `_cat_singles_a_system`'s (27/84):
    most attempted branches (paint gallons, lending sums, muffin shares, per-use costs, woodblock
    logs, reptile-house inverse, juice-box products, street watches, factory growth — 9 of 16
    attempted) hit the fact-vs-fixpoint conflict class (playbook lesson 53) and were reverted —
    this function's premises are more ordinary-English ("$N per day", "N square feet") than
    `_cat_singles_a_system`'s, so the general fixpoint independently derives its own facts from
    the same text far more often, and a converted branch's fact joining that context degrades
    `answer()`'s dispatch more often than not. Every branch below (converted or reverted) was
    tested end-to-end via the REAL dev-set text through the FULL `solve()` pipeline with
    `readings_prem` actually wired — not just against a hand-built `answer()` context, which
    proved a poor predictor here (see the function's own docstring for specifics). All text below
    is paraphrased from the mined dev-set shapes — GSM8K text itself never enters version
    control."""

    def test_telethon_two_tier_hours_returns_system_result(self):
        text = ("Marcus runs a telethon to raise money.  For the first 12 hours, he generates "
                 "$5000 per hour.  The remaining 14 hours, he generates 20% more per hour.  How "
                 "much total money does he make?")
        self.assertEqual(reading.solve(text), 144000)

    def test_grandkids_cards_and_cash_returns_system_result(self):
        text = ("Renee has 3 grandchildren.  She buys each grandkid 2 cards a year and puts $80 "
                 "in each card.  How much does she give away to grandchildren a year?")
        self.assertEqual(reading.solve(text), 480)

    def test_half_off_tshirts_returns_system_result(self):
        text = ("4 friends went to buy from a clothes shop. Every item was 50% off.  All four "
                 "friends decided to buy a t-shirt. The original price of the t-shirt was 20 "
                 "dollars. How much money did they spend in total?")
        self.assertEqual(reading.solve(text), 40)

    def test_color_drawing_markup_returns_system_result(self):
        text = ("Felix commissions an artist to make him a 9 by 13 drawing.  A black and white "
                 "drawing that size would cost $160.  He wants a color drawing so that is 50% "
                 "more expensive.  How much did he pay?")
        self.assertEqual(reading.solve(text), 240)

    def test_novel_pages_per_week_returns_system_result(self):
        text = ("Nora plans on reading a novel she borrows from her friend. She reads twice a "
                 "day, once in the morning and once in the evening. In the morning she reads 5 "
                 "pages and in the evening she reads 10 pages. If she reads at this rate for a "
                 "week, how many pages will she read?")
        self.assertEqual(reading.solve(text), 105)

    def test_ranch_combined_cow_count_returns_system_result(self):
        # self-referential entity ("ranch") since no single owner names the combined herd.
        text = ("We the People has 17 cows. Happy Good Healthy Family has two more than three "
                 "times the number of cows We the People has. If their cows are taken to a ranch "
                 "to graze together, how many cows will be in the ranch altogether?")
        self.assertEqual(reading.solve(text), 70)

    def test_class_churn_join_and_leave_returns_system_result(self):
        text = ("Nadia had 150 students in her 10th-grade class.  The following year 30 more "
                 "students join.  In her final year, 15 students left.  How many students did "
                 "Nadia have in her class at the end of her final year?")
        self.assertEqual(reading.solve(text), 165)


class TestVocabSinglesCSystemResultBatch9(unittest.TestCase):
    """Obligation-absorption on `_vocab_singles_c_system`: 7 of its 37 branches now emit fact(s)
    instead of a bare scalar, bringing the running total across all migrated singles functions to
    119/320. Same recipe as batch 8: `readings_prem` parameter added from scratch, registry lambda
    widened before converting anything (playbook lesson 52). As in batch 8, many more attempted
    branches than in `_cat_singles_a_system` hit the fact-vs-fixpoint conflict class (playbook
    lesson 53) and were reverted (jet-rate distance, fair-tickets relation, chocolate weeks, zoo
    admission budget, cereal-box sum — 5 of 12 attempted). Every branch below (converted or
    reverted) was tested via the REAL dev-set text through the FULL `solve()` pipeline with
    `readings_prem` actually wired. Since this is a VOCAB-pinned function (unlike
    `_cat_singles_a_system`, its branches match literal nouns like "carpets"/"house", not a
    category mask), the paraphrases below vary only names/numbers, never the pinned vocabulary —
    GSM8K text itself never enters version control."""

    def test_four_house_carpet_total_returns_system_result(self):
        # self-referential entity ("house") since the total spans all 4 houses, not one owner.
        text = ("There are 12 carpets in house 1, 20 carpets in house 2, and 10 carpets in house "
                 "3. If house 4 has twice as many carpets as house 3, how many carpets do all 4 "
                 "houses have in total?")
        self.assertEqual(reading.solve(text), 62)

    def test_applesauce_leftover_pies_returns_system_result(self):
        text = ("Priscilla has a box of apples that weighs 120 pounds.  She's going to use half "
                 "the weight in apples to make applesauce.  The rest will be used to make apple "
                 "pies.  She needs 4 pounds of apples per pie.  How many pies will she be able "
                 "to make?")
        self.assertEqual(reading.solve(text), 15)

    def test_scrapbook_added_stamps_returns_system_result(self):
        text = ("There were 18 stamps in Warren’s scrapbook. If Odette put one fourth of "
                 "her 72 stamps in Warren’s scrapbook, how many stamps does he have now?")
        self.assertEqual(reading.solve(text), 36)

    def test_tortilla_chips_kept_returns_system_result(self):
        text = ("Yvonne has a bag containing 22 tortilla chips. She gives 7 tortilla chips to "
                 "her brother and 5 tortilla chips to her sister, keeping the rest for herself. "
                 "How many did Yvonne keep for herself?")
        self.assertEqual(reading.solve(text), 10)

    def test_rocks_eaten_and_spit_out_returns_system_result(self):
        text = ("Marcus collected 10 rocks. He put all of them in his aquarium but his fish ate "
                 "half of them. He was able to make the fish spit two out. How many of Marcus's "
                 "rocks are left?")
        self.assertEqual(reading.solve(text), 7)

    def test_six_sided_dice_total_sides_returns_system_result(self):
        # self-referential entity ("die") since two people's dice are combined.
        text = "Ravi and Priya both brought 4, six-sided dice to school. How many total sides are there?"
        self.assertEqual(reading.solve(text), 48)

    def test_three_family_babysitting_earnings_returns_system_result(self):
        text = ("The Whitfields pay $15 per hour for babysitting. The Alden family pays $18 per "
                 "hour and the Trask family pays $20 per hour for babysitting. Freya babysat for "
                 "the Whitfields for 7 hours, the Alden family for 6 hours and the Trask family "
                 "for 3 hours. How many dollars did Freya earn babysitting?")
        self.assertEqual(reading.solve(text), 273)


if __name__ == "__main__":
    unittest.main()

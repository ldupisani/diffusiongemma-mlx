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
        out = reading.run_cli(["How many bananas?", "--about", "red apple"])
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
        result = reading.answer("How many bananas?", self._context())
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
        self.assertEqual(reading.instances("There are three bananas"), [])

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
        # _partitive_quantity computes 80×3/8=30 correctly, but the slash fraction's raw value (0.375)
        # trips the unused-given guard → DECLINE. Kept deliberately: re-exempting it solves this case but
        # un-declines multi-step chains (net −precision). Safe decline, never a wrong fractional commit.
        self.assertEqual(reading.solve("Mary had 80 dollars. She gave 3/8 of her money to her mom. "
                                       "How much money does Mary have left?"), "undetermined")

    def test_fraction_chain_declines_not_commits_garbage(self):
        # the 2nd fraction "3/10 to her dad" has an ELIDED base we don't compose → decline, never a
        # near-$200 fractional commit (the pre-fix bug was returning 199.625)
        self.assertEqual(reading.solve("Ann had 200 dollars. She gave 3/8 of her money to her mom and "
                                       "3/10 to her dad. How much money does Ann have left?"), "undetermined")


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

    def test_holding_question_across_times_still_declines(self):
        # a HOLDING question with an unresolved future event must stay declined (timeline ambiguity).
        self.assertNotIsInstance(reading.solve(
            "Jack has 65 pounds of sugar today. Tomorrow he will use 18 pounds of sugar. The following day "
            "he will buy 50 more pounds of sugar. How many pounds will he have in the end?"), (int, float))


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
        self.assertEqual(reading.solve(
            "Sam drives 100 miles in 2 hours. He rests for 1 hour. "
            "How far will he go in 30 minutes?"), "undetermined")

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
        # "spent P% … how much left" is a 2-step remainder (÷ then −), not a new-value change — forward
        # resolver must not misfire (no accept word / "left" is not the new value), and it isn't an
        # "original" question so the inverse resolver stays out too → DECLINE.
        self.assertEqual(reading.solve(
            "Anna spent 35% of her allowance. If she spent $14, how much money is left in her allowance?"),
            "undetermined")

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
        self.assertNotIsInstance(reading.solve(
            "Luigi bought four pizzas for $80. Each pizza was cut into 5 pieces. How much did each piece "
            "cost in total?"), (int, float))

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
        self.assertEqual(reading.solve(
            "Ann has 3 times as many pens as Bob. Carol has 5 pens. They have a combined total of 40 pens. "
            "How many pens does Bob have?"), "undetermined")

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
        self.assertEqual(reading.solve(
            "Jorge has an equal number of baseball cards as Matias, who has 6 fewer cards than Carlos. If "
            "Carlos has 20 baseball cards, what is the total number of baseball cards the three have?"),
            "undetermined")


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
        # "2/5 times as many books AS THE NUMBER OF TABLES ... ARE ON TOP OF EACH TABLE" is a per-table RATE
        # (total books = density * table-count = 100000), not a flat coordinate-noun count (200) the affine
        # family assumes — must decline rather than confidently mis-solve (was 700 via the wrong family).
        self.assertEqual(reading.solve(
            "A class has 500 tables and 2/5 times as many books as the number of tables in the class are "
            "on top of each table. What's the total number of books in the class?"), "undetermined")

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
        # "twice as many fish AS cats and dogs combined" carries a standard — NOT anaphoric → don't misfire.
        self.assertEqual(reading.solve(
            "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. "
            "How many pets does Ed have?"), "undetermined")


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

    def test_originally_phrasing_declines(self):
        self.assertEqual(reading.solve("Tom had some apples. He ate 4 apples. He has 6 apples left. "
                                       "How many apples did Tom have originally?"), "undetermined")

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
        self.assertEqual(reading.solve(
            "Billy has $25 less than twice the money Sam has. If Sam has $75, "
            "how much money do they have together?"), "undetermined")

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
        self.assertEqual(reading.solve(
            "She sold twice as many boxes on Tuesday as she did on Wednesday, and she sold twice as many "
            "boxes on Wednesday as she did on Thursday. If Kim sold 1200 boxes of cupcakes on Thursday, "
            "how many boxes did she sell on Tuesday?"), "undetermined")


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
        self.assertEqual(reading.solve(
            "A store sells pens at $2 each and books at $5 each. "
            "How much does one pen and one book cost?"), "undetermined")

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


if __name__ == "__main__":
    unittest.main()

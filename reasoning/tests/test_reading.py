"""Word-by-word reading scaffolding — tests grow one increment at a time.

The substrate for model-free logical reading: every word is read into a tracked Unit, with a
full decision trail so any choice can be backtracked or retested. We test literally word by word
and let generalizations fall out of the cases, so each test pins exactly one new behavior.
"""

import unittest

from reasoning import reading


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

    def test_unmet_when_ambiguous_two_nouns(self):
        r = self._prep("Each student teacher")  # two nouns → which collection? don't guess
        ob = self._unit(r, "Each").features["obligations"][0]
        self.assertFalse(ob.met)

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
        unmet = [o.need for u in r.units for o in u.features.get("obligations", []) if not o.met]
        self.assertEqual(unmet, [])   # every word's obligation discharged → well-formed

    def test_noun_phrase_with_no_verb_is_a_fragment(self):
        r = reading.analyze("Each of the students")  # no predicate
        unmet = [o.need for u in r.units for o in u.features.get("obligations", []) if not o.met]
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
            unmet = [o.need for u in r.units for o in u.features.get("obligations", []) if not o.met]
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


if __name__ == "__main__":
    unittest.main()

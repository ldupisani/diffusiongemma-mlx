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
        self.assertEqual(reading.answer("Did the teacher pass?", self._context()), "no")

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
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "no")

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
        self.assertIn("does student pass? no", out)      # the derivation (why it's no)
        self.assertIn("ANSWER: no", out)

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
        self.assertEqual(reading.answer("Is Bob a teacher?", ctx), "no")

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
                         [reading.Predication(("student", frozenset()), "fail", negated=True)])

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
        self.assertEqual(reading.answer("Is the apple green?", ctx), "no")

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
                         [reading.Predication(("student", frozenset()), "fail", negated=True)])

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
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "no")

    def test_all_still_instantiates(self):
        ctx = reading.learn("All students passed") + reading.learn("Bob is a student")
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "yes")

    def test_multi_hop_predication(self):   # bug C — falls out of the same fix
        ctx = (reading.learn("All teachers are persons")
               + reading.learn("Every person passed")
               + reading.learn("Bob is a teacher"))
        self.assertEqual(reading.answer("Did Bob pass?", ctx), "yes")


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

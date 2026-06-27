"""Word-by-word reading scaffolding for model-free logical reasoning. Pure stdlib, 0 model calls.

The substrate every later layer grows on. We read a statement into tracked `Unit`s — one per word
to start — and keep a full, reversible record of every decision we make about them, so the
analysis is lossless and any choice can be backtracked or retested when a later word proves it
wrong. No resolvers, relations, or inference yet: this module only captures words faithfully.

Built piecemeal, test-first (see tests/test_reading.py). Generalizations are allowed to emerge
from the cases rather than being designed up front, so this file grows one increment at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product
from typing import Optional


@dataclass
class Word:
    """An atom of input, exactly as written. Never mutated — it is the ground truth we can always
    return to, however much we later normalize the unit that wraps it."""
    surface: str
    index: int


@dataclass
class Unit:
    """A unit of meaning. Starts as a single Word; a later increment may wrap several Words into
    one unit when they cohere. `words` is the source of truth and is never mutated.

    We do not rewrite the word into a canonical form — we keep it exactly as written and *attach
    what we discover* as `features` (e.g. {"number": "plural", "singular": "student"} says this
    word is the plural of "student"). Findings only ever accumulate; nothing is overwritten, so
    the surface plus the features always reconstruct the full picture.
    """
    words: list[Word]
    features: dict = field(default_factory=dict)

    @property
    def surface(self) -> str:
        """The original text of this unit, reconstructed from its words — never lossy."""
        return " ".join(w.surface for w in self.words)


@dataclass
class Obligation:
    """A demand a word's sense places on the rest of the statement — something it must find to be
    satisfied (e.g. a quantifier must find the `collection` it ranges over). Unmet until `target`
    points at the unit that discharges it; until then it is a standing reason the statement is
    incomplete."""
    need: str
    target: object = None  # the Unit that satisfies it, once found (None = unmet)

    @property
    def met(self) -> bool:
        return self.target is not None


@dataclass
class Decision:
    """One recorded choice about the reading, with enough to backtrack or retest it. A decision is
    a reversible state-diff on a `target` unit: its features went from `before` to `after`. `kind`
    names the move (annotate | merge | assign-sense | bind | …); `seq` gives stable ordering.
    Reverse it by restoring `target.features = before`."""
    seq: int
    kind: str
    target: "Unit"
    before: dict
    after: dict
    reason: str


@dataclass
class Reading:
    """The workspace for analyzing one statement: the units read so far, the decision trail, and
    the facts discovered. It grows as we read; nothing is discarded."""
    units: list[Unit] = field(default_factory=list)
    decisions: list = field(default_factory=list)
    facts: list = field(default_factory=list)

    def record(self, kind: str, target: Unit, before: str, after: str, reason: str) -> Decision:
        """Append a Decision to the trail and return it. The single chokepoint for change, so every
        mutation of a unit leaves a reversible footprint."""
        d = Decision(seq=len(self.decisions), kind=kind, target=target,
                     before=before, after=after, reason=reason)
        self.decisions.append(d)
        return d


def read(text: str) -> Reading:
    """Read text into a Reading: one Word per whitespace-separated token, each wrapped in its own
    Unit, in order. No normalization or merging yet — those are later, decision-logged steps."""
    words = [Word(surface=tok, index=i) for i, tok in enumerate(text.split())]
    return Reading(units=[Unit(words=[w]) for w in words])


# --- annotation: number ---------------------------------------------------------------------
# We do NOT guess at plurals. A unit is plural iff it appears in this definitive, curated
# dictionary mapping a known plural to its singular. Irregulars ("children"→"child") cost nothing
# here — it stores facts, not patterns. Anything not listed is left UNMARKED: we make no claim
# about its number, so "Socrates" (ends in 's') is never mistaken for a plural. The list grows
# only by deliberate addition; meeting a real plural that isn't here yet is the roadblock that
# tells us to extend the dictionary — not to invent a rule.
PLURALS = {
    "students": "student",
    "children": "child",
    "apples": "apple",
}


_PUNCT = ".,?!;:'\"()"


def _key(surface: str) -> str:
    """The dictionary-lookup key for a surface: lowercased, surrounding punctuation stripped. The
    Word's surface is kept lossless; only LOOKUPS ignore punctuation, so "apples?" still reads as
    the noun "apple" without mutating the word."""
    return surface.lower().strip(_PUNCT)


def _singular_of(surface: str) -> Optional[str]:
    """The known singular of `surface` if it is a recognized plural, else None. Pure lookup."""
    return PLURALS.get(_key(surface))


def annotate_number(reading: Reading) -> Reading:
    """Attach {number: plural, singular: X} to each unit that is a KNOWN plural (in PLURALS),
    without changing the word, logging each as a reversible Decision. Unknown words are left
    unmarked — we never guess about number."""
    for u in reading.units:
        singular = _singular_of(u.surface)
        if singular is None:
            continue  # not a known plural → make no claim, decide nothing
        before = dict(u.features)
        u.features["number"] = "plural"
        u.features["singular"] = singular
        reading.record("annotate", target=u, before=before, after=dict(u.features),
                       reason="known plural")
    return reading


# --- the word-meaning dictionary (lexicon) ---------------------------------------------------
# The definitive dictionary of word meanings — the same pattern as PLURALS, now keyed to MEANING.
# A word maps to a LIST of senses, because a word may mean several things (the list has one entry
# for the unambiguous function words seeded here; an ambiguous word will arrive with more). Known
# words get their senses attached as facts; unknown words are left blank — we never invent meaning.
# Grows only by deliberate addition.
LEXICON: dict[str, list[str]] = {
    "all": ["quantifier"], # quantifiers tend to refer to a collection.
    # "each" — distributive determiner/pronoun: every one of two or more taken INDIVIDUALLY
    # (contrast "all"/"every", which can be collective). Two structural patterns: determiner +
    # singular count noun ("each student"); or pronoun + partitive "of" ("each of the students").
    # Takes singular agreement. Only the quantifier sense is needed so far. (M-W / Grammarly)
    "each": ["quantifier"],
    "every": ["quantifier"],
    "some": ["quantifier"],
    "no": ["quantifier", "boolean"], # As quantifier is analogous to "none"
    "a": ["article"], # singular, even if it is pointing to a singular collection.
    "an": ["article"], # singular, even if it is pointing to a singular collection.
    "the": ["article"], # singular, even if it is pointing to a singular collection.
    # "of" — preposition, a genitive-type link with several senses: partitive / part-whole
    # ("each OF the students" — the one we use), possession ("owner of the house"), composition
    # ("made of wood"), origin ("people of Iceland"). The partitive ties a selector/quantity to
    # the whole it is drawn from. One sense needed for now. (M-W / Grammarly)
    "of": ["preposition"],
    "student": ["noun"],    # open-class content words live here too, tagged by part of speech
    "teacher": ["noun"],
    "person": ["noun"],
    "apple": ["noun"],
    "passed": ["verb"],
    "red": ["adjective"],   # adjectives modify a noun, contributing an attribute (a set-restriction)
    "green": ["adjective"],
    "rotten": ["adjective"],
    "big": ["adjective"],
    "rose": ["noun", "verb"],   # ambiguous: a flower (noun) or the past of "rise" (verb)
    "wilted": ["verb"],
    "how": ["interrogative"],   # asks about the quantity word to its right ("how many …")
    "many": ["quantifier"],     # a quantity quantifier — ranges over a collection like any other
    "there": ["existential"],   # existential "there are …" — introduces instances into the context
    "are": ["verb"],
    "is": ["verb"],
    "one": ["number"], "two": ["number"], "three": ["number"],   # a number is a quantifier with a
    "four": ["number"], "five": ["number"],                      # VALUE (see NUMBERS)
    "did": ["auxiliary"], "does": ["auxiliary"], "do": ["auxiliary"],  # yes/no question markers
    "pass": ["verb"], "fail": ["verb"], "failed": ["verb"],
    "not": ["negation"], "never": ["negation"],   # negators flip a fact's polarity (see NEGATORS)
}

# Number words → their value, for "there are N <collection>" statements that introduce instances.
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def annotate_senses(reading: Reading) -> Reading:
    """Attach each KNOWN word's possible meanings (`senses`) from the LEXICON, without changing the
    word, logging each as a reversible Decision. Unknown words are left unmarked."""
    for u in reading.units:
        senses = LEXICON.get(_key(u.surface))
        if senses is None:
            # fall back to a known singular (set earlier by annotate_number) so a plural inherits
            # its noun sense without listing every plural form in the lexicon. Requires the number
            # pass to run first — see analyze().
            singular = u.features.get("singular")
            senses = LEXICON.get(singular.lower()) if singular else None
        if senses is None and u.surface[:1].isupper() and _key(u.surface).isalpha():
            # a capitalized word we don't otherwise know is a proper noun — a named individual.
            # (Known capitalized words, e.g. sentence-initial "All"/"The", are caught above first.)
            senses = ["name"]
        if senses is None:
            continue  # not in the dictionary → attach no meaning
        before = dict(u.features)
        u.features["senses"] = list(senses)
        reading.record("annotate", target=u, before=before, after=dict(u.features),
                       reason="lexical role")
    return reading


# --- obligations: what a sense must find -----------------------------------------------------
# An obligation is definitional to a ROLE, not special to one word: a quantifier ranges over a
# collection BY VIRTUE of being a quantifier, so "each"/"all"/"some"/"no" all inherit the same
# demand. We attach obligations UNMET; finding what discharges them is a later step. Runs after
# annotate_senses (it reads the senses it left).
ROLE_OBLIGATIONS: dict[str, list[str]] = {
    "quantifier": ["collection"],     # must find the collection it ranges over
    "preposition": ["anchor", "object"],  # a linking preposition relates an `anchor` (the unit on
                                          # its left it attaches to) to an `object` (the noun it
                                          # governs, on its right). e.g. "each [anchor] of [object →
                                          # students]". (your source/target, direction pinned)
    "adjective": ["modifies"],  # must find the head noun it modifies — discharged by MERGE (the
                                # adjective is absorbed into the noun as an attribute), not by binding
    "article": ["determines"],  # determines the noun it introduces (to its right)
    "verb": ["subject"],        # predicates a subject (the noun to its left)
    "noun": ["predicate"],      # wants something asserted of it (a verb to its right); left UNMET
                                # in a bare noun phrase → that is how a fragment is detected.
                                # (Object nouns don't really need this — a later "forego when
                                # subordinated" refinement; current examples don't hit it.)
    "interrogative": ["queries"],  # asks about the quantity word to its right ("how" → "many")
    "number": ["collection"],      # a number ranges over the collection it counts ("three apples")
    "auxiliary": ["asks_predication"],  # a yes/no marker asks whether a verb (its predication) holds
    "name": ["predicate"],         # a named individual is noun-like — it wants a predicate too
}


def annotate_obligations(reading: Reading) -> Reading:
    """For each unit, attach the (unmet) obligations its senses' roles imply, logged reversibly.
    A unit with no obligation-bearing sense is left untouched."""
    for u in reading.units:
        needs = [n for role in u.features.get("senses", []) for n in ROLE_OBLIGATIONS.get(role, [])]
        if not needs:
            continue
        before = dict(u.features)
        u.features["obligations"] = [Obligation(need) for need in needs]
        reading.record("oblige", target=u, before=before, after=dict(u.features),
                       reason="needs " + ", ".join(needs))
    return reading


# --- resolution: discharge an obligation ONLY when unambiguous -------------------------------
# Each obligation has a need; `_find_target` knows how each need is satisfied. We bind ONLY when
# the answer is unambiguous — zero or several candidates leave it unmet, a reportable deficiency
# rather than a guess. We replace the Obligation (not mutate it) so the Decision's before/after
# snapshots stay a clean, reversible diff. Runs after the obligations pass.
def _find_target(need, u, i, units, nouns, verbs, names, pos):
    """The unit that discharges obligation `need` on unit `u` (at position `i`), or None if it is
    absent or ambiguous. We commit only when exactly one candidate fits in the right direction."""
    if need == "collection":      # a quantifier ranges over the one available noun
        cands = [n for n in nouns if n is not u]
        return cands[0] if len(cands) == 1 else None
    if need in ("object", "determines"):   # preposition governs / article determines the unique noun to its right
        right = [n for n in nouns if pos[id(n)] > i]
        return right[0] if len(right) == 1 else None
    if need == "anchor":          # a preposition attaches to the unit on its immediate left
        return units[i - 1] if i > 0 else None   # (PP-attachment can be ambiguous; revisit later)
    if need == "subject":         # a verb's subject is the unique noun OR named individual to its left
        left = [n for n in (nouns + names) if pos[id(n)] < i]
        return left[0] if len(left) == 1 else None
    if need == "predicate":       # a noun's predicate is the unique verb to its right
        right = [v for v in verbs if pos[id(v)] > i]
        return right[0] if len(right) == 1 else None
    if need == "queries":         # an interrogative asks about the unique quantifier to its right
        quants = [q for q in units
                  if "quantifier" in q.features.get("senses", []) and pos[id(q)] > i]
        return quants[0] if len(quants) == 1 else None
    if need == "asks_predication":   # an auxiliary asks whether the unique verb to its right holds
        right = [v for v in verbs if pos[id(v)] > i]
        return right[0] if len(right) == 1 else None
    return None


def resolve(reading: Reading) -> Reading:
    units = reading.units
    pos = {id(u): k for k, u in enumerate(units)}
    nouns = [u for u in units if "noun" in u.features.get("senses", [])]
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    names = [u for u in units if "name" in u.features.get("senses", [])]
    for i, u in enumerate(units):
        obs = u.features.get("obligations")
        if not obs:
            continue
        new_obs, bound = [], False
        for o in obs:
            target = None if o.met else _find_target(o.need, u, i, units, nouns, verbs, names, pos)
            if target is not None:                             # unambiguous → bind
                new_obs.append(Obligation(o.need, target=target))
                bound = True
            else:
                new_obs.append(o)                              # else leave as-is (unmet)
        if bound:
            before = dict(u.features)
            u.features["obligations"] = new_obs
            reading.record("bind", target=u, before=before, after=dict(u.features),
                           reason="obligations resolved")
    return reading


# --- condensation: merge a modifier into its head (structural, reversible) -------------------
# Some obligations resolve by MERGING, not binding. An adjective + its head noun form ONE richer
# concept — "red apple" is the restricted set of red apples, not all apples — so we condense them
# into a single noun unit that keeps the head ("apple") and gains the attribute ("red") as a
# composable set-restriction (attributes stored as a sorted set: parent/child and counts are then
# derivable later without a refactor). A quantifier then ranges over the COMPOUND, keeping the
# restriction. This is the first STRUCTURAL decision: it changes the units LIST (not one unit's
# features), and records the constituents so the merge stays reversible. Multi-pass stacks modifiers.
def _merge_adjective(reading: Reading, adj_i: int, noun_i: int) -> None:
    units = reading.units
    adj, noun = units[adj_i], units[noun_i]
    attribute = adj.surface.lower()
    head = noun.features.get("head") or noun.features.get("singular") or noun.surface.lower()
    attributes = sorted(set(noun.features.get("attributes", [])) | {attribute})
    merged = Unit(words=sorted(adj.words + noun.words, key=lambda w: w.index))
    merged.features = dict(noun.features)            # keep the noun's identity (senses/obligations)
    merged.features["head"] = head
    merged.features["attributes"] = attributes
    lo = min(adj_i, noun_i)
    units[lo:lo + 2] = [merged]                      # replace the adjacent pair with the compound
    reading.record("merge", target=merged,
                   before={"merged_from": [adj, noun], "at": lo},  # structural diff → reversible
                   after={"head": head, "attributes": attributes},
                   reason=f"adjective '{attribute}' modifies '{head}'")


def condense(reading: Reading) -> bool:
    """One condensation step: find the first adjective immediately left of a noun and merge them.
    Returns True if it merged (so `analyze` can loop to a fixed point), False once nothing merges."""
    for i, u in enumerate(reading.units):
        if "adjective" in u.features.get("senses", []) and i + 1 < len(reading.units):
            if "noun" in reading.units[i + 1].features.get("senses", []):
                _merge_adjective(reading, i, i + 1)
                return True
    return False


# --- sense disambiguation: let coverage pick the right meaning --------------------------------
# A word may carry several senses (from the lexicon). We SELECT one per ambiguous word — the
# assignment whose obligations leave the FEWEST unmet across the whole statement (saturation as the
# arbiter; no POS tagger). Each ambiguous word is tried in every sense via throwaway deep-copies;
# the best-covering combination is committed as reversible 'assign-sense' decisions. No-op when
# nothing is ambiguous, so unambiguous sentences pay nothing.
def _unmet_count(reading: Reading) -> int:
    return sum(1 for u in reading.units
               for o in u.features.get("obligations", []) if not o.met)


def _trial_cost(reading: Reading, multi_idx, combo) -> int:
    """Coverage cost of a sense assignment: run obligations + condense + resolve on a COPY with the
    chosen senses forced, and count what stays unmet. Lower is better."""
    trial = copy.deepcopy(reading)
    for k, sense in zip(multi_idx, combo):
        trial.units[k].features["senses"] = [sense]
    annotate_obligations(trial)
    while condense(trial):
        pass
    resolve(trial)
    return _unmet_count(trial)


def disambiguate(reading: Reading) -> Reading:
    """Pick the best-covering sense for each ambiguous word; commit it as a reversible decision."""
    multi_idx = [k for k, u in enumerate(reading.units) if len(u.features.get("senses", [])) > 1]
    if not multi_idx:
        return reading                       # nothing ambiguous → no work
    options = [reading.units[k].features["senses"] for k in multi_idx]
    best = min(product(*options), key=lambda combo: _trial_cost(reading, multi_idx, combo))
    for k, sense in zip(multi_idx, best):
        u = reading.units[k]
        before = dict(u.features)
        u.features["senses"] = [sense]
        reading.record("assign-sense", target=u, before=before, after=dict(u.features),
                       reason=f"sense '{sense}' covers best")
    return reading


# --- the pipeline ----------------------------------------------------------------------------
def analyze(text: str) -> Reading:
    """Read and analyze one statement end to end. Passes run in DEPENDENCY ORDER: number before
    senses (so a plural inherits its noun sense), then DISAMBIGUATE (select one sense per word by
    coverage), then obligations. Then a MULTI-PASS loop: condense modifiers into their heads until
    a fixed point, and finally resolve (bind) the remaining obligations. Every step is logged and
    reversible; 0 model calls."""
    r = read(text)
    annotate_number(r)
    annotate_senses(r)
    disambiguate(r)          # collapse multi-sense words to the reading that covers best
    annotate_obligations(r)
    while condense(r):       # condense "red apple" → one noun unit; loop stacks "big red apple"
        pass
    resolve(r)               # then bind quantifier/preposition obligations over the condensed units
    return r


# --- the answer-or-ask boundary --------------------------------------------------------------
# A reading is complete and understood iff nothing is deficient. Because every word carries an
# obligation and unknown words stay unmarked, deficiency is FULLY characterized by two things:
# words we don't recognize, and obligations left unmet (a stranded word surfaces as its own unmet
# obligation — there is no third failure mode). A non-empty result is exactly what to hand back to
# the user (human or agent) as a clarifying question, instead of guessing.
@dataclass
class Deficiency:
    kind: str       # "unknown-word" | "unmet-obligation"
    unit: "Unit"
    message: str    # specific, human-readable: names the word and what it lacks


_NEED_PHRASING = {
    "collection": "has no collection to range over",
    "object": "has no object to govern",
    "anchor": "has nothing to attach to",
    "subject": "has no subject",
    "predicate": "has nothing asserted of it (missing a predicate)",
    "determines": "has no noun to determine",
    "modifies": "has no noun to modify",
}


def deficiencies(reading: Reading) -> list:
    """Everything that stops this reading from being a complete, understood statement. Empty ⇒
    well-formed."""
    out = []
    for u in reading.units:
        if not u.features.get("senses"):
            out.append(Deficiency("unknown-word", u, f"unrecognized word {u.surface!r}"))
            continue
        for o in u.features.get("obligations", []):
            if not o.met:
                phrase = _NEED_PHRASING.get(o.need, f"is missing its {o.need}")
                out.append(Deficiency("unmet-obligation", u, f"{u.surface!r} {phrase}"))
    return out


def is_well_formed(reading: Reading) -> bool:
    """True iff the statement is complete and fully understood (no deficiencies)."""
    return not deficiencies(reading)


# --- answering: fulfil a question's obligation against parsed text ----------------------------
# A question parses like any statement; its interrogative ("how") carries a `queries` obligation
# that binds to the quantity word, which ranges over a collection. ANSWERING = read that
# collection's descriptor and count the matching instances in the context. It reuses the whole
# pipeline — same reader, same obligations — and inherits the answer-or-ask contract: an
# unrecognized word makes the question unanswerable, so we ASK rather than guess. (Questions ask,
# they don't assert, so we follow the interrogative chain rather than require full saturation.)
def _descriptor(unit: "Unit"):
    """A collection descriptor (head, frozenset(attributes)) for a noun unit."""
    head = unit.features.get("head") or unit.features.get("singular") or _key(unit.surface)
    return (head, frozenset(unit.features.get("attributes", [])))


def collection_descriptor(reading: Reading):
    """The descriptor of the single noun this reading describes (a parsed instance), or None."""
    nouns = [u for u in reading.units if "noun" in u.features.get("senses", [])]
    return _descriptor(nouns[0]) if len(nouns) == 1 else None


def _matches(query, instance) -> bool:
    """A query matches an instance iff same head and the query's attributes are a SUBSET of the
    instance's — so (apple, {red}) matches a {red, rotten} apple, and (apple, {}) matches any apple.
    This is the sub-collection lattice: fewer attributes = bigger set."""
    return query[0] == instance[0] and query[1] <= instance[1]


@dataclass(frozen=True)
class Predication:
    """A known fact "subject does predicate": ((head, attributes), verb-lemma). Tense is normalized
    to the lemma so "passed" and "pass" name one predicate. `negated` is the polarity — set when the
    statement carries a negator, so "Bob did not pass" is NOT the positive "Bob passed"."""
    subject: tuple
    predicate: str
    negated: bool = False


# Past tense → base form, so a fact ("passed") and a question ("pass") name the same predicate.
VERB_LEMMA = {"passed": "pass", "wilted": "wilt", "rose": "rise", "failed": "fail"}

# Words that negate a statement's polarity. "no" both quantifies AND negates ("no students failed"
# = students did NOT fail); detection is by surface, independent of which sense was selected.
NEGATORS = {"not", "never", "no"}


def _lemma(surface: str) -> str:
    return VERB_LEMMA.get(_key(surface), _key(surface))


def _negated(reading: Reading) -> bool:
    return any(_key(u.surface) in NEGATORS for u in reading.units)


def _predication(reading: Reading):
    """The (subject, predicate, polarity) a simple statement asserts, or None — the one verb and its
    subject, negated if a negator (not/never/no) is present."""
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    if len(verbs) != 1:
        return None
    subject = _follow(verbs[0], "subject")
    if subject is None:
        return None
    return Predication(_descriptor(subject), _lemma(verbs[0].surface), _negated(reading))


@dataclass(frozen=True)
class Membership:
    """A known fact "<individual> is a <collection>": a named individual belongs to a collection,
    e.g. "Bob is a student" → Membership("bob", ("student", frozenset()))."""
    member: str
    collection: tuple


def _has_copula(reading: Reading) -> bool:
    return any("verb" in u.features.get("senses", []) and _key(u.surface) in
               ("is", "are", "was", "were") for u in reading.units)


def _membership(reading: Reading):
    """The membership a copula statement asserts ("X is a Y"), or None — exactly one named
    individual, one noun, and a copula (is/are/was/were)."""
    names = [u for u in reading.units if "name" in u.features.get("senses", [])]
    nouns = [u for u in reading.units if "noun" in u.features.get("senses", [])]
    if _has_copula(reading) and len(names) == 1 and len(nouns) == 1:
        return Membership(_key(names[0].surface), _descriptor(nouns[0]))
    return None


@dataclass(frozen=True)
class Subset:
    """A universal-membership fact "Every X is a Y": collection `sub` is contained in `sup`, so
    every member of `sub` is a member of `sup`. e.g. "Every teacher is a person"."""
    sub: tuple
    sup: tuple


def _subset(reading: Reading):
    """The subset a universal copula statement asserts ("Every X is a Y"), or None — a universal
    quantifier, a copula, and exactly two nouns (subject then complement)."""
    nouns = [u for u in reading.units if "noun" in u.features.get("senses", [])]
    has_universal = any("quantifier" in u.features.get("senses", []) and
                        _key(u.surface) in ("every", "all") for u in reading.units)
    if has_universal and _has_copula(reading) and len(nouns) == 2:
        sub, sup = sorted(nouns, key=lambda u: u.words[0].index)
        return Subset(_descriptor(sub), _descriptor(sup))
    return None


def _property(reading: Reading):
    """A predicative-adjective fact ("The apple is red"), or None — a copula, one subject (noun or
    name), and one PREDICATIVE adjective (one that didn't merge into a noun). The predicate is the
    adjective; polarity from any negator. Distinct from attributive "red apple" (which merges)."""
    adjs = [u for u in reading.units if "adjective" in u.features.get("senses", [])]
    subjects = [u for u in reading.units
                if "noun" in u.features.get("senses", []) or "name" in u.features.get("senses", [])]
    if _has_copula(reading) and len(adjs) == 1 and len(subjects) == 1:
        return Predication(_descriptor(subjects[0]), _key(adjs[0].surface), _negated(reading))
    return None


def _is_member(member: str, target: tuple, context: list) -> bool:
    """Is `member` in collection `target`? True if directly a member, or reachable from one of its
    collections by following Subset edges (transitive). The membership-inference engine."""
    frontier = [m.collection for m in context if isinstance(m, Membership) and m.member == member]
    seen = set()
    while frontier:
        c = frontier.pop()
        if c == target:
            return True
        if c in seen:
            continue
        seen.add(c)
        frontier += [f.sup for f in context if isinstance(f, Subset) and f.sub == c]
    return False


def _follow(unit, need):
    """The target of `unit`'s met obligation named `need`, or None."""
    if unit is None:
        return None
    for o in unit.features.get("obligations", []):
        if o.need == need and o.met:
            return o.target
    return None


def answer(question: str, context: list):
    """Answer a question against a context of facts. A yes/no question ("Did the student pass?")
    returns "yes"/"no" by checking a Predication; a count question ("How many red apples?") returns
    an int by counting matching instances. Returns a list of Deficiency (what to ASK) when the
    question can't be answered. Reuses the same parse; dispatches on the interrogative kind."""
    r = analyze(question)
    unknown = [d for d in deficiencies(r) if d.kind == "unknown-word"]
    if unknown:
        return unknown                                   # don't guess about words we don't know

    names = [u for u in r.units if "name" in u.features.get("senses", [])]
    nouns = [u for u in r.units if "noun" in u.features.get("senses", [])]
    if _has_copula(r) and len(names) == 1 and len(nouns) == 1:   # "Is X a Y?" membership question
        return "yes" if _is_member(_key(names[0].surface), _descriptor(nouns[0]), context) else "no"

    prop = _property(r)                                  # "Is <subject> <adjective>?" property question
    if prop is not None:
        return "yes" if Predication(prop.subject, prop.predicate) in context else "no"

    aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
    if aux is not None:                                  # yes/no question
        verb = _follow(aux, "asks_predication")          # did → the verb being asked about
        subject = _follow(verb, "subject") if verb else None
        if subject is None:
            return [Deficiency("unanswerable", aux, "I couldn't tell what is being asked")]
        query = Predication(_descriptor(subject), _lemma(verb.surface))
        if query in context:                             # directly known
            return "yes"
        # universal instantiation: the subject is a member of a collection that has the predicate
        member = query.subject[0]
        for f in context:
            if isinstance(f, Membership) and f.member == member \
                    and Predication(f.collection, query.predicate) in context:
                return "yes"
        return "no"

    interro = next((u for u in r.units if "interrogative" in u.features.get("senses", [])), None)
    if interro is None:
        return [Deficiency("not-a-question", interro, "this is not a question I can answer")]
    collection = _follow(_follow(interro, "queries"), "collection")   # how → many → its collection
    if collection is None:
        return [Deficiency("unanswerable", interro, "I couldn't tell what to count")]
    query = _descriptor(collection)
    return sum(1 for inst in context if isinstance(inst, tuple) and _matches(query, inst))


def instances(statement: str) -> list:
    """Parse a "there are N <collection>" statement into N copies of the collection's descriptor —
    the instance set `answer()` queries. The number word ranges over the collection (same
    `collection` obligation as any quantifier) and carries its value via NUMBERS. Returns [] when no
    count+collection can be read, so an unparseable statement teaches us nothing rather than a guess.
    Concatenate over several statements to build a context."""
    r = analyze(statement)
    for u in r.units:
        value = NUMBERS.get(_key(u.surface))
        if value is None:
            continue
        collection = _follow(u, "collection")
        if collection is not None:
            return [_descriptor(collection)] * value
    return []


def explain(question: str, context: list) -> list:
    """The human-readable derivation steps behind `answer(question, context)` — what was queried,
    which facts were checked, and how the verdict follows."""
    r = analyze(question)
    unknown = [d.message for d in deficiencies(r) if d.kind == "unknown-word"]
    if unknown:
        return unknown
    names = [u for u in r.units if "name" in u.features.get("senses", [])]
    nouns = [u for u in r.units if "noun" in u.features.get("senses", [])]
    if _has_copula(r) and len(names) == 1 and len(nouns) == 1:   # membership derivation
        member, target = _key(names[0].surface), _descriptor(nouns[0])
        cols = [m.collection[0] for m in context if isinstance(m, Membership) and m.member == member]
        verdict = "yes" if _is_member(member, target, context) else "no"
        return [f"is {member!r} a {target[0]}?",
                f"{member!r} is a: {', '.join(cols) or 'nothing known'}",
                f"reachable to {target[0]} via subset edges? {verdict}"]

    aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
    if aux is not None:                                  # yes/no derivation
        verb = _follow(aux, "asks_predication")
        subject = _follow(verb, "subject") if verb else None
        if subject is None:
            return ["couldn't tell what is being asked"]
        subj, pred = _descriptor(subject), _lemma(verb.surface)
        steps = [f"does {subj[0]!r} {pred}?"]
        if Predication(subj, pred) in context:
            return steps + [f"directly known that {subj[0]} {pred} → yes"]
        mems = [f for f in context if isinstance(f, Membership) and f.member == subj[0]]
        if not mems:
            steps.append(f"{subj[0]!r} {pred} is not stated, and it belongs to no collection → no")
        for m in mems:
            ok = Predication(m.collection, pred) in context
            steps.append(f"{subj[0]!r} is a {m.collection[0]}; does {m.collection[0]} {pred}? "
                         + ("yes" if ok else "no"))
        return steps
    interro = next((u for u in r.units if "interrogative" in u.features.get("senses", [])), None)
    if interro is not None:                              # count derivation
        coll = _follow(_follow(interro, "queries"), "collection")
        if coll is not None:
            q = _descriptor(coll)
            n = sum(1 for inst in context if isinstance(inst, tuple) and _matches(q, inst))
            return [f"count of {q}: {n} matching instance(s)"]
    return ["not a question I can answer"]


def learn(statement: str) -> list:
    """Parse a statement into context facts — the one ingestion door. A count statement ("There are
    three red apples") → instances; a subject-verb statement ("The student passed") → a Predication;
    a bare collection ("red apple") → one instance. [] when nothing parses, so we never guess."""
    r = analyze(statement)
    for u in r.units:                                    # count statement?
        value = NUMBERS.get(_key(u.surface))
        coll = _follow(u, "collection") if value is not None else None
        if coll is not None:
            return [_descriptor(coll)] * value
    s = _subset(r)                                       # "Every X is a Y" → universal membership?
    if s is not None:
        return [s]
    m = _membership(r)                                   # "<name> is a <noun>" → membership?
    if m is not None:
        return [m]
    prop = _property(r)                                  # "<subject> is <adjective>" → property?
    if prop is not None:
        return [prop]
    p = _predication(r)                                  # subject-verb statement?
    if p is not None:
        return [p]
    d = collection_descriptor(r)                         # bare collection phrase?
    return [d] if d is not None else []


# --- CLI -------------------------------------------------------------------------------------
# A thin shell over the engine. `run_cli` returns the text to print (factored out of __main__ so it
# is testable): with no --about it analyzes a statement and reports understood / what to clarify;
# with --about it treats the text as a question and answers it against the given instances.
def _obligation_summary(unit: "Unit") -> str:
    parts = [f"{o.need}->{o.target.surface!r}" if o.met else f"{o.need}->?"
             for o in unit.features.get("obligations", [])]
    return ("  " + ", ".join(parts)) if parts else ""


def _sentences(text: str) -> list:
    """Split text into sentences on . ? ! (keeping the terminator). No regex — stdlib only."""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in ".?!" and cur.strip():
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def _answer_text(result) -> str:
    if isinstance(result, list):
        return "I can't answer that:\n" + "\n".join("  - " + d.message for d in result)
    return str(result)


def _trace_lines(reading: Reading) -> list:
    """The full analyzed data for one reading: each unit's senses + obligation bindings (and any
    head/attributes), then the reversible decision trail."""
    lines = ["  units:"]
    for u in reading.units:
        f = u.features
        extra = f"  head={f['head']} attrs={f.get('attributes', [])}" if "head" in f else ""
        lines.append(f"    {u.surface!r:12} {f.get('senses', '?')}{_obligation_summary(u)}{extra}")
    if reading.decisions:
        lines.append("  decisions:")
        for d in reading.decisions:
            tgt = d.target.surface if d.target else "-"
            lines.append(f"    [{d.seq}] {d.kind:11} {tgt!r:12} {d.reason}")
    return lines


_USAGE = 'usage: python -m reasoning.reading "<text>" [--about "<phrase>" ...] [--trace]'


def run_cli(args: list) -> str:
    if not args:
        return _USAGE
    trace = "--trace" in args
    args = [a for a in args if a != "--trace"]
    if not args:
        return _USAGE
    text = args[0]
    if "--about" in args:                                  # explicit context
        premises, question = args[args.index("--about") + 1:], text
    else:                                                  # paragraph: split into sentences
        sentences = _sentences(text)
        qidx = [k for k, s in enumerate(sentences) if s.endswith("?")]
        premises, question = (sentences[:qidx[-1]], sentences[qidx[-1]]) if qidx else ([], None)
    context = [fact for p in premises for fact in learn(p)]

    if question is None:                                   # a single statement → analyze it
        r = analyze(text)
        if trace:
            return "\n".join([f'STATEMENT: "{text}"'] + _trace_lines(r))
        defs = deficiencies(r)
        verdict = ("understood" if not defs
                   else "needs clarification:\n" + "\n".join("  - " + d.message for d in defs))
        return "\n".join([f"  {u.surface!r:14} {u.features.get('senses', '?')}"
                          f"{_obligation_summary(u)}" for u in r.units] + ["", verdict])

    if not trace:                                          # just the answer
        return _answer_text(answer(question, context))

    out = []                                               # the full trace
    for s in premises:
        out += [f'STATEMENT: "{s}"'] + _trace_lines(analyze(s)) + [f"  learned: {learn(s)}", ""]
    out += [f'QUESTION: "{question}"'] + _trace_lines(analyze(question)) + [""]
    out += ["derivation:"] + ["  " + line for line in explain(question, context)]
    out += [f"ANSWER: {_answer_text(answer(question, context))}"]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(run_cli(sys.argv[1:]))

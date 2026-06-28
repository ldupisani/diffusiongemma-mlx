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
    optional: bool = False  # an optional argument (e.g. a verb's direct object) — unmet is NOT a
                            # deficiency, since transitive/intransitive use is a free choice

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


def _merge_compounds(reading: Reading) -> Reading:
    """Merge a known multi-word phrase ("piggy bank") into ONE unit — an INDIVISIBLE compound — before
    the rest of analysis sees it (so it reads as a single noun, not two words). Reversible."""
    changed = True
    while changed:
        changed = False
        units = reading.units
        for i in range(len(units) - 1):
            phrase = _key(units[i].surface) + " " + _key(units[i + 1].surface)
            if phrase in COMPOUNDS:
                merged = Unit(words=sorted(units[i].words + units[i + 1].words, key=lambda w: w.index))
                reading.record("merge", target=merged,
                               before={"merged_from": [units[i], units[i + 1]], "at": i},
                               after={"compound": phrase}, reason=f"indivisible compound {phrase!r}")
                units[i:i + 2] = [merged]
                changed = True
                break
    return reading


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
    "dogs": "dog", "cats": "cat", "birds": "bird", "cars": "car", "houses": "house",
    "trees": "tree", "books": "book", "flowers": "flower", "teachers": "teacher",
    "people": "person", "persons": "person", "men": "man", "women": "woman",
    "times": "time", "clips": "clip", "oranges": "orange",
    "friends": "friend", "dollars": "dollar", "cents": "cent", "hours": "hour", "minutes": "minute",
    "days": "day", "weeks": "week", "months": "month", "years": "year", "items": "item",
    "pages": "page", "miles": "mile", "pounds": "pound", "boxes": "box", "bags": "bag",
    "points": "point", "numbers": "number", "amounts": "amount", "costs": "cost",
    "stores": "store", "shops": "shop", "schools": "school", "classes": "class", "games": "game",
    "teams": "team", "prices": "price", "pieces": "piece", "dozens": "dozen", "pairs": "pair",
    "sets": "set", "cups": "cup", "gallons": "gallon", "kids": "kid", "toys": "toy", "balls": "ball",
    "candies": "candy", "cookies": "cookie", "cupcakes": "cupcake", "slices": "slice", "pies": "pie",
    "cakes": "cake", "eggs": "egg", "bottles": "bottle", "baskets": "basket", "jars": "jar",
    "customers": "customer", "workers": "worker", "tickets": "ticket", "gifts": "gift",
    "cards": "card", "coins": "coin", "bills": "bill", "goats": "goat", "roommates": "roommate",
    "stickers": "sticker", "marbles": "marble", "cars": "car", "children": "child",
    "parents": "parent", "grandparents": "grandparent",
    "trips": "trip", "jobs": "job", "homes": "home", "parties": "party", "fields": "field",
    "banks": "bank", "weights": "weight", "ends": "end", "feet": "foot", "inches": "inch",
    "kilograms": "kilogram", "grams": "gram", "ounces": "ounce", "meters": "meter",
    "centimeters": "centimeter", "liters": "liter", "distances": "distance", "speeds": "speed",
    "ages": "age", "families": "family", "groups": "group", "members": "member", "players": "player",
    "animals": "animal", "horses": "horse", "cows": "cow", "pigs": "pig", "chickens": "chicken",
    "baseballs": "baseball", "basketballs": "basketball", "footballs": "football", "tables": "table",
    "chairs": "chair", "rooms": "room", "buildings": "building", "floors": "floor", "gardens": "garden",
    "parks": "park", "stories": "story", "chapters": "chapter", "pictures": "picture", "photos": "photo",
    "presents": "present", "plates": "plate", "bowls": "bowl", "spoons": "spoon", "forks": "fork",
    "pans": "pan", "pots": "pot", "dolphins": "dolphin", "steps": "step", "levels": "level",
    "scores": "score", "grades": "grade",
}


_PUNCT = ".,?!;:'\"()"


def _key(surface: str) -> str:
    """The dictionary-lookup key for a surface: lowercased, surrounding punctuation stripped. The
    Word's surface is kept lossless; only LOOKUPS ignore punctuation, so "apples?" still reads as
    the noun "apple" without mutating the word."""
    return surface.lower().strip(_PUNCT)


def _as_number(surface: str):
    """The numeric value of a digit/decimal/currency token ("48", "0.2", "1,000", "$12.50"), else
    None. A leading "$"/"£" is stripped (the currency is recorded separately as a money measure)."""
    try:
        return float(_key(surface).replace(",", "").lstrip("$£"))
    except ValueError:
        return None


MONEY = ("money", frozenset())   # the canonical measure for currency ("$12" → 12 of MONEY)


def _as_percent(surface: str):
    """The fraction a percentage token denotes ("20%" → 0.20), else None."""
    s = _key(surface)
    if s.endswith("%"):
        n = _as_number(s[:-1])
        return n / 100 if n is not None else None
    return None


# The mathematical lexicon's operator words → their arithmetic symbol (step 1; grows deliberately).
OPERATORS = {"plus": "+", "minus": "-", "times": "*", "divided": "/"}

# Lexicalized multipliers: a value fused with "× N" — scales a verb's frequency ("jumps twice") or a
# referent quantity ("twice as many"). `half` is the ×0.5 case ("half as many").
MULTIPLIERS = {"once": 1, "twice": 2, "thrice": 3, "half": 0.5}

# Coreference: a pronoun's grammatical gender, and a small name→gender lexicon used to pick a
# compatible antecedent. Both grown deliberately (no guessing from spelling).
PRONOUN_GENDER = {"he": "male", "she": "female", "it": "neuter", "they": "plural",
                  "him": "male", "his": "male", "her": "female",
                  "them": "plural", "their": "plural", "its": "neuter"}
NAME_GENDER = {"tom": "male", "bob": "male", "john": "male", "jim": "male", "jack": "male",
               "mary": "female", "alice": "female", "jane": "female", "carol": "female",
               "sue": "female", "anna": "female"}


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
    "taller": ["comparative"], "older": ["comparative"], "bigger": ["comparative"],
    "more": ["comparative"], "fewer": ["comparative"],   # quantity comparatives ("more X than Y")
    "than": ["preposition"], "by": ["preposition"],   # links comparison / "divided by"
    # --- mathematical lexicon (step 1: binary operators) ---
    "plus": ["operator"], "minus": ["operator"], "times": ["operator"], "divided": ["operator"],
    "how": ["interrogative"],   # asks about the quantity word to its right ("how many …")
    "many": ["quantifier"],     # a quantity quantifier — ranges over a collection like any other
    "who": ["wh_subject"], "what": ["wh_value"],   # wh-questions: each binds to what it asks about
    "which": ["wh_which"],   # "which <noun> …?" restricts to members of that noun
    "there": ["existential"],   # existential "there are …" — introduces instances into the context
    "are": ["verb"], "is": ["verb"], "was": ["verb"], "were": ["verb"],   # copulas (also link a complement)
    "one": ["number"], "two": ["number"], "three": ["number"],   # a number is a quantifier with a
    "four": ["number"], "five": ["number"],                      # VALUE (see NUMBERS)
    "did": ["auxiliary"], "does": ["auxiliary"], "do": ["auxiliary"],  # yes/no question markers
    "and": ["conjunction"],   # conjoins items ("Tom and Mary")
    "altogether": ["aggregator"], "total": ["aggregator"],   # fold a set of quantities (sum)
    "he": ["pronoun"], "she": ["pronoun"], "it": ["pronoun"], "they": ["pronoun"],  # coreference
    "once": ["multiplier"], "twice": ["multiplier"], "thrice": ["multiplier"],  # frequency (value + ×)
    "half": ["multiplier"],   # "half as many" → ×0.5 of a referent quantity
    "as": ["particle"], "much": ["quantifier"],   # "twice as many / as much" relational phrasing
    "time": ["noun"], "times": ["operator", "noun"],   # "times" is × OR the occurrence-noun (ambiguous)
    "jump": ["verb"], "jumps": ["verb"], "jumped": ["verb"], "run": ["verb"], "runs": ["verb"],
    "clip": ["noun"], "orange": ["noun"],   # measures for word problems
    # --- word-problem vocabulary (data-driven from reading_bench's unknown-word frequency list) ---
    # content nouns (units of time / money / measure):
    "friend": ["noun"], "dollar": ["noun"], "cent": ["noun"], "hour": ["noun"], "minute": ["noun"],
    "day": ["noun"], "week": ["noun"], "month": ["noun"], "year": ["noun"], "money": ["noun"],
    "number": ["noun"], "cost": ["noun"], "amount": ["noun"], "item": ["noun"], "page": ["noun"],
    "mile": ["noun"], "pound": ["noun"], "box": ["noun"], "bag": ["noun"], "point": ["noun"],
    # common prepositions (link an anchor to an object; recognised so they stop blocking the parse):
    "to": ["preposition"], "in": ["preposition"], "on": ["preposition"], "at": ["preposition"],
    "from": ["preposition"], "with": ["preposition"], "for": ["preposition"], "after": ["preposition"],
    "into": ["preposition"], "over": ["preposition"], "under": ["preposition"], "about": ["preposition"],
    "around": ["preposition"], "through": ["preposition"],
    # modal auxiliaries (recognised; behave like did/does for question framing):
    "will": ["auxiliary"], "can": ["auxiliary"], "would": ["auxiliary"], "should": ["auxiliary"],
    "could": ["auxiliary"], "must": ["auxiliary"], "may": ["auxiliary"], "might": ["auxiliary"],
    # adverbs / conjunctions (recognised; no obligation — they don't block the parse):
    "then": ["adverb"], "just": ["adverb"], "also": ["adverb"], "only": ["adverb"], "now": ["time"],
    "still": ["adverb"], "again": ["adverb"], "already": ["adverb"], "if": ["conjunction"],
    "but": ["conjunction"], "or": ["conjunction"], "so": ["conjunction"], "less": ["comparative"],
    # ordinals + common adjectives:
    "first": ["adjective"], "second": ["adjective"], "third": ["adjective"], "fourth": ["adjective"],
    "fifth": ["adjective"], "last": ["adjective"], "next": ["adjective"], "new": ["adjective"],
    "old": ["adjective"], "same": ["adjective"], "long": ["adjective"], "short": ["adjective"],
    "full": ["adjective"], "empty": ["adjective"], "large": ["adjective"], "small": ["adjective"],
    "big": ["adjective"],
    # common action verbs (their lemmas drive polarity / event-counting):
    "need": ["verb"], "make": ["verb"], "made": ["verb"], "pay": ["verb"], "paid": ["verb"],
    "take": ["verb"], "took": ["verb"], "go": ["verb"], "went": ["verb"], "want": ["verb"],
    "put": ["verb"], "keep": ["verb"], "kept": ["verb"], "bring": ["verb"], "brought": ["verb"],
    "send": ["verb"], "sent": ["verb"], "read": ["verb"], "write": ["verb"], "wrote": ["verb"],
    "build": ["verb"], "built": ["verb"], "fill": ["verb"], "walk": ["verb"], "see": ["verb"],
    "saw": ["verb"], "drink": ["verb"], "drank": ["verb"], "cut": ["verb"], "save": ["verb"],
    "collect": ["verb"], "count": ["verb"], "pick": ["verb"], "win": ["verb"], "won": ["verb"],
    "cook": ["verb"], "bake": ["verb"], "wash": ["verb"],
    "needs": ["verb"], "makes": ["verb"], "pays": ["verb"], "takes": ["verb"], "goes": ["verb"],
    "wants": ["verb"], "puts": ["verb"], "keeps": ["verb"], "brings": ["verb"], "sends": ["verb"],
    "reads": ["verb"], "writes": ["verb"], "builds": ["verb"], "fills": ["verb"], "walks": ["verb"],
    "sees": ["verb"], "drinks": ["verb"], "cuts": ["verb"], "saves": ["verb"], "collects": ["verb"],
    "counts": ["verb"], "picks": ["verb"], "wins": ["verb"], "cooks": ["verb"], "bakes": ["verb"],
    "washes": ["verb"], "wanted": ["verb"], "needed": ["verb"], "saved": ["verb"], "picked": ["verb"],
    # common GSM8K nouns (measures of word problems):
    "store": ["noun"], "shop": ["noun"], "school": ["noun"], "class": ["noun"], "game": ["noun"],
    "team": ["noun"], "price": ["noun"], "piece": ["noun"], "dozen": ["noun"], "pair": ["noun"],
    "set": ["noun"], "cup": ["noun"], "gallon": ["noun"], "kid": ["noun"], "toy": ["noun"],
    "ball": ["noun"], "candy": ["noun"], "cookie": ["noun"], "cupcake": ["noun"], "slice": ["noun"],
    "pie": ["noun"], "cake": ["noun"], "egg": ["noun"], "bottle": ["noun"], "basket": ["noun"],
    "jar": ["noun"], "customer": ["noun"], "worker": ["noun"], "ticket": ["noun"], "gift": ["noun"],
    "card": ["noun"], "coin": ["noun"], "bill": ["noun"], "goat": ["noun"], "roommate": ["noun"],
    "sticker": ["noun"], "marble": ["noun"], "child": ["noun"], "person": ["noun"], "car": ["noun"],
    "parent": ["noun"], "grandparent": ["noun"], "friend": ["noun"],
    "washing": ["noun"], "babysitting": ["noun"],
    # --- vocabulary push (data-driven from reading_bench): function words + units + common verbs/nouns ---
    "that": ["article"], "this": ["article"], "these": ["article"], "those": ["article"],
    "another": ["article"], "other": ["article"], "both": ["quantifier"], "any": ["article"],
    "be": ["verb"],
    "before": ["preposition"], "during": ["preposition"], "between": ["preposition"],
    "among": ["preposition"], "within": ["preposition"], "across": ["preposition"],
    "while": ["conjunction"], "when": ["conjunction"], "because": ["conjunction"], "since": ["conjunction"],
    "up": ["adverb"], "out": ["adverb"], "off": ["adverb"], "away": ["adverb"], "back": ["adverb"],
    "equally": ["adverb"], "together": ["adverb"], "instead": ["adverb"], "however": ["adverb"],
    # comparatives (age / size / speed / price):
    "younger": ["comparative"], "shorter": ["comparative"], "faster": ["comparative"],
    "slower": ["comparative"], "heavier": ["comparative"], "lighter": ["comparative"],
    "cheaper": ["comparative"], "longer": ["comparative"], "smaller": ["comparative"],
    "larger": ["comparative"], "higher": ["comparative"], "lower": ["comparative"],
    "wider": ["comparative"], "richer": ["comparative"], "poorer": ["comparative"],
    # verbs:
    "decide": ["verb"], "decides": ["verb"], "decided": ["verb"], "increase": ["verb"],
    "increases": ["verb"], "increased": ["verb"], "decrease": ["verb"], "decreases": ["verb"],
    "decreased": ["verb"], "finish": ["verb"], "finishes": ["verb"], "finished": ["verb"],
    "start": ["verb"], "starts": ["verb"], "started": ["verb"], "change": ["verb"], "changes": ["verb"],
    "changed": ["verb"], "calculate": ["verb"], "calculated": ["verb"], "divide": ["verb"],
    "divides": ["verb"], "multiply": ["verb"], "combine": ["verb"], "combines": ["verb"],
    "combined": ["verb"], "travel": ["verb"], "travels": ["verb"], "traveled": ["verb"],
    "weigh": ["verb"], "weighs": ["verb"], "weighed": ["verb"], "measure": ["verb"],
    "measures": ["verb"], "measured": ["verb"], "study": ["verb"], "studies": ["verb"],
    "studied": ["verb"], "watch": ["verb"], "watches": ["verb"], "watched": ["verb"],
    "play": ["verb"], "plays": ["verb"], "played": ["verb"], "ride": ["verb"], "rides": ["verb"],
    "rode": ["verb"],
    # nouns (units of measure + common GSM8K objects):
    "average": ["noun"], "trip": ["noun"], "water": ["noun"], "job": ["noun"], "home": ["noun"],
    "party": ["noun"], "field": ["noun"], "bank": ["noun"], "weight": ["noun"], "end": ["noun"],
    "foot": ["noun"], "inch": ["noun"], "kilogram": ["noun"], "gram": ["noun"], "ounce": ["noun"],
    "meter": ["noun"], "centimeter": ["noun"], "liter": ["noun"], "distance": ["noun"], "speed": ["noun"],
    "age": ["noun"], "family": ["noun"], "group": ["noun"], "member": ["noun"], "player": ["noun"],
    "animal": ["noun"], "fish": ["noun"], "horse": ["noun"], "cow": ["noun"], "pig": ["noun"],
    "chicken": ["noun"], "baseball": ["noun"], "basketball": ["noun"], "football": ["noun"],
    "table": ["noun"], "chair": ["noun"], "room": ["noun"], "building": ["noun"], "floor": ["noun"],
    "garden": ["noun"], "park": ["noun"], "story": ["noun"], "chapter": ["noun"], "picture": ["noun"],
    "photo": ["noun"], "present": ["noun"], "plate": ["noun"], "bowl": ["noun"], "spoon": ["noun"],
    "fork": ["noun"], "pan": ["noun"], "pot": ["noun"], "dolphin": ["noun"], "step": ["noun"],
    "level": ["noun"], "score": ["noun"], "grade": ["noun"],
    # multi-word INDIVISIBLE compounds (merged into one unit before analysis — see _merge_compounds):
    "piggy bank": ["noun"], "ice cream": ["noun"], "high school": ["noun"], "credit card": ["noun"],
    # time-reference NODES — relational, NOT calendar-ranked (order is tracked as separate Before facts).
    # A time carries `associates` (the value it timestamps) + `orders` (its order vs other times).
    "monday": ["time"], "tuesday": ["time"], "wednesday": ["time"], "thursday": ["time"],
    "friday": ["time"], "saturday": ["time"], "sunday": ["time"],
    "yesterday": ["time"], "today": ["time"], "tomorrow": ["time"], "tonight": ["time"],
    "january": ["time"], "february": ["time"], "march": ["time"], "april": ["time"],
    "june": ["time"], "july": ["time"], "august": ["time"], "september": ["time"],
    "october": ["time"], "november": ["time"], "december": ["time"],   # "may" deferred (modal conflict)
    # possessive / object pronouns (recognised, and they extend coreference — "his" → its antecedent):
    "him": ["pronoun"], "his": ["pronoun"], "her": ["pronoun"], "them": ["pronoun"],
    "their": ["pronoun"], "its": ["pronoun"],
    # "what remains" markers — route a count question to the holdings fold:
    "left": ["adjective"], "remaining": ["adjective"], "leftover": ["adjective"], "rest": ["noun"],
    "remain": ["verb"], "remains": ["verb"],
    "earn": ["verb"], "earns": ["verb"], "earned": ["verb"],
    "work": ["verb"], "works": ["verb"], "worked": ["verb"],
    "per": ["per"],   # a rate: "$12 PER hour" — a value distributed over a unit
    "pass": ["verb"], "fail": ["verb"], "failed": ["verb"],
    "not": ["negation"], "never": ["negation"],   # negators flip a fact's polarity (see NEGATORS)
    # --- open-class vocabulary (curated, grown deliberately) ---
    "dog": ["noun"], "cat": ["noun"], "bird": ["noun"], "car": ["noun"], "house": ["noun"],
    "tree": ["noun"], "book": ["noun"], "flower": ["noun"], "man": ["noun"], "woman": ["noun"],
    "child": ["noun"],
    "bark": ["verb"], "barked": ["verb"], "fly": ["verb"], "flew": ["verb"], "run": ["verb"],
    "ran": ["verb"], "swim": ["verb"], "swam": ["verb"], "sleep": ["verb"], "slept": ["verb"],
    "sing": ["verb"], "sang": ["verb"], "sell": ["verb"], "sold": ["verb"], "buy": ["verb"],
    "bought": ["verb"], "own": ["verb"], "owns": ["verb"], "owned": ["verb"],
    "have": ["verb"], "has": ["verb"], "had": ["verb"],
    # accumulation verbs (their lemma's polarity folds in _verb_sign): gains + losses
    "buy": ["verb"], "buys": ["verb"], "get": ["verb"], "gets": ["verb"], "got": ["verb"],
    "find": ["verb"], "finds": ["verb"], "found": ["verb"], "receive": ["verb"], "receives": ["verb"],
    "lose": ["verb"], "loses": ["verb"], "lost": ["verb"], "sells": ["verb"],
    "give": ["verb"], "gives": ["verb"], "gave": ["verb"], "spend": ["verb"], "spends": ["verb"],
    "spent": ["verb"], "eat": ["verb"], "eats": ["verb"], "ate": ["verb"], "use": ["verb"],
    "uses": ["verb"], "used": ["verb"],
    "tall": ["adjective"], "old": ["adjective"], "small": ["adjective"], "happy": ["adjective"],
    "blue": ["adjective"], "yellow": ["adjective"],
}

# Multi-word phrases that are ONE indivisible concept (merged before analysis by _merge_compounds).
COMPOUNDS = {k for k in LEXICON if " " in k}

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
        if senses is None and _as_percent(u.surface) is not None:
            senses = ["percent"]          # "20%" — a fraction (0.20) applied to a base via "of"
        if senses is None and _as_number(u.surface) is not None:
            senses = ["value"]            # a numeric literal ("48", "0.2") is an arithmetic value
        if senses is None and u.surface[:1].isupper() and _key(u.surface).isalpha():
            # a capitalized word we don't otherwise know is a proper noun — a named individual.
            # (Known capitalized words, e.g. sentence-initial "All"/"The", are caught above first.)
            senses = ["name"]
        if senses is None:
            continue  # not in the dictionary → attach no meaning
        before = dict(u.features)
        u.features["senses"] = list(senses)
        if "value" in senses:
            u.features["value"] = _as_number(u.surface)
            if u.surface.strip().startswith(("$", "£")):   # "$12" — a self-measuring money value
                u.features["measure"] = MONEY
        if "operator" in senses:
            u.features["op"] = OPERATORS[_key(u.surface)]
        if "pronoun" in senses:
            u.features["gender"] = PRONOUN_GENDER.get(_key(u.surface))
        if "multiplier" in senses:
            u.features["value"] = MULTIPLIERS[_key(u.surface)]
        if "percent" in senses:
            u.features["value"] = _as_percent(u.surface)
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
    "verb": ["subject", "dobj"],  # predicates a subject (left); may also take a direct object (right,
                                  # OPTIONAL — intransitive verbs simply don't bind it)
    "noun": ["predicate"],      # wants something asserted of it (a verb to its right); left UNMET
                                # in a bare noun phrase → that is how a fragment is detected.
                                # (Object nouns don't really need this — a later "forego when
                                # subordinated" refinement; current examples don't hit it.)
    "interrogative": ["queries"],  # asks about the quantity word to its right ("how" → "many")
    "number": ["collection"],      # a number ranges over the collection it counts ("three apples")
    "auxiliary": ["asks_predication"],  # a yes/no marker asks whether a verb (its predication) holds
    "name": ["predicate"],         # a named individual is noun-like — it wants a predicate too
    "negation": ["negates"],       # "not"/"never" must find the verb (predicate) it scopes over
    "comparative": ["standard"],   # "taller" must find the standard it compares against (after "than")
    "wh_subject": ["asks_predicate"],   # "who" binds the predicate it asks the subject of
    "wh_value": ["asks_individual"],    # "what" binds the individual it asks the type of
    "wh_which": ["restricts"],          # "which" binds the noun (collection) it restricts the query to
    "operator": ["left", "right"],      # a binary operator binds its two operands (the values it combines)
    "value": ["quantifies"],            # a value may quantify a noun ("3 APPLES"); optional (a bare
                                        # operand in "3 plus 2" quantifies nothing)
    "pronoun": ["refers"],              # a pronoun must find its antecedent (resolved cross-sentence
                                        # against the discourse, not within one reading)
    "multiplier": ["scales"],           # a frequency multiplier scales the verb to its left ("jumps TWICE")
    "per": ["rate", "unit"],            # "$12 per hour" — binds the rate value (left) and the unit (right)
    "percent": ["base"],                # "20% of 50" — binds the base the fraction applies to
    "time": ["associates", "orders"],   # a time timestamps a value (`associates`) and orders against
                                        # other times (`orders`); see the temporal model
}

# Obligations that are grammatically OPTIONAL: unmet is not a deficiency. A verb's direct object is
# genuinely optional (intransitive use) — nothing else accounts for its absence. A value's
# `quantifies` is NOT here: it's required, but FOREGONE when the value is an operand (see the
# subordination forego in deficiencies) — being grabbed by an operator discharges its role.
# `associates`/`orders` are optional: a time may timestamp no value (a pure ordering statement) and a
# lone time has nothing to order against. An unmet, DERIVABLE order is filled by the synthesis loop, not
# flagged here; a stated order that's genuinely missing is a later (querying-time) precision concern.
OPTIONAL_NEEDS = {"dobj", "associates", "orders"}


def annotate_obligations(reading: Reading) -> Reading:
    """For each unit, attach the (unmet) obligations its senses' roles imply, logged reversibly.
    A unit with no obligation-bearing sense is left untouched."""
    for u in reading.units:
        needs = [n for role in u.features.get("senses", []) for n in ROLE_OBLIGATIONS.get(role, [])]
        if "verb" in u.features.get("senses", []) and _key(u.surface) in ("is", "are", "was", "were"):
            needs = needs + ["complement"]   # a copula ALSO links a complement (its predicate side)
        if "measure" in u.features:          # a self-measuring value ("$12") needs no `quantifies`
            needs = [n for n in needs if n != "quantifies"]
        if not needs:
            continue
        before = dict(u.features)
        u.features["obligations"] = [Obligation(need, optional=(need in OPTIONAL_NEEDS))
                                     for need in needs]
        reading.record("oblige", target=u, before=before, after=dict(u.features),
                       reason="needs " + ", ".join(needs))
    return reading


# --- resolution: discharge an obligation ONLY when unambiguous -------------------------------
# Each obligation has a need; `_find_target` knows how each need is satisfied. We bind ONLY when
# the answer is unambiguous — zero or several candidates leave it unmet, a reportable deficiency
# rather than a guess. We replace the Obligation (not mutate it) so the Decision's before/after
# snapshots stay a clean, reversible diff. Runs after the obligations pass.
def _find_target(need, u, i, units, pos):
    """The unit that discharges obligation `need` on unit `u` (at position `i`), or None if it is
    absent or ambiguous. We commit only when exactly one candidate fits in the right direction.
    Candidate pools are computed by role so every obligation reads from the same word inventory."""
    def has(x, *s):
        return any(k in x.features.get("senses", []) for k in s)
    def right(*s):
        return sorted((x for x in units if has(x, *s) and pos[id(x)] > i), key=lambda x: pos[id(x)])
    def left(*s):
        return [x for x in units if has(x, *s) and pos[id(x)] < i]

    if need == "collection":      # a quantifier ranges over its HEAD: the nearest noun to its right
        r = right("noun")         # (not "the unique noun anywhere" — that broke on objects/complements)
        return r[0] if r else None
    if need == "object":          # a preposition's object: the nearest noun OR NAME to its right
        r = right("noun", "name")  # (names too, so "than WILLIAM" binds — comparatives compare people)
        return r[0] if r else None
    if need == "determines":      # an article determines the unique noun to its right
        r = right("noun")
        return r[0] if len(r) == 1 else None
    if need == "anchor":          # a preposition attaches to the unit on its immediate left
        return units[i - 1] if i > 0 else None   # (PP-attachment can be ambiguous; revisit later)
    if need == "subject":         # a verb's subject. In a fronted question "<aux> SUBJ <verb>" the
        auxes = [x for x in units if has(x, "auxiliary") and pos[id(x)] < i]   # subject sits BETWEEN
        if auxes:                                                             # the auxiliary and verb
            ax = max(auxes, key=lambda x: pos[id(x)])   # nearest aux to the left — disambiguates
            between = [x for x in units if has(x, "noun", "name", "pronoun")  # "does OLIVER have" from
                       and pos[id(ax)] < pos[id(x)] < i]                      # the fronted "money"
            if len(between) == 1:
                return between[0]
        ll = left("noun", "name", "pronoun")            # else the unique noun/name/pronoun to the LEFT
        if len(ll) == 1:
            return ll[0]
        if not ll:                                      # none to the left → fronted, take the first RIGHT
            r = right("noun", "name", "pronoun")
            return r[0] if r else None
        return None
    if need == "complement":      # a copula's complement (noun/name/adjective/comparative): the nearest
        r = right("noun", "name", "adjective", "comparative")   # to the RIGHT, but if fronted, the
        if left("noun", "name"):                                # SECOND (the first right is the subject)
            return r[0] if r else None
        return r[1] if len(r) > 1 else None
    if need == "standard":        # a comparative's standard: the noun/name after "than" (skipping the
        rns = right("noun", "name")                                       # measure: "more MONEY than
        than = next((x for x in units if pos[id(x)] > i                   # WILLIAM" → William, not money)
                     and _key(x.surface) == "than"), None)
        if than is not None:
            rns = [x for x in rns if pos[id(x)] > pos[id(than)]]
        return rns[0] if rns else None
    if need == "dobj":            # a verb's (optional) direct object: the nearest noun to its right,
        r = right("noun")         # OR, in a fronted wh-question, the noun fronted before the auxiliary
        if r:                     # ("how much MONEY does Oliver have" → money, not "than"/William)
            return r[0]
        auxes = [x for x in units if has(x, "auxiliary") and pos[id(x)] < i]
        if auxes:
            ax = min(auxes, key=lambda x: pos[id(x)])
            before = [x for x in units if has(x, "noun") and pos[id(x)] < pos[id(ax)]]
            if before:
                return before[-1]
        return None
    if need == "predicate":       # a noun's predicate is the unique verb to its right
        r = right("verb")
        return r[0] if len(r) == 1 else None
    if need in ("queries",):      # an interrogative asks about the unique quantifier to its right
        r = right("quantifier")
        return r[0] if len(r) == 1 else None
    if need in ("asks_predication", "asks_predicate"):   # auxiliary / "who": the unique verb to the right
        r = right("verb")
        return r[0] if len(r) == 1 else None
    if need == "asks_individual":   # "what": the unique named individual to its right
        r = right("name")
        return r[0] if len(r) == 1 else None
    if need == "restricts":       # "which": the noun (collection) it restricts the query to
        r = right("noun")
        return r[0] if r else None
    if need == "quantifies":      # a value quantifies the nearest noun to its right ("3 apples")
        r = right("noun")
        return r[0] if r else None
    if need == "scales":          # a multiplier scales a quantifier to its RIGHT ("twice as MANY"),
        rq = right("quantifier")  # else the nearest verb to its LEFT ("jumps TWICE" — frequency)
        if rq:
            return rq[0]
        ll = left("verb")
        return max(ll, key=lambda x: pos[id(x)]) if ll else None
    if need == "rate":            # "per"'s rate value: the nearest value to its LEFT ("$12 per hour")
        ll = left("value")
        return max(ll, key=lambda x: pos[id(x)]) if ll else None
    if need == "unit":            # "per"'s unit: the nearest noun to its RIGHT
        r = right("noun")
        return r[0] if r else None
    if need == "base":            # "20% OF 50" — the nearest value to the percent's right (past "of")
        r = right("value")
        return r[0] if r else None
    if need == "left":            # an operator's left operand: the nearest value to its left
        ll = left("value")
        return max(ll, key=lambda x: pos[id(x)]) if ll else None
    if need == "right":           # an operator's right operand: the nearest value to its right
        rr = right("value")
        return rr[0] if rr else None
    if need == "negates":         # a negator scopes over the NEAREST verb (copula left, or main right)
        verbs = [x for x in units if has(x, "verb")]
        return min(verbs, key=lambda v: abs(pos[id(v)] - i)) if verbs else None
    if need == "associates":      # a time timestamps the NEAREST verb (the value/event it describes)
        verbs = [x for x in units if has(x, "verb")]
        return min(verbs, key=lambda v: abs(pos[id(v)] - i)) if verbs else None
    if need == "orders":          # a time orders against ANOTHER time, only when an after/before cue is
        others = [x for x in units if has(x, "time") and x is not u]   # present (direction read at fact
        cue = any(_key(x.surface) in ("after", "before") for x in units)  # time; else left for synthesis)
        return others[0] if (cue and len(others) == 1) else None
    return None


def resolve(reading: Reading) -> Reading:
    units = reading.units
    pos = {id(u): k for k, u in enumerate(units)}
    for i, u in enumerate(units):
        obs = u.features.get("obligations")
        if not obs:
            continue
        new_obs, bound = [], False
        for o in obs:
            target = None if o.met else _find_target(o.need, u, i, units, pos)
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


def _members(unit: "Unit") -> list:
    """The member entity keys of a unit — a conjoined group's members, else the unit's own key."""
    return unit.features.get("members") or [_key(unit.surface)]


def _merge_conjuncts(reading: Reading, and_i: int) -> None:
    """Merge "X and Y" (the names flanking an `and`) into ONE group unit carrying both as `members`
    — a compound subject. Reversible (records the constituents); stacks for "X and Y and Z"."""
    units = reading.units
    left, conj, right = units[and_i - 1], units[and_i], units[and_i + 1]
    members = _members(left) + _members(right)
    merged = Unit(words=sorted(left.words + conj.words + right.words, key=lambda w: w.index))
    merged.features = {"senses": ["name"], "members": members}   # a group binds like a single entity
    units[and_i - 1:and_i + 2] = [merged]
    reading.record("merge", target=merged,
                   before={"merged_from": [left, conj, right], "at": and_i - 1},
                   after={"members": members}, reason=f"conjunction → group {members}")


def condense(reading: Reading) -> bool:
    """One condensation step: merge an adjective into the noun on its right, OR two names flanking an
    `and` into a group. Returns True if it merged (so `analyze` loops to a fixed point)."""
    for i, u in enumerate(reading.units):
        if "adjective" in u.features.get("senses", []) and i + 1 < len(reading.units):
            if "noun" in reading.units[i + 1].features.get("senses", []):
                _merge_adjective(reading, i, i + 1)
                return True
    for i, u in enumerate(reading.units):                # "Tom and Mary" → one group (names only:
        if "conjunction" in u.features.get("senses", []) and 0 < i < len(reading.units) - 1:  # a
            left, right = reading.units[i - 1], reading.units[i + 1]                          # subject)
            if "name" in left.features.get("senses", []) and "name" in right.features.get("senses", []):
                _merge_conjuncts(reading, i)
                return True
    return False


# --- sense disambiguation: let coverage pick the right meaning --------------------------------
# A word may carry several senses (from the lexicon). We SELECT one per ambiguous word — the
# assignment whose obligations leave the FEWEST unmet across the whole statement (saturation as the
# arbiter; no POS tagger). Each ambiguous word is tried in every sense via throwaway deep-copies;
# the best-covering combination is committed as reversible 'assign-sense' decisions. No-op when
# nothing is ambiguous, so unambiguous sentences pay nothing.
def _unmet_count(reading: Reading) -> int:
    # Coverage uses the SAME notion of "unmet" as the deficiency check: optional and subordinated
    # (foregone) obligations don't count — an operand isn't "uncovered", it's discharged by its role.
    return sum(1 for d in deficiencies(reading) if d.kind == "unmet-obligation")


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
    _merge_compounds(r)      # "piggy bank" → one indivisible unit before anything else reads it
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
    "refers": "has no antecedent — it isn't clear who or what it refers to",
    "quantifies": "doesn't say what it counts (a quantity of what?)",
}


def deficiencies(reading: Reading) -> list:
    """Everything that stops this reading from being a complete, understood statement. Empty ⇒
    well-formed."""
    # A unit that FILLS another word's slot (a verb's direct object, a copula's complement) is
    # subordinated: it's discharged by being that slot-filler, so its own unmet obligations are
    # foregone — "the dog" in "Mary owns a dog" doesn't independently demand a predicate.
    subordinated = {id(o.target) for u in reading.units for o in u.features.get("obligations", [])
                    if o.met and o.need in ("complement", "dobj", "left", "right", "unit", "standard")}
    out = []
    for u in reading.units:
        if not u.features.get("senses"):
            out.append(Deficiency("unknown-word", u, f"I have no information about {u.surface!r}"))
            continue
        if id(u) in subordinated:
            continue
        for o in u.features.get("obligations", []):
            if not o.met and not o.optional:             # optional args (a verb's object) may be unmet
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
_MONEY_WORDS = {"dollar", "cent", "buck", "money", "penny"}   # all normalise to the MONEY measure


def _descriptor(unit: "Unit"):
    """A collection descriptor (head, frozenset(attributes)) for a noun unit. Currency words all
    collapse to the canonical `money` head, so "$30", "30 dollars", and "how much money" agree."""
    head = unit.features.get("head") or unit.features.get("singular") or _key(unit.surface)
    if head in _MONEY_WORDS:
        return MONEY
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
    to the lemma so "passed" and "pass" name one predicate. `negated` is the polarity. `universal`
    is True only when the subject is bound by all/every ("All students passed") — only universal
    facts may be instantiated to a member; an existential ("Some students passed") may NOT."""
    subject: tuple
    predicate: str
    negated: bool = False
    universal: bool = False
    object: tuple = None          # the direct object of a transitive verb ("Mary owns a DOG"), else None


# Past tense → base form, so a fact ("passed") and a question ("pass") name the same predicate.
VERB_LEMMA = {"passed": "pass", "wilted": "wilt", "rose": "rise", "failed": "fail",
              "barked": "bark", "flew": "fly", "ran": "run", "swam": "swim", "slept": "sleep",
              "sang": "sing", "sold": "sell", "bought": "buy", "owns": "own", "owned": "own",
              "has": "have", "had": "have", "buys": "buy", "gets": "get", "got": "get",
              "finds": "find", "found": "find", "receives": "receive", "loses": "lose",
              "lost": "lose", "sells": "sell", "gives": "give", "gave": "give", "spends": "spend",
              "spent": "spend", "eats": "eat", "ate": "eat", "uses": "use", "used": "use",
    "jumps": "jump", "jumped": "jump", "runs": "run", "earns": "earn", "earned": "earn",
    "works": "work", "worked": "work",
    "needs": "need", "needed": "need", "makes": "make", "made": "make", "pays": "pay", "paid": "pay",
    "takes": "take", "took": "take", "goes": "go", "went": "go", "wants": "want", "wanted": "want",
    "puts": "put", "keeps": "keep", "kept": "keep", "brings": "bring", "brought": "bring",
    "sends": "send", "sent": "send", "reads": "read", "writes": "write", "wrote": "write",
    "builds": "build", "built": "build", "fills": "fill", "walks": "walk", "sees": "see", "saw": "see",
    "drinks": "drink", "drank": "drink", "cuts": "cut", "saves": "save", "saved": "save",
    "collects": "collect", "counts": "count", "picks": "pick", "picked": "pick", "wins": "win",
    "won": "win", "cooks": "cook", "bakes": "bake", "washes": "wash",
    "decides": "decide", "decided": "decide", "increases": "increase", "increased": "increase",
    "decreases": "decrease", "decreased": "decrease", "finishes": "finish", "finished": "finish",
    "starts": "start", "started": "start", "changes": "change", "changed": "change",
    "calculated": "calculate", "divides": "divide", "divided": "divide", "combines": "combine",
    "combined": "combine", "travels": "travel", "traveled": "travel", "weighs": "weigh",
    "weighed": "weigh", "measures": "measure", "measured": "measure", "studies": "study",
    "studied": "study", "watches": "watch", "watched": "watch", "plays": "play", "played": "play",
    "rides": "ride", "rode": "ride"}

def _lemma(surface: str) -> str:
    return VERB_LEMMA.get(_key(surface), _key(surface))


def _copula_unit(reading: Reading):
    """The copula verb (is/are/was/were) unit, or None — the anchor a property/membership negates."""
    for u in reading.units:
        if "verb" in u.features.get("senses", []) and _key(u.surface) in ("is", "are", "was", "were"):
            return u
    return None


def _verb_negated(verb, reading: Reading) -> bool:
    """Is `verb` negated? Both paths are scoped by BINDING, not by presence:
      - a "not"/"never" whose `negates` obligation bound to THIS verb, or
      - a "no" quantifier whose `collection` bound to this verb's SUBJECT ("no students failed" =
        the students-collection did not fail). Scoping via the bindings means "no" negates only its
        own predication, never every verb in reach (the global-flag trap the principle warns of)."""
    for u in reading.units:
        if "negation" in u.features.get("senses", []):
            if any(o.need == "negates" and o.met and o.target is verb
                   for o in u.features.get("obligations", [])):
                return True
    subject = _follow(verb, "subject")
    if subject is not None:
        for u in reading.units:
            if "quantifier" in u.features.get("senses", []) and _key(u.surface) == "no":
                if any(o.need == "collection" and o.met and o.target is subject
                       for o in u.features.get("obligations", [])):
                    return True
    return False


def _universal(verb, reading: Reading) -> bool:
    """Is `verb`'s predication universally quantified (regardless of polarity)? True iff its SUBJECT
    is the collection an all/every/no quantifier bound to — read from the bindings, like
    _verb_negated. "all students passed" and "no students passed" are both universal (the latter a
    negative universal); "some"/"the"/a number/a bare plural are NOT, so they don't instantiate."""
    subject = _follow(verb, "subject")
    if subject is None:
        return False
    for u in reading.units:
        if "quantifier" in u.features.get("senses", []) and _key(u.surface) in ("all", "every", "no"):
            if any(o.need == "collection" and o.met and o.target is subject
                   for o in u.features.get("obligations", [])):
                return True
    return False


@dataclass(frozen=True)
class Referent:
    """A resolved pronoun's antecedent — the stand-in `target` for the cross-sentence `refers`
    obligation. Its `.surface` is the entity name, so trace / obligation-summary read it like a Unit."""
    surface: str


def _entity(unit) -> str:
    """The entity a subject unit denotes: a resolved pronoun's antecedent if bound, else its own key."""
    if "pronoun" in unit.features.get("senses", []):
        ref = _follow(unit, "refers")
        if ref is not None:
            return _key(ref.surface)
    return _key(unit.surface)


def _pick_antecedent(gender, discourse: list):
    """The antecedent for a pronoun of `gender` from `discourse` (entity names, most-recent LAST): the
    most recent gender-compatible entity, else the most recent. None if the discourse is empty."""
    compatible = [n for n in discourse
                  if gender in (None, "plural") or NAME_GENDER.get(n) in (None, gender)]
    pool = compatible or discourse
    return pool[-1] if pool else None


def _resolve_references(reading: Reading, discourse: list) -> None:
    """Discharge each pronoun's `refers` obligation against the discourse — binding it to a Referent
    and logging a reversible Decision, so coreference shows in --trace like any other binding."""
    for u in reading.units:
        if "pronoun" not in u.features.get("senses", []):
            continue
        obs = u.features.get("obligations", [])
        ref_ob = next((o for o in obs if o.need == "refers"), None)
        if ref_ob is None or ref_ob.met:
            continue
        if u.features.get("gender") == "plural":         # "they"/"them" → the GROUP of all prior
            members = list(dict.fromkeys(discourse))     # distinct entities (a collective, like a
            if not members:                              # conjoined subject) → _question_subjects expands
                continue
            antecedent = " and ".join(members)
            before = dict(u.features)
            u.features["members"] = members
        else:
            antecedent = _pick_antecedent(u.features.get("gender"), discourse)
            if antecedent is None:
                continue
            before = dict(u.features)
        u.features["obligations"] = [Obligation(o.need, target=Referent(antecedent)) if o is ref_ob
                                     else o for o in obs]
        reading.record("bind", target=u, before=before, after=dict(u.features),
                       reason=f"{u.surface!r} refers to {antecedent!r}")


# A quantity verb's arithmetic polarity: a LOSS verb subtracts, everything else (gains + the "have"
# baseline) adds. Keyed by lemma; grown deliberately like the other dictionaries.
LOSS_VERBS = {"lose", "sell", "give", "spend", "eat", "use", "drop"}

# A comparative's arithmetic direction: −1 lowers the referent ("10 years YOUNGER" → −10), +1 raises it.
COMPARATIVE_SIGN = {"less": -1, "fewer": -1, "younger": -1, "shorter": -1, "lighter": -1,
                    "slower": -1, "cheaper": -1, "smaller": -1, "lower": -1, "poorer": -1}


def _verb_sign(lemma: str) -> int:
    """+1 for a gain or the 'have' baseline, -1 for a loss verb — how this quantity folds."""
    return -1 if lemma in LOSS_VERBS else 1


# Words that ask for what REMAINS — a holdings question ("how many are left / remaining") in another
# surface form. ("rest" = "the rest"; "remain"/"remains" the verb form.)
_REMAINDER = {"left", "remaining", "leftover", "remain", "remains", "rest"}


def _holdings(entity: str, measure: tuple, context: list):
    """The net holdings of `entity` for `measure` — baseline + gains − losses — or None if unknown.
    Timeless / single-state fold (order-independent); the time-aware variant is `_holdings_at`."""
    qs = [f for f in context if isinstance(f, Quantity) and f.entity == entity and f.measure == measure]
    return sum(_verb_sign(f.verb) * f.amount for f in qs) if qs else None


# A STATE verb asserts the value AT a time ("has 30 on Monday" — Monday's known total); every other
# verb is a CHANGE (a delta applied to the running value). The distinction only matters once values are
# spread across an ordered timeline (`_holdings_at`); the timeless fold treats all as signed deltas.
STATE_VERBS = {"have", "be"}


def _time_order(times: list, context: list):
    """A linear (chronological) ordering of the given time nodes from the Before closure, or None if
    they are NOT totally ordered (some pair's order is unknown — we never guess an order). Insertion
    sort using `_before`."""
    ordered = []
    for t in dict.fromkeys(x for x in times if x is not None):
        placed = False
        for k, o in enumerate(ordered):
            if _before(t, o, context):                   # t earlier than o → goes before it
                ordered.insert(k, t)
                placed = True
                break
            if not _before(o, t, context):               # neither before nor after → incomparable
                return None
        if not placed:
            ordered.append(t)
    return ordered


def _holdings_at(entity: str, measure: tuple, context: list, at_time: str):
    """Holdings of `entity`/`measure` VALID AT time node `at_time`, folded along the timeline: a STATE
    verb SETS the running value (a known total), a CHANGE verb adds its signed delta. Events strictly
    after `at_time` are excluded; a timeless event is an always-present baseline. None (decline) if an
    event's order vs `at_time` is unknown, or the applicable times aren't linearly orderable."""
    events = [f for f in context if isinstance(f, Quantity) and f.entity == entity and f.measure == measure]
    if not events:
        return None
    applicable = []
    for q in events:
        if q.time is None or q.time == at_time or _before(q.time, at_time, context):
            applicable.append(q)
        elif _before(at_time, q.time, context):
            continue                                     # strictly after the query time → not in effect
        else:
            return None                                  # order vs the query time unknown → can't place
    order = _time_order([q.time for q in applicable], context)
    if order is None:
        return None
    rank = {t: k for k, t in enumerate(order)}
    chronological = sorted(applicable, key=lambda q: rank.get(q.time, -1))   # stable: None first
    running = 0
    for q in chronological:
        if q.verb in STATE_VERBS:
            running = _verb_sign(q.verb) * q.amount      # a stated state replaces the running total
        else:
            running += _verb_sign(q.verb) * q.amount     # a change accumulates onto it
    return running


@dataclass(frozen=True)
class Quantity:
    """A counted-amount fact: "Tom has 3 apples" → Quantity("tom", ("apple", frozenset()), 3.0,
    "have"). The measure is a collection descriptor ("3 RED apples"); `verb` (the lemma) carries the
    arithmetic intent (have=baseline, buy=gain, lose=loss) used when folding. Arithmetic analogue of
    Predication."""
    entity: str
    measure: tuple
    amount: float
    verb: str = "have"
    time: str = None      # the time NODE this value is valid at (a `time` unit's `associates` binding);
                          # None = timeless / always-valid. Order between nodes lives in `Before` facts.


@dataclass(frozen=True)
class Before:
    """A temporal ORDER fact: `Before("monday", "tuesday")` means the Monday-node is earlier than the
    Tuesday-node. RELATIONAL (read off the text's after/before cues), never a calendar rank — days
    wrap. Transitive; the closure is derived by the synthesis loop, not assumed."""
    earlier: str
    later: str


@dataclass(frozen=True)
class Frequency:
    """A how-many-times fact: "Mary jumps twice" → Frequency("mary", "jump", 2). Counts OCCURRENCES of
    an action (the verb), distinct from Quantity (a count of a noun)."""
    entity: str
    action: str
    count: float


def _frequency(reading: Reading):
    """The frequency a "<entity> <verb>s <multiplier>" statement asserts, or None — a multiplier that
    scales a verb whose subject is known. ("N times" — the frequency-noun form — comes later.)"""
    mult = next((u for u in reading.units if "multiplier" in u.features.get("senses", [])), None)
    if mult is None:
        return None
    verb = _follow(mult, "scales")
    if verb is None or "verb" not in verb.features.get("senses", []):   # scaling a quantifier instead
        return None                                                     # ("twice as many") → not a freq
    subject = _follow(verb, "subject")
    if subject is None:
        return None
    return Frequency(_entity(subject), _lemma(verb.surface), mult.features["value"])


def _relational_quantity(reading: Reading, context: list):
    """A DERIVED quantity — "Mary has twice as many" → `factor × referent`, the referent being the most
    recent prior Quantity (resolved like coreference, but for a quantity). The referent supplies the
    measure; the multiplier supplies the factor. None if there's no multiplier-on-a-quantifier or no
    referent to scale."""
    mult = next((u for u in reading.units if "multiplier" in u.features.get("senses", [])), None)
    if mult is None:
        return None
    scaled = _follow(mult, "scales")                     # must scale "as many" (a quantifier), not a verb
    if scaled is None or "quantifier" not in scaled.features.get("senses", []):
        return None
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    # the referent: a NAMED quantity ("half as many as TIM" — a name other than the subject), else the
    # most recent prior quantity ("twice as many" — implicit, like coreference).
    named = [u for u in reading.units if "name" in u.features.get("senses", []) and u is not subject]
    if named:
        ref_ent = _entity(named[-1])
        referent = next((f for f in reversed(context or [])
                         if isinstance(f, Quantity) and f.entity == ref_ent), None)
    else:
        referent = next((f for f in reversed(context or []) if isinstance(f, Quantity)), None)
    if referent is None:
        return None
    return Quantity(_entity(subject), referent.measure,
                    mult.features["value"] * referent.amount, _lemma(verbs[0].surface))


def _partitive_quantity(reading: Reading, context: list):
    """A PARTITIVE quantity — "loses half OF them" → `fraction × current holding`, a relational CHANGE
    (not a Frequency: the "of" makes "half" a fraction of a set, not a verb multiplier). The referent is
    a noun after "of", else (a pronoun like "them"/"it") the subject's most-recent measure; its current
    holdings supply the base. None if there's no referent holding to take a fraction of."""
    units = reading.units
    mult = next((u for u in units if "multiplier" in u.features.get("senses", [])), None)
    if mult is None:
        return None
    mi = units.index(mult)
    if mi + 1 >= len(units) or _key(units[mi + 1].surface) != "of":   # require "<multiplier> OF …"
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    ent = _entity(subject)
    noun = next((u for u in units[mi + 2:] if "noun" in u.features.get("senses", [])), None)
    if noun is not None:                                  # "half of the APPLES" → that measure
        measure = _descriptor(noun)
    else:                                                 # "half of THEM" → the subject's latest measure
        recent = next((f for f in reversed(context or [])
                       if isinstance(f, Quantity) and f.entity == ent), None)
        measure = recent.measure if recent is not None else None
    if measure is None:
        return None
    holding = _holdings(ent, measure, context or [])
    if holding is None:
        return None
    return Quantity(ent, measure, mult.features["value"] * holding,
                    _lemma(verbs[0].surface), _time_at(reading, verbs[0]))


def _comparative_quantity(reading: Reading, context: list):
    """A DERIVED quantity from a comparison — "Tim has 30 less apples than Martha" → `referent ± N`,
    the referent being the NAMED standard's holdings (resolved by name via `than`). None unless there's
    a comparative + a value + a known standard. "more" → +N; "less"/"fewer" → −N."""
    comp = next((u for u in reading.units if "comparative" in u.features.get("senses", [])), None)
    if comp is None:
        return None
    if any(_key(u.surface) == "times" for u in reading.units):   # "25 TIMES more" is multiplicative,
        return None                                              # not our additive ±N — don't guess
    standard = _follow(comp, "standard")
    value_unit = next((u for u in reading.units if "value" in u.features.get("senses", [])), None)
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if standard is None or value_unit is None or subject is None:
        return None
    if "measure" in value_unit.features:                 # "$10 more than William" — money value
        measure = value_unit.features["measure"]
    else:                                                # "30 less apples than Martha" — value+noun
        noun = _follow(value_unit, "quantifies")
        if noun is None:
            return None
        measure = _descriptor(noun)
    referent = _holdings(_entity(standard), measure, context)
    if referent is None:
        return None                                      # standard's count unknown (e.g. forward ref)
    sign = COMPARATIVE_SIGN.get(_key(comp.surface), 1)   # "younger"/"less" → −, "older"/"more" → +
    return Quantity(_entity(subject), measure, referent + sign * value_unit.features["value"],
                    _lemma(verbs[0].surface))


@dataclass(frozen=True)
class Rate:
    """A per-unit rate: "Weng earns $12 per hour" → Rate("weng", 12, MONEY, "hour", "earn"). The total
    is `amount × (the quantity of per_unit)` — composed at answer time, not stored."""
    entity: str
    amount: float
    measure: tuple
    per_unit: str
    verb: str


def _rate(reading: Reading):
    """The rate a "<entity> <verb>s $R per <unit>" statement asserts, or None. Slice 1 is money rates
    (the rate value carries its own MONEY measure); "N items per unit" comes later."""
    per = next((u for u in reading.units if "per" in u.features.get("senses", [])), None)
    if per is None:
        return None
    rate_val, unit = _follow(per, "rate"), _follow(per, "unit")
    if rate_val is None or unit is None or "measure" not in rate_val.features:
        return None
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    return Rate(_entity(subject), rate_val.features["value"], rate_val.features["measure"],
                _descriptor(unit)[0], _lemma(verbs[0].surface))


def _percent_quantity(reading: Reading, context: list):
    """A percentage OF A HOLDING — "puts 20% more money" → 0.2 × the matching prior quantity. The
    percent has NO explicit base value (that case, "20% of 50", is arithmetic) — its base is the
    referent: the most recent prior Quantity whose measure a noun here names."""
    pct = next((u for u in reading.units if "percent" in u.features.get("senses", [])), None)
    if pct is None or _follow(pct, "base") is not None:
        return None
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    measures = {_descriptor(u) for u in reading.units if "noun" in u.features.get("senses", [])}
    referent = next((f for f in reversed(context or [])
                     if isinstance(f, Quantity) and f.measure in measures), None)
    if referent is None:
        return None
    return Quantity(_entity(subject), referent.measure,
                    pct.features["value"] * referent.amount, _lemma(verbs[0].surface))


def _distributed_quantity(reading: Reading):
    """A DISTRIBUTED quantity — "gave 4 stickers EACH to 3 friends" → 4 × 3 = 12. "each" makes the
    per-recipient amount (the value-noun just before it) multiply by the recipient count (the other
    value). None unless "each" sits after a noun with two value-noun pairs around it."""
    units = reading.units
    each = next((u for u in units if _key(u.surface) == "each"), None)
    if each is None:
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    ei = units.index(each)
    if subject is None or ei == 0 or "noun" not in units[ei - 1].features.get("senses", []):
        return None
    per_noun = units[ei - 1]                              # "4 STICKERS each" — the distributed measure
    per_val = next((u for u in units if "value" in u.features.get("senses", [])
                    and _follow(u, "quantifies") is per_noun), None)
    count_val = next((u for u in units if "value" in u.features.get("senses", []) and u is not per_val
                      and _follow(u, "quantifies") is not None), None)   # "to 3 FRIENDS" — the count
    if per_val is None or count_val is None:
        return None
    total = per_val.features["value"] * count_val.features["value"]
    return Quantity(_entity(subject), _descriptor(per_noun), total, _lemma(verbs[0].surface))


def _time_at(reading: Reading, verb_unit) -> str:
    """The time NODE a value under `verb_unit` is valid at — the `time` unit whose `associates`
    obligation binds this verb (so value@time is read off the binding, not scanned ad-hoc). None if
    no time timestamps this verb (a timeless value)."""
    for u in reading.units:
        if "time" in u.features.get("senses", []) and _follow(u, "associates") is verb_unit:
            return _key(u.surface)
    return None


def _temporal_order(reading: Reading):
    """The `Before` fact a "<timeA> is after/before <timeB>" statement asserts, or None — read off the
    discharged `orders` binding plus the after/before cue (the cue fixes the direction). Contingent on
    the obligation actually being met, so it stays a binding-derived fact."""
    times = [u for u in reading.units if "time" in u.features.get("senses", [])]
    if len(times) != 2 or _follow(times[0], "orders") is None:
        return None
    a, b = _key(times[0].surface), _key(times[1].surface)             # text order
    if any(_key(u.surface) == "after" for u in reading.units):        # "A is after B" → B earlier
        return Before(b, a)
    if any(_key(u.surface) == "before" for u in reading.units):       # "A is before B" → A earlier
        return Before(a, b)
    return None


def _quantity(reading: Reading):
    """The quantities a "<entity> <verb>s <amount> …" statement asserts — ONE per value: a
    self-measuring money value ("earns $12") or a value quantifying a noun ("buys 2 apples"). A list,
    so a CONJOINED object yields several ("has 3 apples and 2 oranges" → two). The verb lemma carries
    the polarity. [] if there's no single verb with a subject."""
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    if len(verbs) != 1:
        return []
    subject = _follow(verbs[0], "subject")
    if subject is None:
        return []
    # Two ADJACENT bare values ("10 $20" = ten twenty-dollar bills) are a compound denomination we
    # don't parse — never a genuine conjoined object ("3 apples AND 2 oranges" separates them). Decline.
    vi = [k for k, u in enumerate(reading.units) if "value" in u.features.get("senses", [])]
    if any(b - a == 1 for a, b in zip(vi, vi[1:])):
        return []
    ent, lemma, when = _entity(subject), _lemma(verbs[0].surface), _time_at(reading, verbs[0])
    out = []
    for u in reading.units:
        if "value" not in u.features.get("senses", []):
            continue
        if "measure" in u.features:                      # "$12" — a self-measuring money value
            out.append(Quantity(ent, u.features["measure"], u.features["value"], lemma, when))
        else:
            noun = _follow(u, "quantifies")              # "3 APPLES" — the value's measure noun
            if noun is not None:
                out.append(Quantity(ent, _descriptor(noun), u.features["value"], lemma, when))
    return out


def _predication(reading: Reading):
    """The (subject, predicate, polarity, scope) a simple statement asserts, or None — the one verb
    and its subject; negated if a negator is present, universal if an all/every quantifier scopes it."""
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    if len(verbs) != 1:
        return None
    subject = _follow(verbs[0], "subject")
    if subject is None:
        return None
    obj = _follow(verbs[0], "dobj")
    return Predication(_descriptor(subject), _lemma(verbs[0].surface),
                       _verb_negated(verbs[0], reading), _universal(verbs[0], reading),
                       _descriptor(obj) if obj is not None else None)


@dataclass(frozen=True)
class Membership:
    """A known fact "<individual> is a <collection>": a named individual belongs to a collection,
    e.g. "Bob is a student" → Membership("bob", ("student", frozenset())). `negated` for "is not"."""
    member: str
    collection: tuple
    negated: bool = False


def _has_copula(reading: Reading) -> bool:
    return _copula_unit(reading) is not None


def _has_sense(unit, sense) -> bool:
    return unit is not None and sense in unit.features.get("senses", [])


def _copula_args(reading: Reading):
    """(subject, complement) units bound by the copula — read from its BINDINGS, not by counting.
    Works for statements ("Bob is a student") and fronted questions ("Is Bob a student?")."""
    cop = _copula_unit(reading)
    if cop is None:
        return (None, None)
    return (_follow(cop, "subject"), _follow(cop, "complement"))


def _membership(reading: Reading):
    """The membership a copula statement asserts ("X is a Y"), or None — read off the copula's bound
    subject (a name) and complement (a noun); polarity from a negation bound to the copula."""
    if any("value" in u.features.get("senses", []) for u in reading.units):
        return None                                      # "Kris is 24 years old" is a QUANTITY, not a
    subj, comp = _copula_args(reading)                   # membership — defer to the quantity extractors
    if _has_sense(subj, "name") and _has_sense(comp, "noun"):
        return Membership(_key(subj.surface), _descriptor(comp),
                          _verb_negated(_copula_unit(reading), reading))
    return None


@dataclass(frozen=True)
class Subset:
    """A universal relation between two collections. Positive ("Every X is a Y") → `sub` ⊆ `sup`:
    every member of `sub` is a member of `sup`. Negated ("No X are Y") → `sub` and `sup` are
    DISJOINT: no member of one is a member of the other."""
    sub: tuple
    sup: tuple
    negated: bool = False


def _subset(reading: Reading):
    """The subset/disjointness a universal copula statement asserts, or None — read off the copula's
    bound subject and complement (both nouns), with an all/every/no quantifier present. "Every X is a
    Y" → Subset(X, Y); "No X are Y" → Subset(X, Y, negated=True) (disjoint)."""
    has_universal = any("quantifier" in u.features.get("senses", []) and
                        _key(u.surface) in ("every", "all", "no") for u in reading.units)
    subj, comp = _copula_args(reading)
    if has_universal and _has_sense(subj, "noun") and _has_sense(comp, "noun"):
        return Subset(_descriptor(subj), _descriptor(comp), _verb_negated(_copula_unit(reading), reading))
    return None


def _property(reading: Reading):
    """A predicative-adjective fact ("The apple is red"), or None — a copula, one subject (noun or
    name), and one PREDICATIVE adjective (one that didn't merge into a noun). The predicate is the
    adjective; polarity from any negator. Distinct from attributive "red apple" (which merges)."""
    subj, comp = _copula_args(reading)
    if (_has_sense(subj, "noun") or _has_sense(subj, "name")) and _has_sense(comp, "adjective"):
        cop = _copula_unit(reading)
        return Predication(_descriptor(subj), _key(comp.surface),
                           _verb_negated(cop, reading), _universal(cop, reading))
    return None


def _collections_of(member: str, context: list) -> set:
    """Every collection `member` belongs to — directly, and transitively up POSITIVE Subset edges."""
    frontier = [m.collection for m in context
                if isinstance(m, Membership) and m.member == member and not m.negated]
    seen = set()
    while frontier:
        c = frontier.pop()
        if c in seen:
            continue
        seen.add(c)
        frontier += [f.sup for f in context if isinstance(f, Subset) and not f.negated and f.sub == c]
    return seen


def _is_member(member: str, target: tuple, context: list) -> bool:
    """Is `member` in collection `target`? True iff `target` is one of its (transitive) collections."""
    return target in _collections_of(member, context)


def _disjoint(a: tuple, b: tuple, context: list) -> bool:
    """Are collections `a` and `b` known disjoint? A negated Subset between them, either direction
    ("No X are Y" makes X and Y mutually exclusive)."""
    return any(isinstance(f, Subset) and f.negated and {f.sub, f.sup} == {a, b} for f in context)


@dataclass(frozen=True)
class Relation:
    """A comparative relation between two individuals, e.g. "Alice is taller than Bob" →
    Relation("alice", "taller", "bob"). Transitive (see _related)."""
    subject: str
    relation: str
    object: str


def _relation(reading: Reading):
    """The comparison a "<A> is <comparative> than <B>" statement asserts, or None — read off the
    bindings: the copula's subject (A), its complement (the comparative), and the comparative's
    bound `standard` (B)."""
    subj, comp = _copula_args(reading)
    if not _has_sense(comp, "comparative"):
        return None
    standard = _follow(comp, "standard")
    if subj is None or standard is None:
        return None
    return Relation(_key(subj.surface), _key(comp.surface), _key(standard.surface))


def _related(subj: str, rel: str, obj: str, context: list) -> bool:
    """Does `subj rel obj` hold, transitively? Reachable from subj to obj along `rel` edges (start
    from subj's direct successors, so nothing is its own comparative)."""
    frontier = [f.object for f in context if isinstance(f, Relation) and f.relation == rel
                and f.subject == subj]
    seen = set()
    while frontier:
        x = frontier.pop()
        if x == obj:
            return True
        if x in seen:
            continue
        seen.add(x)
        frontier += [f.object for f in context if isinstance(f, Relation) and f.relation == rel
                     and f.subject == x]
    return False


def _follow(unit, need):
    """The target of `unit`'s met obligation named `need`, or None."""
    if unit is None:
        return None
    for o in unit.features.get("obligations", []):
        if o.need == need and o.met:
            return o.target
    return None


def _verdict(yes: bool, no: bool) -> str:
    """Fold derivable-true / derivable-false into one verdict: both → "conflict" (the premises
    contradict), true → "yes", false → "no", neither → "undetermined" (we lack the information)."""
    if yes and no:
        return "conflict"
    return "yes" if yes else "no" if no else "undetermined"


def _predication_verdict(subject: tuple, predicate: str, context: list, obj: tuple = None) -> str:
    """The verdict for "does <subject> <predicate> [<object>]?". A universal fact transfers iff the
    subject is a member of its collection (transitively via Subset); the object must match too. yes
    AND no both derivable → "conflict"."""
    member = subject[0]
    preds = [f for f in context if isinstance(f, Predication)
             and f.predicate == predicate and f.object == obj]
    yes = Predication(subject, predicate, object=obj) in context \
        or any(f.universal and not f.negated and _is_member(member, f.subject, context) for f in preds)
    no = Predication(subject, predicate, negated=True, object=obj) in context \
        or any(f.universal and f.negated and _is_member(member, f.subject, context) for f in preds)
    return _verdict(yes, no)


def _question_subjects(reading: Reading, aux):
    """The entities a fronted question asks about + its verb lemma. Reads the verb's bound `subject`
    (the STRUCTURAL binding the parse produced — not a positional rescan) when it resolves; falls back
    to the collection between aux and verb only for a conjoined subject ("Tom and Mary") that the
    single-target `subject` obligation can't yet bind."""
    verb = _follow(aux, "asks_predication")
    q_verb = _lemma(verb.surface) if verb is not None else "have"
    subject = _follow(verb, "subject") if verb is not None else None
    if subject is not None:                              # a group's `members`, else the single entity
        return (subject.features.get("members") or [_entity(subject)]), q_verb
    units = reading.units
    ai, vi = units.index(aux), (units.index(verb) if verb is not None else len(units))
    entities = []                                        # flatten any group's members (comma-lists)
    for u in units[ai + 1:vi]:
        if any(s in u.features.get("senses", []) for s in ("name", "noun", "pronoun")):
            entities.extend(u.features.get("members") or [_entity(u)])
    return entities, q_verb


def answer(question: str, context: list, discourse: list = None):
    """Answer a question against a context of facts. A yes/no question ("Did the student pass?")
    returns "yes"/"no" by checking a Predication; a count question ("How many red apples?") returns
    an int by counting matching instances. Returns a list of Deficiency (what to ASK) when the
    question can't be answered. Reuses the same parse; dispatches on the interrogative kind. `discourse`
    resolves the question's own pronouns ("how much does SHE have?" → the entity she refers to)."""
    r = analyze(question)
    if discourse is not None:
        _resolve_references(r, discourse)
    unknown = [d for d in deficiencies(r) if d.kind == "unknown-word"]
    if unknown:
        return unknown                                   # don't guess about words we don't know

    op_unit = next((u for u in r.units if "operator" in u.features.get("senses", [])), None)
    if op_unit is not None:                              # arithmetic: "What is 3 plus 2?"
        a, b = _follow(op_unit, "left"), _follow(op_unit, "right")
        if a is None or b is None:
            return [Deficiency("unanswerable", op_unit, "I couldn't read the arithmetic")]
        from reasoning import translate                  # reuse the audited safe evaluator
        val = translate.run_program(
            f"answer = {a.features['value']} {op_unit.features['op']} {b.features['value']}")
        return int(val) if val == int(val) else val
    percent = next((u for u in r.units if "percent" in u.features.get("senses", [])), None)
    if percent is not None:                              # "What is 20% of 50?" → fraction × base
        base = _follow(percent, "base")
        if base is None or "value" not in base.features:
            return [Deficiency("unanswerable", percent, "I couldn't find what the percentage is of")]
        from reasoning import translate
        val = translate.run_program(
            f"answer = {percent.features['value']} * {base.features['value']}")
        return int(val) if val == int(val) else val
    if any("interrogative" in u.features.get("senses", []) for u in r.units) \
            and any(_key(u.surface) == "old" for u in r.units):   # "How old is/will X be?" → age (years)
        shift = ("next", "ago", "last", "later")         # a future/past age SHIFT we don't compute
        if any(_key(u.surface) in shift or any(a in shift for a in u.features.get("attributes", []))
               for u in r.units):                        # "next year" merges → "next" is an attribute
            return "undetermined"
        aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
        if aux is not None:
            entities, _ = _question_subjects(r, aux)
        else:                                            # "How old is X?" — copula, subject is fronted
            subj, _ = _copula_args(r)
            entities = [_entity(subj)] if subj is not None else []
        total = 0
        for ent in entities:
            h = _holdings(ent, ("year", frozenset()), context)
            if h is None:
                return "undetermined"
            total += h
        return (int(total) if total == int(total) else total) if entities else "undetermined"

    who = next((u for u in r.units if "wh_subject" in u.features.get("senses", [])), None)
    if who is not None:                                  # "Who <predicate>?" → subjects with it
        verb = _follow(who, "asks_predicate")
        if verb is None:
            return [Deficiency("unanswerable", who, "I couldn't tell what is being asked")]
        pred = _lemma(verb.surface)
        subs = sorted({f.subject[0] for f in context if isinstance(f, Predication)
                       and f.predicate == pred and not f.negated})
        return ", ".join(subs) if subs else "(no one)"
    what = next((u for u in r.units if "wh_value" in u.features.get("senses", [])), None)
    if what is not None:
        aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
        if aux is not None:                              # "What does <subj> <verb>?" → the objects
            verb = _follow(aux, "asks_predication")
            subject = _follow(verb, "subject") if verb else None
            if subject is None:
                return [Deficiency("unanswerable", what, "I couldn't tell what is being asked")]
            subj, pred = _descriptor(subject), _lemma(verb.surface)
            objs = sorted({f.object[0] for f in context if isinstance(f, Predication)
                           and f.subject == subj and f.predicate == pred and f.object and not f.negated})
            return ", ".join(objs) if objs else "(nothing)"
        name = _follow(what, "asks_individual")          # "What is <name>?" → its collections
        if name is None:
            return [Deficiency("unanswerable", what, "I couldn't tell what is being asked")]
        member = _key(name.surface)
        cols = sorted({m.collection[0] for m in context
                       if isinstance(m, Membership) and m.member == member and not m.negated})
        return ", ".join(cols) if cols else "(unknown)"
    which = next((u for u in r.units if "wh_which" in u.features.get("senses", [])), None)
    if which is not None:        # "Which <noun> <predicate>?" → members of the noun that satisfy it
        coll_unit = _follow(which, "restricts")
        adjs = [u for u in r.units if "adjective" in u.features.get("senses", [])]
        verbs = [u for u in r.units if "verb" in u.features.get("senses", [])
                 and _key(u.surface) not in ("is", "are", "was", "were")]
        pred = _key(adjs[0].surface) if adjs else (_lemma(verbs[0].surface) if verbs else None)
        if coll_unit is None or pred is None:
            return [Deficiency("unanswerable", which, "I couldn't tell what is being asked")]
        coll = _descriptor(coll_unit)
        hits = sorted({f.subject[0] for f in context if isinstance(f, Predication)
                       and f.predicate == pred and not f.negated and f.object is None
                       and _is_member(f.subject[0], coll, context)})
        return ", ".join(hits) if hits else "(none)"

    qcomp = next((u for u in r.units if _key(u.surface) in ("more", "fewer")), None)
    if qcomp is not None and any(_key(u.surface) == "than" for u in r.units):  # "more X than Y?" count
        ns = sorted((u for u in r.units if "noun" in u.features.get("senses", [])),
                    key=lambda u: u.words[0].index)
        if len(ns) >= 2:
            counts = [sum(1 for inst in context if isinstance(inst, tuple)
                          and _matches(_descriptor(n), inst)) for n in ns[:2]]
            more = counts[0] > counts[1] if _key(qcomp.surface) == "more" else counts[0] < counts[1]
            return "yes" if more else "no"

    # The copula yes/no questions — but NOT when a "how many" interrogative is present: "How many
    # apples ARE LEFT?" is a count question, not the property question "are apples left?".
    has_interrogative = any("interrogative" in u.features.get("senses", []) for u in r.units)
    subj, comp = _copula_args(r)
    if not has_interrogative and _has_sense(comp, "comparative"):  # "Is A <comparative> than B?"
        standard = _follow(comp, "standard")
        if subj is not None and standard is not None:
            a, rel, b = _key(subj.surface), _key(comp.surface), _key(standard.surface)
            return _verdict(_related(a, rel, b, context),
                            _related(b, rel, a, context))   # B>A ⇒ NOT A>B (asymmetry)
    if not has_interrogative and _has_sense(subj, "name") and _has_sense(comp, "noun"):  # "Is X a Y?"
        member, target = _key(subj.surface), _descriptor(comp)
        yes = _is_member(member, target, context)
        no = any(isinstance(m, Membership) and m.member == member         # explicitly "not a Y", or
                 and m.collection == target and m.negated for m in context) \
            or any(_disjoint(c, target, context) for c in _collections_of(member, context))  # disjoint
        return _verdict(yes, no)
    if not has_interrogative:
        prop = _property(r)                              # "Is <subject> <adjective>?" property question
        if prop is not None:                             # a property is a Predication (adjective as
            return _predication_verdict(prop.subject, prop.predicate, context)   # predicate) → verdict

    aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
    if aux is not None and not has_interrogative:        # "Did X <verb> [Y]?" → yes / no / undetermined
        verb = _follow(aux, "asks_predication")          # did → the verb being asked about (an
                                                         # interrogative present → it's a "how many …
                                                         # does …?" count/quantity question instead)
        subject = _follow(verb, "subject") if verb else None
        if subject is None:
            return [Deficiency("unanswerable", aux, "I couldn't tell what is being asked")]
        obj = _follow(verb, "dobj")
        return _predication_verdict(_descriptor(subject), _lemma(verb.surface), context,
                                    _descriptor(obj) if obj is not None else None)

    interro = next((u for u in r.units if "interrogative" in u.features.get("senses", [])), None)
    if interro is None:
        return [Deficiency("not-a-question", interro, "this is not a question I can answer")]
    collection = _follow(_follow(interro, "queries"), "collection")   # how → many → its collection
    query = _descriptor(collection) if collection is not None else None
    much = next((u for u in r.units if _key(u.surface) == "much"), None)
    if much is not None:                                 # "how much" → a MONEY question, UNLESS a mass
        coll = _follow(much, "collection")               # noun sits immediately after it ("how much
        mi = r.units.index(much)                         # WATER"); a non-adjacent noun ("how much did
        if coll is None or r.units.index(coll) != mi + 1:   # GRANDPARENTS give") is the subject, not it
            query = MONEY
    remainder = any(_key(u.surface) in _REMAINDER for u in r.units)
    aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
    comp = next((u for u in r.units if "comparative" in u.features.get("senses", [])), None)
    if comp is not None and aux is not None and query is not None:    # "How much more <measure> does
        standard = _follow(comp, "standard")                         # A have than B?" → A's − B's
        subjects, _ = _question_subjects(r, aux)                     # A = the verb's bound subject
        if standard is not None and len(subjects) == 1:
            a = _holdings(subjects[0], query, context)
            b = _holdings(_entity(standard), query, context)
            if a is not None and b is not None:
                diff = a - b if _key(comp.surface) == "more" else b - a   # "less"/"fewer" → B − A
                return int(diff) if diff == int(diff) else diff
        return "undetermined"
    if aux is not None and query is not None and query[0] == "time":   # "How many times does X <verb>?"
        subjects, action = _question_subjects(r, aux)
        if subjects:
            counts = [f.count for f in context if isinstance(f, Frequency)
                      and f.entity == subjects[0] and f.action == action]
            if counts:
                return int(counts[0]) if counts[0] == int(counts[0]) else counts[0]
            return "undetermined"
    if aux is not None and query is not None:            # "How many <measure> do <entities> <verb>?"
        entities, q_verb = _question_subjects(r, aux)    # the verb's bound subject(s), structurally
        q_time = next((_key(u.surface) for u in r.units if "time" in u.features.get("senses", [])), None)
        if entities:
            total = 0
            for ent in entities:
                if q_verb in ("have", "own") and q_time is not None:   # TIME-AWARE: the value VALID at
                    h = _holdings_at(ent, query, context, q_time)      # the queried time, folded along
                    if h is None:                                      # the timeline (state vs change)
                        return "undetermined"
                    total += h
                    continue
                qs = [f for f in context if isinstance(f, Quantity)
                      and f.entity == ent and f.measure == query]
                # POLARITY IS RELATIVE TO THE QUESTION: "how many do you HAVE" folds holdings
                # (baseline + gains - losses); "how many did you SELL" COUNTS the sell events (+).
                if q_verb in ("have", "own"):
                    folded = [_verb_sign(f.verb) * f.amount for f in qs]
                else:
                    folded = [f.amount for f in qs if f.verb == q_verb]   # events of THIS verb
                if folded:
                    total += sum(folded)
                    continue
                rate = next((f for f in context if isinstance(f, Rate) and f.entity == ent     # rate ×
                             and f.measure == query and f.verb == q_verb), None)                # per-unit
                per_qty = _holdings(ent, (rate.per_unit, frozenset()), context) if rate else None
                if per_qty is None:
                    return "undetermined"                # no relevant quantity/rate for this conjunct
                total += rate.amount * per_qty
            return int(total) if total == int(total) else total
    if remainder:                                        # "How many [<measure>] are left?" → the net
        measure = query                                  # HOLDINGS of the (implicit) owner — no aux/entity
        if measure is None:                              # "how many are left" — measure from the latest qty
            recent = next((f for f in reversed(context) if isinstance(f, Quantity)), None)
            measure = recent.measure if recent is not None else None
        if measure is not None:
            owners = {f.entity for f in context if isinstance(f, Quantity) and f.measure == measure}
            if len(owners) == 1:                         # one holder → its net holdings; else ambiguous
                total = _holdings(next(iter(owners)), measure, context)
                if total is not None:
                    return int(total) if total == int(total) else total
        return "undetermined"
    if query is None:
        return [Deficiency("unanswerable", interro, "I couldn't tell what to count")]
    if not any(isinstance(inst, tuple) for inst in context):
        return "undetermined"        # no instance data to count — we don't know (never a fabricated 0)
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


def _verdict_steps(subject: tuple, predicate: str, context: list, q: str, fact: str,
                   obj: tuple = None) -> list:
    """Derivation steps for a (subject, predicate[, object]) yes/no query — shared by the verb (aux)
    and the adjective (property) branches. `q` opens the question ("does"/"is"); `fact` is how a
    stated fact reads ("" for a verb, "is" for an adjective)."""
    def line(text):
        return " ".join(text.split())                    # tidy doubled spaces from an empty `fact`
    obj_s = f" {obj[0]}" if obj else ""
    verdict = _predication_verdict(subject, predicate, context, obj)
    steps = [line(f"{q} {subject[0]!r} {predicate}{obj_s}? → {verdict}")]
    if Predication(subject, predicate, object=obj) in context:
        steps.append(line(f"directly stated: {subject[0]} {fact} {predicate}{obj_s}"))
    if Predication(subject, predicate, negated=True, object=obj) in context:
        steps.append(line(f"directly stated: {subject[0]} {fact} NOT {predicate}{obj_s}"))
    for f in context:                                    # universal facts that reach the subject
        if isinstance(f, Predication) and f.universal and f.predicate == predicate and f.object == obj \
                and _is_member(subject[0], f.subject, context):
            steps.append(line(f"{subject[0]!r} is a {f.subject[0]}; "
                              f"{'no ' if f.negated else 'all '}{f.subject[0]} {fact} {predicate}"))
    if verdict == "undetermined":
        steps.append("nothing in context settles it — lack of information")
    if verdict == "conflict":
        steps.append("both true and false are derivable — the premises contradict")
    return steps


def explain(question: str, context: list) -> list:
    """The human-readable derivation steps behind `answer(question, context)` — what was queried,
    which facts were checked, and how the verdict follows."""
    r = analyze(question)
    unknown = [d.message for d in deficiencies(r) if d.kind == "unknown-word"]
    if unknown:
        return unknown
    subj, comp = _copula_args(r)
    if _has_sense(comp, "comparative"):                  # comparative derivation
        standard = _follow(comp, "standard")
        if subj is not None and standard is not None:
            a, rel, b = _key(subj.surface), _key(comp.surface), _key(standard.surface)
            fwd, rev = _related(a, rel, b, context), _related(b, rel, a, context)
            return [f"is {a!r} {rel} than {b!r}? → {_verdict(fwd, rev)}",
                    f"{a} {rel} than {b} (transitively)? {'yes' if fwd else 'no'}",
                    f"{b} {rel} than {a}, so NOT the reverse (asymmetry)? {'yes' if rev else 'no'}"]
    if _has_sense(subj, "name") and _has_sense(comp, "noun"):    # membership derivation
        member, target = _key(subj.surface), _descriptor(comp)
        yes = _is_member(member, target, context)
        no = any(isinstance(m, Membership) and m.member == member and m.collection == target
                 and m.negated for m in context) \
            or any(_disjoint(c, target, context) for c in _collections_of(member, context))
        cols = sorted(c[0] for c in _collections_of(member, context))
        steps = [f"is {member!r} a {target[0]}? → {_verdict(yes, no)}",
                 f"{member!r} is a: {', '.join(cols) or 'nothing known'}"]
        if no:
            steps.append(f"and it is stated/derivable that {member!r} is NOT a {target[0]}")
        return steps
    prop = _property(r)                                  # property derivation (adjective predicate)
    if prop is not None:
        return _verdict_steps(prop.subject, prop.predicate, context, "is", "are")

    aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
    if aux is not None:                                  # yes/no derivation
        verb = _follow(aux, "asks_predication")
        subject = _follow(verb, "subject") if verb else None
        if subject is None:
            return ["couldn't tell what is being asked"]
        obj = _follow(verb, "dobj")
        return _verdict_steps(_descriptor(subject), _lemma(verb.surface), context, "does", "",
                              _descriptor(obj) if obj is not None else None)
    interro = next((u for u in r.units if "interrogative" in u.features.get("senses", [])), None)
    if interro is not None:                              # count derivation
        coll = _follow(_follow(interro, "queries"), "collection")
        if coll is not None:
            q = _descriptor(coll)
            n = sum(1 for inst in context if isinstance(inst, tuple) and _matches(q, inst))
            return [f"count of {q}: {n} matching instance(s)"]
    return ["not a question I can answer"]


def learn(statement: str, discourse: list = None, context: list = None) -> list:
    """Parse a statement into context facts — the one ingestion door. A count statement ("There are
    three red apples") → instances; a subject-verb statement ("The student passed") → a Predication;
    a bare collection ("red apple") → one instance. [] when nothing parses, so we never guess.

    `discourse` (entity names, most-recent LAST) threads coreference across sentences: a pronoun's
    `refers` obligation is resolved against it before extraction, and the statement's own subject is
    appended after, so a later pronoun can find it. `context` (the facts so far) lets a RELATIONAL
    quantity ("twice as many") resolve its referent quantity."""
    r = analyze(statement)
    if discourse is not None:
        _resolve_references(r, discourse)
    facts = _facts_from(r, context)
    if discourse is not None:                            # remember this statement's subject entity
        verbs = [u for u in r.units if "verb" in u.features.get("senses", [])]
        subj = _follow(verbs[0], "subject") if len(verbs) == 1 else None
        if subj is not None:
            discourse.append(_entity(subj))
    return facts


# --- the read–SYNTHESIZE loop: reading BINDS what's present, synthesis DERIVES what's implied --------
# Reading discharges obligations by binding to words that are THERE. Some obligations have no such word
# — their target exists only by IMPLICATION (a time never explicitly ordered but derivable by
# transitivity; later: an implied time, an "earlier value" standard). The synthesize() driver runs to a
# fixpoint: it dispatches the registered synthesizers (keyed by the fact KIND they produce) over the
# facts gathered so far, folds in whatever they derive, and repeats. The OUTSTANDING obligations are the
# agenda (passed to each synthesizer). New reasoning is added by REGISTERING a synthesizer — never a
# one-off check. Anything still unfilled after the fixpoint stays a deficiency, never a guess.
def _before(a: str, b: str, facts: list) -> bool:
    """True if time-node `a` is earlier than `b` under the TRANSITIVE closure of the Before facts
    (relational order — no calendar rank). DFS over Before edges."""
    adj = {}
    for f in facts:
        if isinstance(f, Before):
            adj.setdefault(f.earlier, set()).add(f.later)
    seen, stack = set(), [a]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y == b:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def _synth_transitive_order(facts: list, readings: list, agenda: set) -> list:
    """One-hop transitive ordering: `a before b` ∧ `b before c` ⟹ `a before c`. Iterated to full
    closure by the driver's fixpoint, so a chain (Tue→Sat→Mon→Wed) is filled over successive rounds —
    front-loaded multi-step reasoning. (Registered for the `orders`/Before kind.)"""
    befores = [f for f in facts if isinstance(f, Before)]
    out = []
    for f1 in befores:
        for f2 in befores:
            if f1.later == f2.earlier and f1.earlier != f2.later:
                cand = Before(f1.earlier, f2.later)
                if cand not in facts and cand not in out:
                    out.append(cand)
    return out


def _synth_narrative_order(facts: list, readings: list, agenda: set) -> list:
    """Values stated in SEQUENCE are narrated chronologically: consecutive DISTINCT value@time nodes (in
    the order their facts appear) → a Before edge — UNLESS those two times are already ordered (so an
    explicit/transitive order always wins, and we never add a contradicting edge). The default that lets
    "On Monday … On Tuesday … On Wednesday …" be queryable without an explicit "after" in every sentence."""
    seq = []
    for f in facts:
        if isinstance(f, Quantity) and f.time is not None and f.time not in seq:
            seq.append(f.time)
    out = []
    for a, b in zip(seq, seq[1:]):
        if a != b and not _before(a, b, facts) and not _before(b, a, facts):
            cand = Before(a, b)
            if cand not in out:
                out.append(cand)
    return out


# obligation/fact KIND → derivation (facts, readings, agenda) -> new facts. Register here to add reasoning.
SYNTHESIZERS = {"orders": _synth_transitive_order, "narrative": _synth_narrative_order}


def _outstanding(readings: list) -> set:
    """The set of still-UNMET obligation needs across the readings — the synthesis agenda (what KINDS of
    fact we still need to try to derive)."""
    return {o.need for r in (readings or []) for u in r.units
            for o in u.features.get("obligations", []) if not o.met}


def synthesize(facts: list, readings: list = None) -> list:
    """The DERIVE step of the read–synthesize loop: to a fixpoint, run the registered synthesizers over
    the facts gathered so far and fold in what they imply. Returns the augmented fact list (existing +
    derived). Idempotent once closed."""
    facts = list(facts)
    changed = True
    while changed:
        changed = False
        agenda = _outstanding(readings)
        for fn in SYNTHESIZERS.values():
            for f in fn(facts, readings, agenda):
                if f not in facts:
                    facts.append(f)
                    changed = True
    return facts


def _facts_from(reading: Reading, context: list = None) -> list:
    r = reading
    to = _temporal_order(r)                              # "<time> is after/before <time>" → Before fact
    if to is not None:
        return [to]
    verbs = [u for u in r.units if "verb" in u.features.get("senses", [])]
    if len(verbs) == 1:                                  # a group-subject STATEMENT ("Tom and Mary have
        subj = _follow(verbs[0], "subject")              # 5 apples") is distributive/collective —
        if subj is not None and "members" in subj.features:   # ambiguous — so decline (no guessed fact)
            return []
    for u in r.units:                                    # count statement?
        value = NUMBERS.get(_key(u.surface))
        coll = _follow(u, "collection") if value is not None else None
        if coll is not None:
            return [_descriptor(coll)] * value
    mult = next((u for u in r.units if "multiplier" in u.features.get("senses", [])), None)
    scaled = _follow(mult, "scales") if mult is not None else None
    if scaled is not None and "quantifier" in scaled.features.get("senses", []):
        rq = _relational_quantity(r, context)            # "<entity> has twice as many" → derived, or
        return [rq] if rq is not None else []            # decline (no referent) — never a bare predication
    if mult is not None:                                 # "<mult> OF <referent>" — a partitive fraction
        mi = r.units.index(mult)                         # of a holding ("loses half of them" → 0.5×held,
        if mi + 1 < len(r.units) and _key(r.units[mi + 1].surface) == "of":   # NOT a Frequency)
            pq = _partitive_quantity(r, context)
            return [pq] if pq is not None else []        # unresolved → defer (fixpoint retries; else declines)
    s = _subset(r)                                       # "Every X is a Y" → universal membership?
    if s is not None:
        return [s]
    m = _membership(r)                                   # "<name> is a <noun>" → membership?
    if m is not None:
        return [m]
    prop = _property(r)                                  # "<subject> is <adjective>" → property?
    if prop is not None:
        return [prop]
    rel = _relation(r)                                   # "<A> is <comparative> than <B>" → relation?
    if rel is not None:
        return [rel]
    fr = _frequency(r)                                   # "<entity> <verb>s twice" → frequency?
    if fr is not None:
        return [fr]
    rate = _rate(r)                                      # "<entity> earns $12 per hour" → rate?
    if rate is not None:
        return [rate]
    cq = _comparative_quantity(r, context)               # "<A> has 30 less <X> than <B>" → derived?
    if cq is not None:
        return [cq]
    comp = next((u for u in r.units if "comparative" in u.features.get("senses", [])), None)
    if comp is not None and _follow(comp, "standard") is not None \
            and any("value" in u.features.get("senses", []) for u in r.units):
        return []                                        # a comparative-quantity structure we couldn't
        # resolve (the standard's count is unknown — a forward ref) → DEFER (don't fall through to a
        # flat _quantity); the solve() fixpoint will retry it once the standard is learned.
    pq = _percent_quantity(r, context)                   # "puts 20% more money" → 0.2 × holding?
    if pq is not None:
        return [pq]
    pct = next((u for u in r.units if "percent" in u.features.get("senses", [])), None)
    if pct is not None and _follow(pct, "base") is None \
            and len([u for u in r.units if "verb" in u.features.get("senses", [])]) == 1 \
            and any("noun" in u.features.get("senses", []) for u in r.units):
        return []                                        # percent-of-holding, referent not yet known → defer
    dq = _distributed_quantity(r)                        # "gave 4 stickers each to 3 friends" → 4×3?
    if dq is not None:
        return [dq]
    q = _quantity(r)                                     # "<entity> has 3 apples and 2 oranges" → list
    if q:
        return q
    p = _predication(r)                                  # subject-verb statement?
    if p is not None:
        return [p]
    d = collection_descriptor(r)                         # bare collection phrase?
    return [d] if d is not None else []


def _has_verb(text: str) -> bool:
    return any("verb" in u.features.get("senses", []) for u in analyze(text).units)


def _main_verb(text: str):
    """The surface of the last verb in a clause — the action to lend to an elided neighbour."""
    verbs = [u for u in analyze(text).units if "verb" in u.features.get("senses", [])]
    return verbs[-1].surface if verbs else None


def _is_elided_clause(text: str) -> bool:
    """A verb-less clause whose verb is ELIDED: it has a SUBJECT (name/noun) BEFORE its first quantity
    ("her grandparents twice as much", "Mary 5 apples") — distinct from a bare conjoined object
    ("2 oranges": the quantity comes first, no subject)."""
    units = analyze(text).units
    qi = next((k for k, u in enumerate(units)
               if any(s in u.features.get("senses", []) for s in ("value", "multiplier"))), None)
    return qi is not None and any(
        any(s in units[k].features.get("senses", []) for s in ("name", "noun")) for k in range(qi))


def _supply_verb(clause: str, verb: str):
    """Insert `verb` after the elided clause's subject (its first name/noun) — "Mary 5 apples" +
    "has" → "Mary has 5 apples". None if there's no subject word to anchor to."""
    if verb is None:
        return None
    subj = next((u for u in analyze(clause).units
                 if any(s in u.features.get("senses", []) for s in ("name", "noun"))), None)
    if subj is None:
        return None
    idx, words = subj.words[-1].index, clause.split()
    if idx >= len(words):
        return None
    return " ".join(words[:idx + 1] + [verb] + words[idx + 1:])


def _split_clauses(sentence: str) -> list:
    """Split a sentence into INDEPENDENT clauses joined by ';'/'and'/comma — ONLY where the left has its
    own verb AND the right either has one too OR is an elided clause that can BORROW the left's verb. A
    phrase-joining "and" ("3 apples and 2 oranges", "Tom and Mary") does NOT split. Recursive, so a
    chain of clauses (and a leading "If P, Q?" premise) all separate."""
    for sep in ("; ", ", and ", ", but ", ", so ", " and ", ", "):
        i = sentence.find(sep)
        while i != -1:
            left, right = sentence[:i].strip(), sentence[i + len(sep):].strip()
            if _has_verb(left):
                if _has_verb(right):
                    return _split_clauses(left) + _split_clauses(right)
                if _is_elided_clause(right):           # the right borrows the left's verb (elision)
                    synth = _supply_verb(right, _main_verb(left))
                    if synth:
                        return _split_clauses(left) + _split_clauses(synth)
            i = sentence.find(sep, i + 1)
    return [sentence]


def solve(text: str):
    """Read a whole word-problem paragraph (premises + a trailing question) and answer the question —
    threading coreference (`discourse`) and the growing `context` across the premises (split into
    independent CLAUSES first, so the fixpoint sees each fact separately). Returns the raw `answer()`
    result. The programmatic entry the GSM8K coverage harness scores. None if there's no question."""
    sentences = _sentences(text)
    if not any(s.endswith("?") for s in sentences):
        return None
    clauses = [c for s in sentences for c in _split_clauses(s)]   # split into independent clauses
    qpos = max((k for k, c in enumerate(clauses) if c.rstrip().endswith("?")), default=None)
    if qpos is None:
        return None
    question = clauses[qpos]                              # the question clause; the rest are premises
    premises = clauses[:qpos] + clauses[qpos + 1:]        # (a leading "If P," clause is now a premise)
    readings = [analyze(c) for c in premises + [question]]
    # PRECISION GUARD 1 — unrecognized word: a partial parse may have dropped the fact that mattered,
    # so we DECLINE rather than commit off a fragment ("definitive-or-decline": never answer what we
    # didn't fully read).
    if any(d.kind == "unknown-word" for r in readings for d in deficiencies(r)):
        return "undetermined"
    discourse, context = [], []
    facts_of = [None] * len(premises)                    # per-premise facts (None = not yet resolved)
    for i, p in enumerate(premises):                     # first pass: sequential, builds the discourse
        f = learn(p, discourse, context)
        if f:
            facts_of[i] = f
            context += f
    progress = True                                      # FIXPOINT: retry the unresolved (forward refs,
    while progress:                                      # chains) against the growing context
        progress = False
        for i, p in enumerate(premises):
            if facts_of[i] is None:
                f = learn(p, None, context)              # discourse already built in the first pass
                if f:
                    facts_of[i], progress = f, True
                    context += f
        derived = [f for f in synthesize(context, readings) if f not in context]  # SYNTHESIZE step:
        if derived:                                      # derive implied facts (transitive order, …) and
            context += derived                           # keep looping — one unified read–synthesize loop
            progress = True
    # GUARD 2 — a "than" comparison that collapsed to a FLAT quantity (a verbatim stated value, no
    # Relation, not a derived ±) means the relation ("25 times more stickers THAN Kristoff") was
    # dropped → we misread it. A comparative-derived quantity (amount ≠ any stated value) is OK.
    for i, p in enumerate(premises):
        if "than" in [w.strip(".,?!:;") for w in p.lower().split()]:
            facts = facts_of[i] or []
            raw = {u.features["value"] for u in readings[i].units if "value" in u.features.get("senses", [])}
            if any(isinstance(f, Quantity) and f.amount in raw for f in facts) \
                    and not any(isinstance(f, Relation) for f in facts):
                return "undetermined"
    # GUARD 3 — unused given: every NUMBER stated in the premises must surface as a fact amount; a
    # dropped number means we didn't capture the whole problem ("Oliver has 10 $20 and 3 $5 bills").
    # A premise with a comparative/multiplier is exempt — its stated value is an offset/factor consumed
    # into a DERIVED amount ("30 less … than Martha" → 38), so it won't appear verbatim.
    given = set()
    for rp in readings[:len(premises)]:
        if any(s in u.features.get("senses", []) for u in rp.units for s in ("comparative", "multiplier")) \
                or any(_key(u.surface) == "each" for u in rp.units):   # "each" multiplies its values
            continue
        given |= {u.features["value"] for u in rp.units if "value" in u.features.get("senses", [])}
    used = {f.amount for f in context if isinstance(f, (Quantity, Rate))} \
        | {f.count for f in context if isinstance(f, Frequency)}
    if given - used:
        return "undetermined"
    # GUARD 4 — a premise that STATES a number (value/percent/multiplier) but produced no quantity fact
    # dropped information we can't model ("…increases by 50% every year"; "gave his three friends total"
    # → spurious instances) → decline rather than answer off the part we did parse.
    for i, rp in enumerate(readings[:len(premises)]):
        if any(s in u.features.get("senses", []) for u in rp.units
               for s in ("value", "percent", "multiplier")) \
                and not any(isinstance(f, (Quantity, Rate, Frequency)) for f in (facts_of[i] or [])):
            return "undetermined"
    # GUARD 5 — temporal safety (now that we FOLD along the timeline, not blanket-decline). (a) A TIMED
    # premise that states a value/multiplier/percent but produced no Quantity dropped a time-tagged event
    # ("On Tuesday loses half of them" → a misparsed Frequency) — folding would silently skip it → decline.
    for i, rp in enumerate(readings[:len(premises)]):
        if any("time" in u.features.get("senses", []) for u in rp.units) \
                and any(s in u.features.get("senses", []) for u in rp.units
                        for s in ("value", "percent", "multiplier")) \
                and not any(isinstance(f, Quantity) for f in (facts_of[i] or [])):
            return "undetermined"
    # (b) ≥2 distinct times but the QUESTION names none → ambiguous which state to report → decline. With
    # a named time the time-aware fold (`_holdings_at`) answers it; an unorderable timeline returns None.
    times = {f.time for f in context if isinstance(f, Quantity) and f.time is not None}
    q_has_time = any("time" in u.features.get("senses", []) for u in readings[-1].units)
    if len(times) >= 2 and not q_has_time:
        return "undetermined"
    return answer(question, context, discourse)


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
    discourse, context = [], []
    for p in premises:                                     # thread coreference + the growing context
        context += learn(p, discourse, context)

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
    disc, ctx = [], []
    for s in premises:
        rs = analyze(s)
        _resolve_references(rs, disc)                      # resolve on THIS reading so trace shows it
        learned = _facts_from(rs, ctx)
        ctx += learned
        verbs = [u for u in rs.units if "verb" in u.features.get("senses", [])]
        subj = _follow(verbs[0], "subject") if len(verbs) == 1 else None
        if subj is not None:
            disc.append(_entity(subj))
        out += [f'STATEMENT: "{s}"'] + _trace_lines(rs) + [f"  learned: {learned}", ""]
    out += [f'QUESTION: "{question}"'] + _trace_lines(analyze(question)) + [""]
    out += ["derivation:"] + ["  " + line for line in explain(question, context)]
    out += [f"ANSWER: {_answer_text(answer(question, context))}"]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(run_cli(sys.argv[1:]))

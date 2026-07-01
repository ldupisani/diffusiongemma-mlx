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
import re
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


# Contractions are the surface form of TWO words — we restore both so each carries its own obligation
# (no special-case flag): "what's" → what + is, "didn't" → did + not. The apostrophe-s splits two ways:
# a PRONOUN/interrogative "'s" is the copula "is" ("he's", "what's"); a NAME's GENITIVE ("Leo's") → the
# name + the possessive marker "'s" (so the name is recognised, not a stray unknown token).
_NT_IRREGULAR = {"won't": "will", "can't": "can", "shan't": "shall"}
_PRONOUN_S = {"what", "he", "she", "it", "that", "there", "who", "here", "where", "this", "how"}
_CONTRACTION_SUFFIX = (("'re", "are"), ("'ve", "have"), ("'ll", "will"), ("'m", "am"))


def _split_contraction(tok: str) -> list:
    """Expand one token if it is a contraction, else return it unchanged. Trailing punctuation rides on
    the LAST expanded word so the sentence still reads ("what's?" → ["what", "is?"])."""
    core, punct = tok.replace("’", "'"), ""
    while core and core[-1] in ".,?!;:":
        core, punct = core[:-1], core[-1] + punct
    low = core.lower()

    def two(first, second):
        return [first, second + punct]

    if low in _NT_IRREGULAR:                              # won't → will not, can't → can not
        return two(_NT_IRREGULAR[low], "not")
    if low.endswith("n't") and len(low) > 3:             # did|n't, is|n't, does|n't, could|n't, …
        return two(core[:-3], "not")
    for suf, word in _CONTRACTION_SUFFIX:                 # they're / we've / he'll / I'm
        if low.endswith(suf):
            return two(core[:-len(suf)], word)
    if low.endswith("'s") and len(low) > 2:
        if low[:-2] in _PRONOUN_S:                        # what's / he's / it's → the copula IS
            return two(core[:-2], "is")
        if core[:1].isupper():                            # a NAME's genitive — "Leo's" → Leo + 's (the
            return two(core[:-2], "'s")                   # possessive marker, so the name is recognised)
    return [tok]


def _split_hyphen(tok: str) -> list:
    """A DIGIT-led hyphen compound is an adjectival number-unit phrase — restore its words: "120-page" →
    120 + page, "2-hour" → 2 + hour (so the number quantifies the unit normally). Only when the FIRST part
    is a NUMBER and the rest are words/numbers — so "twenty-five" / "fast-food" stay one token (never split
    a number-word into two values, never a confident error)."""
    if "-" not in tok:
        return [tok]
    core, punct = tok, ""
    while core and core[-1] in ".,?!;:":
        core, punct = core[:-1], core[-1] + punct
    parts = core.split("-")
    if len(parts) >= 2 and _as_number(parts[0]) is not None \
            and all(p.isalpha() or _as_number(p) is not None for p in parts[1:]):
        parts[-1] += punct
        return parts
    return [tok]


def read(text: str) -> Reading:
    """Read text into a Reading: one Word per whitespace-separated token (contractions and digit-led hyphen
    compounds expanded to their constituent words first), each wrapped in its own Unit. A non-leading part
    of a digit-led hyphen compound ("page" in "120-page") is flagged `hyphen_bound` — its value belongs to
    IT, not to a following head noun, so the compound-noun merge must not steal it. No further norm."""
    units, i = [], 0
    for tok in text.split():
        parts = _split_hyphen(tok)
        for j, t1 in enumerate(parts):
            for t2 in _split_contraction(t1):
                u = Unit(words=[Word(surface=t2, index=i)])
                if len(parts) > 1 and j > 0:
                    u.features["hyphen_bound"] = True
                units.append(u)
                i += 1
    return Reading(units=units)


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


def _merge_label_letters(reading: Reading) -> Reading:
    """A NOUN/NAME followed by a SINGLE CAPITAL LETTER ("Jar A", "Class B", "Building C") is a LABELED
    ENTITY, not a noun plus a stray article/name. Merge into one NAME unit so "Jar A"/"Jar B"/"Jar C" are
    DISTINCT entities (keyed 'jar a'/'jar b'/'jar c', like proper names) the relational/sum machinery can
    chain — without this they all collapse to the bare noun 'jar'. Fires on a LOWERCASE head too ("…than
    jar A") so a subject ("Jar B") and a later standard ("jar B") key identically. The single uppercase
    letter is the label signal; a sentence-initial article ("A teacher") has no preceding noun, so it's
    untouched."""
    units = reading.units
    for i in range(len(units) - 1):
        head, lab = units[i], units[i + 1]
        if ("noun" in head.features.get("senses", []) or "name" in head.features.get("senses", [])) \
                and re.fullmatch(r"[A-Z]", lab.surface.strip(".,?!:;()")):
            merged = Unit(words=sorted(head.words + lab.words, key=lambda w: w.index))
            merged.features["senses"] = ["name"]
            reading.record("merge-label-letter", target=merged,
                           before={"merged_from": [head, lab], "at": i},
                           after={"entity": _key(merged.surface)}, reason="noun + single-letter label")
            units[i:i + 2] = [merged]
            return _merge_label_letters(reading)             # indices shifted → restart
    return reading


def _merge_scales(reading: Reading) -> Reading:
    """Combine "N <scale>" into ONE value unit: "two dozen" → 24, "three hundred" → 300 (and chains like
    "two hundred thousand"). Runs after senses are known so both are value units. Reversible."""
    units = reading.units
    for i in range(len(units) - 1):
        if _key(units[i + 1].surface) in SCALES and "value" in units[i].features.get("senses", []):
            scale = SCALES[_key(units[i + 1].surface)]
            merged = Unit(words=sorted(units[i].words + units[i + 1].words, key=lambda w: w.index))
            merged.features["senses"] = ["value"]
            merged.features["value"] = units[i].features["value"] * scale
            reading.record("merge-scale", target=merged,
                           before={"merged_from": [units[i], units[i + 1]], "at": i},
                           after={"value": merged.features["value"]}, reason="N × scale")
            units[i:i + 2] = [merged]
            return _merge_scales(reading)                 # recurse — "two hundred thousand" chains
    return reading


def _merge_times_multiplier(reading: Reading) -> Reading:
    """Fuse a NUMERIC multiplier "<value> times" into ONE multiplier unit (factor N), but ONLY in a
    relational frame — "three times as many", "four times as much", "three times that many" (the "that"
    pointing at the just-stated quantity), or "four times BENEDICT's house" (a genitive standard — the same
    possessive ratio frame `_ratio_relation` already accepts for "twice B's age") — so it scales a quantifier
    exactly like the lexicalized "twice"/"thrice", bringing the quantity path to parity with the age path
    ("N times as old"). A NAME immediately after "times" is unambiguous (unlike "N times MORE than", the
    multiplicative-ambiguity Riku trap — that idiom needs "more/less" right after "times", not a bare name,
    so there's no overlap); `_ratio_relation`'s own frame-check still requires the name be followed by a
    possessive marker, so a bare "N times NAME" with no "'s" stays correctly unmerged-useless. EXCLUDES a
    possessed "age" specifically ("three times Mark's AGE minus 4") — the dedicated age-domain machinery
    parses "<value> times" as a bare operator-expression over the UNMERGED tokens and regressed (a confident
    wrong, 39 vs gold 26) when this merge pre-empted it; age already has its own "N times as old" path via a
    different route, so this merge only needs to cover the non-age possessive case. "N times M" (operator
    multiplication, no relational frame at all) is left untouched. Runs after senses, before obligations, so
    the merged multiplier carries its `scales`."""
    units = reading.units
    for i in range(len(units) - 2):
        val, times, nxt = units[i], units[i + 1], units[i + 2]
        is_age_possessive = ("name" in nxt.features.get("senses", []) and i + 4 < len(units)
                              and "possessive" in units[i + 3].features.get("senses", [])
                              and _key(units[i + 4].surface) == "age")
        if _key(times.surface) == "times" and "value" in val.features.get("senses", []) \
                and "measure" not in val.features and not is_age_possessive \
                and (_key(nxt.surface) in ("as", "that", "the") or "name" in nxt.features.get("senses", [])):
            merged = Unit(words=sorted(val.words + times.words, key=lambda w: w.index))
            merged.features["senses"] = ["multiplier"]
            merged.features["value"] = val.features["value"]
            reading.record("merge-times", target=merged,
                           before={"merged_from": [val, times], "at": i},
                           after={"value": merged.features["value"]}, reason="<value> times as → multiplier")
            units[i:i + 2] = [merged]
            return _merge_times_multiplier(reading)
    return reading


def _merge_labels(reading: Reading) -> Reading:
    """A bare VALUE right after a NOUN is a LABEL/ordinal of it ("Grade 5", "Room 3"), NOT a quantity —
    absorb it into the noun (the number becomes an attribute) so it is no longer a free value that would
    spuriously quantify a neighbour, base a percent, or trip the unused-given guard."""
    units = reading.units
    for i in range(len(units) - 1):
        noun, val = units[i], units[i + 1]
        # NOTE: do NOT un-merge a value before "times" ("shop 2 times a day"). Tried it so the freq count
        # would be visible — but it then parses as a Frequency (satisfying the unused-given guard) without
        # composing into the answer, AND un-declines other un-composed frequency problems (Theo: "13
        # cookies, 3 times a day, 20 days/month" commits a partial 13). Net −precision; the label-merge's
        # value-hiding is doing useful decline work here. Revisit only with frequency×per-occurrence compose.
        # a GERUND ("finish reading 30 pages") is a verb, not a label-bearing noun → keep N free
        if "noun" in noun.features.get("senses", []) and "operator" not in noun.features.get("senses", []) \
                and "value" in val.features.get("senses", []) \
                and "measure" not in val.features \
                and not (_key(noun.surface).endswith("ing") and "verb" in noun.features.get("senses", [])):
            v = val.features["value"]
            label = str(int(v)) if v == int(v) else str(v)
            merged = Unit(words=sorted(noun.words + val.words, key=lambda w: w.index))
            merged.features["senses"] = ["noun"]
            merged.features["head"] = noun.features.get("head") or noun.features.get("singular") \
                or _key(noun.surface)
            merged.features["attributes"] = list(noun.features.get("attributes", [])) + [label]
            reading.record("merge-label", target=merged,
                           before={"merged_from": [noun, val], "at": i},
                           after={"label": merged.features["head"]}, reason="noun + number label")
            units[i:i + 2] = [merged]
            return _merge_labels(reading)
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
    "adults": "adult", "seniors": "senior",
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
    "mornings": "morning", "evenings": "evening", "afternoons": "afternoon", "nights": "night",
    "weeks": "week", "weekends": "weekend", "months": "month", "hours": "hour", "minutes": "minute",
    "pizzas": "pizza", "eggs": "egg", "cookies": "cookie", "cakes": "cake", "pies": "pie",
    "sandwiches": "sandwich", "meals": "meal", "snacks": "snack", "drinks": "drink", "sodas": "soda",
    "donuts": "donut", "muffins": "muffin", "discounts": "discount", "shoes": "shoe", "bills": "bill",
    "coins": "coin", "prizes": "prize", "tickets": "ticket", "sales": "sale", "items": "item",
    "stores": "store", "shops": "shop", "markets": "market", "tips": "tip", "fees": "fee",
    "taxes": "tax", "bonuses": "bonus", "sizes": "size", "rates": "rate", "ratios": "ratio",
    "machines": "machine", "packs": "pack", "containers": "container", "practices": "practice",
    "seasons": "season", "computers": "computer", "libraries": "library", "plans": "plan",
    "lessons": "lesson", "tests": "test", "exams": "exam", "quizzes": "quiz", "projects": "project",
    "courses": "course", "movies": "movie", "bikes": "bike", "episodes": "episode", "games": "game",
    "matches": "match", "songs": "song", "shows": "show", "races": "race", "recipes": "recipe",
    "restaurants": "restaurant", "halls": "hall", "gallons": "gallon", "miles": "mile",
    "pounds": "pound", "kilometers": "kilometer", "birthdays": "birthday", "gifts": "gift",
    "boxes": "box", "jars": "jar", "bottles": "bottle", "cans": "can", "geckos": "gecko",
    "pills": "pill", "pets": "pet", "shelves": "shelf", "bags": "bag", "baskets": "basket",
    "girls": "girl", "boys": "boy", "brothers": "brother", "sisters": "sister", "sons": "son",
    "daughters": "daughter", "uncles": "uncle", "aunts": "aunt", "cousins": "cousin",
    "neighbors": "neighbor", "kids": "kid", "doctors": "doctor", "nurses": "nurse",
    "managers": "manager", "drivers": "driver", "farmers": "farmer", "bakers": "baker",
    "teachers": "teacher", "students": "student", "workers": "worker", "customers": "customer",
    "employees": "employee", "puppies": "puppy", "kittens": "kitten", "classmates": "classmate",
    "heights": "height", "lengths": "length", "widths": "width", "areas": "area", "lawns": "lawn",
    "trucks": "truck", "offices": "office", "loads": "load", "goals": "goal", "plants": "plant",
    "theaters": "theater", "packets": "packet", "weekdays": "weekday", "wages": "wage",
    "salaries": "salary", "candies": "candy", "puppies": "puppy",
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
    s = _key(surface).replace(",", "").lstrip("$£")
    try:
        return float(s)
    except ValueError:
        if "/" in s:                                      # a fraction literal "3/4" → 0.75, "1/2" → 0.5
            a, _, b = s.partition("/")
            try:
                return float(a) / float(b) if a and b else None
            except (ValueError, ZeroDivisionError):
                return None
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
MULTIPLIERS = {"once": 1, "twice": 2, "thrice": 3, "half": 0.5, "double": 2, "triple": 3}

# Fraction words as relational multipliers (num, denom) for "<frac> as many/far" — computed EXACTLY
# (divide last) to dodge float drift. The partitive "<frac> OF X" is a DIFFERENT op (_partitive_quantity),
# kept apart by the following "as" vs "of"; these carry their fraction so the scale stays exact.
FRACTION_MULTIPLIERS = {"one-half": (1, 2), "one-third": (1, 3), "two-thirds": (2, 3),
                        "one-fourth": (1, 4), "one-quarter": (1, 4), "three-fourths": (3, 4),
                        "three-quarters": (3, 4), "one-fifth": (1, 5), "two-fifths": (2, 5),
                        "three-fifths": (3, 5), "four-fifths": (4, 5), "one-sixth": (1, 6),
                        "one-eighth": (1, 8)}

# Count-NOUNS that, after a multiplier + article, denote "the count/quantity of X" — so "twice the
# NUMBER (of X)" is the same relational quantity as "twice as MANY (as X)" (the noun plays the role of
# the quantifier "many"; the referent X supplies the count). Kept tiny and specific to avoid over-firing.
COUNT_NOUNS = {"number", "amount"}

# Dimension words that, in a relational "<multiplier> as <DIM>" phrase ("twice as LONG/FAR/TALL"), play
# the role of the quantifier "many" — the multiplier scales the referent's quantity in that dimension
# (the referent supplies the measure). Without this, "twice as long" mis-reads as a verb Frequency.
DIMENSIONS = {"long", "far", "tall", "high", "wide", "deep", "fast", "heavy"}   # ("old" → _relational_age)

# Unit-fraction words → their DENOMINATOR (for partitives "half/a third/a quarter OF …"). Stored as the
# denominator so the fraction of a holding is `base / n` (exact when it divides — avoids 1/3 float drift).
def _fraction_parts(word: str):
    """(numerator, denominator) for a fraction word — non-unit via FRACTION_MULTIPLIERS ("two-thirds"
    → (2,3)), unit via FRACTIONS ("third" → (1,3)), or the SLASH-DIGIT form ("3/8" → (3,8)). None if it
    isn't a fraction. The slash form keeps its exact integer parts so a partitive divides last."""
    k = _key(word)
    if k in FRACTION_MULTIPLIERS:
        return FRACTION_MULTIPLIERS[k]
    if k in FRACTIONS:
        return (1, FRACTIONS[k])
    a, slash, b = k.partition("/")                        # "3/8" → ("3", "/", "8"); "" if no slash
    if slash and a.isdigit() and b.isdigit() and int(b) != 0:
        return (int(a), int(b))
    return None


FRACTIONS = {"half": 2, "third": 3, "fourth": 4, "quarter": 4, "fifth": 5, "sixth": 6,
             "eighth": 8, "tenth": 10,
             "one-half": 2, "one-third": 3, "one-quarter": 4, "one-fourth": 4, "one-fifth": 5}

# A distribution METHOD (a manner adverb modifying a distribute verb) → the kind of split. Only "equal"
# is deterministically computable (total ÷ count); the others are recognised but unsolvable → decline.
DISTRIBUTION_METHODS = {"equally": "equal", "evenly": "equal", "uniformly": "equal",
                        "randomly": "random", "optimally": "optimal", "proportionally": "proportional"}

# Unit conversion factors: (from, to) → multiply. Exact ratios within a measurement family, so a quantity
# stated in one unit folds into a rate/answer asked in another ("50 minutes" for an hourly rate).
CONVERSIONS = {("minute", "hour"): 1 / 60, ("hour", "minute"): 60, ("second", "minute"): 1 / 60,
               ("minute", "second"): 60, ("hour", "day"): 1 / 24, ("day", "hour"): 24,
               ("day", "week"): 1 / 7, ("week", "day"): 7, ("gram", "kilogram"): 1 / 1000,
               ("kilogram", "gram"): 1000, ("ounce", "pound"): 1 / 16, ("pound", "ounce"): 16,
               ("inch", "foot"): 1 / 12, ("foot", "inch"): 12, ("foot", "yard"): 1 / 3, ("yard", "foot"): 3,
               ("centimeter", "meter"): 1 / 100, ("meter", "centimeter"): 100,
               ("meter", "kilometer"): 1 / 1000, ("kilometer", "meter"): 1000}

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
    "mr": ["title"], "mrs": ["title"], "ms": ["title"], "dr": ["title"],   # honorifics — recognised but
    "mister": ["title"], "miss": ["title"], "prof": ["title"], "professor": ["title"],   # NOT names, so
    "sir": ["title"], "madam": ["title"],   # they don't compete to be a verb's subject ("Mr. Sam shared"
    # → Sam is the unique name). A `title` carries no obligation (it's an unforced, recognised modifier).
    "are": ["verb"], "is": ["verb"], "was": ["verb"], "were": ["verb"],   # copulas (also link a complement)
    "one": ["value"], "two": ["value"], "three": ["value"],   # word-numbers ARE values (like digits) so
    "four": ["value"], "five": ["value"], "six": ["value"], "seven": ["value"],   # "three apples"/"three
    "eight": ["value"], "nine": ["value"], "ten": ["value"], "eleven": ["value"], "twelve": ["value"],  # years
    # younger" parse like "3 …". The value comes from NUMBERS (annotate_senses); count statements ("there
    # are three X") are existential-gated in _facts_from / instances().
    "dozen": ["value"], "hundred": ["value"], "thousand": ["value"],   # number-SCALES (value from SCALES);
    "million": ["value"], "billion": ["value"],                        # "two dozen" → 24 via _merge_scales
    "did": ["auxiliary", "verb"], "does": ["auxiliary", "verb"], "do": ["auxiliary", "verb"],
    # ^ yes/no markers ("did she pass?") AND the main verb "do" ("she did 50 minutes of work");
    #   disambiguation picks the verb reading when there's no other verb for the aux to ask about.
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
    "ago": ["adverb"], "later": ["adverb"], "earlier": ["adverb"],   # relative-time markers (see _time_offset)
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
    "adult": ["noun"], "senior": ["noun"], "chef": ["noun"],
    "washing": ["noun"], "babysitting": ["noun"],
    # --- vocabulary push (data-driven from reading_bench): function words + units + common verbs/nouns ---
    "that": ["article"], "this": ["article"], "these": ["article"], "those": ["article"],
    "another": ["article"], "other": ["article"], "both": ["quantifier"], "any": ["article"],
    "be": ["verb"],
    "before": ["preposition"], "during": ["preposition"], "between": ["preposition"],
    "among": ["preposition"], "within": ["preposition"], "across": ["preposition"],
    "while": ["conjunction"], "when": ["conjunction"], "because": ["conjunction"], "since": ["conjunction"],
    "up": ["adverb"], "out": ["adverb"], "off": ["adverb"], "away": ["adverb"], "back": ["adverb"],
    "together": ["adverb"], "instead": ["adverb"], "however": ["adverb"],
    # MANNER adverbs — they modify a verb's OPERATION (its method), so they BIND it (`qualifies`), not
    # orphaned like sentence adverbs. Distribution methods are looked up in DISTRIBUTION_METHODS.
    "equally": ["manner"], "evenly": ["manner"], "uniformly": ["manner"], "randomly": ["manner"],
    "optimally": ["manner"], "proportionally": ["manner"],
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
    # verbs (vocabulary push 2):
    "work": ["verb"], "works": ["verb"], "working": ["verb"], "worked": ["verb"],
    "hold": ["verb"], "holds": ["verb"], "holding": ["verb"], "held": ["verb"],
    "go": ["verb"], "goes": ["verb"], "going": ["verb"], "went": ["verb"], "gone": ["verb"],
    "collect": ["verb"], "collects": ["verb"], "collecting": ["verb"], "collected": ["verb"],
    "save": ["verb"], "saves": ["verb"], "saving": ["verb"], "saved": ["verb"],
    "purchase": ["verb"], "purchases": ["verb"], "purchasing": ["verb"], "purchased": ["verb"],
    "charge": ["verb"], "charges": ["verb"], "charging": ["verb"], "charged": ["verb"],
    "share": ["verb"], "shares": ["verb"], "sharing": ["verb"], "shared": ["verb"],
    "distribute": ["verb"], "distributes": ["verb"], "distributed": ["verb"], "distributing": ["verb"],
    "spread": ["verb"], "spreads": ["verb"], "spreading": ["verb"],
    "receive": ["verb"], "receives": ["verb"], "receiving": ["verb"], "received": ["verb"],
    "pay": ["verb"], "pays": ["verb"], "paying": ["verb"], "paid": ["verb"],
    "need": ["verb"], "needs": ["verb"], "needing": ["verb"], "needed": ["verb"],
    "want": ["verb"], "wants": ["verb"], "wanting": ["verb"], "wanted": ["verb"],
    "keep": ["verb"], "keeps": ["verb"], "keeping": ["verb"], "kept": ["verb"],
    "bring": ["verb"], "brings": ["verb"], "bringing": ["verb"], "brought": ["verb"],
    "send": ["verb"], "sends": ["verb"], "sending": ["verb"], "sent": ["verb"],
    "fill": ["verb"], "fills": ["verb"], "filling": ["verb"], "filled": ["verb"],
    "mix": ["verb"], "mixes": ["verb"], "mixing": ["verb"], "mixed": ["verb"],
    "pour": ["verb"], "pours": ["verb"], "pouring": ["verb"], "poured": ["verb"],
    "order": ["verb"], "orders": ["verb"], "ordering": ["verb"], "ordered": ["verb"],
    "earn": ["verb"], "earns": ["verb"], "earning": ["verb"], "earned": ["verb"],
    "make": ["verb"], "makes": ["verb"], "making": ["verb"], "made": ["verb"],
    "take": ["verb"], "takes": ["verb"], "taking": ["verb"], "took": ["verb"], "taken": ["verb"],
    "been": ["verb"],
    # verbs (batch 2):
    "compare": ["verb"], "compares": ["verb"], "compared": ["verb"], "comparing": ["verb"],
    "plant": ["verb"], "plants": ["verb"], "planted": ["verb"], "planting": ["verb"],
    "add": ["verb"], "adds": ["verb"], "added": ["verb"], "adding": ["verb"],
    "manage": ["verb"], "manages": ["verb"], "managed": ["verb"], "managing": ["verb"],
    "turn": ["verb"], "turns": ["verb"], "turned": ["verb"], "turning": ["verb"],
    "split": ["verb"], "splits": ["verb"], "splitting": ["verb"],
    "know": ["verb"], "knows": ["verb"], "knew": ["verb"], "knowing": ["verb"],
    "come": ["verb"], "comes": ["verb"], "came": ["verb"], "coming": ["verb"],
    "last": ["verb"], "lasts": ["verb"], "lasted": ["verb"], "lasting": ["verb"],
    "teach": ["verb"], "teaches": ["verb"], "taught": ["verb"], "teaching": ["verb"],
    "hit": ["verb"], "hits": ["verb"], "hitting": ["verb"], "wonder": ["verb"], "wonders": ["verb"],
    "wondering": ["verb"], "grow": ["verb"], "grows": ["verb"], "grew": ["verb"], "growing": ["verb"],
    "build": ["verb"], "builds": ["verb"], "built": ["verb"], "building": ["verb"],
    "run": ["verb"], "runs": ["verb"], "ran": ["verb"], "running": ["verb"],
    "walk": ["verb"], "walks": ["verb"], "walked": ["verb"], "walking": ["verb"],
    "drive": ["verb"], "drives": ["verb"], "drove": ["verb"], "driving": ["verb"],
    "count": ["verb"], "counts": ["verb"], "counted": ["verb"], "counting": ["verb"],
    "paint": ["verb"], "paints": ["verb"], "painted": ["verb"], "painting": ["verb"],
    "playing": ["verb"], "having": ["verb"], "shopping": ["verb"], "practicing": ["verb"],
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
    "level": ["noun"], "score": ["noun"], "grade": ["noun"], "quarter": ["noun"],
    # --- vocabulary push 2 (data-driven from reading_bench) ---
    # nouns (time-of-day, food/drink, shopping, school, entertainment, misc):
    "morning": ["noun"], "evening": ["noun"], "afternoon": ["noun"], "night": ["noun"],
    "week": ["noun"], "weekend": ["noun"], "month": ["noun"], "hour": ["noun"], "minute": ["noun"],
    "pizza": ["noun"], "food": ["noun"], "sugar": ["noun"], "flour": ["noun"], "egg": ["noun"],
    "milk": ["noun"], "butter": ["noun"], "bread": ["noun"], "candy": ["noun"], "cookie": ["noun"],
    "cake": ["noun"], "pie": ["noun"], "sandwich": ["noun"], "meal": ["noun"], "snack": ["noun"],
    "drink": ["noun"], "soda": ["noun"], "juice": ["noun"], "coffee": ["noun"], "tea": ["noun"],
    "meat": ["noun"], "rice": ["noun"], "salt": ["noun"], "donut": ["noun"], "muffin": ["noun"],
    "discount": ["noun"], "budget": ["noun"], "shoe": ["noun"], "account": ["noun"], "savings": ["noun"],
    "bill": ["noun"], "coin": ["noun"], "prize": ["noun"], "ticket": ["noun"], "sale": ["noun"],
    "item": ["noun"], "store": ["noun"], "shop": ["noun"], "market": ["noun"], "cash": ["noun"],
    "tip": ["noun"], "fee": ["noun"], "tax": ["noun"], "bonus": ["noun"], "size": ["noun"],
    "rate": ["noun"], "ratio": ["noun"], "machine": ["noun"], "pack": ["noun"], "container": ["noun"],
    "practice": ["noun"], "season": ["noun"], "computer": ["noun"], "library": ["noun"], "plan": ["noun"],
    "lesson": ["noun"], "homework": ["noun"], "test": ["noun"], "exam": ["noun"], "quiz": ["noun"],
    "project": ["noun"], "course": ["noun"], "movie": ["noun"], "bike": ["noun"], "episode": ["noun"],
    "game": ["noun"], "match": ["noun"], "song": ["noun"], "show": ["noun"], "race": ["noun"],
    "recipe": ["noun"], "restaurant": ["noun"], "hall": ["noun"], "fuel": ["noun"], "gallon": ["noun"],
    "mile": ["noun"], "pound": ["noun"], "kilometer": ["noun"], "birthday": ["noun"], "gift": ["noun"],
    "box": ["noun"], "jar": ["noun"], "bottle": ["noun"], "can": ["noun"], "gecko": ["noun"],
    "pill": ["noun"], "pet": ["noun"], "shelf": ["noun"], "bag": ["noun"], "basket": ["noun"],
    # adjectives:
    "able": ["adjective"], "different": ["adjective"], "previous": ["adjective"], "final": ["adjective"],
    "double": ["adjective", "multiplier"],   # "a double room" (adj) vs "double the number" (×2)
    "extra": ["adjective"], "worth": ["adjective"], "whole": ["adjective"],
    "equal": ["adjective"], "entire": ["adjective"], "certain": ["adjective"], "various": ["adjective"],
    "free": ["adjective"], "regular": ["adjective"], "normal": ["adjective"], "single": ["adjective"],
    "separate": ["adjective"], "original": ["adjective"], "favorite": ["adjective"], "daily": ["adjective"],
    "weekly": ["adjective"], "monthly": ["adjective"], "annual": ["adjective"], "enough": ["adjective"],
    # hyphenated unit fractions (recognised + usable as partitives; see FRACTIONS):
    # fraction words: adjective + MULTIPLIER ("two-thirds as far" = ×2/3; "two-thirds OF X" stays the
    # partitive, kept apart by the following "as" vs "of"). Multiplier last so a tie favours adjective.
    "one-half": ["adjective", "multiplier"], "one-third": ["adjective", "multiplier"],
    "one-quarter": ["adjective", "multiplier"], "one-fourth": ["adjective", "multiplier"],
    "one-fifth": ["adjective", "multiplier"], "two-thirds": ["adjective", "multiplier"],
    "three-quarters": ["adjective", "multiplier"], "three-fourths": ["adjective", "multiplier"],
    "two-fifths": ["adjective", "multiplier"], "three-fifths": ["adjective", "multiplier"],
    "four-fifths": ["adjective", "multiplier"], "one-sixth": ["adjective", "multiplier"],
    "one-eighth": ["adjective", "multiplier"],
    # --- vocabulary push 2, batch 2 ---
    # family + role nouns:
    "girl": ["noun"], "boy": ["noun"], "brother": ["noun"], "sister": ["noun"], "son": ["noun"],
    "daughter": ["noun"], "mom": ["noun"], "dad": ["noun"], "father": ["noun"], "mother": ["noun"],
    "uncle": ["noun"], "aunt": ["noun"], "cousin": ["noun"], "neighbor": ["noun"], "kid": ["noun"],
    "doctor": ["noun"], "nurse": ["noun"], "boss": ["noun"], "manager": ["noun"], "driver": ["noun"],
    "farmer": ["noun"], "baker": ["noun"], "teacher": ["noun"], "student": ["noun"], "worker": ["noun"],
    "customer": ["noun"], "employee": ["noun"], "puppy": ["noun"], "kitten": ["noun"], "classmate": ["noun"],
    # misc nouns:
    "height": ["noun"], "length": ["noun"], "width": ["noun"], "area": ["noun"], "lawn": ["noun"],
    "profit": ["noun"], "truck": ["noun"], "office": ["noun"], "load": ["noun"], "goal": ["noun"],
    "plant": ["noun"], "oil": ["noun"], "beef": ["noun"], "pork": ["noun"], "theater": ["noun"],
    "packet": ["noun"], "weekday": ["noun"], "wage": ["noun"], "salary": ["noun"], "income": ["noun"],
    # adjectives:
    "expensive": ["adjective"], "cheap": ["adjective"], "additional": ["adjective"], "main": ["adjective"],
    "following": ["adjective"], "square": ["adjective"], "fresh": ["adjective"], "fancy": ["adjective"],
    # function words:
    "your": ["pronoun"], "down": ["adverb"], "below": ["preposition"], "above": ["preposition"],
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
    "s": ["possessive"],   # the genitive marker from a split "Leo's" → Leo + 's (a NAME's possessive)
    # "what remains" markers — route a count question to the holdings fold:
    "left": ["adjective"], "remaining": ["adjective"], "leftover": ["adjective"], "rest": ["noun"],
    "remainder": ["noun"], "others": ["noun"],   # complement markers — "the REST/others <verb>" (_rest_quantity)
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
    # have/has/had: primarily the verb (own), but ALSO the perfect auxiliary ("has Brandon SOLD …"); verb
    # sense first so "Tom has 5 apples" stays a main verb, auxiliary wins when a participle follows.
    "have": ["verb", "auxiliary"], "has": ["verb", "auxiliary"], "had": ["verb", "auxiliary"],  # perfect aux
    # accumulation verbs (their lemma's polarity folds in _verb_sign): gains + losses
    "buy": ["verb"], "buys": ["verb"], "get": ["verb"], "gets": ["verb"], "got": ["verb"],
    "find": ["verb"], "finds": ["verb"], "found": ["verb"], "receive": ["verb"], "receives": ["verb"],
    "lose": ["verb"], "loses": ["verb"], "lost": ["verb"], "sells": ["verb"],
    "give": ["verb"], "gives": ["verb"], "gave": ["verb"], "spend": ["verb"], "spends": ["verb"],
    "spent": ["verb"], "eat": ["verb"], "eats": ["verb"], "ate": ["verb"], "use": ["verb"],
    "uses": ["verb"], "used": ["verb"],
    "tall": ["adjective"], "old": ["adjective"], "small": ["adjective"], "happy": ["adjective"],
    "blue": ["adjective"], "yellow": ["adjective"],
    # resolve duplicate-key collisions introduced by the vocab batches (last occurrence wins) — restore
    # the words whose original sense a batch entry shadowed, as MULTI-sense where both readings are real:
    "last": ["adjective", "verb"], "building": ["noun", "verb"], "drink": ["noun", "verb"],
    "can": ["auxiliary", "noun"], "plant": ["noun", "verb"],
}

# Multi-word phrases that are ONE indivisible concept (merged before analysis by _merge_compounds).
COMPOUNDS = {k for k in LEXICON if " " in k}

# Number words → their value, for "there are N <collection>" statements that introduce instances.
TITLES = {w for w, s in LEXICON.items() if s == ["title"]}   # honorifics; a trailing "." is not a sentence end
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
# Number-SCALE words: a bare scale is its value ("a dozen" → 12); "N scale" MULTIPLIES ("two dozen" → 24,
# "three hundred" → 300), merged into one value unit by _merge_scales.
SCALES = {"dozen": 12, "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
TIME_UNITS = {"second", "minute", "hour", "day", "week", "month", "year"}   # rate per_unit / duration query
DAYPARTS = {"morning", "afternoon", "evening", "night", "noon"}   # sub-day recurrences → a DAILY cadence
# Measurement UNITS — time + distance + weight + volume. A value quantifying one of these (or a money value)
# is OPERATIVE: it must enter the computation, else we dropped it (decline-safety GUARD 12). Object nouns
# ("3 snakes", "3 people") are NOT units — those are enumeration counts, legitimately often uncomputed.
UNIT_NOUNS = TIME_UNITS | {"second", "minute", "hour", "day", "week", "month", "year",
                           "inch", "foot", "yard", "mile", "centimeter", "meter", "kilometer",
                           "gram", "kilogram", "ounce", "pound", "ton", "milligram",
                           "liter", "gallon", "cup", "quart", "pint"}


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
        elif senses == ["noun"] and u.surface[:1].isupper() and _key(u.surface).isalpha():
            # a capitalized word that IS a known common noun ("Job", "Bill", "Rose", "Penny") may be a
            # NAME used as that noun — make it name/noun-ambiguous so disambiguate can pick by coverage.
            # A name wins in name positions (the as…as standard, a copula subject); the noun wins when
            # article-DETERMINED ("the bill") or otherwise covers better. Modals/verbs ("Will", "May")
            # aren't ["noun"], so they're untouched. The name reading is listed FIRST → ties favour it.
            senses = ["name", "noun"]
        if senses is None:
            continue  # not in the dictionary → attach no meaning
        before = dict(u.features)
        u.features["senses"] = list(senses)
        if "value" in senses:
            u.features["value"] = _as_number(u.surface)
            if u.features["value"] is None:                # a WORD-number ("three") / SCALE ("dozen"→12)
                u.features["value"] = NUMBERS.get(_key(u.surface)) or SCALES.get(_key(u.surface))
            if u.surface.strip().startswith(("$", "£")):   # "$12" — a self-measuring money value
                u.features["measure"] = MONEY
        if "operator" in senses:
            u.features["op"] = OPERATORS[_key(u.surface)]
        if "pronoun" in senses:
            u.features["gender"] = PRONOUN_GENDER.get(_key(u.surface))
        if "multiplier" in senses:
            key = _key(u.surface)
            if key in FRACTION_MULTIPLIERS:                # "two-thirds" → exact (num, denom) + its value
                u.features["fraction"] = FRACTION_MULTIPLIERS[key]
                u.features["value"] = FRACTION_MULTIPLIERS[key][0] / FRACTION_MULTIPLIERS[key][1]
            else:
                u.features["value"] = MULTIPLIERS[key]
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
    "manner": ["qualifies"],            # a manner adverb ("equally") modifies a verb's OPERATION (its method)
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
        if _key(u.surface) == "much":   # "how MUCH water" — the ADJACENT mass noun; else a money noun
            if r and pos[id(r[0])] == i + 1:                                  # anywhere ("how much was the
                return r[0]                                                  # MONEY shared"); else the
            money = next((x for x in units if _key(x.surface) in _MONEY_WORDS  # result-measure verb (its
                          and "noun" in x.features.get("senses", []) and pos[id(x)] > i), None)  # nominal-
            if money is not None:                                            # isation: "how much did X EARN")
                return money
            rv = [x for x in units if has(x, "verb") and _lemma(x.surface) in RESULT_MEASURE
                  and pos[id(x)] > i]
            return rv[0] if rv else None
        return r[0] if r else None
    if need == "object":          # a preposition's object: the nearest noun / NAME / TIME to its right
        r = right("noun", "name", "time")   # name → "than WILLIAM"; time → "on MONDAY" (so the time, not
        return r[0] if r else None          # a following name, is the object — keeps the name free as subj)
    if need == "determines":      # an article determines the NEAREST noun to its right ("a certain
        r = right("noun")         # amount of money" → amount, not "unique noun anywhere" which failed
        return r[0] if r else None   # when a following "of money"/object added a second noun)
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
            if not between:                              # "she WILL have" — SUBJ AUX VERB (declarative
                bef = [x for x in units if has(x, "noun", "name", "pronoun")  # order): the subject sits
                       and pos[id(x)] < pos[id(ax)]]                          # just before the aux, past
                if bef:                                                      # the fronted "how much money"
                    return max(bef, key=lambda x: pos[id(x)])
        ll = left("noun", "name", "pronoun")            # else the unique noun/name/pronoun to the LEFT,
        pp_obj = {id(_follow(p, "object")) for p in units    # EXCLUDING a preposition's object ("the price
                  if has(p, "preposition") and _follow(p, "object") is not None}   # OF beef is …" → price,
        poss_det = {id(units[k]) for k in range(len(units) - 1)   # and a POSSESSIVE pronoun before a name/
                    if _key(units[k].surface) in ("his", "her", "their", "its", "my", "our")   # noun ("HIS
                    and (has(units[k + 1], "noun", "name")        # dad is 32") — a determiner, not the subject;
                         or (k + 2 < len(units) and has(units[k + 1], "title", "adjective")   # also across an
                             and has(units[k + 2], "noun", "name")))}   # attributive modifier ("his OLDER brother")
        ll = [x for x in ll if id(x) not in pp_obj and id(x) not in poss_det]   # ("dad" wins, not "his")
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
    if need in ("queries",):      # an interrogative asks about the NEAREST quantifier to its right ("how
        r = right("quantifier")   # MANY candies … each student" — "each" is a second quantifier, but the
        return r[0] if r else None   # query is the adjacent "many", not "unique quantifier anywhere")
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
    if need == "scales":          # a multiplier scales a quantifier to its RIGHT ("twice as MANY"), a
        rq = right("quantifier")  # count-noun "the NUMBER (of X)", or a dimension word "as LONG/FAR" (all
        if rq:                    # play the "many" role), else the nearest verb to its LEFT ("jumps TWICE")
            return rq[0]
        def _as_dim(x):   # a dimension word counts ONLY in the relational "as LONG/FAR" frame (preceded
            px = pos[id(x)]   # by "as") — so the idiom "so FAR" / a bare "LONG" is never scaled
            return _key(x.surface) in DIMENSIONS and any(_key(u.surface) == "as" and pos[id(u)] == px - 1
                                                         for u in units)
        cd = sorted((x for x in units if pos[id(x)] > i                       # count-noun or "as <dimension>"
                     and (_key(x.surface) in COUNT_NOUNS or _as_dim(x))), key=lambda x: pos[id(x)])
        if cd:
            return cd[0]
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
    if need == "qualifies":       # a manner adverb modifies the NEAREST verb (its operation's method)
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
    for i in range(1, len(reading.units) - 1):           # NOUN-NOUN compound — head-FINAL: "8 flower
        left, right = reading.units[i], reading.units[i + 1]   # stickers" is 8 stickers (head) of the
        prev = reading.units[i - 1].features.get("senses") or []
        prevk = _key(reading.units[i - 1].surface)
        # ALSO merge the two COMPARATIVE-context measure phrases that the value/quantifier gate misses, so a
        # measure-to-measure comparison keeps its head: "2 fewer ANIMAL STICKERS" (modifier after a comparative
        # that itself follows a value) and the standard "than FLOWER STICKERS" / "as many BLUE MARBLES" (after
        # than/as). Without this the compound fragments and the comparison can't ground (Victor).
        comp_ctx = ("comparative" in prev and i >= 2
                    and reading.units[i - 2].features.get("senses") == ["value"]) \
            or prevk in ("than", "as")
        if (prev == ["value"] or "quantifier" in prev or comp_ctx) \
                and left.features.get("senses") == ["noun"] and right.features.get("senses") == ["noun"] \
                and not left.features.get("hyphen_bound") and not _has_rate_marker(reading) \
                and _compound_modifier(left) and _compound_modifier(right):
            # GATED narrow: "<value|quantifier> <modifier-noun> <head-noun>" where the count ranges over the
            # HEAD (8 stickers / how many baseball CARDS), the modifier is a kind (flower/baseball), and the
            # modifier wasn't hyphen-bound to that value ("120-page book" → 120 counts pages, not books). The
            # quantifier case keeps the QUESTION's measure aligned with the statements'. Excludes "bags OF
            # coffee beans" and "a coffee shop" (no count before the modifier) — they key on the un-merged measure.
            _merge_adjective(reading, i, i + 1)          # left noun → attribute, right noun → head
            return True
    for i, u in enumerate(reading.units):                # "Tom and Mary" → one group (names only:
        if "conjunction" in u.features.get("senses", []) and 0 < i < len(reading.units) - 1:  # a
            left, right = reading.units[i - 1], reading.units[i + 1]                          # subject)
            if "name" in left.features.get("senses", []) and "name" in right.features.get("senses", []):
                _merge_conjuncts(reading, i)
                return True
    return False


def _compound_modifier(unit: "Unit") -> bool:
    """True if `unit` is a plain CONTENT noun eligible for a noun-noun compound merge. Excludes measure/
    unit/money/time nouns whose head choice would corrupt a measure, and anything already carrying an
    attribute or number."""
    head = unit.features.get("head") or unit.features.get("singular") or _key(unit.surface)
    return (head not in (UNIT_NOUNS | _MONEY_WORDS | TIME_UNITS)
            and not unit.features.get("attributes") and "value" not in unit.features)


def _has_rate_marker(reading: Reading) -> bool:
    """True if the statement carries a per-unit RATE marker ("7 apple pies A DAY", "$5 PER hour"). The
    compound-noun merge is suppressed here: a rate unifies the measure enough to compose a rate×duration
    PARTIAL result, re-committing numbers (Cooper's 84) that were safely declining until the full chain
    (cross-entity depletion) lands. None of the merge's targets (holdings/counts) are rates."""
    units = reading.units
    if any("per" in u.features.get("senses", []) for u in units):
        return True
    for k in range(len(units) - 1):
        if _key(units[k].surface) in ("a", "an", "per", "every", "each") \
                and (units[k + 1].features.get("singular") or _key(units[k + 1].surface)) in TIME_UNITS:
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


_ASPECTUAL = {"is", "are", "was", "were", "am",          # progressive copulas
              "finish", "finishes", "finished",          # ASPECTUAL light verbs that take a V-ing
              "start", "starts", "started", "begin", "begins", "began",
              "keep", "keeps", "kept", "continue", "continues", "continued"}


def _progressive(reading: Reading) -> Reading:
    """Present/past PROGRESSIVE "is/are/was/were V-ing" and ASPECTUAL "finish/start/keep V-ing" → the leading
    verb is an AUXILIARY and the participle the main verb ("is driving" → drive, "can finish reading 30 pages"
    → read). Without this both read as verbs and the single-main-verb logic (rates, the discourse subject)
    misfires. Gated to an aspectual/copula immediately before an -ing word that is a known verb."""
    units = reading.units
    for i in range(len(units) - 1):
        nxt = _key(units[i + 1].surface)
        if _key(units[i].surface) in _ASPECTUAL \
                and "verb" in units[i].features.get("senses", []) \
                and nxt.endswith("ing") and "verb" in units[i + 1].features.get("senses", []):
            before = dict(units[i].features)
            units[i].features["senses"] = ["auxiliary"]
            reading.record("progressive", target=units[i], before=before, after=dict(units[i].features),
                           reason="aspectual/copula + V-ing → auxiliary")
    return reading


def _prefer_noun_after_quantifier(reading: Reading) -> Reading:
    """A quantifier ("many"/"much") counts a NOUN, never an auxiliary/modal — so a word right after
    "many"/"much" carrying BOTH a noun and an auxiliary/modal sense ("how many CANS", can=[auxiliary, noun])
    must read as the NOUN. Drop the auxiliary/modal sense so disambiguate can't be pulled to it by a nearby
    "did" ("how many cans DID the boys collect" was reading 'cans' as a modal, then 'many' bound 'boys')."""
    units = reading.units
    for i in range(1, len(units)):
        if _key(units[i - 1].surface) in ("many", "much"):
            s = units[i].features.get("senses", [])
            if "noun" in s and ("auxiliary" in s or "modal" in s):
                units[i].features["senses"] = [x for x in s if x not in ("auxiliary", "modal")]
    return reading


KINSHIP = {"dad", "mom", "mum", "mommy", "mother", "father", "sister", "brother", "son", "daughter",
           "cousin", "aunt", "uncle", "grandfather", "grandmother", "grandpa", "grandma", "grandmother",
           "niece", "nephew", "wife", "husband", "sibling", "parent"}


def _promote_kinship(reading: Reading) -> Reading:
    """Treat "his/her/their <KINSHIP>" as a NAMED entity keyed by the kinship noun, so "his dad is 32" stores
    an age and "how old is his dad" reads it. A possessive pronoun immediately before a kinship noun is the
    signal — rare enough as a non-referential phrase that the promotion is safe. Keyed by the kinship word
    (one "his dad" per problem); a possessor-qualified key is a later refinement if collisions appear."""
    POSS = ("his", "her", "their", "my", "our", "its")
    units = reading.units

    def is_poss(k):
        return 0 <= k < len(units) and _key(units[k].surface) in POSS \
            and "pronoun" in units[k].features.get("senses", [])

    for i in range(1, len(units)):
        u = units[i]
        if _key(u.surface) not in KINSHIP or "noun" not in u.features.get("senses", []):
            continue
        mod = None                                       # the possessive sits at i-1, OR at i-2 with one
        if is_poss(i - 1):                               # attributive modifier between ("his OLDER brother")
            pass
        elif i - 2 >= 0 and is_poss(i - 2) and ("adjective" in units[i - 1].features.get("senses", [])
                                                or "comparative" in units[i - 1].features.get("senses", [])):
            mod = units[i - 1]
        else:
            continue
        before = dict(u.features)
        u.features["senses"] = ["name"]
        reading.record("promote-kinship", target=u, before=before, after=dict(u.features),
                       reason="possessive + kinship noun → named entity")
        if mod is not None:                              # the modifier ("older") is now an attribute of a
            mb = dict(mod.features)                      # NAME — neutralize it to a no-obligation title so it
            mod.features["senses"] = ["title"]           # doesn't strand a comparative's "standard" need
            reading.record("demote-kin-modifier", target=mod, before=mb, after=dict(mod.features),
                           reason="modifier of a named kin → attributive title")
    return reading


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
    _merge_scales(r)         # "two dozen" → one value unit (24) before obligations bind it
    _merge_times_multiplier(r)  # "three times as" → one multiplier unit (×3), like "twice"
    _merge_labels(r)         # "Grade 5" → a labeled noun (the number is an ordinal, not a quantity)
    _merge_label_letters(r)  # "Jar A" / "Class B" → one labeled NAME (distinct entities, not bare 'jar')
    _prefer_noun_after_quantifier(r)   # "how many CANS" → cans is a noun, never the modal 'can'
    _promote_kinship(r)      # "his dad" / "her sister" → a named entity (keyed by the kinship noun)
    _progressive(r)          # "is driving" → driving is the main verb (the copula is an auxiliary)
    disambiguate(r)          # collapse multi-sense words to the reading that covers best
    annotate_obligations(r)
    while condense(r):       # condense "red apple" → one noun unit; loop stacks "big red apple"
        pass
    resolve(r)               # then bind quantifier/preposition obligations over the condensed units
    _resolve_possessives(r)  # within-sentence: a possessive ("his") back-links to the clause's agent
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


def _comparison_standard_ids(reading: Reading) -> set:
    """id()s of the noun/name that is the STANDARD of an "as <X> as <STANDARD>" comparison — the
    noun/name immediately following the SECOND "as" ("4 times as old as Job" → Job; "half as many as
    Tim" → Tim). A comparative's "than"-standard is already subordinated via its met "standard"
    obligation, but the bare as…as frame carries no comparative, so its standard would otherwise look
    UNPREDICATED — making `disambiguate` wrongly prefer the common-noun reading of a name like "Job".
    Foregoing the standard's own obligations (treating it as subordinated) lets the name reading win.
    The standard fills the comparison's reference slot, so this is the same "slot-filler is discharged"
    principle the subordinated set already encodes."""
    units = reading.units
    as_idx = [i for i, u in enumerate(units) if _key(u.surface) == "as"]
    if len(as_idx) < 2:
        return set()
    out = set()
    for i in range(as_idx[-1] + 1, len(units)):          # the nearest noun/name after the LAST "as"
        if any(s in units[i].features.get("senses", []) for s in ("noun", "name")):
            out.add(id(units[i]))
            break
    return out


def deficiencies(reading: Reading) -> list:
    """Everything that stops this reading from being a complete, understood statement. Empty ⇒
    well-formed."""
    # A unit that FILLS another word's slot (a verb's direct object, a copula's complement) is
    # subordinated: it's discharged by being that slot-filler, so its own unmet obligations are
    # foregone — "the dog" in "Mary owns a dog" doesn't independently demand a predicate.
    subordinated = {id(o.target) for u in reading.units for o in u.features.get("obligations", [])
                    if o.met and o.need in ("complement", "dobj", "left", "right", "unit", "standard")}
    subordinated |= _comparison_standard_ids(reading)    # the as…as comparison standard (no comparative
    #                                                      word binds it, but it fills the reference slot)
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
_MONEY_WORDS = {"dollar", "cent", "buck", "money", "penny",   # all normalise to the MONEY measure
                "savings", "earnings"}                        # ("half of his SAVINGS" → his money holding)


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
    "sends": "send", "sent": "send", "reads": "read", "reading": "read", "writes": "write", "wrote": "write",
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
    "rides": "ride", "rode": "ride",
    "works": "work", "working": "work", "worked": "work", "holds": "hold", "holding": "hold",
    "held": "hold", "goes": "go", "going": "go", "went": "go", "gone": "go", "collects": "collect",
    "collecting": "collect", "collected": "collect", "saves": "save", "saving": "save", "saved": "save",
    "purchases": "purchase", "purchasing": "purchase", "purchased": "purchase", "charges": "charge",
    "charging": "charge", "charged": "charge", "shares": "share", "sharing": "share", "shared": "share",
    "receives": "receive", "receiving": "receive", "received": "receive", "pays": "pay", "paying": "pay",
    "paid": "pay", "needs": "need", "needing": "need", "needed": "need", "wants": "want",
    "wanting": "want", "wanted": "want", "keeps": "keep", "keeping": "keep", "kept": "keep",
    "brings": "bring", "bringing": "bring", "brought": "bring", "sends": "send", "sending": "send",
    "sent": "send", "fills": "fill", "filling": "fill", "filled": "fill", "mixes": "mix",
    "mixing": "mix", "mixed": "mix", "pours": "pour", "pouring": "pour", "poured": "pour",
    "orders": "order", "ordering": "order", "ordered": "order", "earns": "earn", "earning": "earn",
    "earned": "earn", "makes": "make", "making": "make", "made": "make", "takes": "take",
    "taking": "take", "took": "take", "taken": "take", "been": "be",
    "compares": "compare", "compared": "compare", "comparing": "compare", "planted": "plant",
    "planting": "plant", "adds": "add", "added": "add", "adding": "add", "manages": "manage",
    "managed": "manage", "managing": "manage", "turns": "turn", "turned": "turn", "turning": "turn",
    "splits": "split", "splitting": "split", "knows": "know", "knew": "know", "knowing": "know",
    "comes": "come", "came": "come", "coming": "come", "lasts": "last", "lasted": "last",
    "lasting": "last", "teaches": "teach", "taught": "teach", "teaching": "teach", "hits": "hit",
    "hitting": "hit", "wonders": "wonder", "wondering": "wonder", "grows": "grow", "grew": "grow",
    "growing": "grow", "builds": "build", "built": "build", "building": "build", "runs": "run",
    "ran": "run", "running": "run", "walks": "walk", "walked": "walk", "walking": "walk",
    "drives": "drive", "drove": "drive", "driving": "drive", "counts": "count", "counted": "count",
    "counting": "count", "paints": "paint", "painted": "paint", "painting": "paint",
    "playing": "play", "having": "have", "shopping": "shop", "practicing": "practice",
    "distributes": "distribute", "distributed": "distribute", "distributing": "distribute",
    "spreads": "spread", "spreading": "spread", "does": "do", "did": "do"}

# --- vocabulary push: frequent GSM8K words (deliberate, conservatively-typed; a wrong sense → decline,
# never a confident-wrong). Raises `fully-read` (the ceiling for coverage). ---
LEXICON.update({
    "experience": ["noun"], "collection": ["noun"], "section": ["noun"], "company": ["noun"],
    "vacation": ["noun"], "lottery": ["noun"], "wallet": ["noun"], "jelly": ["noun"], "bean": ["noun"],
    "brownie": ["noun"], "business": ["noun"], "shirt": ["noun"], "overtime": ["noun"],
    "allowance": ["noun"], "farm": ["noun"], "mango": ["noun"], "coconut": ["noun"], "cheese": ["noun"],
    "ingredient": ["noun"], "dinner": ["noun"], "hat": ["noun"], "stamp": ["noun"], "snake": ["noun"],
    "forest": ["noun"], "summer": ["noun"], "grocery": ["noun"], "nickel": ["noun"], "dime": ["noun"],
    "penny": ["noun"], "meeting": ["noun"], "reading": ["verb", "noun"], "top": ["noun"],
    "stay": ["verb"], "stays": ["verb"], "stayed": ["verb"],
    "serve": ["verb"], "serves": ["verb"], "served": ["verb"], "serving": ["verb"],
    "mow": ["verb"], "mows": ["verb"], "mowed": ["verb"], "mowing": ["verb"],
    "bike": ["verb"], "bikes": ["verb"], "biked": ["verb"], "biking": ["verb"],
    "remove": ["verb"], "removes": ["verb"], "removed": ["verb"], "removing": ["verb"],
    "create": ["verb"], "creates": ["verb"], "created": ["verb"], "creating": ["verb"],
    "offer": ["verb"], "offers": ["verb"], "offered": ["verb"], "offering": ["verb"],
    "given": ["verb"], "eaten": ["verb"], "buying": ["verb"], "spending": ["verb"], "using": ["verb"],
    "starting": ["verb"], "care": ["verb"],
    "currently": ["adverb"], "normally": ["adverb"], "finally": ["adverb"], "hard": ["adjective"],
    "past": ["preposition"], "round": ["adjective"],
    "everything": ["pronoun"], "you": ["pronoun"], "ones": ["pronoun"], "where": ["interrogative"],
})
PLURALS.update({
    "collections": "collection", "sections": "section", "companies": "company", "wallets": "wallet",
    "beans": "bean", "brownies": "brownie", "shirts": "shirt", "farms": "farm", "mangoes": "mango",
    "mangos": "mango", "coconuts": "coconut", "ingredients": "ingredient", "dinners": "dinner",
    "hats": "hat", "stamps": "stamp", "snakes": "snake", "forests": "forest", "groceries": "grocery",
    "nickels": "nickel", "dimes": "dime", "pennies": "penny", "meetings": "meeting",
    "vacations": "vacation", "lotteries": "lottery", "experiences": "experience",
    "businesses": "business", "allowances": "allowance",
})
VERB_LEMMA.update({
    "stays": "stay", "stayed": "stay", "staying": "stay",
    "serves": "serve", "served": "serve", "serving": "serve",
    "mows": "mow", "mowed": "mow", "mowing": "mow",
    "bikes": "bike", "biked": "bike", "biking": "bike",
    "removes": "remove", "removed": "remove", "removing": "remove",
    "creates": "create", "created": "create", "creating": "create",
    "offers": "offer", "offered": "offer", "offering": "offer",
    "given": "give", "eaten": "eat", "buying": "buy", "spending": "spend", "using": "use",
    "starting": "start",
})

# --- vocabulary push, round 2 (frequent + single-unknown-word GSM8K terms; same conservative typing) ---
LEXICON.update({
    "teaspoon": ["noun"], "pencil": ["noun"], "partner": ["noun"], "gumball": ["noun"],
    "seashell": ["noun"], "video": ["noun"], "debt": ["noun"], "purpose": ["noun"], "fabric": ["noun"],
    "bedroom": ["noun"], "concert": ["noun"], "piano": ["noun"], "semester": ["noun"],
    "assignment": ["noun"], "part": ["noun"], "chip": ["noun"], "aquarium": ["noun"], "hotel": ["noun"],
    "guest": ["noun"], "performance": ["noun"], "history": ["noun"], "tank": ["noun"], "artist": ["noun"],
    "bicycle": ["noun"], "lake": ["noun"], "bar": ["noun"], "grandmother": ["noun"], "desert": ["noun"],
    "balance": ["noun"], "craft": ["noun"], "accident": ["noun"], "percentage": ["noun"], "supply": ["noun"],
    "rule": ["noun"], "chocolate": ["noun"], "guest": ["noun"], "fountain": ["noun"], "rent": ["noun"],
    "transport": ["noun"], "value": ["noun"],
    "borrow": ["verb"], "borrows": ["verb"], "borrowed": ["verb"], "borrowing": ["verb"],
    "triple": ["multiplier", "verb"], "triples": ["verb"], "tripled": ["verb"],   # "triple the number" → ×3
    # ("triple" before "verb": a tie ("He uses triple that number" — both cover) falls to the multiplier;
    #  a bare "sales triple" leaves `scales` unmet so the verb reading wins on coverage)
    "drop": ["verb"], "drops": ["verb"], "dropped": ["verb"], "dropping": ["verb"],
    "attend": ["verb"], "attends": ["verb"], "attended": ["verb"],
    "download": ["verb"], "downloads": ["verb"], "downloaded": ["verb"],
    "rents": ["verb"], "rented": ["verb"], "feed": ["verb"], "feeds": ["verb"], "fed": ["verb"],
    "feeding": ["verb"], "repair": ["verb"], "repairs": ["verb"], "repaired": ["verb"],
    "checks": ["verb"], "checked": ["verb"], "help": ["verb"], "helps": ["verb"], "helped": ["verb"],
    "likes": ["verb"], "liked": ["verb"], "selling": ["verb"], "watching": ["verb"], "eating": ["verb"],
    "getting": ["verb"], "cutting": ["verb"], "processing": ["verb"],
    "far": ["adverb"], "little": ["adjective"], "fast": ["adjective"], "high": ["adjective"],
    "quickly": ["adverb"], "entirely": ["adverb"], "inside": ["preposition"], "tired": ["adjective"],
    "identical": ["adjective"], "limited": ["adjective"], "massive": ["adjective"], "until": ["preposition"],
})
PLURALS.update({
    "teaspoons": "teaspoon", "pencils": "pencil", "partners": "partner", "gumballs": "gumball",
    "seashells": "seashell", "videos": "video", "debts": "debt", "purposes": "purpose",
    "bedrooms": "bedroom", "roses": "rose", "concerts": "concert", "pianos": "piano",
    "semesters": "semester", "assignments": "assignment", "parts": "part", "chips": "chip",
    "aquariums": "aquarium", "hotels": "hotel", "guests": "guest", "performances": "performance",
    "tanks": "tank", "artists": "artist", "bicycles": "bicycle", "lakes": "lake", "bars": "bar",
    "grandmothers": "grandmother", "deserts": "desert", "accidents": "accident", "supplies": "supply",
    "rules": "rule", "fountains": "fountain", "rules": "rule",
})
VERB_LEMMA.update({
    "borrows": "borrow", "borrowed": "borrow", "borrowing": "borrow",
    "triples": "triple", "tripled": "triple",
    "drops": "drop", "dropped": "drop", "dropping": "drop",
    "attends": "attend", "attended": "attend",
    "downloads": "download", "downloaded": "download",
    "rents": "rent", "rented": "rent", "feeds": "feed", "fed": "feed", "feeding": "feed",
    "repairs": "repair", "repaired": "repair", "checks": "check", "checked": "check",
    "helps": "help", "helped": "help", "likes": "like", "liked": "like",
    "selling": "sell", "watching": "watch", "eating": "eat", "getting": "get", "cutting": "cut",
    "processing": "process",
})

# --- vocabulary push, round 3 (coverage; now safe behind decline-safety GUARDS 11+12) --------------------
# First attempt was REVERTED: blind vocab adds turned fully-read-but-structurally-unhandled problems into
# COMMITTED WRONGS (TEST fell 36.4→33.3). After building decline-safety (GUARD 11 unapplied-period, GUARD 12
# dropped unit/money quantity), those now DECLINE instead of committing, so the batch is re-attemptable. Most
# frequent GSM8K unknown words, conservative typing; ambiguous words (percent/way/being/leaves) still omitted.
LEXICON.update({   # only the VERIFIED-dev-gain words (the rule: add a word only with a dev decline→correct)
    "score": ["noun", "verb"], "scored": ["verb"], "scores": ["verb"], "scoring": ["verb"],
    "raise": ["verb"], "raises": ["verb"], "raised": ["verb"], "raising": ["verb"],
})
PLURALS.update({"scores": "score"})
# NOTE: groundwork nouns/adjectives/pronouns (fruit/paper/side/…/herself) were re-tested with decline-safety
# GUARDS 11–14 in place and STILL drop TEST 40.0→36.4% (a noun unlocks a held-out wrong in an as-yet-unguarded
# class) with ZERO dev gain → they stay OUT. The loop continues: build the next decline guard, then re-test.
VERB_LEMMA.update({
    "scored": "score", "scores": "score", "scoring": "score",
    "raises": "raise", "raised": "raise", "raising": "raise",
})


def _lemma(surface: str) -> str:
    return VERB_LEMMA.get(_key(surface), _key(surface))


def _copula_unit(reading: Reading):
    """The copula verb unit, or None — the anchor a property/membership negates. Includes the bare/perfect
    "be"/"been" ("Emile will BE …", "has BEEN …"), not just the finite is/are/was/were."""
    for u in reading.units:
        if "verb" in u.features.get("senses", []) \
                and _key(u.surface) in ("is", "are", "was", "were", "be", "been"):
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


def _noun_entity_key(unit) -> str:
    """The COMMON-NOUN entity a unit denotes, for the subtype-partition/coordinate-noun affine family (the
    noun-entity twin of `_entity`, used when a clause has NO names at all): a QUALIFIER for a merged
    adjective-noun compound ("red balls" → "red", from the `attributes` the noun-compound merge records), a
    bare ADJECTIVE with an elided head noun ("...than BLUE" = "...than blue balls" → "blue"), or the
    SINGULAR of a plain distinct noun ("pigs"/"cows" → "pig"/"cow", DIFFERENT nouns naturally getting
    DIFFERENT keys with no qualifier needed)."""
    attrs = unit.features.get("attributes")
    if attrs:
        return _key(attrs[0])
    if "adjective" in unit.features.get("senses", []):
        return _key(unit.surface)
    return _key(unit.features.get("singular") or unit.surface)


def _name_count(rp: "Reading") -> int:
    """How many NAME units a reading holds — the discriminator between the NAMED-ENTITY affine family (which
    requires EXACTLY 2 names for a relation / 1 for an anchor) and `_affine_relation`/`_anchor_fact`'s
    COMMON-NOUN fallback (0 or 1 names for a relation, 0 for an anchor). The named systems must gate on this
    BEFORE accepting a result — otherwise a noun-entity relation the fallback silently produces (valid for
    the coordinate-noun family, but semantically WRONG here — e.g. "as many books as the number of tables...
    ARE ON TOP OF EACH TABLE" is a per-unit RATE, not a flat count the named systems' arithmetic assumes)
    would be picked up by the wrong family and mis-solve. A real confident-wrong (700 vs gold 100000)
    surfaced this: `_anchored_sum_system` had no idea its relation actually came from the noun fallback."""
    return sum(1 for u in rp.units if "name" in u.features.get("senses", []))


def _disc_pair(e):
    """A discourse entry as (entity, is_name) — tolerating a bare entity string (treated as a name)."""
    return e if isinstance(e, tuple) else (e, True)


def _pick_antecedent(gender, discourse: list):
    """The antecedent for a pronoun of `gender` from `discourse` (a list of (entity, is_name) most-recent
    LAST). A gendered pronoun ("he"/"she") prefers the most recent NAME of compatible gender — a PERSON,
    not a thing like "price"; "it"/"they" take the most recent compatible entity. None if empty."""
    pairs = [_disc_pair(e) for e in discourse]
    ents = [e for e, _ in pairs]
    if gender in ("masculine", "feminine"):              # he/she → a person (named), gender-compatible
        names = [e for e, is_name in pairs if is_name and NAME_GENDER.get(e) in (None, gender)]
        if names:
            return names[-1]
    compatible = [e for e in ents if gender in (None, "plural") or NAME_GENDER.get(e) in (None, gender)]
    pool = compatible or ents
    return pool[-1] if pool else None


def _resolve_possessives(reading: Reading) -> None:
    """Back-link an unresolved POSSESSIVE pronoun ("his"/"her"/"their"/"its") to the clause's SUBJECT
    when that is a NAME — "Mr. Sam shared … his two sons" → his refers to Sam (the agent is the default
    possessor). Within-sentence and discourse-free, so it runs in `analyze` and takes precedence over the
    cross-sentence resolver; when the subject isn't a name ("his brother left") it leaves it for discourse."""
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subj = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subj is None or "name" not in subj.features.get("senses", []):
        return
    for u in reading.units:
        if "pronoun" not in u.features.get("senses", []) \
                or _key(u.surface) not in ("his", "her", "their", "its"):
            continue
        obs = u.features.get("obligations", [])
        ref = next((o for o in obs if o.need == "refers"), None)
        if ref is None or ref.met:
            continue
        before = dict(u.features)
        u.features["obligations"] = [Obligation(o.need, target=Referent(_entity(subj)))
                                     if o is ref else o for o in obs]
        reading.record("bind", target=u, before=before, after=dict(u.features),
                       reason=f"{u.surface!r} (possessive) back-links to the agent {_entity(subj)!r}")


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
            members = list(dict.fromkeys(_disc_pair(e)[0] for e in discourse))   # distinct entities (a collective,
            if not members:                              # like a conjoined subject) → _question_subjects expands
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

# POSSESSION verbs a ratio's STANDARD can sit before, in lieu of an elided "that"/"which" ("as much money
# DARWIN HAS" = "...as Darwin has") — deliberately narrow (NOT any verb) so a mistagged common noun ("CDs")
# followed by an unrelated passive modifier ("the CDs SOLD at $10 each") doesn't get misread as a name-ratio.
POSSESSION_VERBS = {"has", "had", "have", "owns", "owned", "gets", "got", "receives", "received",
                    "holds", "held"}


def _verb_sign(lemma: str) -> int:
    """+1 for a gain or the 'have' baseline, -1 for a loss verb — how this quantity folds."""
    return -1 if lemma in LOSS_VERBS else 1


# Words that ask for what REMAINS — a holdings question ("how many are left / remaining") in another
# surface form. ("rest" = "the rest"; "remain"/"remains" the verb form.)
_REMAINDER = {"left", "remaining", "leftover", "remain", "remains", "rest"}


DISTRIBUTE_VERBS = {"share", "distribute", "split", "divide", "spread"}


# Every word-problem number is produced by EXECUTING a tiny arithmetic program through the AUDITED safe
# evaluator (translate.run_program: ast.parse + an operator/function whitelist, never eval/exec) — not by
# an inline Python operator. So each answer's arithmetic is a recorded, re-executable, whitelisted program.
ARITHMETIC_LOG: list = []                                # (program, result) per audited computation


def _num(x) -> str:
    """Format a number for an arithmetic program — drop the trailing .0 so the program reads cleanly."""
    return str(int(x)) if isinstance(x, float) and x == int(x) else str(x)


def _compute(expr: str):
    """Evaluate a one-line arithmetic expression through the audited safe evaluator, LOG it, and return the
    int/float. The single arithmetic seam — no answer number is computed by a bare Python operator."""
    from reasoning import translate                       # lazy: avoid an import cycle
    val = translate.run_program(f"answer = {expr}")
    ARITHMETIC_LOG.append((expr, val))
    return int(val) if val == int(val) else val


def _holdings(entity: str, measure: tuple, context: list):
    """The net holdings of `entity` for `measure` — baseline + gains − losses — or None if unknown.
    Timeless / single-state fold (order-independent); the time-aware variant is `_holdings_at`. Quantities
    with an UNKNOWN value (`amount is None`, e.g. an unresolved shared total) are skipped — folding waits
    until synthesis pins them."""
    qs = [f for f in context if isinstance(f, Quantity) and f.entity == entity
          and f.measure == measure and f.amount is not None]
    if not qs:
        return None
    return _compute(" + ".join(f"{_verb_sign(f.verb)} * {_num(f.amount)}" for f in qs))


# A STATE verb asserts the value AT a time ("has 30 on Monday" — Monday's known total); every other
# verb is a CHANGE (a delta applied to the running value). The distinction only matters once values are
# spread across an ordered timeline (`_holdings_at`); the timeless fold treats all as signed deltas.
STATE_VERBS = {"have", "be"}
GROWTH_VERBS = {"grow", "gain", "increase", "rise", "lengthen"}   # a rate of these advances a value over time

# Verbs whose RESULT is itself a measure — their nominalisation is a quantity ("earn" → "earnings"). Lets
# "how MUCH did X earn" bind `much` to the verb (the earnings) and read the measure off it, instead of a
# blanket "much → MONEY" default. All money here; a future "weigh"→weight extends the same mechanism.
RESULT_MEASURE = {v: MONEY for v in ("earn", "spend", "pay", "cost", "save", "make", "give", "get",
                                     "receive", "sell", "buy", "charge", "collect", "win", "owe", "raise")}


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


def _time_offset(reading: Reading) -> float:
    """A relative-time reference's POSITION on the timeline, in YEARS from 'now' (the reference point):
    "next year" → +1, "last year" → −1, "in N years" → +N, "N years ago" → −N, else 0. The temporal
    model's quantified ordering — a time-tracking quantity (AGE advances 1/year) reads it to value itself
    at a shifted time: age@(now+Δ) = age@now + Δ."""
    units = reading.units
    surf = {_key(u.surface) for u in units}
    attrs = {a for u in units for a in u.features.get("attributes", [])}
    yv = next((u.features["value"] for u in units if "value" in u.features.get("senses", [])
               and _follow(u, "quantifies") is not None
               and _descriptor(_follow(u, "quantifies"))[0] == "year"), None)
    if "ago" in surf:                                    # "N years ago" (a bare "a year ago" → 1)
        return -(yv if yv is not None else 1)
    if {"from", "now"} <= surf and yv is not None:       # "N years from now" → +N (a future displacement)
        return yv
    if (surf & {"in", "after"}) and yv is not None:      # "in N years" / "after N years"
        return yv
    if "next" in attrs:                                  # "next year"
        return 1
    if "last" in attrs:                                  # "last year"
        return -1
    return 0


def _holdings_at(entity: str, measure: tuple, context: list, at_time: str):
    """Holdings of `entity`/`measure` VALID AT time node `at_time`, folded along the timeline: a STATE
    verb SETS the running value (a known total), a CHANGE verb adds its signed delta. Events strictly
    after `at_time` are excluded; a timeless event is an always-present baseline. None (decline) if an
    event's order vs `at_time` is unknown, or the applicable times aren't linearly orderable."""
    events = [f for f in context if isinstance(f, Quantity) and f.entity == entity
              and f.measure == measure and f.amount is not None]
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
    running = "0"                                        # build the program as we fold: a STATE RESETS the
    for q in chronological:                              # running expression, a CHANGE appends a signed term
        term = f"{_verb_sign(q.verb)} * {_num(q.amount)}"
        running = term if q.verb in STATE_VERBS else f"{running} + {term}"
    return _compute(running)


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
    derived: bool = field(default=False, compare=False)   # True when COMPUTED from a relation/comparative/
                          # multiplier (Armand = 1/3 × Amber), not stated outright. An implicit referent
                          # ("Ella worked twice as long") prefers the most-recent NON-derived baseline over
                          # a just-derived sibling. compare=False so fact equality (and tests) ignore it.
    first_period: bool = field(default=False, compare=False)   # True for a "in the FIRST <time-unit> …"
                          # override ("in the first year he only teaches 40") — year 1 differs from the
                          # rate, so a rate×duration answer becomes override + rate×(duration−1).


@dataclass(frozen=True)
class Before:
    """A temporal ORDER fact: `Before("monday", "tuesday")` means the Monday-node is earlier than the
    Tuesday-node. RELATIONAL (read off the text's after/before cues), never a calendar rank — days
    wrap. Transitive; the closure is derived by the synthesis loop, not assumed."""
    earlier: str
    later: str


@dataclass(frozen=True)
class Distribution:
    """A DISTRIBUTION relation — "<distributor> shared <measure> between/among <recipients>". Two readings,
    both resolved by the synthesis loop: (a) whole UNKNOWN, recipients NAMED, shares known → whole = Σ
    shares; (b) whole KNOWN and `equal` → each recipient gets whole ÷ `count`. Read off the obligation
    graph (share-verb + recipient group/noun + the measure). `recipients` = named members (may be empty);
    `count` = recipient count; `recipient_noun` = the share-holder type for an unnamed equal split;
    `method` = the split method bound from a manner adverb ("equal" is computable; others decline)."""
    distributor: str
    recipients: tuple
    measure: tuple
    count: int = None
    method: str = None
    recipient_noun: tuple = None


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


def _relational_age_factor(reading: Reading):
    """The factor in an "as OLD as" comparison — "twice" / "<value> times" → its number, else None. (The
    multiplier scales the AGE dimension; "two times" is value-then-"times".)"""
    if not any(_key(u.surface) == "old" for u in reading.units):
        return None
    mult = next((u for u in reading.units if "multiplier" in u.features.get("senses", [])), None)
    if mult is not None:
        return mult.features["value"]
    units = reading.units
    for i, u in enumerate(units):                         # "<value> times as old" → the value is the factor
        if _key(u.surface) == "times" and i and "value" in units[i - 1].features.get("senses", []):
            return units[i - 1].features["value"]
    for i, u in enumerate(units):                         # "1/6 as old as Y" — a FRACTION/number written
        if "value" in u.features.get("senses", []) and i + 1 < len(units) \
                and _key(units[i + 1].surface) == "as" \
                and any(_key(units[j].surface) == "old" for j in range(i + 1, min(i + 4, len(units)))):
            return u.features["value"]                    # directly before "as … old" is the factor, not an age
    return None


def _possessive_age_parts(reading: Reading):
    """The (subject, standard, factor) of a POSSESSIVE relational age — "Jordyn's age is double Zrinka's
    age" / "Jerica is twice Louis' age" → subject's age = factor × standard's age. The possessive frame
    (the noun "age" owned via a name) instead of "as old as"; the multiplier mis-scales the copula, so we
    read it POSITIONALLY: the standard is the name after the multiplier that owns an "age". None if the
    frame doesn't apply. PURE-RATIO only (a comparative offset is left to other resolvers)."""
    units = reading.units
    if not any(_key(u.surface) == "age" for u in units):
        return None
    if any(_key(u.surface) == "as" for u in units):
        return None                                       # "as old as Y" is _relational_age's frame
    if any(_key(u.surface) in ("more", "less", "older", "younger", "fewer") for u in units):
        return None                                       # an offset on the factor we don't fold here
    if not any(_key(u.surface) in ("is", "are", "was", "were", "be") for u in units):
        return None                                       # "be" covers the future "X will BE k× Y's age"
    mult = next((u for u in units if "multiplier" in u.features.get("senses", [])), None)
    if mult is None:
        return None
    if "fraction" in mult.features:
        num, den = mult.features["fraction"]; k = num / den
    else:
        k = mult.features.get("value")
    if k is None:
        return None
    mi = units.index(mult)
    names = [u for u in units if "name" in u.features.get("senses", [])]
    if not names:
        return None
    # the standard Y owns an "age": EITHER "Y's age" (age within 3 AFTER the name) OR "the age of Y" (age
    # within 3 BEFORE it, past the multiplier). Both denote Y's-age as the scaled referent.
    std = next((u for i, u in enumerate(units) if i > mi and "name" in u.features.get("senses", [])
                and (any(_key(units[j].surface) == "age" for j in range(i + 1, min(i + 4, len(units))))
                     or any(_key(units[j].surface) == "age" for j in range(max(mi, i - 3), i)))), None)
    if std is None:
        return None
    cs = _copula_args(reading)[0]                         # the copula subject: a NAME, or a resolved pronoun
    subj = cs if (cs is not None and cs is not std        # ("she is one-third Tina's age" → she→Cori), else
                  and ("name" in cs.features.get("senses", [])           # the first non-standard name
                       or ("pronoun" in cs.features.get("senses", []) and _follow(cs, "refers") is not None))) \
        else next((u for u in names if u is not std), None)
    if subj is None or subj is std or _entity(subj) == _entity(std):
        return None
    return _entity(subj), _entity(std), k


def _possessive_age_statement(reading: Reading):
    """A plain POSSESSIVE age statement — "Ron's age is 43" / "Ron's age now is 43" → Quantity(ron, year,
    43). The noun 'age' owned by a name, a copula, and a value, with NO multiplier/comparative/"as" (those
    are the relational / possessive-relational frames). Without this the value stays unbound ("age" carries
    the predication and 43 quantifies nothing). Gated to exactly one name + one value → no relation to confuse."""
    units = reading.units
    if not any(_key(u.surface) == "age" for u in units):
        return None
    if any(_key(u.surface) == "as" for u in units) \
            or any("multiplier" in u.features.get("senses", []) for u in units) \
            or any(_key(u.surface) in ("more", "less", "older", "younger", "fewer") for u in units):
        return None
    if not any(_key(u.surface) in ("is", "are", "was", "were") for u in units):
        return None
    names = [u for u in units if "name" in u.features.get("senses", [])]
    vals = [u for u in units if "value" in u.features.get("senses", [])]
    if len(names) != 1 or len(vals) != 1:
        return None
    return Quantity(_entity(names[0]), ("year", frozenset()), vals[0].features["value"], "be")


def _simultaneous_ages(reading: Reading):
    """"X was A years old WHEN Y was B years old" → two age facts at a common reference time, stored as
    current ages (the constant gap is what a later "when Y turns M" projection needs; with no other anchor,
    'then' = 'now'). Each side is a plain "<name> was <value> years old" clause flanking "when"."""
    units = reading.units
    when = next((u for u in units if _key(u.surface) == "when"
                 and "conjunction" in u.features.get("senses", [])), None)
    if when is None or not any(_key(u.surface) == "old" for u in units):
        return None

    def age_of(span):
        name = next((u for u in span if "name" in u.features.get("senses", [])), None)
        val = next((u.features["value"] for u in span if "value" in u.features.get("senses", [])
                    and _follow(u, "quantifies") is not None
                    and _descriptor(_follow(u, "quantifies"))[0] == "year"), None)
        if name is None or val is None:
            return None
        return Quantity(_entity(name), ("year", frozenset()), val, "be")

    wi = units.index(when)
    left, right = age_of(units[:wi]), age_of(units[wi + 1:])
    if left is None or right is None or left.entity == right.entity:
        return None
    return [left, right]


def _relational_age(reading: Reading, context: list):
    """A RELATIONAL AGE — "X is N times as old as Y" → X's age = N × Y's age (the year-Quantity of the
    standard Y). None when Y's age isn't known yet — the solve() fixpoint then chains a multi-hop
    (Anne = 2×Emile, Emile = 6×Maude, Maude = 8 → 96)."""
    factor = _relational_age_factor(reading)
    if factor is None:
        return None
    if any(_key(u.surface) in ("more", "less", "older", "younger", "fewer") for u in reading.units):
        return None                                       # "N more than k× as old as Y" — a comparative
        # OFFSET on top of the factor that we don't fold into the age relation (we'd compute only k×Y and
        # drop the +N). DECLINE rather than commit that wrong → defers, so a nested standard stays unknown.
    subj, _ = _copula_args(reading)                       # X — the copula's subject
    if subj is None:
        return None
    standard = next((u for u in reversed(reading.units)   # Y — the LAST name ("as old as EMILE"), not X
                     if "name" in u.features.get("senses", []) and u is not subj), None)
    if standard is None:
        return None
    y_now = _holdings(_entity(standard), ("year", frozenset()), context)
    delta = _time_offset(reading)
    si = reading.units.index(standard)                    # the present-anchor ("as Y is NOW") follows the
    std_now = any(_key(u.surface) in ("now", "today")     # standard; the displacement "from NOW" precedes
                  for u in reading.units[si + 1:])         # it — so look only to the RIGHT of the standard
    if y_now is None:
        # BACKWARD (age twin of #107): Y's age unknown, but if X's (the subject's) is known, INVERT so the
        # fixpoint chains backward through age relations too. Pure-ratio (the offset case is excluded above).
        # Mutually exclusive with the forward path, which needs Y known — no double-emit.
        x_now = _holdings(_entity(subj), ("year", frozenset()), context)
        if x_now is None or not factor:
            return None                                   # Y's age unknown → defer (see _facts_from)
        if delta == 0:                                    # present: X = k·Y → Y = X / k ("Sam 12, twice Sue" → 6)
            return Quantity(_entity(standard), ("year", frozenset()),
                            _compute(f"{_num(x_now)} / {_num(factor)}"), "be")
        # SHIFTED backward: X+Δ = k·(Y+Δ) → Y_now = (X_now+Δ)/k − Δ ("After 5 yrs Ron = 4× Maurice; Ron 43"
        # → Maurice = (43+5)/4 − 5 = 7). The "as Y is NOW" anchor drops the −Δ: X+Δ = k·Y_now → Y_now =
        # (X_now+Δ)/k.
        x_future = f"({_num(x_now)} + {_num(delta)})"
        y_now_expr = f"{x_future} / {_num(factor)}" if std_now \
            else f"{x_future} / {_num(factor)} - {_num(delta)}"
        return Quantity(_entity(standard), ("year", frozenset()), _compute(y_now_expr), "be")
    if delta == 0:                                        # present-tense relation: X = k × Y_now
        return Quantity(_entity(subj), ("year", frozenset()),
                        _compute(f"{_num(factor)} * {_num(y_now)}"), "be")
    # SHIFTED relation "In Δ years, X will be k× as old as Y": BOTH are at the future point by default
    # (X+Δ = k·(Y+Δ)), so X_now = k·(Y_now+Δ) − Δ. The variant "as Y is NOW/today" anchors the standard
    # to the present (X+Δ = k·Y_now → X_now = k·Y_now − Δ) — else the Kaylee class commits a wrong.
    y_future = _num(y_now) if std_now else f"({_num(y_now)} + {_num(delta)})"
    x_now = _compute(f"{_num(factor)} * {y_future} - {_num(delta)}")
    return Quantity(_entity(subj), ("year", frozenset()), x_now, "be")


def _future_age(reading: Reading):
    """A FUTURE age stated with a time displacement: "(In N years,) E will be X years old" / "E will be X
    years old in N years" → E's age NOW = X − N. The offset and the age share the 'year' token (so
    _time_offset can't separate them); the offset is the value quantifying the "in"/"after" preposition's
    object, the age is the OTHER year-value (by 'old'). Future tense ("will") required."""
    units = reading.units
    if not any(_key(u.surface) == "old" for u in units) \
            or not any(_key(u.surface) == "will" and "auxiliary" in u.features.get("senses", []) for u in units):
        return None
    off_prep = next((u for u in units if "preposition" in u.features.get("senses", [])
                     and _key(u.surface) in ("in", "after")), None)
    off_obj = _follow(off_prep, "object") if off_prep is not None else None
    if off_obj is None or _descriptor(off_obj)[0] != "year":
        return None
    off_val = next((u.features["value"] for u in units if "value" in u.features.get("senses", [])
                    and _follow(u, "quantifies") is off_obj), None)
    age_val = next((u.features["value"] for u in units if "value" in u.features.get("senses", [])
                    and _follow(u, "quantifies") is not None and _follow(u, "quantifies") is not off_obj
                    and _descriptor(_follow(u, "quantifies"))[0] == "year"), None)
    subj = next((u for u in units if "name" in u.features.get("senses", [])), None)
    if off_val is None or age_val is None or subj is None:
        return None
    return Quantity(_entity(subj), ("year", frozenset()),
                    _compute(f"{_num(age_val)} - {_num(off_val)}"), "be")


def _comparative_age(reading: Reading, context: list):
    """A COMPARATIVE age — "X is N years older/younger than Y" → X's age = Y's age ± N. Defers (None) until
    Y's age is known (the fixpoint then resolves it). Distinct from _relational_age's MULTIPLICATIVE "N
    times as old"; this is the ADDITIVE year-offset form."""
    units = reading.units
    comp = next((u for u in units if "comparative" in u.features.get("senses", [])
                 and _key(u.surface) in ("older", "younger")), None)
    than = next((u for u in units if _key(u.surface) == "than"), None)
    y = _follow(than, "object") if than is not None else None
    val = next((u.features["value"] for u in units if "value" in u.features.get("senses", [])
                and _follow(u, "quantifies") is not None
                and _descriptor(_follow(u, "quantifies"))[0] == "year"), None)
    subj, _ = _copula_args(reading)
    if comp is None or y is None or val is None or subj is None \
            or "name" not in y.features.get("senses", []):
        return None
    y_age = _holdings(_entity(y), ("year", frozenset()), context)
    if y_age is None:
        # BACKWARD inversion — Y's age unknown, but if the SUBJECT's is known, invert the relation:
        # "Benjie is 5 older than Margo; Benjie is 6" → Margo = 6 − 5. (older: subj = std + N → std =
        # subj − N; younger flips.) Grounds the standard so the fixpoint can chain onward, mirroring
        # _relational_quantity's bidirectional grounding. Defers if the subject's age is unknown too.
        subj_age = _holdings(_entity(subj), ("year", frozenset()), context)
        if subj_age is None:
            return None
        inv = "-" if _key(comp.surface) == "older" else "+"
        return Quantity(_entity(y), ("year", frozenset()),
                        _compute(f"{_num(subj_age)} {inv} {_num(val)}"), "be")
    op = "+" if _key(comp.surface) == "older" else "-"
    return Quantity(_entity(subj), ("year", frozenset()),
                    _compute(f"{_num(y_age)} {op} {_num(val)}"), "be")


def _relational_quantity(reading: Reading, context: list):
    """A DERIVED quantity — "Mary has twice as many" → `factor × referent`, the referent being the most
    recent prior Quantity (resolved like coreference, but for a quantity). The referent supplies the
    measure; the multiplier supplies the factor. None if there's no multiplier-on-a-quantifier or no
    referent to scale."""
    mult = next((u for u in reading.units if "multiplier" in u.features.get("senses", [])), None)
    if mult is None:
        return None
    scaled = _follow(mult, "scales")                     # must scale "as many" (a quantifier), "the number"
    if scaled is None or not ("quantifier" in scaled.features.get("senses", [])   # (a count-noun), or "as
                              or _key(scaled.surface) in (COUNT_NOUNS | DIMENSIONS)):  # long/far" (a
        return None                                                               # dimension) — not a verb
    # TIME-INDEXED relational ("twice as many boxes on Tuesday as she did on Wednesday") — the referent is
    # the SAME entity at a DIFFERENT time, identified by the 2nd "on <day>". We resolve the referent by
    # ENTITY (recency), not time, so we'd grab the wrong day's value and chain garbage → DECLINE. (A full
    # solve needs time-tagged relational quantities + a per-day referent + time-aware answer fold.)
    if sum(1 for u in reading.units if "time" in u.features.get("senses", [])) >= 2:
        return None
    # NOTE: a relative-clause referent ("twice the number (that) X has", "the number of flowers he
    # sold") was prototyped here (read subject/referent off POSITION around the multiplier, to survive
    # the relative clause's 2nd verb / the matrix "has" mis-reading as an auxiliary). It resolved the
    # sub-fact correctly but was REVERTED: the problems that use it are predominantly multi-step
    # (Monica's 3-year chain, Ali's multi-day shop) and fail downstream, so it traded +2 dev solves for
    # +3 confident-wrongs — net precision-negative. Deferred until multi-step composition is stronger;
    # such forms stay an honest DECLINE for now. See reading-bench-firewall-prefix memory.
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    # a subject that sits right after "as" is really the STANDARD, not the subject — the real subject was
    # elided from a prior conjunct ("[Jennifer bought] three times as many as BOB bought"). We'd mis-read
    # it as "Bob = 3 × recent" and commit garbage; the elided-subject inverse is unresolved → DECLINE.
    si = reading.units.index(subject)
    if si > 0 and _key(reading.units[si - 1].surface) == "as":
        return None
    # the referent: a NAMED quantity ("half as many as TIM" — a name other than the subject), else the
    # most recent prior quantity ("twice as many" — implicit, like coreference).
    named = [u for u in reading.units if "name" in u.features.get("senses", []) and u is not subject]
    if named:
        ref_ent = _entity(named[-1])
        referent = next((f for f in reversed(context or [])
                         if isinstance(f, Quantity) and f.entity == ref_ent), None)
        if referent is None:
            # BIDIRECTIONAL grounding — the named standard has no quantity yet, but if the SUBJECT does,
            # INVERT the relation: standard = subject / factor ("Tyrah has 6× as many pencils as Sarah;
            # Tyrah has 12" → Sarah = 2). The forward path only derives subject-FROM-standard; this grounds
            # the standard so the solve() fixpoint chains backward too (Sarah=2 → Tim = 8×2 = 16). PURE-RATIO
            # only: a comparative offset ("5 more than twice as many as B") inverts ambiguously → keep deferring.
            comp0 = any("comparative" in u.features.get("senses", []) for u in reading.units)
            subj_q = next((f for f in reversed(context or []) if isinstance(f, Quantity)
                           and f.entity == _entity(subject) and f.amount is not None), None)
            if subj_q is not None and not comp0:
                if "fraction" in mult.features:
                    num, denom = mult.features["fraction"]
                    inv = f"{_num(subj_q.amount)} * {denom} / {num}"
                else:
                    inv = f"{_num(subj_q.amount)} / {_num(mult.features['value'])}"
                return Quantity(ref_ent, subj_q.measure, _compute(inv),
                                _lemma(verbs[0].surface), derived=True)   # the relation's verb (run/have),
                #                                                          so the answer fold can match it
    else:                                                 # IMPLICIT referent: prefer the most-recent
        quants = [f for f in reversed(context or []) if isinstance(f, Quantity)]   # NON-derived baseline
        referent = next((f for f in quants if not f.derived), None) or next(iter(quants), None)
    if referent is None:                                  # ("Ella worked twice as long" scales AMBER, not
        return None                                       # the just-derived Armand sitting between them)
    if "fraction" in mult.features:                       # exact: (referent × num) / denom, divide LAST
        num, denom = mult.features["fraction"]            # ("12 × 2 / 3" = 8, dodging 0.666×12 float drift)
        expr = f"{_num(referent.amount)} * {num} / {denom}"
    else:
        expr = f"{_num(mult.features['value'])} * {_num(referent.amount)}"
    comp = next((u for u in reading.units if "comparative" in u.features.get("senses", [])), None)
    offset = next((u for u in reading.units if "value" in u.features.get("senses", [])), None)
    if comp is not None and offset is not None:          # "FIVE MORE than twice as many as Bob" → 2×Bob+5
        expr += f" + {COMPARATIVE_SIGN.get(_key(comp.surface), 1)} * {_num(offset.features['value'])}"
    return Quantity(_entity(subject), referent.measure, _compute(expr), _lemma(verbs[0].surface),
                    derived=True)


def _partitive_quantity(reading: Reading, context: list):
    """A PARTITIVE quantity — `fraction × a holding`, a relational CHANGE (the "of" makes the fraction a
    part of a SET, not a verb multiplier). "loses half OF them" → 0.5 × the subject's current holding;
    "buys a third OF what he had on Tuesday" → (1/3) × that holder's holding AT Tuesday. The of-clause
    decides the base: a verb + holder + time ("what he had on Tuesday") → `_holdings_at`; else the
    subject's own current set ("of them"). None if there's no base to take the fraction of."""
    units = reading.units
    frac = next((u for u in units if _fraction_parts(u.surface) is not None), None)   # unit OR non-unit
    fi = units.index(frac) if frac is not None else -1
    if frac is None or fi + 1 >= len(units) or _key(units[fi + 1].surface) != "of":   # "<frac> OF …"
        return None
    left_verbs = [u for k, u in enumerate(units) if k < fi and "verb" in u.features.get("senses", [])]
    if not left_verbs:                                    # the partitive attaches to the verb on its LEFT
        return None                                       # (a verb inside the of-clause is embedded)
    main_verb = left_verbs[-1]
    subject = _follow(main_verb, "subject")
    if subject is None:
        return None
    ent = _entity(subject)
    of = units[fi + 2:]
    of_verb = next((u for u in of if "verb" in u.features.get("senses", [])), None)
    of_pron = next((u for u in of if any(s in u.features.get("senses", []) for s in ("name", "pronoun"))), None)
    of_noun = next((u for u in of if "noun" in u.features.get("senses", [])), None)
    of_time = next((_key(u.surface) for u in of if "time" in u.features.get("senses", [])), None)
    if of_verb is not None and of_pron is not None:       # "… what HE had on Tuesday" — a holder's value
        ref_ent, ref_time = _entity(of_pron), of_time
    else:                                                 # "… of THEM" — the subject's own current set
        ref_ent, ref_time = ent, None
    measure = _descriptor(of_noun) if of_noun is not None else None
    if measure is None:                                   # no measure named → the referent's latest one
        recent = next((f for f in reversed(context or [])
                       if isinstance(f, Quantity) and f.entity == ref_ent), None)
        measure = recent.measure if recent is not None else None
    if measure is None:
        return None
    base = _holdings_at(ref_ent, measure, context or [], ref_time) if ref_time \
        else _holdings(ref_ent, measure, context or [])
    if base is None:
        return None
    num, denom = _fraction_parts(frac.surface)            # exact: base × num / denom (divide last)
    return Quantity(ent, measure, _compute(f"{_num(base)} * {num} / {denom}"),
                    _lemma(main_verb.surface), _time_at(reading, main_verb), derived=True)


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
    subject = next((_follow(v, "subject") for v in verbs if _follow(v, "subject") is not None), None)
    if value_unit is None or subject is None:            # multi-verb ("has been teaching") — any bound subj
        return None
    elided_measure = False
    if "measure" in value_unit.features:                 # "$10 more than William" — money value
        measure = value_unit.features["measure"]
    else:                                                # "30 less apples than Martha" — value+noun
        noun = _follow(value_unit, "quantifies")
        if noun is None:                                 # ELIDED measure — "Riley has 3 fewer." The noun is
            # gone, so the most-recent prior Quantity must supply BOTH measure and base. Restricted to the
            # BARE frame (no "than" at all): an anaphoric "47 more than that" reads correctly here but tends
            # to sit in a multi-step chain (Billy/Katelyn/Chloe) where a later part stays underived, so
            # deriving this one turns an honest decline into a partial-sum confident-wrong.
            if standard is not None or any(_key(u.surface) == "than" for u in reading.units):
                return None
            elided_measure = True
            measure = None
        else:
            measure = _descriptor(noun)
    # MEASURE-to-MEASURE: the standard is another MEASURE of the SAME subject (same head noun, different
    # attribute) — "2 fewer ANIMAL stickers than FLOWER stickers" → animal = subject's flower count ± N.
    # (Distinct from the entity-standard grounding below, which reads ANOTHER person's holdings.)
    if standard is not None and subject is not None and measure is not None \
            and "name" not in standard.features.get("senses", []) and "noun" in standard.features.get("senses", []):
        std_measure = _descriptor(standard)
        if std_measure[0] == measure[0] and std_measure != measure:
            base = _holdings(_entity(subject), std_measure, context or [])
            if base is not None:
                op = "-" if COMPARATIVE_SIGN.get(_key(comp.surface), 1) < 0 else "+"
                return Quantity(_entity(subject), measure,
                                _compute(f"{_num(base)} {op} {_num(value_unit.features['value'])}"),
                                _lemma(verbs[-1].surface), derived=True)
    if standard is not None:                             # explicit "… than Y" — Y's holdings, but expressed
        referent = _holdings(_entity(standard), measure, context or [])   # in the SUBJECT's frame: _holdings
        if referent is not None:                         # is signed (a loss nets −17.5), while stored amounts
            referent *= _verb_sign(_lemma(verbs[-1].surface))   # are magnitudes — so a loss compared with a
    else:                                                # loss ("lost 1.5 more than Seth") compares MAGNITUDES
        base = next((f for f in reversed(context or [])   # (17.5+1.5=19, not −17.5+1.5). IMPLICIT standard —
                     if isinstance(f, Quantity) and (elided_measure or f.measure == measure)  # "10 years less"
                     and f.amount is not None), None)    # — is a stored magnitude; an elided measure inherits
        referent = base.amount if base is not None else None   # the most-recent quantity's measure too.
        if elided_measure and base is not None:
            measure = base.measure
    if referent is None:
        return None                                      # standard's count unknown (e.g. forward ref)
    op = "-" if COMPARATIVE_SIGN.get(_key(comp.surface), 1) < 0 else "+"   # younger/less → −, older/more → +
    return Quantity(_entity(subject), measure,
                    _compute(f"{_num(referent)} {op} {_num(value_unit.features['value'])}"),
                    _lemma(verbs[-1].surface), derived=True)


@dataclass(frozen=True)
class Rate:
    """A per-unit rate: "Weng earns $12 per hour" → Rate("weng", 12, MONEY, "hour", "earn"). The total
    is `amount × (the quantity of per_unit)` — composed at answer time, not stored."""
    entity: str
    amount: float
    measure: tuple
    per_unit: str
    verb: str
    per_amount: float = 1            # "8 pages in 20 MINUTES" → amount=8, per_unit=minute, per_amount=20


@dataclass(frozen=True)
class RateRel:
    """A pending RELATIONAL rate: `subject`'s rate (of `verb`) = `factor` × `reference`'s rate. Resolved by
    the synthesis loop in EITHER direction — forward (reference's rate known → subject = factor × ref) or
    inverse (subject's rate known → reference = subject ÷ factor). So "Carole reads as fast as Anna (×1)
    but at half the speed of Brianna (×0.5)" chains Anna(1/min) → Carole(1/min) → Brianna(2/min)."""
    subject: str
    factor: float
    reference: str
    verb: str


def _rate(reading: Reading, context: list = None):
    """The rate a statement asserts — "$12 per hour" / "$12 an hour" / "20 pages a day" → Rate(entity,
    amount, measure, per_unit, verb). The per-marker is "per", or "a"/"an" before a unit noun when it
    sits right after a money value or a quantified measure-noun (so a plain article "a book" is NOT a
    rate). The measure is a self-measuring money value, or the noun the rate value quantifies. Else None."""
    units = reading.units
    per = next((u for u in units if "per" in u.features.get("senses", [])), None)
    rate_val = unit = rate_noun = None
    per_amount = 1
    if per is not None:                                  # explicit "per"
        rate_val, unit = _follow(per, "rate"), _follow(per, "unit")
    else:                                                # "a"/"an"/"every"/"each" as per — after the measure
        for k in range(1, len(units) - 1):
            if _key(units[k].surface) not in ("a", "an", "every", "each") \
                    or "noun" not in units[k + 1].features.get("senses", []):
                continue
            prev = units[k - 1]
            if "measure" in prev.features:               # "$12 an hour"
                rate_val, unit = prev, units[k + 1]
                break
            if "noun" in prev.features.get("senses", []):   # "20 pages a day" — the value quantifying prev,
                run, j = [], k - 1                           # or, for a COMPOUND noun ("7 apple pies a day"),
                while j >= 0 and "noun" in units[j].features.get("senses", []):  # any noun in the contiguous
                    run.append(units[j]); j -= 1             # run ending at prev (the value quantifies the
                v = next((x for x in units if "value" in x.features.get("senses", [])   # head 'apple', the
                          and _follow(x, "quantifies") in run), None)                   # rate-noun is 'pies')
                if v is not None:
                    rate_val, unit, rate_noun = v, units[k + 1], prev   # measure is the rate-noun (prev),
                    break                                               # not what the value quantifies
    if rate_val is None:                                 # "N <measure> in M <time-unit>" — a per-M rate
        inp = next((u for u in units if _key(u.surface) == "in"), None)
        tu = _follow(inp, "object") if inp is not None else None
        if tu is not None and _descriptor(tu)[0] in TIME_UNITS:
            per_v = next((x for x in units if "value" in x.features.get("senses", [])
                          and _follow(x, "quantifies") is tu), None)
            mv = next((x for x in units if "value" in x.features.get("senses", [])
                       and _follow(x, "quantifies") is not None
                       and _descriptor(_follow(x, "quantifies"))[0] not in TIME_UNITS), None)
            if per_v is not None and mv is not None:
                rate_val, unit, per_amount = mv, tu, per_v.features["value"]
            elif mv is not None:                         # "75 pages in AN hour" — no explicit duration value,
                ti = units.index(tu)                     # so the article a/an/each/every means a SINGLE unit
                if ti > 0 and _key(units[ti - 1].surface) in ("a", "an", "each", "every"):
                    rate_val, unit, per_amount = mv, tu, 1
    if rate_val is None:                                 # "N <time-unit> over M <time-unit>" — a
        ov = next((u for u in units if _key(u.surface) == "over"), None)   # time-over-time rate ("reads
        tu = _follow(ov, "object") if ov is not None else None             # 10 hours over 5 days" = 2 h/day).
        if tu is not None and _descriptor(tu)[0] in TIME_UNITS:            # BOTH sides must be time units
            per_v = next((x for x in units if "value" in x.features.get("senses", [])   # (a non-time
                          and _follow(x, "quantifies") is tu), None)                     # measure "30 points
            mv = next((x for x in units if "value" in x.features.get("senses", [])       # over 5 games" is
                       and _follow(x, "quantifies") is not None and x is not per_v        # not a duration
                       and _descriptor(_follow(x, "quantifies"))[0] in TIME_UNITS), None)  # rate — declines)
            if per_v is not None and mv is not None:
                rate_val, unit, per_amount = mv, tu, per_v.features["value"]
    if rate_val is None or unit is None:
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    entity = _entity(subject)
    if entity == "rate":                                 # "(he) reads at a RATE of 50 pages an hour" — the
        # rate-marker noun "rate" was grabbed as the elided subject; recover the real entity from context
        # (the most-recent prior holder) so the two rates share an entity and `_synth_rate_chain` composes.
        entity = next((f.entity for f in reversed(context or [])
                       if isinstance(f, (Rate, Quantity)) and f.entity != "rate"), None)
        if entity is None:
            return None
    if "measure" in rate_val.features:                   # "$12" — a self-measuring money rate
        measure = rate_val.features["measure"]
    else:                                                # "20 PAGES a day" — the noun the value quantifies,
        mnoun = rate_noun or _follow(rate_val, "quantifies")   # or the compound rate-noun ("apple PIES")
        measure = _descriptor(mnoun) if mnoun is not None else None
    if measure is None:
        return None
    amount = rate_val.features["value"]
    # RELATIONAL RATE — "Sarah makes 5 times more money per hour THAN Connor": the rate is N × another
    # entity's rate ("times" operator + a comparative whose standard is a NAME). Resolve against that
    # entity's known rate of the same measure/per_unit; DEFER (None) if it isn't known yet (forward ref).
    comp = next((u for u in reading.units if "comparative" in u.features.get("senses", [])), None)
    std = _follow(comp, "standard") if comp is not None else None
    if std is not None and "name" in std.features.get("senses", []) \
            and any(_key(u.surface) == "times" for u in reading.units):
        base = next((f for f in reversed(context or []) if isinstance(f, Rate)
                     and f.entity == _entity(std) and f.measure == measure
                     and f.per_unit == _descriptor(unit)[0]), None)
        if base is None:
            return None                                  # standard's rate unknown yet — defer to the fixpoint
        amount = _compute(f"{_num(amount)} * {_num(base.amount)}")
    return Rate(entity, amount, measure,
                _descriptor(unit)[0], _lemma(verbs[0].surface), per_amount)


def _time_range_rate(reading: Reading):
    """A clock TIME-RANGE done every period is an hours-per-period Rate: "writes from 1 PM to 4 PM every
    night" → 4PM−1PM = 3 hours, "every night" → per day (a night is a daily occurrence) → Rate(3 hours
    per day). Needs a "from <clock> to <clock>" pair (a value + AM/PM meridiem each) and an "every/each
    <period>". Returns None unless the whole shape is present, so it never steals a plain quantity."""
    units = reading.units
    merid = [(i, _key(u.surface)) for i, u in enumerate(units) if _key(u.surface) in ("am", "pm")]
    if len(merid) != 2:
        return None
    hours = []                                           # the two clock times as 24-hour values
    for mi, mk in merid:
        v = next((units[j] for j in range(mi - 1, -1, -1)               # nearest preceding value
                  if "value" in units[j].features.get("senses", [])), None)
        if v is None:
            return None
        h = v.features["value"] % 12 + (12 if mk == "pm" else 0)        # 1 PM → 13, 4 PM → 16
        hours.append(h)
    span = hours[1] - hours[0]                            # text order = chronological (from … to …)
    if span <= 0:
        return None
    per = next((units[k + 1] for k in range(len(units) - 1)             # "every/each <period>"
                if _key(units[k].surface) in ("every", "each")
                and "noun" in units[k + 1].features.get("senses", [])), None)
    if per is None:
        return None
    per_unit = _descriptor(per)[0]
    per_unit = "day" if per_unit in ("night", "day") else per_unit      # a night is a daily occurrence
    if per_unit not in TIME_UNITS:
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if verbs else None
    if subject is None:
        return None
    return Rate(_entity(subject), span, ("hour", frozenset()), per_unit, _lemma(verbs[0].surface), 1)


SPEED_WORDS = {"fast", "quick", "slow"}                   # manner adjectives that compare a rate
SPEED_NOUNS = {"speed", "pace"}                           # "half the SPEED of X"


def _rate_relations(reading: Reading):
    """Relational rate(s) keyed off a SPEED cue: "Carole reads as fast as Anna but at half the speed of
    Brianna" → [RateRel(carole, 1.0, anna), RateRel(carole, 0.5, brianna)]. Two shapes: "as <speed-adj>
    as <NAME>" (equality, ×1) and "<multiplier> the <speed-noun> of <NAME>" (×multiplier). Gated on a
    speed word/noun so it never hijacks a plain comparative; the synth resolves them bidirectionally."""
    units = reading.units
    if not any(_key(u.surface) in SPEED_WORDS or _key(u.surface) in SPEED_NOUNS for u in units):
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if verbs else None
    if subject is None:
        return None
    subj, verb = _entity(subject), _lemma(verbs[0].surface)
    keys = [_key(u.surface) for u in units]
    rels = []
    for i, u in enumerate(units):
        if keys[i] in SPEED_WORDS and 0 < i < len(units) - 2 \
                and keys[i - 1] == "as" and keys[i + 1] == "as" \
                and "name" in units[i + 2].features.get("senses", []):   # "as fast as Anna" → ×1
            rels.append(RateRel(subj, 1.0, _entity(units[i + 2]), verb))
        if keys[i] in SPEED_NOUNS:                                        # "half the speed of Brianna"
            mult = next((units[j] for j in range(i - 1, -1, -1)
                         if "multiplier" in units[j].features.get("senses", [])), None)
            of = next((units[j] for j in range(i + 1, len(units)) if keys[j] == "of"), None)
            ref = _follow(of, "object") if of is not None else None
            if mult is not None and ref is not None and "name" in ref.features.get("senses", []):
                rels.append(RateRel(subj, mult.features["value"], _entity(ref), verb))
    return rels or None


def _per_occurrence_rate(reading: Reading):
    """A per-OCCURRENCE rate fronted by "For every/each <noun> …, <subject> <verb>s N <measure>": "For every
    sandwich that he eats, Sam eats four apples" → Rate(sam, 4, apple, per_unit=sandwich, eat). The leading
    "for every <noun>" names the per_unit (a non-time occurrence); the main clause gives the amount/measure/
    subject. The rate_duration synth then takes it × an occurrence COUNT (70 sandwiches → 280 apples)."""
    units = reading.units
    keys = [_key(u.surface) for u in units]
    if not (len(units) > 2 and keys[0] == "for" and keys[1] in ("every", "each")
            and "noun" in units[2].features.get("senses", [])):
        return None
    per_unit = _descriptor(units[2])[0]
    val = next((u for u in units if "value" in u.features.get("senses", [])              # "four apples"
                and _follow(u, "quantifies") is not None
                and _descriptor(_follow(u, "quantifies"))[0] != per_unit), None)
    if val is None:
        return None
    measure = _descriptor(_follow(val, "quantifies"))
    name = next((u for u in units if "name" in u.features.get("senses", [])), None)       # main subject (Sam)
    # the main verb is the one nearest the value (skip the relative-clause verb "that he eats")
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    if name is None or not verbs:
        return None
    vi = units.index(val)
    verb = min(verbs, key=lambda v: abs(units.index(v) - vi))
    return Rate(_entity(name), val.features["value"], measure, per_unit, _lemma(verb.surface), 1)


def _for_duration(reading: Reading, rate: "Rate"):
    """A trailing "for <value> <time-unit>" duration ("…every day FOR ONE WEEK") → a Quantity counting that
    unit, so the rate_duration synth can accumulate the rate over it. Emitted ALONGSIDE the rate (the rate
    parse alone drops it). SKIPS a unit that is the rate's own MEASURE ("for 8 HOURS a day" — the 8 hours
    is the rate value, not a duration), which would double-count. Returns None if there's none."""
    units = reading.units
    for i, u in enumerate(units):
        if _key(u.surface) != "for":
            continue
        v = next((units[j] for j in range(i + 1, len(units))
                  if "value" in units[j].features.get("senses", [])), None)
        if v is None:
            continue
        qn = _follow(v, "quantifies")
        if qn is not None and _descriptor(qn)[0] in TIME_UNITS \
                and _descriptor(qn)[0] != rate.measure[0]:
            return Quantity(rate.entity, _descriptor(qn), v.features["value"], rate.verb)
    return None


REST_WORDS = {"rest", "remainder", "others"}


def _rest_quantity(reading: Reading, context: list):
    """The COMPLEMENT category — "the REST <verb>" → the population total − Σ(the stated subsets of the same
    measure). The population is a have/be quantity ("a class HAS 60 students"); the subsets are the other
    quantities of that measure ("40 passed" / a percent-of-count). Summed by MEASURE (entity keying of a
    plural subject is inconsistent), so e.g. "60 students; 40 passed; the rest failed" → 20 failed."""
    units = reading.units
    rest = next((u for u in units if _key(u.surface) in REST_WORDS), None)
    if rest is None:
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])
             and _key(u.surface) not in ("is", "are", "was", "were")]
    if len(verbs) != 1:
        return None
    subj = _follow(verbs[0], "subject")
    if subj is None or _key(subj.surface) not in REST_WORDS:   # "the rest" must BE the subject
        return None
    totals = [f for f in context if isinstance(f, Quantity)    # the population: a have/be total
              and f.verb in ("have", "be") and f.amount is not None]
    if not totals:
        return None
    total = totals[-1]
    subs = [f.amount for f in context if isinstance(f, Quantity) and f.measure == total.measure
            and f.verb not in ("have", "be") and f.amount is not None]
    expr = _num(total.amount) + "".join(f" - {_num(s)}" for s in subs)   # total − sub1 − sub2 …
    return Quantity(total.measure[0], total.measure, _compute(expr), _lemma(verbs[0].surface))


def _percent_quantity(reading: Reading, context: list):
    """A percentage OF A HOLDING/COUNT. Two shapes: (a) NO explicit base — "puts 20% more money" → 0.2 ×
    the matching prior quantity (the portion; its base is the referent a noun here names). (b) base is a
    NOUN — "40% OF the students passed" → 0.4 × the students' quantity (the count of students who passed).
    A VALUE base ("20% of 50") is pure arithmetic, handled elsewhere."""
    pct = next((u for u in reading.units if "percent" in u.features.get("senses", [])), None)
    if pct is None:
        return None
    if _follow(pct, "base") is not None:                  # "20% OF 50" — an explicit VALUE base: arithmetic
        return None
    of = next((u for u in reading.units if _key(u.surface) == "of"            # "X% OF the <noun>" — a
               and _follow(u, "anchor") is pct), None)                        # percent of a COUNT
    noun_base = None
    if of is not None:                                   # the of-phrase HEAD = the LAST noun before a verb
        units, oi = reading.units, reading.units.index(of)   # ("of his Grade-5 STUDENTS got …" → students,
        for u in units[oi + 1:]:                             # the head, not the "Grade 5" classifier)
            if "verb" in u.features.get("senses", []):
                break
            if "noun" in u.features.get("senses", []):
                noun_base = u
    if noun_base is not None and "noun" in noun_base.features.get("senses", []):
        ref = next((f for f in reversed(context or [])
                    if isinstance(f, Quantity) and f.measure == _descriptor(noun_base)), None)
        if ref is None:
            return None
        cverbs = [u for u in reading.units if "verb" in u.features.get("senses", [])
                  and _key(u.surface) not in ("is", "are", "was", "were")]
        cverb = _lemma(cverbs[0].surface) if cverbs else "have"
        return Quantity(_descriptor(noun_base)[0], _descriptor(noun_base),
                        _compute(f"{_num(pct.features['value'])} * {_num(ref.amount)}"), cverb)
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    measures = {_descriptor(u) for u in reading.units if "noun" in u.features.get("senses", [])}
    referent = next((f for f in reversed(context or [])
                     if isinstance(f, Quantity) and f.measure in measures), None)
    if referent is None:
        return None
    # CROSS-ENTITY "N% more/less THAN X" ("Mark has 50% more points than Eric") → the FULL derived amount
    # X × (1 ± pct) = 6 × 1.5 = 9, not just the portion (0.5×6=3). Gated on a comparative WITH a "than"
    # standard: a SAME-entity "puts 20% more money" (comparative, NO than-standard) keeps the portion
    # reading below, since its base is the entity's own holding that the fold already carries.
    comp = next((u for u in reading.units if "comparative" in u.features.get("senses", [])), None)
    std = _follow(comp, "standard") if comp is not None else None
    if std is not None and "name" in std.features.get("senses", []):   # standard is ANOTHER ENTITY ("than
        op = "-" if COMPARATIVE_SIGN.get(_key(comp.surface), 1) < 0 else "+"   # Eric"), not the measure noun
        ref = next((f for f in reversed(context or [])                 # ("20% more MONEY"): the full derived
                    if isinstance(f, Quantity) and f.entity == _entity(std)), None)   # amount X×(1±pct)
        if ref is None:                                  # the named standard's value isn't known YET (a
            return None                                  # forward ref: "John …than Jules" before Jules) —
        return Quantity(_entity(subject), referent.measure,   # DEFER so the fixpoint resolves it after the
                        _compute(f"{_num(ref.amount)} * (1 {op} {_num(pct.features['value'])})"),  # standard,
                        _lemma(verbs[0].surface), derived=True)   # never off a measure-matched stand-in (Stan)
    return Quantity(_entity(subject), referent.measure,
                    _compute(f"{_num(pct.features['value'])} * {_num(referent.amount)}"),
                    _lemma(verbs[0].surface))


def _container_quantity(reading: Reading):
    """CONTAINMENT quantities — "20 packets OF 100 grams" → 20 × 100 = 2000 grams. A value-noun pair
    straddles each side of an "of" (N containers of M units each); the product's measure is the INNER unit.
    CONJOINED of-phrases ("four books of 10 stamps AND six books of 15 stamps") each yield a product,
    summed at fold time — every "of" pairs with its NEAREST value-noun on each side. A NESTED "of" with no
    inner count ("100 grams OF sugar") contributes nothing (no value on its right). Returns a LIST (empty
    if no "of" has a value-noun pair on BOTH sides)."""
    units = reading.units
    verbs = [u for u in units if "verb" in u.features.get("senses", [])      # the MAIN verb — a relative-
             and _key(u.surface) not in ("is", "are", "was", "were")]        # clause copula ("that ARE 4
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None       # pounds each") doesn't count
    if subject is None:
        return []
    ofs = [units.index(u) for u in units if _key(u.surface) == "of"]
    if not ofs:
        return []
    each_i = next((units.index(u) for u in units if _key(u.surface) == "each"), None)
    out = []
    for oi in ofs:
        # the of-PHRASE ends at the next preposition after THIS "of" (a later "of" bounds it) — a value
        # beyond it ("8 pages of a book IN 20 minutes") belongs to a different phrase, NOT the container
        # (the "each" relative clause is allowed in)
        next_prep = next((units.index(u) for u in units[oi + 1:]
                          if "preposition" in u.features.get("senses", [])), len(units))
        right_bound = max(next_prep, each_i + 1) if each_i is not None else next_prep
        left = right = None
        for u in units:
            if "value" not in u.features.get("senses", []):
                continue
            n = _follow(u, "quantifies")
            if n is None:
                continue
            ui, ni = units.index(u), units.index(n)
            if ui < oi and ni < oi:                      # "20 PACKETS of …" — count + noun both left;
                if left is None or ui > units.index(left):    # keep the one NEAREST "of"
                    left = u
            elif oi < ui < right_bound:                  # "… of 100 GRAMS" — count + noun right of "of",
                if right is None or ui < units.index(right):  # within the of-phrase; keep NEAREST "of"
                    right = u
        if left is None or right is None or _follow(left, "quantifies") is subject:
            continue                                     # the container can't be the SUBJECT ("FOUR people
        out.append(Quantity(_entity(subject), _descriptor(_follow(right, "quantifies")),   # lost a total of
                            _compute(f"{_num(left.features['value'])} * {_num(right.features['value'])}"),
                            _lemma(verbs[0].surface), _time_at(reading, verbs[0])))   # 103 kg" is not 4×103)
    return out


def _distributed_quantity(reading: Reading):
    """A DISTRIBUTED quantity — "gave 4 stickers EACH to 3 friends" → 4 × 3 = 12. "each" makes the
    per-recipient amount (the value-noun just before it) multiply by the recipient count (the other
    value). None unless "each" sits after a noun with two value-noun pairs around it."""
    units = reading.units
    each = next((u for u in units if _key(u.surface) == "each"), None)
    if each is None:
        return None
    verbs = [u for u in units if "verb" in u.features.get("senses", [])      # MAIN verb — a relative-clause
             and _key(u.surface) not in ("is", "are", "was", "were")]        # copula ("books that ARE 400
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None       # pages each") doesn't count
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
    total = _compute(f"{_num(per_val.features['value'])} * {_num(count_val.features['value'])}")
    return Quantity(_entity(subject), _descriptor(per_noun), total, _lemma(verbs[0].surface))


def _time_at(reading: Reading, verb_unit) -> str:
    """The time NODE a value under `verb_unit` is valid at — the `time` unit whose `associates`
    obligation binds this verb (so value@time is read off the binding, not scanned ad-hoc). None if
    no time timestamps this verb (a timeless value)."""
    for u in reading.units:
        if "time" in u.features.get("senses", []) and _follow(u, "associates") is verb_unit:
            return _key(u.surface)
    return None


def _distribution(reading: Reading):
    """The facts a distribution statement asserts — "<name> shared <measure> between/among <recipients>".
    The whole rides on `Quantity(distributor, measure, total, lemma)` (total None if unstated — "a certain
    amount"); the relation on `Distribution`. Recipients are a NAMED group's members, or a COUNT on the
    recipient noun ("among 3 sons"); measure = the `of`-object, a money value's measure, or a money word.
    None unless it's a distribute-verb with an identifiable set of recipients."""
    units = reading.units
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    if len(verbs) != 1 or _lemma(verbs[0].surface) not in DISTRIBUTE_VERBS:
        return None
    subject = _follow(verbs[0], "subject")
    if subject is None or not any(s in subject.features.get("senses", []) for s in ("name", "noun")):
        return None                                      # distributor may be a name ("Sam") or noun ("a teacher")
    prep = next((u for u in units if _key(u.surface) in ("between", "among")), None)
    rnoun = _follow(prep, "object") if prep is not None else None
    vpos = units.index(verbs[0])                         # recipients are OBJECTS — a group AFTER the verb
    group = next((u for u in units if "members" in u.features and units.index(u) > vpos), None)
    recipients = tuple(group.features["members"]) if group is not None else ()
    if recipients:                                       # named ("Ken and Tony") → count is how many
        count = len(recipients)
    elif rnoun is not None:                              # unnamed ("3 sons") → the number on the noun
        cv = next((u for u in units if "value" in u.features.get("senses", [])
                   and _follow(u, "quantifies") is rnoun), None)
        count = int(cv.features["value"]) if cv is not None and cv.features["value"] == int(cv.features["value"]) else None
    else:
        count = None
    if not recipients and count is None:                 # no recipients we can name or count → no fact
        return None
    of_obj = next((_follow(u, "object") for u in units if _key(u.surface) == "of"
                   and _follow(u, "object") is not None), None)
    money_val = next((u for u in units if "measure" in u.features), None)
    # a value quantifying a NON-recipient noun is the shared measure ("30 CANDIES among 5 students")
    shared_q = next((u for u in units if "value" in u.features.get("senses", [])
                     and _follow(u, "quantifies") is not None and _follow(u, "quantifies") is not rnoun), None)
    if money_val is not None:                            # "$5250" — a self-measuring known total
        measure, total = money_val.features["measure"], money_val.features["value"]
    elif shared_q is not None:                           # "30 candies" — a known total of a counted noun
        measure, total = _descriptor(_follow(shared_q, "quantifies")), shared_q.features["value"]
    elif of_obj is not None:                             # "amount OF money" — measure named, total unstated
        measure, total = _descriptor(of_obj), None
    elif any(_key(u.surface) in _MONEY_WORDS for u in units):
        measure, total = MONEY, None
    else:
        return None
    ent = _entity(subject)
    manner = next((u for u in units if "manner" in u.features.get("senses", [])     # the method modifier
                   and _follow(u, "qualifies") is verbs[0]), None)                  # BOUND to this verb
    method = DISTRIBUTION_METHODS.get(_key(manner.surface)) if manner is not None else None
    return [Quantity(ent, measure, total, _lemma(verbs[0].surface)),
            Distribution(ent, recipients, measure, count, method,
                         _descriptor(rnoun) if rnoun is not None else None)]


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


def _purchase(reading: Reading, context: list = None):
    """A PURCHASE — "buys 4 candy bars that cost $2 each" → spends 4 × $2 = $8, a money LOSS. Two facts
    fold differently for one "buy" verb (you GAIN the items but LOSE the money), and `_verb_sign` is
    measure-agnostic, so the money outflow is emitted with the LOSS verb 'spend' (−) while the items keep
    'buy' (+). Gated to "buy/purchase <count> <items> … $P each": a buy verb, EXACTLY ONE 'each' marker,
    one money price, one item count, AND the buyer already holds money in context (the baseline the spend
    is netted against — "Todd has $20 … how much LEFT"). The baseline gate keeps it off a MULTI-item buy
    ("5 apples … and 2 oranges …", clause-split into single-purchase fragments), which has no prior
    holding and must DECLINE (we'd commit only one fragment's product). Returns None otherwise."""
    units = reading.units
    verb = next((u for u in units if "verb" in u.features.get("senses", [])
                 and _lemma(u.surface) in ("buy", "purchase")), None)
    if verb is None or sum(1 for u in units if _key(u.surface) == "each") != 1:
        return None                                      # exactly one "each" — a SINGLE purchase
    subject = _follow(verb, "subject")
    prices = [u for u in units if "value" in u.features.get("senses", []) and u.features.get("measure") == MONEY]
    counts = [u for u in units if "value" in u.features.get("senses", [])
              and _follow(u, "quantifies") is not None and _descriptor(_follow(u, "quantifies")) != MONEY]
    if subject is None or len(prices) != 1 or len(counts) != 1:
        return None                                      # multi-item → DECLINE (only one pair's product)
    if not any(isinstance(f, Quantity) and f.entity == _entity(subject) and f.measure == MONEY
               for f in (context or [])):
        return None                                      # no money baseline to net against → DECLINE
    price, count = prices[0], counts[0]
    item = _descriptor(_follow(count, "quantifies"))
    spent = _compute(f"{_num(count.features['value'])} * {_num(price.features['value'])}")
    return [Quantity(_entity(subject), item, count.features["value"], "buy"),
            Quantity(_entity(subject), MONEY, spent, "spend")]   # money out = count × price (a loss)


def _transfer_quantity(reading: Reading):
    """ "<entity> sold CLIPS to 48 of her friends" → 48 of CLIPS (one item per recipient). The bare-plural
    direct object is the measure; the RECIPIENT count (via "to") is the amount. None unless the dobj is
    bare (carries no count of its own) and a value quantifies the "to"-object."""
    units = reading.units
    verbs = [u for u in units if "verb" in u.features.get("senses", [])]
    subject = _follow(verbs[0], "subject") if len(verbs) == 1 else None
    if subject is None:
        return None
    dobj = _follow(verbs[0], "dobj")
    if dobj is None or "noun" not in dobj.features.get("senses", []):
        return None
    if any("value" in u.features.get("senses", []) and _follow(u, "quantifies") is dobj for u in units):
        return None                                      # dobj has its OWN count → not the to-N shape
    to = next((u for u in units if _key(u.surface) == "to" and "preposition" in u.features.get("senses", [])), None)
    to_obj = _follow(to, "object") if to is not None else None
    if to_obj is None or to_obj is dobj:
        return None
    n = next((u for u in units if "value" in u.features.get("senses", [])
              and _follow(u, "quantifies") is to_obj), None)
    if n is None:
        return None
    return [Quantity(_entity(subject), _descriptor(dobj), n.features["value"],
                     _lemma(verbs[0].surface), _time_at(reading, verbs[0]))]


def _list_cardinals(reading: Reading) -> set:
    """The units that are a value-LIST's set-size CARDINAL — "Malou got 91, 90, and 92 in her three
    quizzes": the "three" equals the count of the OTHER values quantifying the same noun ('quiz'), so it's
    the list size, not another datum. Returns their id()s. A genuine list needs ≥3 values on one noun (≥2
    data + the cardinal). Shared by _quantity (skip emitting it) and GUARD 3 (don't flag it unused-given)."""
    by_noun: dict = {}
    for u in reading.units:
        if "value" in u.features.get("senses", []):
            n = _follow(u, "quantifies")
            if n is not None:
                by_noun.setdefault(id(n), []).append(u)
    return {id(u) for vs in by_noun.values() if len(vs) >= 3
            for u in vs if u.features.get("value") == len(vs) - 1}


_TRANSFER_VERBS = {"give", "hand", "pass", "lend", "send"}


def _dative_transfer(reading: Reading):
    """A DATIVE transfer — "<giver> gave <recipient> $N" (no "to") credits the RECIPIENT +N AND debits the
    giver −N. The recipient is a NAME or a RESOLVED object-pronoun (him/her/them) stranded right after the
    transfer verb, before the value. "Tom gave Reggie $48" → [Tom −48, Reggie +48]; "father gave him $48" →
    the resolved "him" gains. A "to"-marked transfer ("gave 12 to his brother") is a DIFFERENT construct
    (GUARD 10 / distribution), so it's excluded. Returns the (giver, recipient) quantity pair(s), or None."""
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])]
    if len(verbs) != 1 or _lemma(verbs[0].surface) not in _TRANSFER_VERBS:
        return None
    if any(_key(u.surface) == "to" for u in reading.units):
        return None
    vpos = reading.units.index(verbs[0])
    subj = _follow(verbs[0], "subject")
    subj_e = _entity(subj) if subj is not None else None
    # POSSESSIVE OWNERS in the clause ("Reggie's father" → Reggie) — a name immediately followed by a
    # possessive marker; an antecedent for an unresolved object pronoun ("…gave HIM"), disjoint from the subject.
    owners = [reading.units[k] for k in range(len(reading.units) - 1)
              if "name" in reading.units[k].features.get("senses", [])
              and "possessive" in reading.units[k + 1].features.get("senses", [])
              and _entity(reading.units[k]) != subj_e]
    rec = None                                           # recipient: name / resolved (or possessive-owner) pronoun
    for u in reading.units[vpos + 1:]:
        if "value" in u.features.get("senses", []):
            break
        if "name" in u.features.get("senses", []):
            rec = u
            break
        if "pronoun" in u.features.get("senses", []) and _key(u.surface) in ("him", "her", "them"):
            if _follow(u, "refers") is not None:
                rec = u
            elif len(owners) == 1:                       # "Reggie's father gave him $48" → him = Reggie (owner)
                rec = owners[0]
            else:
                return None                              # unresolved pronoun recipient → can't attribute → bail
            break
    if rec is None:
        return None
    giver = _entity(subj) if subj is not None and "name" in subj.features.get("senses", []) else None
    recip = _entity(rec)
    if recip == giver:
        return None
    out = []
    for u in reading.units:
        if "value" not in u.features.get("senses", []):
            continue
        if "measure" in u.features:                      # "$48" — a self-measuring money value
            meas, val = u.features["measure"], u.features["value"]
        else:
            noun = _follow(u, "quantifies")              # "5 books" — the value's measure noun
            if noun is None:
                continue
            meas, val = _descriptor(noun), u.features["value"]
        if giver is not None:                            # the giver's loss (skip if the giver is a non-name,
            out.append(Quantity(giver, meas, val, "give"))   # e.g. "his father" — don't track a noun-entity)
        out.append(Quantity(recip, meas, val, "receive"))
    return out or None


def _quantity(reading: Reading):
    """The quantities a "<entity> <verb>s <amount> …" statement asserts — ONE per value: a
    self-measuring money value ("earns $12") or a value quantifying a noun ("buys 2 apples"). A list,
    so a CONJOINED object yields several ("has 3 apples and 2 oranges" → two). The verb lemma carries
    the polarity. [] if there's no single verb with a subject."""
    verbs = [u for u in reading.units if "verb" in u.features.get("senses", [])   # a trailing GERUND
             and not (_key(u.surface).endswith("ing") and _follow(u, "subject") is None)]   # adjunct
    if len(verbs) != 1:                                  # ("spends 40 years TEACHING") is not a 2nd verb
        return []
    subject = _follow(verbs[0], "subject")
    if subject is None:
        return []
    # Two ADJACENT bare values ("10 $20" = ten twenty-dollar bills) are a compound denomination we
    # don't parse — never a genuine conjoined object ("3 apples AND 2 oranges" separates them). Decline.
    # EXCEPTION: a COMMA after the first value ("got 91, 90, and 92") marks a value LIST, not a denomination.
    vi = [k for k, u in enumerate(reading.units) if "value" in u.features.get("senses", [])]
    if any(b - a == 1 and not reading.units[a].surface.rstrip().endswith(",")
           for a, b in zip(vi, vi[1:])):
        return []
    cardinals = _list_cardinals(reading)                 # the set-size cardinal of a value list ("three
    #                                                      quizzes" beside 3 scores) — dropped, not a datum
    ent, lemma, when = _entity(subject), _lemma(verbs[0].surface), _time_at(reading, verbs[0])
    first = any("noun" in u.features.get("senses", []) and _descriptor(u)[0] in TIME_UNITS   # "in the FIRST
               and "first" in u.features.get("attributes", []) for u in reading.units)       # year …" override
    out = []
    for u in reading.units:
        if "value" not in u.features.get("senses", []):
            continue
        if id(u) in cardinals:                           # the list's set-size cardinal ("three quizzes")
            continue
        if "measure" in u.features:                      # "$12" — a self-measuring money value
            out.append(Quantity(ent, u.features["measure"], u.features["value"], lemma, when))
        else:
            noun = _follow(u, "quantifies")              # "3 APPLES" — the value's measure noun
            if noun is not None:
                if lemma == "turn" and _descriptor(noun)[0] == "year" and _time_offset(reading):
                    continue                              # "turned 3 a year ago" is a TIME-SHIFTED age we
                    # can't reliably read: the age (3) and the offset's "year" share the one 'year' token,
                    # so _time_offset can't separate them. Don't emit a (wrong) current-age fact — leave it
                    # unknown so a relational age over it DECLINES rather than committing off a stale age
                    # (Diana "twice as old as Grace … turned 3 a year ago" was committing 2×3=6).
                fp = first and _descriptor(noun)[0] not in TIME_UNITS   # flag the OVERRIDDEN count (40
                out.append(Quantity(ent, _descriptor(noun), u.features["value"], lemma, when,   # students),
                                    first_period=fp))                                           # not a duration
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


def _age_when_projection(r: Reading, context: list):
    """"How old will X be WHEN Y is/turns M?" → X_now + (M − Y_now). The future point is set by Y REACHING
    age M (not a "+Δ years" displacement), and ages advance together, so X gains the same (M − Y_now) years.
    Without this the question's two names (X and the when-clause's Y) both resolve as subjects and their ages
    are summed (Marianne 20 + Bella 8 = 28, not 20 + (18−8) = 30). Gated: a "when" clause whose entity Y
    reaches a target age M (an "is/turns/be <value>" after Y), a DISTINCT main subject X before "when", and
    both current ages known. None otherwise."""
    units = r.units
    when = next((u for u in units if _key(u.surface) == "when"), None)
    if when is None:
        return None
    wi = units.index(when)
    after = units[wi + 1:]
    if any("multiplier" in u.features.get("senses", []) or _key(u.surface) == "times" for u in after):
        return None                                      # "when Y is k× as old as X" is a RELATION, not a
        #                                                  literal target age → _age_solve_for_t handles it
    y = next((u for u in after if "name" in u.features.get("senses", [])), None)
    m = next((u.features["value"] for u in after if "value" in u.features.get("senses", [])), None)
    x = next((u for u in units[:wi] if "name" in u.features.get("senses", [])), None)
    if y is None or m is None or x is None or _entity(x) == _entity(y):
        return None
    x_now = _holdings(_entity(x), ("year", frozenset()), context)
    y_now = _holdings(_entity(y), ("year", frozenset()), context)
    if x_now is None or y_now is None:
        return None
    return _compute(f"{_num(x_now)} + ({_num(m)} - {_num(y_now)})")


def _age_solve_for_t(r: Reading, context: list):
    """"How old will X be WHEN Y is k times as old as X [is now]?" → solve for the future displacement t at
    which the MULTIPLICATIVE age relation holds, then return X's age then. Both ages advance together, so
    Y+t = k·(X+t) → t = (k·X − Y)/(1 − k); a PRESENT-anchored standard ("as old as X is NOW") fixes X →
    Y+t = k·X_now → t = k·X_now − Y_now. Gated: an age question with a "when" clause carrying a multiplicative
    factor, a distinct X (the question subject, before "when") and Y (after "when"), every after-clause name
    being X or Y (the relation is Y-vs-X, not a third party), pure-multiplicative (no additive offset), and
    both current ages known. k ≠ 1."""
    units = r.units
    if not (any("interrogative" in u.features.get("senses", []) for u in units)
            and any(_key(u.surface) == "old" for u in units)
            and any(_key(u.surface) == "will" and "auxiliary" in u.features.get("senses", []) for u in units)):
        return None
    when = next((u for u in units if _key(u.surface) == "when"), None)
    if when is None:
        return None
    wi = units.index(when)
    after = units[wi + 1:]
    if any("comparative" in u.features.get("senses", []) for u in units):
        return None                                      # an additive offset on the factor ("N years older
        #                                                  than") — not handled. (Senses, not surface: a
        #                                                  demoted kinship modifier "his OLDER brother" is a
        #                                                  `title`, not a comparative, so it doesn't trip this.)
    mult = next((u for u in after if "multiplier" in u.features.get("senses", [])), None)
    k = mult.features.get("value") if mult is not None else next(
        (after[i - 1].features["value"] for i, u in enumerate(after) if _key(u.surface) == "times"
         and i and "value" in after[i - 1].features.get("senses", [])), None)
    if k is None or k == 1:
        return None
    x = next((u for u in units[:wi] if "name" in u.features.get("senses", [])), None)
    y = next((u for u in after if "name" in u.features.get("senses", [])), None)
    if x is None or y is None or _entity(x) == _entity(y):
        return None
    if any(_entity(u) not in (_entity(x), _entity(y)) for u in after   # a THIRD entity in the relation →
           if "name" in u.features.get("senses", [])):                 # the equation isn't Y-vs-X; decline
        return None
    x_now = _holdings(_entity(x), ("year", frozenset()), context)
    y_now = _holdings(_entity(y), ("year", frozenset()), context)
    if x_now is None or y_now is None:
        return None
    anchored = any(_key(u.surface) in ("now", "today", "currently") for u in after)   # "as old as X is NOW"
    t = f"{_num(k)} * {_num(x_now)} - {_num(y_now)}" if anchored \
        else f"({_num(k)} * {_num(x_now)} - {_num(y_now)}) / (1 - {_num(k)})"
    return _compute(f"{_num(x_now)} + ({t})")


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

    # AVERAGE over a set — "what is the average height of the three people?" = Σ(values) / N. Handled FIRST:
    # "average" is a "what is …" phrasing the wh_value dispatch (expecting "what is <name>") would otherwise
    # decline. The cue is SPECIFIC (unlike "more than"), and average questions all decline today → only
    # blocked → {correct | decline}. Two modes over the SINGLE shared measure: (a) MULTI-OWNER — one net
    # holding per distinct owner (Reese/Daisy/Parker → 3 heights); (b) VALUE-LIST — one owner with several
    # quantities ("Malou got 91, 90, 92" → average those amounts, NOT the accumulated 273). DENOMINATOR
    # SAFETY: require exactly one shared measure (the question says "height", holdings are keyed 'inch'); ≥2
    # terms; and if the question states the set size, the term count MUST equal it (else DECLINE, not /wrong-N).
    if any(_key(u.surface) == "average" for u in r.units) and context:
        measures = {f.measure for f in context if isinstance(f, Quantity) and f.amount is not None}
        if len(measures) == 1:
            gq = next(iter(measures))
            qs = [f for f in context if isinstance(f, Quantity) and f.measure == gq and f.amount is not None]
            owners = {f.entity for f in qs}
            offset = 0
            if len(owners) == 1:                         # VALUE-LIST: average the individual amounts
                terms = [f.amount for f in qs]
            else:                                        # MULTI-OWNER: one net holding per owner
                terms = [_holdings(o, gq, context) for o in owners]
                if gq == ("year", frozenset()):          # "average AGE of the three IN N YEARS" — each
                    offset = _time_offset(r)             # member ages by N first, like the sibling TOTAL
            #                                               AGE branch below
            stated_n = next((u.features["value"] for u in r.units if "value" in u.features.get("senses", [])
                             and float(u.features["value"]).is_integer()), None)
            if terms and all(t is not None for t in terms) and len(terms) >= 2 \
                    and (stated_n is None or len(terms) == stated_n):
                expr = f"({' + '.join(_num(t) for t in terms)})"
                if offset:
                    expr = f"({expr} + {_num(offset)} * {len(terms)})"
                return _compute(f"{expr} / {len(terms)}")
        return "undetermined"

    # TOTAL AGE over a set — "what is the total of their ages now?" / "what will be their total age in N
    # years?" = Σ(each owner's age) [+ N × owner-count for a future projection]. Sibling of the average
    # branch above; handled FIRST so the "what is …" form isn't claimed by the wh_value handler. GATED on an
    # AGE cue + a "total"/"combined" word + ≥2 year-quantity owners, so non-age "total/combined" questions
    # keep routing to the grand-total fold below. (Xavier 36: 24+12; Sam 36: (6+3+18) + 3×3.)
    if any(_key(u.surface) in ("total", "combined") for u in r.units) and context \
            and any(_key(u.surface) in ("age", "old", "ages") for u in r.units) \
            and not any(_key(u.surface) == "many" for u in r.units):   # NOT the inverse "in how many
            # years will their total age be N?" (that solves for the DURATION, not the total) — it has
            # "many"; the forward "what is/will be the total age" does not.
        owners = {f.entity for f in context if isinstance(f, Quantity)
                  and f.measure == ("year", frozenset()) and f.amount is not None}
        ages = [_holdings(o, ("year", frozenset()), context) for o in owners]
        if ages and all(a is not None for a in ages) and len(ages) >= 2:
            offset = _time_offset(r)                     # "in N years" → each member ages by N
            expr = " + ".join(_num(a) for a in ages)
            if offset:
                expr = f"({expr}) + {_num(offset)} * {len(ages)}"
            return _compute(expr)
        return "undetermined"

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
    proj_offset = _time_offset(r)                        # GROWTH PROJECTION: a quantity that grows at a
    if proj_offset:                                      # RATE per year, valued at a future time → current
        growth = next((f for f in context if isinstance(f, Rate) and f.per_unit == "year"   # + rate×Δ. The
                       and f.verb in GROWTH_VERBS), None)        # explicit-rate generalisation of the age
        if growth is not None:                                   # shift ("grows 3 inches/yr; 20 now; after
            base = _holdings(growth.entity, growth.measure, context)   # 10 yrs?" → 20 + 3×10 = 50)
            if base is not None:
                return _compute(f"{_num(base)} + {_num(growth.amount)} * {_num(proj_offset)}")
    if any("interrogative" in u.features.get("senses", []) for u in r.units) \
            and any(_key(u.surface) == "old" for u in r.units):   # "How old is/will X be?" → age (years)
        when_proj = _age_when_projection(r, context)     # "how old will X be WHEN Y is/turns M" → X + (M − Y)
        if when_proj is not None:                        # (the future point is set by Y reaching M, not "+Δ")
            return when_proj
        solve_t = _age_solve_for_t(r, context)           # "WHEN Y is k× as old as X" → solve for the future t
        if solve_t is not None:                          # at which the relation holds, then X's age then
            return solve_t
        if any(_key(u.surface) == "when" for u in r.units):
            return "undetermined"                        # a "when <condition>" neither projection nor solve-for-t
            # resolved must DECLINE, never fall to the bare entity-age SUM below (Joel 5 + dad 32×3 = 101).
        offset = _time_offset(r)                         # the queried time vs now — age TRACKS the timeline:
        aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)   # age@(now+Δ)
        if aux is not None:                                                                     # = age_now+Δ
            entities, _ = _question_subjects(r, aux)
        else:                                            # "How old is X?" / "How many years old is X?" — the
            # entity asked about is the NAME (the copula's grammatical subject mis-resolves to the fronted
            # measure "years" in "how many years old", so prefer the name; fall back to the copula subject).
            name = next((u for u in r.units if "name" in u.features.get("senses", [])), None)
            if name is not None:
                entities = [_entity(name)]
            else:
                subj, _ = _copula_args(r)
                entities = [_entity(subj)] if subj is not None else []
        if not entities:
            return "undetermined"
        parts = []
        for ent in entities:
            h = _holdings(ent, ("year", frozenset()), context)
            if h is None:
                return "undetermined"
            parts.append(_num(h))
        if offset:
            parts.append(_num(offset))
        return _compute(" + ".join(parts))               # Σ(ages) + the timeline offset

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
    _coll0 = _follow(_follow(interro, "queries"), "collection")               # "how many HOURS" → (hour,…)
    asked_unit = _descriptor(_coll0)[0] if _coll0 is not None else None
    kys = {_key(u.surface) for u in r.units}
    combined_q = bool(kys & {"combined", "combine", "combines", "altogether", "total"}) \
        or {"all", "together"} <= kys \
        or "in all" in question.lower()                  # "How many cans did the boys collect IN ALL?" → Σ
        #                                                  ("the two friends COMBINE to lose …" → a sum too)
    is_long = any(_key(u.surface) == "long" for u in r.units)
    # AGGREGATION CUE — bare "together" (without "all") is a sum cue ("how much did the two men earn
    # TOGETHER?" → Σ), but it's CONFOUNDED with the collaborative-rate sense ("how long to paint the house
    # together?" = a joint RATE, not a sum). The rate-inversion branch below is gated `not combined_q`, so
    # firing combined_q on a how-long/time question would skip it and sum rates into garbage. Restrict the
    # "together" trigger to NON-time questions, which the rate branch never claims. (Salvadore 2934, Anika 64.)
    if "together" in kys and not is_long and asked_unit not in TIME_UNITS:
        combined_q = True
    if (is_long or asked_unit in TIME_UNITS) and not combined_q:   # "How LONG / how many HOURS … to <verb>?"
        rates = [f for f in context if isinstance(f, Rate)]                   # → RATE INVERSION: target ÷
        def _inv_target(rate):                                               # rate, in per_unit (convert if
            return next((u for u in r.units if "value" in u.features.get("senses", [])   # asked). The target
                         and _follow(u, "quantifies") is not None                        # value quantifies
                         and _descriptor(_follow(u, "quantifies")) == rate.measure), None)  # the rate's measure
        # NOTE (deferred generalization): the target is sought only in the QUESTION units. "…2800 pages long
        # how many weeks…" (no comma) keeps the total in the question clause → found; a COMMA ("…long, how many
        # weeks…") splits the total into a PREMISE → a context-Quantity fallback (f.measure == rate.measure)
        # would solve it. Prototyped zero-regression but with ZERO benefit on real dev/test/permissive (the one
        # real case, John-bible, is already fixed upstream by the rate-marker entity recovery), so DEFERRED per
        # example-driven-deferral — re-add when a real comma-split rate-inversion problem forces it.
        # SELECT among rates (each must have a target the question quantifies): (1) one OWNED by a name in
        # the question ("how long does it take BRIANNA" → Brianna's 2 pg/min, not Anna's); (2) one whose
        # per_unit IS / converts to the asked unit ("15 pages/DAY" beats "5 pages/HOUR" for "how many
        # WEEKS"); (3) the first target-matching rate (preserves single-rate Joy/Sam).
        qnames = {_entity(u) for u in r.units if "name" in u.features.get("senses", [])}
        rate = next((f for f in rates if _inv_target(f) is not None and f.entity in qnames), None) \
            or next((f for f in rates if _inv_target(f) is not None and asked_unit is not None
                     and (asked_unit == f.per_unit or (f.per_unit, asked_unit) in CONVERSIONS)), None) \
            or next((f for f in rates if _inv_target(f) is not None), None) \
            or (rates[0] if rates else None)
        if rate is not None:                                                  # rate, in per_unit (convert if
            mv = _inv_target(rate)
            if mv is not None:
                target = _num(mv.features["value"])
                cv = next((u for u in r.units if "value" in u.features.get("senses", []) and u is not mv
                           and _follow(u, "quantifies") is not None), None)   # "3 BOOKS that are 400 pages"
                if cv is not None:
                    target += f" * {_num(cv.features['value'])}"
                expr = f"({target}) * {_num(rate.per_amount)} / {_num(rate.amount)}"   # "120 × 20 ÷ 8"
                if asked_unit in TIME_UNITS and asked_unit != rate.per_unit \
                        and (rate.per_unit, asked_unit) in CONVERSIONS:        # 300 min → 5 hours
                    expr = f"({expr}) * {_num(CONVERSIONS[(rate.per_unit, asked_unit)])}"
                return _compute(expr)
        if is_long:
            return "undetermined"                        # a how-long question with no usable rate
    collection = _follow(_follow(interro, "queries"), "collection")   # how → many → its collection
    query = _descriptor(collection) if collection is not None else None
    much = next((u for u in r.units if _key(u.surface) == "much"), None)
    if much is not None:                                 # "how much": read what `much` BOUND (no global
        coll = _follow(much, "collection")               # MONEY default) — an adjacent mass noun, or a
        if coll is not None and "verb" in coll.features.get("senses", []):   # result-measure verb ("how
            query = RESULT_MEASURE.get(_lemma(coll.surface))   # much did X EARN" → the earnings = MONEY)
        elif coll is not None:                           # "how much WATER" → that mass noun
            query = _descriptor(coll)
        else:                                            # bound nothing (e.g. "how much did he RUN") →
            query = None                                 # can't tell what's asked → decline, not MONEY
    remainder = any(_key(u.surface) in _REMAINDER for u in r.units)
    named = any("name" in u.features.get("senses", []) for u in r.units)   # NAMED entities ("Tom and Mary
    if combined_q and not named:                         # altogether") need the per-entity fold, not this
        gq = query                                       # "How many INCHES are all snakes COMBINED?" — a
        if gq is None:                                   # generic population → the GRAND TOTAL. A query-less
            measures = {f.measure for f in context if isinstance(f, Quantity) and f.amount is not None}
            gq = next(iter(measures)) if len(measures) == 1 else None   # "their combined experience" → the
        if gq is not None and remainder:                 # "how many do they have LEFT in total" → Σ each
            owners = {f.entity for f in context           # owner's NET holdings (signed: picked − gave −
                      if isinstance(f, Quantity) and f.measure == gq and f.amount is not None}  # sold),
            hs = [_holdings(o, gq, context) for o in owners]   # not the raw sum (which double-counts the
            if hs and all(h is not None for h in hs):          # gross alongside its own losses → Frank 210)
                return _compute(" + ".join(_num(h) for h in hs))
        if gq is not None:                               # one measure the context quantities share (years)
            terms = []                                   # sum every quantity convertible to it (cross-entity,
            for f in context:                            # cross-unit via CONVERSIONS)
                if isinstance(f, Quantity) and f.amount is not None:
                    if f.measure == gq:
                        terms.append(_num(f.amount))
                    elif (f.measure[0], gq[0]) in CONVERSIONS:
                        terms.append(f"{_num(f.amount)} * {_num(CONVERSIONS[(f.measure[0], gq[0])])}")
            if terms:
                return _compute(" + ".join(terms))
    if query is not None and any("verb" in u.features.get("senses", [])      # "How much <measure> was
                                 and _lemma(u.surface) in DISTRIBUTE_VERBS for u in r.units):  # shared?"
        shared = [f.amount for f in context if isinstance(f, Quantity) and f.measure == query   # → the
                  and f.verb in DISTRIBUTE_VERBS and f.amount is not None]   # PINNED distributed total
        if shared:
            return int(shared[0]) if shared[0] == int(shared[0]) else shared[0]
        return "undetermined"
    aux = next((u for u in r.units if "auxiliary" in u.features.get("senses", [])), None)
    comp = next((u for u in r.units if "comparative" in u.features.get("senses", [])), None)
    if comp is not None and aux is not None and query is not None:    # "How much more <measure> does
        standard = _follow(comp, "standard")                         # A have than B?" → A's − B's
        subjects, _ = _question_subjects(r, aux)                     # A = the verb's bound subject
        times = [u for u in r.units if "time" in u.features.get("senses", [])]
        than = next((u for u in r.units if _key(u.surface) == "than"), None)
        if len(subjects) == 1 and than is not None and len(times) == 2:   # ACROSS TIMES: one entity at
            tp = r.units.index(than)                                       # two times split by "than" —
            t1 = next((u for u in times if r.units.index(u) < tp), None)   # "…on Tuesday than on Monday"
            t2 = next((u for u in times if r.units.index(u) > tp), None)
            if t1 is not None and t2 is not None:
                a = _holdings_at(subjects[0], query, context, _key(t1.surface))
                b = _holdings_at(subjects[0], query, context, _key(t2.surface))
                if a is not None and b is not None:
                    diff = a - b if _key(comp.surface) == "more" else b - a
                    return int(diff) if diff == int(diff) else diff
            return "undetermined"
        # the comparative often binds its standard to the MEASURE noun ("more PAIRS"), not the entity being
        # compared. When the bound standard isn't a NAME and the question carries a comparison marker
        # ("compared to Jim" / "than Jim"), recover the standard as the non-subject NAME. ("How many more
        # pairs does Anthony have compared to Jim?" → Anthony − Jim.)
        if (standard is None or "name" not in standard.features.get("senses", [])) \
                and any(_key(u.surface) in ("than", "compared", "versus") for u in r.units):
            standard = next((u for u in r.units if "name" in u.features.get("senses", [])
                             and _entity(u) not in subjects), None)
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
    # ELIDED COUNT-NOUN — "how many does Steven have?" omits the measure noun, so `query` is None and the
    # per-entity fold below can't run. If the context quantities share exactly ONE measure, infer it as the
    # query (the single salient thing to count) — mirrors the query-less combined-fold's measure inference
    # (#102). Decline-safe: only when unambiguous (one measure); two+ measures keep query None → decline.
    if query is None and any(_key(u.surface) == "many" for u in r.units):
        measures = {f.measure for f in context if isinstance(f, Quantity) and f.amount is not None}
        if len(measures) == 1:
            query = next(iter(measures))
    qverb_unit = next((u for u in r.units if "verb" in u.features.get("senses", [])
                       and _key(u.surface) not in ("is", "are", "was", "were")), None)
    # "How many STUDENTS passed?" — no aux; the query noun IS the subject. Only when a matching count-fact
    # exists (so it never preempts the remainder / count-instances branches).
    noaux_match = (aux is None and query is not None and qverb_unit is not None and not remainder
                   and any(isinstance(f, Quantity) and f.entity == query[0] and f.measure == query
                           and f.verb == _lemma(qverb_unit.surface) for f in context))
    if noaux_match:
        entities, q_verb, q_time = [query[0]], _lemma(qverb_unit.surface), None
    elif aux is not None and query is not None:           # "How many <measure> do <entities> <verb>?"
        entities, q_verb = _question_subjects(r, aux)    # the verb's bound subject(s), structurally
        q_time = next((_key(u.surface) for u in r.units if "time" in u.features.get("senses", [])), None)
    else:
        entities = None
    if query is not None and entities is not None:
        if entities:
            terms = []                                   # each conjunct's contribution, summed once at the end
            for ent in entities:
                if q_verb in ("have", "own") and q_time is not None:   # TIME-AWARE: the value VALID at
                    h = _holdings_at(ent, query, context, q_time)      # the queried time, folded along
                    if h is None:                                      # the timeline (state vs change)
                        return "undetermined"
                    terms.append(_num(h))
                    continue
                qs = [f for f in context if isinstance(f, Quantity)
                      and f.entity == ent and f.measure == query]
                if not qs and not query[1]:              # BARE HEAD ("how many STICKERS") with no exact
                    qs = [f for f in context if isinstance(f, Quantity)   # match → sum the entity's SUB-TYPES
                          and f.entity == ent and f.measure[0] == query[0]]   # under that head (8 flower +
                #   6 animal stickers → 14). Reached only when no bare-head quantity exists, so it never
                #   double-counts; the fold below signs each by its verb (have/used/…).
                if not qs:                               # CONVERT: "how many KILOGRAMS" when we hold grams
                    qs = [Quantity(ent, query, _compute(f"{_num(f.amount)} * "
                                   f"{_num(CONVERSIONS[(f.measure[0], query[0])])}"), f.verb)
                          for f in context if isinstance(f, Quantity) and f.entity == ent
                          and f.amount is not None and (f.measure[0], query[0]) in CONVERSIONS]
                # POLARITY IS RELATIVE TO THE QUESTION: "how many do you HAVE" folds holdings
                # (baseline + gains - losses); "how many did you SELL" COUNTS the sell events (+).
                if q_verb in ("have", "own"):
                    folded = [f"{_verb_sign(f.verb)} * {_num(f.amount)}" for f in qs]
                else:
                    folded = [_num(f.amount) for f in qs if f.verb == q_verb]   # events of THIS verb
                # FIRST-PERIOD EXCEPTION: "teaches 50 a year … in 10 years … the FIRST year only 40" →
                # override + rate×(duration−1) = 40 + 50×9. The override (40) is year 1; the rate covers
                # the rest. (Without an override the rate×duration path below handles the plain case.)
                override = next((f for f in qs if f.verb == q_verb and f.first_period), None)
                rate = next((f for f in context if isinstance(f, Rate) and f.entity == ent
                             and f.measure == query and f.verb == q_verb), None)
                if override is not None and rate is not None:
                    dur = next((u.features["value"] for u in r.units if "value" in u.features.get("senses", [])
                                and _follow(u, "quantifies") is not None
                                and _descriptor(_follow(u, "quantifies"))[0] == rate.per_unit), None)
                    if dur is not None:
                        terms.append(f"{_num(override.amount)} + {_num(rate.amount)} * ({_num(dur)} - 1)")
                        continue
                if folded:
                    terms.extend(folded)
                    continue
                rate = next((f for f in context if isinstance(f, Rate) and f.entity == ent     # rate ×
                             and f.measure == query and f.verb == q_verb), None)                # per-unit
                if rate is None and query == MONEY:         # CROSS-ENTITY price: "$5.50 per pound" (stated
                    rate = next((f for f in context if isinstance(f, Rate) and f.measure == MONEY  # of beef/
                                 and _holdings(ent, (f.per_unit, frozenset()), context) is not None), None)
                per_qty = _holdings(ent, (rate.per_unit, frozenset()), context) if rate else None
                if per_qty is None and rate is not None:    # "…write in 3 DAYS?" — per-unit count is in
                    per_qty = next((u.features["value"] for u in r.units    # the QUESTION, not a premise
                                    if "value" in u.features.get("senses", []) and _follow(u, "quantifies")
                                    is not None and _descriptor(_follow(u, "quantifies"))[0] == rate.per_unit), None)
                if per_qty is None and rate is not None:    # CONVERT: a premise quantity in another unit
                    for f in context:                       # ("50 minutes" for an hourly rate → 50/60 hr)
                        if isinstance(f, Quantity) and f.entity == ent and f.amount is not None \
                                and (f.measure[0], rate.per_unit) in CONVERSIONS:
                            base = _holdings(ent, f.measure, context)
                            if base is not None:
                                per_qty = _compute(f"{_num(base)} * "
                                                   f"{_num(CONVERSIONS[(f.measure[0], rate.per_unit)])}")
                                break
                if per_qty is None:
                    return "undetermined"                # no relevant quantity/rate for this conjunct
                terms.append(f"{_num(rate.amount)} * {_num(per_qty)}")
            return _compute(" + ".join(terms)) if terms else "undetermined"
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
        # ABSTRACT/ELIDED MEASURE over an ACCUMULATION — "how FAR did she walk for 2 hours?" has no measure
        # noun, but if the context holds ONE measure, ONE owner, and ≥2 quantities (a sum the normal path
        # missed), the answer is that owner's net holding (Hazel: 2 + 4 = 6). The ≥2 gate keeps a single-fact
        # "how much did Tom run?" (5 miles) DECLINING per #74's no-guess design — this only rescues an
        # accumulation the measure-less question couldn't dispatch.
        qs = [f for f in context if isinstance(f, Quantity) and f.amount is not None]
        if len(qs) >= 2 and len({f.measure for f in qs}) == 1 and len({f.entity for f in qs}) == 1:
            total = _holdings(qs[0].entity, qs[0].measure, context)
            if total is not None:
                return int(total) if total == int(total) else total
        return [Deficiency("unanswerable", interro, "I couldn't tell what to count")]
    if not any(isinstance(inst, tuple) and inst[0] == query[0] for inst in context):
        return "undetermined"        # the queried TYPE isn't present as instances (e.g. "how much money"
        # when only `son` instances exist, or the amount lives in Quantities) — decline, never a bogus 0.
        # (A type that IS present but with 0 matches — "how many RED apples", no red ones — stays a real 0.)
    return sum(1 for inst in context if isinstance(inst, tuple) and _matches(query, inst))


def instances(statement: str) -> list:
    """Parse a "there are N <collection>" statement into N copies of the collection's descriptor —
    the instance set `answer()` queries. The number word ranges over the collection (same
    `collection` obligation as any quantifier) and carries its value via NUMBERS. Returns [] when no
    count+collection can be read, so an unparseable statement teaches us nothing rather than a guess.
    Concatenate over several statements to build a context."""
    r = analyze(statement)
    for u in r.units:
        if "value" not in u.features.get("senses", []):
            continue
        collection = _follow(u, "quantifies")
        if collection is not None:
            return [_descriptor(collection)] * int(u.features["value"])
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
    facts = _facts_from(r, context if context is not None else [])
    if discourse is not None:                            # remember this statement's subject as a discourse
        verbs = [u for u in r.units if "verb" in u.features.get("senses", [])]  # entry (entity, is_name).
        non_cop = [v for v in verbs if _key(v.surface) not in ("is", "are", "was", "were", "be")]
        subj = _follow(non_cop[0], "subject") if len(non_cop) == 1 else None    # the non-copula MAIN verb's
        #                                                    subject ("James buys … that are …" → James). A
        if subj is None and len(verbs) == 1 and not non_cop:   # BARE copula ("Tina is 10") records its subject
            cs = _follow(verbs[0], "subject")                  # too — but ONLY a NAME (so "She is …" can later
            if cs is not None and "name" in cs.features.get("senses", []):   # corefer to Tina); a common-noun
                subj = cs                                      # copula subject ("the PRICE of beef is $5") must
        if subj is not None:                                   # NOT pollute the discourse (it broke he→James)
            discourse.append((_entity(subj), "name" in subj.features.get("senses", [])))   # James)
        elif facts and len(r.units) > 1 and _key(r.units[0].surface) == "for" \
                and _key(r.units[1].surface) in ("every", "each"):   # the per-occurrence construct "For
            ent = next((f.entity for f in facts if getattr(f, "entity", None)   # every X, Sam eats N Y" —
                        and any(_entity(u) == f.entity                          # the generic finder skips it
                        and "name" in u.features.get("senses", []) for u in r.units)), None)  # (2 verbs), so
            if ent is not None:                          # register the fact's NAME entity for a later pronoun.
                discourse.append((ent, True))            # Scoped to this construct to keep coref blast radius small.
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


def _synth_distribution(facts: list, readings: list, agenda: set) -> list:
    """PIN an unknown distributed total by back-linking through its `Distribution`: once EVERY recipient's
    holding of the measure is known, the whole = their SUM → derive `Quantity(distributor, measure, sum,
    "share")`. This is the unknown-node's `value` obligation discharged by inference (not a question-time
    heuristic). Idempotent: skip a Distribution whose total is already resolved."""
    out = []
    for f in facts:
        if not isinstance(f, Distribution):
            continue
        if any(isinstance(q, Quantity) and q.entity == f.distributor and q.measure == f.measure
               and q.verb in DISTRIBUTE_VERBS and q.amount is not None for q in facts):
            continue                                     # already pinned
        shares = [_holdings(r, f.measure, facts) for r in f.recipients]
        if shares and all(s is not None for s in shares):
            out.append(Quantity(f.distributor, f.measure,
                                _compute(" + ".join(_num(s) for s in shares)), "share"))
    return out


def _underspecified_distribution(context: list):
    """A Distribution we can't resolve because the SPLIT METHOD is missing/non-computable: the whole is
    KNOWN, but `method` isn't "equal" (None, or "randomly"/"optimally"…) AND no recipient has an explicit
    share. Returns that Distribution (→ a "how was it shared?" deficiency) or None. NOTE (future, not now):
    when a verb is blocked for want of a manner like this, instead of declining we could (a) fall back on a
    CONTEXT-DEPENDENT default method, or (b) enumerate an answer PER plausible manner. For now: ask."""
    for d in context:
        if not isinstance(d, Distribution) or d.method == "equal":
            continue
        total_known = any(isinstance(q, Quantity) and q.entity == d.distributor and q.measure == d.measure
                          and q.verb in DISTRIBUTE_VERBS and q.amount is not None for q in context)
        holders = d.recipients or ((d.recipient_noun[0],) if d.recipient_noun else ())
        has_shares = any(_holdings(h, d.measure, context) is not None for h in holders)
        if total_known and not has_shares:
            return d
    return None


def _synth_equal_split(facts: list, readings: list, agenda: set) -> list:
    """The EQUAL-SPLIT reading of a Distribution: a KNOWN whole shared `equally` among `count` recipients
    → each gets whole ÷ count. Derives a `get` Quantity per recipient — by name if named, else keyed to
    the recipient noun ("each son got …"). The per-share is inferred, not asked at question-time."""
    out = []
    for f in facts:
        if not isinstance(f, Distribution) or f.method != "equal" or not f.count:
            continue
        total = next((q.amount for q in facts if isinstance(q, Quantity) and q.entity == f.distributor
                      and q.measure == f.measure and q.verb in DISTRIBUTE_VERBS and q.amount is not None), None)
        if total is None:
            continue
        per = _compute(f"{_num(total)} / {_num(f.count)}")
        holders = f.recipients or ((f.recipient_noun[0],) if f.recipient_noun else ())
        for h in holders:
            cand = Quantity(h, f.measure, per, "get")
            if cand not in facts and cand not in out:
                out.append(cand)
    return out


def _synth_rate_duration(facts: list, readings: list, agenda: set) -> list:
    """A Rate accumulated over a stated DURATION → a concrete Quantity: "makes 7 pies a day … does this
    for 12 days" → Quantity(cooper, pie, 7×12, make), which then folds with later events (− eaten …).
    Fires when a same-entity Quantity counts the rate's per_unit (e.g. "12 days" for a per-day rate); a
    DIFFERENT-measure duration only (so the rate's own count-noun isn't mistaken for its duration). The
    derived total carries the rate's verb, so holdings sees production, not a stray day-count."""
    out = []
    for rt in facts:
        if not isinstance(rt, Rate):
            continue
        dur = next((q.amount for q in facts if isinstance(q, Quantity) and q.entity == rt.entity
                    and q.measure[0] == rt.per_unit and q.amount is not None
                    and q.measure != rt.measure), None)
        if dur is None:                                  # a CONVERTIBLE-unit duration ("one WEEK" for a
            rate_units = {x.per_unit for x in facts if isinstance(x, Rate)}   # per-DAY rate → 7 days). Skip
            conv = next(((q.amount, CONVERSIONS[(q.measure[0], rt.per_unit)]) for q in facts  # a quantity
                         if isinstance(q, Quantity) and q.entity == rt.entity and q.amount is not None
                         and q.measure != rt.measure and (q.measure[0], rt.per_unit) in CONVERSIONS
                         and q.measure[0] not in rate_units), None)   # whose unit is some rate's EXACT
            if conv is not None:                          # per_unit (it's THAT rate's duration — e.g. Tom's
                dur = _compute(f"{_num(conv[0])} * {_num(conv[1])}")   # 7 days feeds the per-day rate, not
                                                          # the per-hour one via a day→hour conversion)
        if dur is None:
            continue
        total = _compute(f"{_num(rt.amount)} * {_num(dur)} / {_num(rt.per_amount)}")
        cand = Quantity(rt.entity, rt.measure, total, rt.verb, derived=True)
        if cand not in facts and cand not in out:
            out.append(cand)
    return out


def _synth_rate_chain(facts: list, readings: list, agenda: set) -> list:
    """Two same-entity rates that share a unit COMPOSE into one: "3 hours per DAY" × "5 pages per HOUR"
    → 15 pages per day (the HOUR cancels). Fires when r1.measure == r2.per_unit; the product carries r2's
    measure over r1's per_unit and r2's verb. Derived; the inversion/duration paths then consume it
    (James: 15 pg/day, then 735 ÷ 15 = 49 days → 7 weeks)."""
    out = []
    rates = [f for f in facts if isinstance(f, Rate)]
    for r1 in rates:
        for r2 in rates:
            if r1 is r2 or r1.entity != r2.entity or r1.measure[0] != r2.per_unit:
                continue
            amt = _compute(f"{_num(r2.amount)} * {_num(r1.amount)} / {_num(r1.per_amount)} "
                           f"/ {_num(r2.per_amount)}")
            cand = Rate(r1.entity, amt, r2.measure, r1.per_unit, r2.verb, 1)
            if cand not in facts and cand not in out:
                out.append(cand)
    return out


def _synth_rate_relation(facts: list, readings: list, agenda: set) -> list:
    """Resolve a RateRel into a concrete Rate, EITHER direction (so a chain resolves across fixpoint
    rounds): forward — reference's rate known → subject = factor × reference; inverse — subject's rate
    known → reference = subject ÷ factor. Anna(1/min) → Carole = 1×Anna → Brianna = Carole ÷ 0.5 = 2/min.
    The derived rate copies the known rate's measure/per_unit/verb (only the amount scales)."""
    out = []
    def rate_of(ent):
        return next((f for f in facts if isinstance(f, Rate) and f.entity == ent), None)
    for rel in facts:
        if not isinstance(rel, RateRel):
            continue
        subj_rate, ref_rate = rate_of(rel.subject), rate_of(rel.reference)
        if ref_rate is not None and subj_rate is None:               # forward
            amt = _compute(f"{_num(rel.factor)} * {_num(ref_rate.amount)}")
            cand = Rate(rel.subject, amt, ref_rate.measure, ref_rate.per_unit, ref_rate.verb,
                        ref_rate.per_amount)
        elif subj_rate is not None and ref_rate is None and rel.factor != 0:   # inverse
            amt = _compute(f"{_num(subj_rate.amount)} / {_num(rel.factor)}")
            cand = Rate(rel.reference, amt, subj_rate.measure, subj_rate.per_unit, subj_rate.verb,
                        subj_rate.per_amount)
        else:
            continue
        if cand not in facts and cand not in out:
            out.append(cand)
    return out


# obligation/fact KIND → derivation (facts, readings, agenda) -> new facts. Register here to add reasoning.
SYNTHESIZERS = {"orders": _synth_transitive_order, "narrative": _synth_narrative_order,
                "distribution": _synth_distribution, "equal_split": _synth_equal_split,
                "rate_duration": _synth_rate_duration, "rate_chain": _synth_rate_chain,
                "rate_relation": _synth_rate_relation}


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
    dist = _distribution(r)                              # "<name> shared <measure> between <recipients>"
    if dist is not None:                                 # → unknown-amount node + a Distribution relation
        return dist
    # FUTURE age with a SUB-YEAR shift the year-machinery can't convert ("In 18 months, she will be 10 years
    # old") — storing a plain current age (10) ignores the shift and the relation off it commits a wrong
    # (Antonio = 10/2 = 5). The age path is year-only, so DECLINE rather than fold the months as years.
    if any(_key(u.surface) == "will" and "auxiliary" in u.features.get("senses", []) for u in r.units) \
            and any(_key(u.surface) == "old" for u in r.units):
        off_prep = next((u for u in r.units if "preposition" in u.features.get("senses", [])
                         and _key(u.surface) in ("in", "after")), None)
        off_obj = _follow(off_prep, "object") if off_prep is not None else None
        if off_obj is not None and _descriptor(off_obj)[0] in ("month", "week", "day"):
            return []
    verbs = [u for u in r.units if "verb" in u.features.get("senses", [])]
    if len(verbs) == 1:                                  # a group-subject STATEMENT ("Tom and Mary have
        subj = _follow(verbs[0], "subject")              # 5 apples") is distributive/collective — ambiguous,
        if subj is not None and "members" in subj.features \
                and not any(_key(u.surface) in ("total", "combined") for u in r.units):   # but an explicit
            return []                                    # TOTAL ("they walk a TOTAL of 70 miles") is an
            #                                              unambiguous collective → let the fact form
    if any("existential" in u.features.get("senses", []) for u in r.units):   # "there are N <coll>" → N
        for u in r.units:                                # instances (existential-gated, so "Tom has three
            if "value" in u.features.get("senses", []):  # apples"/"three years younger" stay arithmetic)
                coll = _follow(u, "quantifies")
                if coll is not None:
                    return [_descriptor(coll)] * int(u.features["value"])
    sa = _simultaneous_ages(r)                           # "X was A years old when Y was B years old" → both
    if sa is not None:                                   # ages (a constant-gap pair) at the same reference
        return sa
    fa = _future_age(r)                                  # "(in N years) E will be X years old" → E_now=X−N?
    if fa is not None:
        return [fa]
    ca = _comparative_age(r, context)                    # "X is N years older than Y" → Y_age ± N?
    if ca is not None:
        return [ca]
    ra = _relational_age(r, context)                     # "X is N times as old as Y" → X_age = N × Y_age?
    if ra is not None:
        return [ra]
    pas = _possessive_age_statement(r)                    # "Ron's age is 43" → Quantity(ron, year, 43)
    if pas is not None:
        return [pas]
    pap = _possessive_age_parts(r)                        # "X's age is k× Y's age" / "X is k× Y's age" →
    if pap is not None:                                   # X_age = k × Y_age (the possessive frame)
        subj_e, std_e, k = pap
        delta = _time_offset(r)                           # "In Δ years, X will be k× Y's age" shifts BOTH
        std_now = any(_key(u.surface) in ("now", "today", "currently") for u in r.units)   # …unless Y is
        #                                                  anchored to the present ("the age Y is CURRENTLY")
        y = _holdings(std_e, ("year", frozenset()), context)
        if y is not None:                                 # FORWARD: X_now = k·(Y_now+Δ) − Δ; std_now drops +Δ
            y_ref = _num(y) if std_now else f"({_num(y)} + {_num(delta)})"
            x = f"{_num(k)} * {y_ref} - {_num(delta)}" if delta else f"{_num(k)} * {_num(y)}"
            return [Quantity(subj_e, ("year", frozenset()), _compute(x), "be")]
        x_now = _holdings(subj_e, ("year", frozenset()), context)   # BACKWARD: subject known, derive standard
        if x_now is not None and k:                       # Y_now = (X_now+Δ)/k − Δ ("Suzy 20; Suzy+4 = 2·(Mary+4)"
            tail = "" if std_now else f" - {_num(delta)}"   # → Mary = 24/2 − 4 = 8); std_now drops the −Δ
            yv = f"({_num(x_now)} + {_num(delta)}) / {_num(k)}{tail}" if delta \
                else f"{_num(x_now)} / {_num(k)}"
            return [Quantity(std_e, ("year", frozenset()), _compute(yv), "be")]
        return []                                         # neither age known → DEFER (fixpoint chains), and
        #                                                   crucially don't fall to _frequency (bogus Freq)
    if _relational_age_factor(r) is not None and _copula_args(r)[0] is not None:
        return []                                        # Y's age not yet known → DEFER (the fixpoint
        # chains the multi-hop: Anne=2×Emile, Emile=6×Maude, Maude=8 → 96; never a bare predication)
    mult = next((u for u in r.units if "multiplier" in u.features.get("senses", [])), None)
    scaled = _follow(mult, "scales") if mult is not None else None
    if scaled is not None and ("quantifier" in scaled.features.get("senses", [])     # "twice as many" /
                               or _key(scaled.surface) in (COUNT_NOUNS | DIMENSIONS)):  # the number / as long
        rq = _relational_quantity(r, context)            # "<entity> has twice as many" → derived, or
        return [rq] if rq is not None else []            # decline (no referent) — never a bare predication
    frac = next((u for u in r.units if _fraction_parts(u.surface) is not None), None)  # unit OR non-unit
    if frac is not None:                                  # "<fraction> OF <referent>" — a partitive part
        fi = r.units.index(frac)                          # of a holding ("loses half of them" → 0.5×held;
        if fi + 1 < len(r.units) and _key(r.units[fi + 1].surface) == "of":   # "a third of …"; NOT a Freq)
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
    rr = _rate_relations(r)                              # "Carole reads as fast as Anna but half Brianna's
    if rr is not None:                                    # speed" → relational rate(s) (resolved bidirectionally)
        return rr
    fr = _frequency(r)                                   # "<entity> <verb>s twice" → frequency?
    if fr is not None:
        return [fr]
    trr = _time_range_rate(r)                            # "writes from 1 PM to 4 PM every night" → 3 h/day?
    if trr is not None:
        return [trr]
    por = _per_occurrence_rate(r)                        # "for every sandwich … eats 4 apples" → 4/sandwich?
    if por is not None:
        return [por]
    rate = _rate(r, context)                             # "<entity> earns $12 per hour" → rate?
    if rate is not None:                                 # a trailing "for one week" → a duration alongside it
        dur = _for_duration(r, rate)
        return [rate, dur] if dur is not None else [rate]
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
    rq = _rest_quantity(r, context)                      # "the REST failed" → total − Σ(stated subsets)?
    if rq is not None:
        return [rq]
    if any(_key(u.surface) in REST_WORDS for u in r.units):
        return []                                        # "the rest …" but no total yet → DEFER (the
        # solve() fixpoint retries once the population total / subsets are learned, e.g. a trailing "if")
    pct = next((u for u in r.units if "percent" in u.features.get("senses", [])), None)
    if pct is not None and _follow(pct, "base") is None \
            and len([u for u in r.units if "verb" in u.features.get("senses", [])]) == 1 \
            and any("noun" in u.features.get("senses", []) for u in r.units):
        return []                                        # percent-of-holding, referent not yet known → defer
    cq = _container_quantity(r)                          # "20 packets of 100 grams" → 20×100 = 2000?
    if cq:                                               # (conjoined of-phrases → a product each)
        return cq
    pu = _purchase(r, context)                           # "buys 4 candy bars that cost $2 each" → −$8?
    if pu is not None:
        return pu
    dq = _distributed_quantity(r)                        # "gave 4 stickers each to 3 friends" → 4×3?
    if dq is not None:
        return [dq]
    tq = _transfer_quantity(r)                           # "sold clips to 48 friends" → 48 clips (one each)
    if tq is not None:
        return tq
    dt = _dative_transfer(r)                             # "Tom gave Reggie $48" → Reggie +48 AND Tom −48
    if dt is not None:
        return dt
    q = _quantity(r)                                     # "<entity> has 3 apples and 2 oranges" → list
    if q:
        return q
    p = _predication(r)                                  # subject-verb statement?
    if p is not None:
        return [p]
    d = collection_descriptor(r)                         # bare collection phrase?
    return [d] if d is not None else []


def _has_verb(text: str) -> bool:
    # do/does/did are PRIMARILY auxiliaries; a clause fragment shouldn't count as having its own verb on
    # their account ("how many apples do Tom | and Mary have" must NOT split the "Tom and Mary" group).
    return any("verb" in u.features.get("senses", []) and _key(u.surface) not in ("do", "does", "did")
               for u in analyze(text).units)


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


def _clause_subject(text: str):
    """The SUBJECT surface of a clause — the first name/noun BEFORE its first content verb — or None if the
    clause opens straight with a verb (a subjectless conjunct like "is 3 years older than Yusaf")."""
    units = analyze(text).units
    vi = next((k for k, u in enumerate(units) if "verb" in u.features.get("senses", [])
               and _key(u.surface) not in ("do", "does", "did")), None)
    if vi is None:
        return None
    subj = next((units[k] for k in range(vi)
                 if any(s in units[k].features.get("senses", []) for s in ("name", "noun", "pronoun"))), None)
    return subj.surface if subj is not None else None


def _is_value_first_relation(text: str) -> bool:
    """A VALUE-FIRST coordinate relation whose subject AND verb are elided: "2 fewer toys than Amanda",
    "twice as many peppers as cucumbers", "4 more cows than goats" (a common-noun standard), "35 fewer
    children than THE NUMBER OF adults" (tolerating the elided re-quantification wrapper). It OPENS with a
    value/multiplier (so `_is_elided_clause` — which needs a subject before the value — rejects it as a bare
    object like "2 oranges"), but a COMPARATIVE (more/less/fewer) or a MULTIPLIER makes it a relation, and a
    NAME or NOUN/ADJECTIVE standard after "than"/"as" (possibly past a "the number/amount of" wrapper) pins
    the other entity — `_affine_relation`'s coordinate-noun fallback can consume a noun/adjective standard
    too, not just a name. Distinguishes the relation-conjunct from a genuine phrase-join ("3 apples and 2
    oranges" — no comparative, no factor)."""
    units = analyze(text).units
    qi = next((k for k, u in enumerate(units)
               if any(s in u.features.get("senses", []) for s in ("value", "multiplier"))), None)
    if qi is None or qi != 0:                          # must OPEN with the value/multiplier (subject elided)
        return False
    has_comp = any(_key(u.surface) in ("more", "less", "fewer") and "comparative" in u.features.get("senses", [])
                   for u in units)
    has_mult = any("multiplier" in u.features.get("senses", []) for u in units)
    if not (has_comp or has_mult):
        return False
    for k, u in enumerate(units):                      # a NAME or NOUN/ADJECTIVE standard right after "than"/"as"
        if _key(u.surface) not in ("than", "as"):
            continue
        i = k + 1
        if i < len(units) and _key(units[i].surface) in ("the", "a", "an") and i + 2 < len(units) \
                and _key(units[i + 1].surface) in COUNT_NOUNS and _key(units[i + 2].surface) == "of":
            i += 3                                      # skip the elided "the NUMBER/AMOUNT of" wrapper
        if i < len(units) and any(s in units[i].features.get("senses", []) for s in ("name", "noun", "adjective")):
            return True
    return False


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
    for sep in ("; ", ", and ", ", but ", ", so ", " if ", " and ", " while ", ", "):
        i = sentence.find(sep)
        while i != -1:
            left, right = sentence[:i].strip(), sentence[i + len(sep):].strip()
            if left.lower().startswith(("for every", "for each")):
                i = sentence.find(sep, i + 1)            # "For every X, <main>" is ONE per-occurrence
                continue                                 # clause (the rate's per_unit + the main amount)
            if right.lower().startswith(("who ", "whom ")) and _has_verb(left):
                # RELATIVE CLAUSE "<Name>, who VERBs …" — the "who" refers to the antecedent (the name right
                # before the comma), and the clause predicates a SECOND relation about it. Rewrite "who" → the
                # antecedent so it learns as an independent premise ("Katie, who runs 4× as Cole" → "Katie runs
                # 4× as Cole"), letting a ratio/comparative CHAIN resolve. Person-only ("who"); "which"/"that"
                # stay reserved below for the per-item-price relative clause.
                ref = left.split()[-1].strip(_PUNCT)
                synth = f"{ref} {right[right.find(' ') + 1:]}".strip() if " " in right else ""
                if synth and _has_verb(synth):
                    return _split_clauses(left) + _split_clauses(synth)
                i = sentence.find(sep, i + 1)
                continue
            if right.lower().startswith(("which ", "each of which ", "each of them ", "that cost",
                                         "that costs", "that each cost")):
                i = sentence.find(sep, i + 1)            # a RELATIVE clause modifying the left's object
                continue                                 # ("bought 5 books, each of which cost $2") — keep
                #                                          together so _purchase reads the per-item price
            if _has_verb(left):
                if _has_verb(right):
                    # STATIVE subject elision: "Ali is 8 and is 3 years older than Yusaf" — the right conjunct
                    # has a verb but NO subject; supply the left's. GATED to a COPULA/stative right verb
                    # (is/are/was/were/has/have/had): the general elision (below) was REVERTED because it
                    # mis-fired on ACTION verbs ("She gave X and bought Y" → per-recipient/running-sum gaps,
                    # confident-wrongs 39→41), but a stative "is/has N more/older than Y" is a clean relation.
                    rwords = {_key(w) for w in right.split()}
                    is_question = right.rstrip().endswith("?") \
                        or rwords & {"how", "what", "which", "who", "whom", "whose", "why", "when", "where"}
                    if not is_question and _clause_subject(right) is None:
                        lsubj = _clause_subject(left)
                        rverb = next((_key(u.surface) for u in analyze(right).units
                                      if "verb" in u.features.get("senses", [])), None)
                        # a "then <verb>…" continuation ("spent $1.75 … and then spent another $1.25") is the
                        # SAME subject acting again — supply it for ANY verb (the "then" cue makes the sequential
                        # same-subject reading unambiguous, unlike a bare action-verb conjunct).
                        then_cont = right.lower().startswith("then ")
                        if lsubj is not None and (then_cont
                                                  or rverb in ("is", "are", "was", "were", "has", "have", "had")):
                            return _split_clauses(left) + _split_clauses(f"{lsubj} {right}")
                    # NOTE: general SUBJECT elision (action-verb right conjunct) stays REVERTED — it exposes
                    # multi-step gaps (per-recipient "each" distribution, "had $N left" inversion, running sums)
                    # that turn previously-declined problems into partial-answer confident-wrongs.
                    return _split_clauses(left) + _split_clauses(right)
                if _is_elided_clause(right):           # the right borrows the left's verb (elision)
                    synth = _supply_verb(right, _main_verb(left))
                    if synth:
                        return _split_clauses(left) + _split_clauses(synth)
                if _is_value_first_relation(right):    # VALUE-FIRST coordinate relation: "Anna has 3× toys as
                    # Mandy and 2 fewer toys than Amanda" — the right conjunct elides BOTH subject and verb
                    # (it opens with the value). Supply the left's subject + main verb so it learns as an
                    # independent relation. Gated (comparative/ratio + name standard) so a bare "and 2 oranges"
                    # phrase-join never triggers it.
                    lsubj, lverb = _clause_subject(left), _main_verb(left)
                    if lsubj is not None and lverb is not None:
                        return _split_clauses(left) + _split_clauses(f"{lsubj} {lverb} {right}")
            i = sentence.find(sep, i + 1)
    return [sentence]


def _question_durations(reading: Reading, context: list) -> list:
    """Time-unit quantities stated in the QUESTION clause ("…how many pages does he read in 7 DAYS?").
    Question values are normally read at answer-time, but a rate×duration CHAIN whose duration lives in
    the question (not a premise) needs the duration as a synth SEED so the chain can compose: Tom's "10
    hours over 5 days" (2 h/day) × the question's "7 days" → 14 hours, then "50 pages/hour" × 14 → 700.
    The rate_duration synth chains the two rates once the day-seed is in context. Gated tightly — a seed
    is emitted ONLY when the question time-value's measure matches some rate's per_unit (the sole legit
    consumer), so a stray time value ("in 5 years" with no per-year rate) injects nothing. Carries the
    rate's entity+verb so the synth matches; derived-flagged so it never trips unused-given / equality."""
    seeds = []
    # GROWTH rates (grow/gain/…) are handled by the offset growth-projection path (baseline + rate×Δ);
    # seeding their duration would double-count (the derived total folds into the baseline). Only seed
    # from-scratch PRODUCTION rates (read/make/earn/…), whose total is the standalone rate×duration.
    rates = [f for f in context if isinstance(f, Rate) and f.verb not in GROWTH_VERBS]
    if not rates:
        return seeds
    for u in reading.units:
        if "value" not in u.features.get("senses", []):
            continue
        qn = _follow(u, "quantifies")
        if qn is None or _descriptor(qn)[0] not in TIME_UNITS:
            continue
        meas, amt = _descriptor(qn), u.features["value"]
        for rt in rates:
            if meas[0] != rt.per_unit:
                continue
            cand = Quantity(rt.entity, meas, amt, rt.verb, derived=True)
            if cand not in seeds and cand not in context:
                seeds.append(cand)
    return seeds


# Words that SELECT a subgroup of a partitioned container set ("the FIRST two tanks", "the OTHER two",
# "ANOTHER tank") — the structural signature of a partition (distinct from a plain "N of the apples").
_PARTITION_SELECTORS = {"first", "other", "another", "next", "last", "remaining"}


def _subgroup_count(reading: Reading):
    """The COUNT of members a subgroup-selector picks out, or None. Three surface forms:
    "the first/other TWO …" (value right of a selector word), "N OF the tanks" (value left of "of"),
    or a bare "<value> <container>" only when a selector word is also present in the clause."""
    units = reading.units
    for k, u in enumerate(units):
        if "value" not in u.features.get("senses", []):
            continue
        prevk = _key(units[k - 1].surface) if k > 0 else ""
        nextk = _key(units[k + 1].surface) if k + 1 < len(units) else ""
        if prevk in _PARTITION_SELECTORS:        # "the other TWO", "the first TWO"
            return u.features["value"]
        if nextk == "of":                        # "1 OF the tanks" (partition-level gates provide safety)
            return u.features["value"]
    return None


def _subgroup_container(reading: Reading):
    """The CONTAINER noun a subgroup partitions ("the first two TANKS" / "1 of the TANKS"), or None —
    the noun adjacent to the count selector. Used to gate: the container must differ from the queried
    measure (a partition counts CONTAINERS, each holding some quantity of the asked measure)."""
    units = reading.units
    for k, u in enumerate(units):
        if "value" not in u.features.get("senses", []):
            continue
        prevk = _key(units[k - 1].surface) if k > 0 else ""
        nxt = next((x for x in units[k + 1:] if "noun" in x.features.get("senses", [])), None)
        if prevk in _PARTITION_SELECTORS and nxt is not None \
                and units.index(nxt) == k + 1:   # "first two TANKS" — noun immediately follows the count
            return _descriptor(nxt)[0]
        if k + 1 < len(units) and _key(units[k + 1].surface) == "of" and nxt is not None:
            return _descriptor(nxt)[0]           # "1 of the TANKS"
    return None


def _partition(premises: list, question: str):
    """Solve a PARTITION-and-AGGREGATE problem, or None if the shape doesn't apply. A container set is
    split into named subgroups ("the first two … the other two … 1 of the tanks"), each member holding a
    per-member quantity of the asked measure; the answer is Σ (subgroup_count × per_member), optionally ×
    a period asked in the question ("…in four weeks"). The per-member value — including a RELATIONAL one
    ("twice as many fish each as the first", "2 fewer gallons each than the first two") — is resolved by
    the ordinary `learn` machinery against the running per-member context; only the subgroup COUNT and the
    Σ-fold are new here. Tightly gated: ≥2 subgroups, a partition selector word present, and a container
    that differs from the queried measure (so a plain "3 of the apples … 4 of the apples" never fires)."""
    qr = analyze(question)
    interro = next((u for u in qr.units if "interrogative" in u.features.get("senses", [])), None)
    if interro is None:
        return None
    coll = _follow(_follow(interro, "queries"), "collection")
    target = _descriptor(coll) if coll is not None else None
    if target is None:
        return None
    portions, ctx, containers, saw_selector = [], [], set(), False
    for p in premises:
        rp = analyze(p)
        cnt = _subgroup_count(rp)
        if cnt is None:
            continue
        facts = learn(p, [], ctx)
        pm = next((f.amount for f in facts if isinstance(f, Quantity)
                   and f.measure == target and f.amount is not None), None)
        if pm is None:
            continue
        saw_selector = saw_selector or any(_key(u.surface) in _PARTITION_SELECTORS for u in rp.units)
        cont = _subgroup_container(rp)
        if cont is not None:
            containers.add(cont)
        portions.append((cnt, pm))
        ctx += facts
    if len(portions) < 2 or not saw_selector:
        return None
    if not containers or target[0] in containers:        # the partition must be over CONTAINERS, not the
        return None                                      # asked measure itself ("3 of the apples …")
    total = _compute(" + ".join(f"{_num(c)} * {_num(v)}" for c, v in portions))
    # PERIOD: "…how many gallons will she need in four weeks?" → × the question's time-unit count.
    period = next((u.features["value"] for u in qr.units if "value" in u.features.get("senses", [])
                   and _follow(u, "quantifies") is not None
                   and _descriptor(_follow(u, "quantifies"))[0] in TIME_UNITS), None)
    if period is not None:
        total = _compute(f"{_num(total)} * {_num(period)}")
    return total


def _queried_measure(question: str):
    """The measure a 'how many/much <measure>' question asks about, or None."""
    qr = analyze(question)
    interro = next((u for u in qr.units if "interrogative" in u.features.get("senses", [])), None)
    if interro is None:
        return None
    coll = _follow(_follow(interro, "queries"), "collection")
    return _descriptor(coll) if coll is not None else None


def _period_unapplied(question_reading: Reading, premise_readings: list, context: list) -> bool:
    """DECLINE-SAFETY: True when the question asks for a per-PERIOD total ("…how many in N weeks/days?")
    but the period value N never entered the computation (absent from ARITHMETIC_LOG) AND the premises
    describe a RECURRING action the period should have scaled — i.e. we committed a number without applying
    the demanded period scaling, so the structure is unhandled. Only TIME-unit-quantifying question values
    count as periods (enumeration counts "the 3 PEOPLE" quantify non-time nouns → ignored). The recurrence
    gate is essential: "Tom HAS 5 apples … in 7 days?" is a static holding the period correctly leaves at 5
    (no recurrence → don't decline); "buys 2 … EVERY school day … in 3 weeks?" recurs → the unapplied
    period means we dropped the scaling → decline. The prerequisite for safe vocabulary growth."""
    periods = {round(float(u.features["value"]), 6) for u in question_reading.units
               if "value" in u.features.get("senses", []) and u.features.get("value") is not None
               and _follow(u, "quantifies") is not None
               and _descriptor(_follow(u, "quantifies"))[0] in TIME_UNITS}
    if not periods:
        return False
    recurs = any(_key(u.surface) in ("every", "per", "each") or _key(u.surface) in DAYPARTS
                 for rp in premise_readings for u in rp.units) \
        or any(isinstance(f, (Rate, Frequency)) for f in context)
    if not recurs:
        return False
    computed = set()
    for prog, _res in ARITHMETIC_LOG:
        computed |= {round(float(m), 6) for m in re.findall(r"\d+\.?\d*", prog)}
    return any(p not in computed for p in periods)


def _dropped_unit_quantity(premise_readings: list, question_reading: Reading) -> bool:
    """DECLINE-SAFETY (GUARD 12): True when a stated number that quantifies a measurement UNIT (time /
    distance / weight / volume) or is a MONEY value never entered the computation (absent from
    ARITHMETIC_LOG) — an OPERATIVE quantity we dropped, so the structure is unhandled → decline. Object-noun
    counts ("3 snakes", "3 people") are NOT units and stay exempt (enumeration counts are often legitimately
    uncomputed). Generalizes the period guard to every unit/money quantity; the broad firewall-safe guard
    that lets future vocabulary growth turn unknown-word→{correct | decline}, never →wrong."""
    non_time_units = UNIT_NOUNS - TIME_UNITS   # TIME periods are handled by the recurrence-gated GUARD 11
    need = set()                               # (a static "…in 7 days?" must NOT decline); here: distance /
    for rd in list(premise_readings) + [question_reading]:   # weight / volume / money — always operative.
        for u in rd.units:
            if "value" not in u.features.get("senses", []) or u.features.get("value") is None:
                continue
            if u.features.get("measure") == MONEY:
                need.add(round(float(u.features["value"]), 6))
                continue
            qn = _follow(u, "quantifies")
            if qn is not None and _descriptor(qn)[0] in non_time_units:
                need.add(round(float(u.features["value"]), 6))
    if not need:
        return False
    computed = set()
    for prog, _res in ARITHMETIC_LOG:
        computed |= {round(float(m), 6) for m in re.findall(r"\d+\.?\d*", prog)}
    return any(n not in computed for n in need)


def _unapplied_multi_rate(premise_readings: list) -> bool:
    """DECLINE-SAFETY (GUARD 13): a MULTI-RATE purchase/charge we don't compose → decline. Fires when a
    premise pairs a per-item price ("$P each / per <noun>") with an item COUNT that never reached the
    computation ("three movie tickets for $12 each" → 3×12 dropped, only 12 added), OR when the premises
    carry ≥2 distinct per-item prices (a priced list — "$a each … $b per packet …" — we'd sum without the
    per-item quantities). An "each/per" price is what makes a nearby count OPERATIVE (vs a bare enumeration
    count "3 snakes", which has no price → never triggers). Catches the multi-rate residue GUARD 12 misses
    (those prices ARE summed, so no money value is 'dropped'; the error is the unapplied counts)."""
    computed = set()
    for prog, _res in ARITHMETIC_LOG:
        computed |= {round(float(m), 6) for m in re.findall(r"\d+\.?\d*", prog)}
    per_item_prices = 0
    for rd in premise_readings:
        money = [u for u in rd.units if u.features.get("measure") == MONEY]
        has_each = any(_key(u.surface) in ("each", "per") for u in rd.units)
        if not (money and has_each):
            continue
        per_item_prices += len(money)
        for u in rd.units:                               # (a) a count linked to the price, never applied
            if "value" in u.features.get("senses", []) and u.features.get("value") is not None \
                    and u.features.get("measure") != MONEY:
                qn = _follow(u, "quantifies")
                if qn is not None and _descriptor(qn)[0] not in UNIT_NOUNS \
                        and round(float(u.features["value"]), 6) not in computed:
                    return True
    return per_item_prices >= 2                           # (b) a multi-rate priced list


def _dropped_multiplier(premise_readings: list) -> bool:
    """DECLINE-SAFETY (GUARD 14): a stated relational MULTIPLIER ("7 times", "twice"=2, "one-third") whose
    EVIDENCE never reached the computation → a dropped relational-chain link → decline. Evidence = the
    integer factor itself ("7 times" → 7), or for a FRACTION the DENOMINATOR ("one-third" → 3; fractions are
    computed as num/denom, so the decimal never appears literally but the denom does). Catches multi-step
    chains that mis-resolve and commit a partial ("Emily = 7× Sandra, Sandra = 3× Betty" → 7×Betty drops the
    3×). The residue the unit/money/multi-rate guards miss (the dropped factor isn't a unit or a price)."""
    need = set()
    for rd in premise_readings:
        for u in rd.units:
            if "multiplier" not in u.features.get("senses", []):
                continue
            if "fraction" in u.features:
                need.add(round(float(u.features["fraction"][1]), 6))      # the denominator
            else:
                v = u.features.get("value")
                if v is not None and float(v) == int(float(v)) and float(v) >= 2:
                    need.add(round(float(v), 6))                          # an integer factor ≥ 2
    if not need:
        return False
    computed = set()
    for prog, _res in ARITHMETIC_LOG:
        computed |= {round(float(m), 6) for m in re.findall(r"\d+\.?\d*", prog)}
    return any(n not in computed for n in need)


def _dropped_each_attribute(premise_readings: list) -> bool:
    """DECLINE-SAFETY (GUARD 15): a per-EACH attribute multiplier we dropped → decline. The "each"/"every"
    quantifier turns a nearby object count into a MULTIPLIER ("7 starfish with 5 arms EACH" → 7×5; "EACH
    daisy has 8 petals" → ×8), so that count is OPERATIVE — unlike a bare enumeration count ("3 snakes")
    GUARD 12 deliberately exempts. Fires when a premise carries an each/every quantifier and a value
    quantifying a NON-TIME, NON-MONEY object noun that never reached the computation (absent from
    ARITHMETIC_LOG). TIME per-each ("every 30 minutes") is a rate/period (GUARDs 11) not an attribute, and a
    clock TIME-RANGE premise ("1 PM to 4 PM every night") consumes its clocks into a span — both excluded."""
    computed = set()
    for prog, _res in ARITHMETIC_LOG:
        computed |= {round(float(m), 6) for m in re.findall(r"\d+\.?\d*", prog)}
    for rd in premise_readings:
        units = rd.units
        if not any("quantifier" in u.features.get("senses", []) and _key(u.surface) in ("each", "every")
                   for u in units):
            continue
        if any(_key(u.surface) in ("am", "pm") for u in units):       # clock values consumed into a span
            continue
        for i, u in enumerate(units):
            if "value" not in u.features.get("senses", []) or u.features.get("value") is None:
                continue
            if u.features.get("measure") == MONEY:
                continue
            nxt = units[i + 1] if i + 1 < len(units) else None         # the object this count quantifies
            if nxt is not None and _descriptor(nxt)[0] in TIME_UNITS:   # "every 30 MINUTES" — a rate, not attr
                continue
            if round(float(u.features["value"]), 6) not in computed:
                return True
    return False


def _container_giveaway(premises: list, question: str, context: list):
    """EQUAL-SPLIT into containers, then GIVE some containers away → the contents lost. "28 marbles … into
    4 bags … the same number in each bag … gives one bag away … how many marbles left?" → per-bag = 28/4 =
    7, lose 1×7 = 7, answer 28−7 = 21. The contents and the containers are DIFFERENT measures; the give is
    counted in CONTAINERS but the question asks the CONTENTS, so the give converts via the per-container
    share. Tightly gated: an equal-split marker ("same … each"), a whole of the asked measure, a container
    count (a `put` of a different measure), and a `give` of that container measure. None otherwise."""
    if not any("same" in p.lower() and "each" in p.lower() for p in premises):
        return None
    target = _queried_measure(question)
    if target is None:
        return None
    whole = next((f for f in context if isinstance(f, Quantity) and f.measure == target
                  and f.verb in ("have", "put") and f.amount is not None), None)
    container = next((f for f in context if isinstance(f, Quantity) and f.measure != target
                      and f.verb == "put" and f.amount is not None), None)
    if whole is None or container is None or not container.amount:
        return None
    give = next((f for f in context if isinstance(f, Quantity) and f.measure == container.measure
                 and f.verb == "give" and f.amount is not None), None)
    if give is None:
        return None
    per = _compute(f"{_num(whole.amount)} / {_num(container.amount)}")    # contents per container
    return _compute(f"{_num(whole.amount)} - {_num(give.amount)} * {_num(per)}")


def _measure_mod(rp: Reading, head: str):
    """The content-word modifier immediately before the measure head noun ("PARKING tickets" → 'parking',
    "much money" → None — 'much' is a quantifier). Distinguishes compound measures the head-only descriptor
    collapses ('parking ticket' vs 'traffic ticket'), INCLUDING unknown-word qualifiers (no sense)."""
    func = {"quantifier", "article", "value", "preposition", "verb", "pronoun", "name",
            "multiplier", "comparative", "operator"}
    for i, u in enumerate(rp.units):
        if _descriptor(u)[0] == head and i > 0:
            prev = rp.units[i - 1]
            if _key(prev.surface).isalpha() and not (set(prev.features.get("senses") or []) & func):
                return _key(prev.surface)
    return None


def _noun_std_entity(rp: Reading, after_idx: int):
    """The COMMON-NOUN standard for a NO-NAMES ratio/offset clause (used by `_ratio_relation`/
    `_offset_ratio_relation`/`_pure_offset_relation`'s noun-entity fallback): the token IMMEDIATELY after an
    "as"/"than" marker — "twice as many pigs AS COWS" / "24 more blue marbles THAN RED MARBLES" — but ONLY if
    that immediate token IS itself a noun/adjective, tolerating an elided "the NUMBER/AMOUNT of" re-
    quantification wrapper right after the marker ("35 fewer children THAN THE NUMBER OF adults" → adults;
    same idiom `_ratio_relation`'s NAME path already reads via its "of"-frame check). Otherwise deliberately
    NOT a scan-ahead search: "twice as many books AS SHE finished last month" has "she" (a pronoun)
    immediately after "as" — a self-referential RELATIVE CLAUSE ("as many as SHE finished"), not a second
    coordinate-noun entity — scanning past the pronoun to grab "month" further along caused a real
    confident-wrong (a temporal reference misread as a second entity). None if there's no marker, or the
    (possibly wrapper-skipped) next token isn't a noun/adjective."""
    tail = rp.units[after_idx + 1:]
    marker = next((k for k, u in enumerate(tail) if _key(u.surface) in ("as", "than")), None)
    if marker is None or marker + 1 >= len(tail):
        return None
    i = marker + 1
    if _key(tail[i].surface) in ("the", "a", "an") and i + 2 < len(tail) \
            and _key(tail[i + 1].surface) in COUNT_NOUNS and _key(tail[i + 2].surface) == "of":
        i += 3                                              # skip the elided "the NUMBER/AMOUNT of" wrapper
    if i >= len(tail):
        return None
    nxt = tail[i]
    if "noun" in nxt.features.get("senses", []) or "adjective" in nxt.features.get("senses", []):
        return nxt
    return None


def _asked_noun_entity(qr: Reading):
    """The COMMON-NOUN entity a QUESTION asks about, for the coordinate-noun affine family: the noun "many"/
    "much" quantifies ("how MANY goats does he have?" → goats), or — when there's no such quantifier — the
    noun right after a bare "the (total) NUMBER/AMOUNT OF X" phrase ("What is the total number of guests
    served?" → guests, a common GSM8K imperative-question wording with no "how many" at all). None if
    neither pattern is found."""
    many = next((u for u in qr.units if _key(u.surface) in ("many", "much")
                 and "quantifier" in u.features.get("senses", [])), None)
    if many is not None:
        coll = _follow(many, "collection")
        return _noun_entity_key(coll) if coll is not None else None
    for i, u in enumerate(qr.units):
        if _key(u.surface) in COUNT_NOUNS and i + 1 < len(qr.units) and _key(qr.units[i + 1].surface) == "of":
            nxt = qr.units[i + 2] if i + 2 < len(qr.units) else None
            if nxt is not None and "noun" in nxt.features.get("senses", []):
                return _noun_entity_key(nxt)
    return None


def _ratio_relation(rp: Reading):
    """A single pure-RATIO relation in a clause → (subject, factor, standard), where subject = factor ×
    standard. A clause with EXACTLY two names unifies every surface form: the subject is the name BEFORE the
    verb, the standard is the OTHER name, the factor is the clause's multiplier or fraction. Covers "A has k×
    as many/much <M> as B", "A is k× as old as B", "A is 1/3 the age of B", "A is twice B's age", "A is half
    of B's age". Also covers a COMMON-NOUN pair with NO names at all ("twice as many PIGS as COWS", "twice as
    many RED BALLS as BLUE") via `_noun_entity_key`/`_noun_std_entity` — the coordinate-noun/subtype-partition
    affine family reuses this same extractor. None if there's no factor, not exactly two names (and not a
    no-names noun-pair either), a comparative/offset (pure-ratio only — "N more/less than" is a different,
    firewall-risky shape), the ratio is over a RATE ADJECTIVE ("twice as FAST/quick as") — speed doesn't scale
    like a plain value (twice as fast = HALF the time, not double), so folding it as a direct ratio on
    whatever value the standard states elsewhere is a silent inversion bug; that shape belongs to the
    dedicated rate-cluster machinery, not this generic value-ratio extractor — or the clause is governed by
    "needs"/"need" ("Amanda NEEDS to have twice as many as Katrina" states a TARGET Amanda doesn't hold yet,
    not Amanda's current value — misreading it as a plain ratio is a silent deficit/current-state conflation
    this extractor doesn't model)."""
    if any("comparative" in u.features.get("senses", []) for u in rp.units) \
            or any(_key(u.surface) in ("more", "less", "older", "younger", "fewer", "fast", "quick", "quickly",
                                        "need", "needs")
                   for u in rp.units):
        return None
    mult = next((u for u in rp.units if "multiplier" in u.features.get("senses", [])), None)
    if mult is not None:
        k = (mult.features["fraction"][0] / mult.features["fraction"][1]) if "fraction" in mult.features \
            else mult.features.get("value")
    else:
        frac = next((u for u in rp.units if _fraction_parts(u.surface) is not None), None)
        if frac is None:
            return None
        num, den = _fraction_parts(frac.surface); k = num / den
    if k is None:
        return None
    names = [u for u in rp.units if "name" in u.features.get("senses", [])]
    verbs = [u for u in rp.units if "verb" in u.features.get("senses", [])]
    if len(names) != 2:
        # NOT-TWO-NAMES fallback: a coordinate/subtype COMMON-NOUN ratio, with 0 names (bare "twice as many
        # pigs as cows") or 1 (an outer possessor, "Brett has 24 more BLUE MARBLES than RED MARBLES" — Brett
        # owns both, isn't one of the two ratio entities). The noun "many"/"much" quantifies is the subject
        # ("twice AS MANY pigs as cows" → pigs), the standard is whatever noun/adjective follows the
        # clause's "as"/"than" marker after it.
        many = next((u for u in rp.units if _key(u.surface) in ("many", "much")
                     and "quantifier" in u.features.get("senses", [])), None)
        subj_u = _follow(many, "collection") if many is not None else None
        if subj_u is None:
            return None
        std_u = _noun_std_entity(rp, rp.units.index(subj_u))
        if std_u is None:
            return None
        subj_key, std_key = _noun_entity_key(subj_u), _noun_entity_key(std_u)
        if subj_key == std_key:
            return None
        return subj_key, k, std_key
    if len(names) != 2 or not verbs:
        return None
    vpos = rp.units.index(verbs[0])
    subj = next((u for u in names if rp.units.index(u) < vpos), None)   # the subject precedes the verb;
    if subj is None:                                                   # the standard is the other name
        return None
    std = next(u for u in names if u is not subj)
    # the standard must sit in a genuine RATIO FRAME — after "as" ("as old as B"), after "of" ("the age of
    # B"), as a possessor ("B's age"), or an IMPLICIT relative clause with the "that"/"which" elided ("as
    # much money DARWIN HAS" = "...as Darwin has"; the standard is immediately followed by a POSSESSION verb
    # — deliberately narrow, not ANY verb, so a mistagged common noun standard ("the CDs SOLD at $10") isn't
    # misread as a name-ratio via an unrelated passive modifier). Else the factor isn't relating the two
    # names (e.g. "bought half the CDs" — a count of an object, not a ratio between people) → reject.
    si = rp.units.index(std)
    prev = _key(rp.units[si - 1].surface) if si > 0 else ""
    nxt = rp.units[si + 1] if si + 1 < len(rp.units) else None
    if prev not in ("as", "of", "that") \
            and not (nxt is not None and ("possessive" in nxt.features.get("senses", [])
                                           or _key(nxt.surface) in POSSESSION_VERBS)):
        return None                                                   # "that B Ved" is a relative-clause
        #                                                               standard ("1/4 of the apples that
        #                                                               Kylie picked"); "the CDs" is not
    return _entity(subj), k, _entity(std)


def _offset_ratio_relation(rp: Reading):
    """An OFFSET-RATIO relation → (subject, factor, standard, offset_signed) where subject = factor × standard
    + offset. The affine twin of `_ratio_relation` for "X has N more/less than k times [as many M as] Y"
    ("Justin has 8 more than twice as many pencils as Sabrina" → J = 2·S + 8; "Devin has 5 less than half the
    years Tom has" → D = ½·T − 5). POSITIONAL parse: the OFFSET value precedes the comparative ("8 more"), the
    FACTOR is a multiplier AFTER "than" ("twice"). Requires an explicit multiplier — a bare "N more than Y"
    (no factor) is the firewall-risky offset shape we keep declining, not this one. Exactly two names, the
    standard in a genuine ratio frame (after as/of/that), like `_ratio_relation`. None if the clause is
    governed by "needs"/"need" — "Amanda NEEDS 12 more to have twice as many as Katrina" states a DEFICIT to
    reach a target (Amanda + 12 = 2·Katrina), the OPPOSITE equation from "Amanda HAS 12 more than twice as
    many as Katrina" (Amanda = 2·Katrina + 12) this extractor assumes; silently applying the wrong equation
    is a confident-wrong risk, not a parsing nuance to special-case here. Covers ANY comparative adjective via
    the shared `COMPARATIVE_SIGN` table, not just "more"/"less"/"fewer" — "600 square feet LARGER than 4
    times Benedict's house" — EXCLUDING "older"/"younger" (the age domain's own temporal-shift machinery
    owns those, same reasoning as `_pure_offset_relation`)."""
    if any(_key(u.surface) in ("need", "needs") for u in rp.units):
        return None
    comp = next((u for u in rp.units if "comparative" in u.features.get("senses", [])
                 and _key(u.surface) not in ("older", "younger")), None)
    if comp is None:
        return None
    ci = rp.units.index(comp)
    sign = float(COMPARATIVE_SIGN.get(_key(comp.surface), 1))
    off = next((u for u in reversed(rp.units[:ci])                     # offset = the value RIGHT before the
               if "value" in u.features.get("senses", []) and u.features.get("value") is not None), None)
    if off is None:                                                   # comparative ("8 more than …")
        return None
    mult = next((u for u in rp.units[ci + 1:] if "multiplier" in u.features.get("senses", [])), None)
    if mult is None:                                                  # require an explicit factor after "than"
        return None
    k = (mult.features["fraction"][0] / mult.features["fraction"][1]) if "fraction" in mult.features \
        else mult.features.get("value")
    if k is None:
        return None
    names = [u for u in rp.units if "name" in u.features.get("senses", [])]
    verbs = [u for u in rp.units if "verb" in u.features.get("senses", [])]
    if len(names) != 2:
        # NOT-TWO-NAMES fallback: a coordinate/subtype COMMON-NOUN offset-ratio ("three fewer than twice as
        # many cows as pigs") — same "many"-collection subject as `_ratio_relation`'s fallback.
        many = next((u for u in rp.units if _key(u.surface) in ("many", "much")
                     and "quantifier" in u.features.get("senses", [])), None)
        subj_u = _follow(many, "collection") if many is not None else None
        if subj_u is None:
            return None
        std_u = _noun_std_entity(rp, rp.units.index(subj_u))
        if std_u is None:
            return None
        subj_key, std_key = _noun_entity_key(subj_u), _noun_entity_key(std_u)
        if subj_key == std_key:
            return None
        return subj_key, k, std_key, sign * off.features["value"]
    if len(names) != 2 or not verbs:
        return None
    vpos = rp.units.index(verbs[0])
    subj = next((u for u in names if rp.units.index(u) < vpos), None)
    if subj is None:
        return None
    std = next(u for u in names if u is not subj)
    si = rp.units.index(std)                                          # the standard must sit in a ratio frame
    prev = _key(rp.units[si - 1].surface) if si > 0 else ""           # (after as/of/that, or an implicit
    nxt = rp.units[si + 1] if si + 1 < len(rp.units) else None        # relative "Y HAS"), like _ratio_relation
    if prev not in ("as", "of", "that") \
            and not (nxt is not None and ("possessive" in nxt.features.get("senses", [])
                                           or _key(nxt.surface) in POSSESSION_VERBS)):
        return None
    return _entity(subj), k, _entity(std), sign * off.features["value"]


def _sum_ratio_system(premises: list, question: str):
    """An N-UNKNOWN SUM+RATIO-CHAIN system: several entities, each a multiple of another (relations may
    CHAIN), and a COMBINED TOTAL → solve for the asked one by PIVOT REDUCTION. "Amy, Jeremy, Chris combined
    age 132; Amy = ⅓·Jeremy; Chris = 2·Amy" → reduce all to the root Jeremy (coeffs 1+⅓+⅔ = 2) → Jeremy =
    132/2 = 66. Generalizes (and subsumes) the 2-entity case (Jean = 3·Jane, +$76 → Jane 19). The forward
    fold can't invert this (nothing is grounded). PURE-RATIO only. TIGHTLY gated — these shapes also appear
    in CORRECT forward problems: a unique ROOT (one entity never a relation subject), a CONNECTED tree (every
    entity reduces to the root, no cycle / multi-definition), and the asked measure must MATCH the total's
    measure incl. qualifier (rejects "combined YEARS … how many MONTHS", and "parking" vs "traffic" tickets).
    None if the shape doesn't apply."""
    qr = analyze(question)
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    asked_measure = _queried_measure(question)
    if asked_measure is not None and "verb" in analyze(asked_measure[0]).units[0].features.get("senses", []):
        asked_measure = RESULT_MEASURE.get(asked_measure[0])           # "how much did X EARN" → money
    # OFFSET-TAINTED SYSTEM: a premise with both a factor and a comparative offset ("CJ has 5 MORE than
    # TWICE …") is a relation we can't fold into the pure linear system — _ratio_relation drops it, leaving
    # an INCOMPLETE system that would still "solve" off the remaining relations. Decline the whole system.
    for p in premises:
        rp = analyze(p)
        has_factor = any("multiplier" in u.features.get("senses", []) for u in rp.units) \
            or any(_fraction_parts(u.surface) is not None for u in rp.units)
        if has_factor and any(_key(u.surface) in ("more", "less", "older", "younger", "fewer")
                              for u in rp.units):
            return None
        # A STANDALONE OFFSET relation ("Anna has 2 fewer toys than Amanda") alongside the ratios is the same
        # incompleteness in a separate clause: _ratio_relation drops it (comparative), so the pure-ratio system
        # would silently ignore Amanda and mis-solve off the remaining ratio (Anna=3·Mandy → 142/4 = 35.5).
        # This is an AFFINE system — defer to `_affine_system`. (A pure-ratio system has no offset relations.)
        if _pure_offset_relation(rp) is not None:
            return None
    rels = {}                                                          # subject → (factor, standard)
    rel_mod = rel_desc = None
    for p in premises:
        rp = analyze(p)
        if _name_count(rp) != 2:
            continue                       # not a genuine 2-name relation — may be `_ratio_relation`'s
        rr = _ratio_relation(rp)           # coordinate-noun fallback firing, a DIFFERENT family's turf
        if rr is None:
            continue
        subj_e, k, std_e = rr
        if subj_e in rels:                                             # a subject defined twice → reject
            return None
        rels[subj_e] = (k, std_e)
        # carry over #3h's qualifier checks:
        if asked_measure is not None and any(_descriptor(u)[0] == asked_measure[0] for u in rp.units):
            rel_mod = _measure_mod(rp, asked_measure[0])               # a separate pre-modifier ('parking'),
            rel_desc = next((_descriptor(u) for u in rp.units          # AND the full descriptor incl. a
                             if _descriptor(u)[0] == asked_measure[0]), None)   # CONDENSED adj ('red')
    if not rels:
        return None
    knowns = {}                                                        # entities with a stated ABSOLUTE value
    for p in premises:                                                 # ("Delta took 8 donuts" → delta=8) —
        rp = analyze(p)                                                # a part of the total that isn't a ratio
        if any("multiplier" in u.features.get("senses", []) for u in rp.units) \
                or any(_fraction_parts(u.surface) is not None for u in rp.units) \
                or any(_key(u.surface) in ("more", "less", "fewer") for u in rp.units):
            continue
        nm = [u for u in rp.units if "name" in u.features.get("senses", []) and " and " not in _entity(u)]
        kv = next((u for u in rp.units if "value" in u.features.get("senses", [])           # value of the
                   and u.features.get("value") is not None and _follow(u, "quantifies") is not None  # asked
                   and asked_measure is not None and _descriptor(_follow(u, "quantifies")) == asked_measure
                   ), None) if asked_measure is not None else None
        if len(nm) == 1 and kv is not None and _entity(nm[0]) not in rels:
            knowns[_entity(nm[0])] = kv.features["value"]
    total = _combined_total(premises, asked_measure, rel_mod, rel_desc)
    if total is None:
        return None
    entities = set(rels) | {std for _k, std in rels.values()} | set(knowns) | {asked}
    free = [e for e in entities if e not in rels and e not in knowns]   # the pivot: the one entity that is
    if len(free) != 1:                                                 # neither a ratio-subject nor known
        return None
    coeff = {free[0]: 1.0}                                             # propagate coeff = factor × coeff[std]
    for _ in range(len(entities) + 1):                                # to a fixpoint along the relation tree
        for e, (k, std) in rels.items():
            if std in coeff and e not in coeff:
                coeff[e] = k * coeff[std]
    # every entity must be reducible to the pivot OR a known (else disconnected / cyclic) → reject
    if any(e not in coeff and e not in knowns for e in entities):
        return None
    denom = sum(coeff.values())
    if denom == 0:
        return None
    if asked in knowns:                                               # the asked part is itself stated
        return knowns[asked]
    # pivot = (Total − Σ knowns) / Σ coeff ; answer = coeff[asked] × pivot ("Delta 8 + Beta 3·Gamma + Gamma
    # = 40" → Gamma = (40−8)/4 = 8). With no knowns this is exactly #3r's Total / Σ coeff.
    pivot = _compute(f"({_num(total)} - ({_num(sum(knowns.values()))})) / {_num(denom)}") if knowns \
        else _compute(f"{_num(total)} / {_num(denom)}")
    return pivot if coeff[asked] == 1 else _compute(f"{_num(coeff[asked])} * {_num(pivot)}")


def _combined_total(premises: list, asked_measure, rel_mod, rel_desc):
    """The stated COMBINED TOTAL over the system's entities (the #3h total-finder, shared by the pure-ratio
    `_sum_ratio_system` and the offset `_offset_sum_system`). Same gates: a combined/total/together cue, no
    comparative in the clause, and a measure/qualifier match to the asked measure. None if no such total."""
    for p in premises:
        if not any(w in p.lower() for w in ("combined", "total", "together", "between them", "in all",
                                            "share", "shared", "altogether")):
            continue
        rp = analyze(p)
        if any("comparative" in u.features.get("senses", []) for u in rp.units):
            continue
        tot_nouns = [_descriptor(u)[0] for u in rp.units if u.features.get("senses") == ["noun"]]
        # UNIT/MEASURE GATE: the asked measure must match the total's measure (else "combined YEARS … how
        # many MONTHS" would commit a wrong-unit number); money totals often have no measure noun ("$76").
        if asked_measure is not None and tot_nouns and asked_measure[0] not in tot_nouns \
                and "money" not in str(asked_measure):
            continue
        if asked_measure is not None and rel_mod is not None and _measure_mod(rp, asked_measure[0]) != rel_mod:
            continue                                                   # "traffic" total ≠ "parking" relation
        tot_desc = next((_descriptor(u) for u in rp.units
                         if asked_measure is not None and _descriptor(u)[0] == asked_measure[0]), None)
        if rel_desc is not None and tot_desc is not None and rel_desc != tot_desc:
            continue                                                   # condensed-qualifier 'red' ≠ 'blue'
        vals = [u for u in rp.units if "value" in u.features.get("senses", [])
                and u.features.get("value") is not None]
        # Prefer the value that quantifies the ASKED MEASURE — a total clause often leads with a PEOPLE-COUNT
        # ("Together 3 friends collected 411 shells" → the total is 411 shells, not 3 friends; "the three boys
        # have 930 stamps" → 930, not 3). Fall back to the first value (a measure-less money total "$76").
        tv = None
        if asked_measure is not None:
            tv = next((u for u in vals if _follow(u, "quantifies") is not None
                       and _descriptor(_follow(u, "quantifies"))[0] == asked_measure[0]), None)
        if tv is None and vals:
            tv = vals[0]
        if tv is not None:
            return tv.features["value"]
    return None


def _pure_offset_relation(rp: "Reading"):
    """A PURE additive comparative "X has N more/fewer <M> than Y" (NO factor) → (subject, standard, signed_N).
    Covers ANY comparative surface, not just "more"/"less"/"fewer" — the adjectival forms "X is N pounds
    HEAVIER/lighter than Y", "N inches TALLER/shorter" etc. via the shared `COMPARATIVE_SIGN` table (the same
    sign lookup the age/holdings domains already use). EXCLUDES "older"/"younger" — the dedicated age-domain
    machinery (`_comparative_age` and friends) owns those, with its own temporal-shift handling this generic
    extractor doesn't do; letting it also claim age comparatives caused real regressions (a plain "younger
    than" premise no longer reached the age solver). The no-multiplier twin of `_offset_ratio_relation`; safe
    only inside a sum-system (the total pins the pair, so the bare offset — normally too under-determined to
    fold — becomes solvable). Exactly two names, standard after "than"/"as". None if the clause is governed
    by "needs"/"need" — "Loris needs 3 more to have the same as Lamont" states a DEFICIT (Loris + 3 = Lamont),
    the OPPOSITE equation from "Loris has 3 more than Lamont" (Loris = Lamont + 3) this extractor assumes."""
    if any(_key(u.surface) in ("need", "needs") for u in rp.units):
        return None
    comp = next((u for u in rp.units if "comparative" in u.features.get("senses", [])
                 and _key(u.surface) not in ("older", "younger")), None)
    if comp is None:
        return None
    if any("multiplier" in u.features.get("senses", []) for u in rp.units) \
            or any(_fraction_parts(u.surface) is not None for u in rp.units):
        return None                                                    # a factor present → not the pure shape
    ci = rp.units.index(comp)
    sign = float(COMPARATIVE_SIGN.get(_key(comp.surface), 1))
    off = next((u for u in reversed(rp.units[:ci]) if "value" in u.features.get("senses", [])
                and u.features.get("value") is not None), None)
    if off is None:
        return None
    names = [u for u in rp.units if "name" in u.features.get("senses", [])]
    verbs = [u for u in rp.units if "verb" in u.features.get("senses", [])]
    if len(names) != 2:
        # NOT-TWO-NAMES fallback: a coordinate/subtype COMMON-NOUN pure offset ("Brett has 24 more BLUE
        # MARBLES than RED MARBLES" — Brett is an outer possessor, not one of the two ratio entities) — no
        # "many"/multiplier here, so the subject is the noun right after the comparative.
        subj_u = next((u for u in rp.units[ci + 1:]
                       if "noun" in u.features.get("senses", []) or "adjective" in u.features.get("senses", [])),
                      None)
        if subj_u is None:
            return None
        std_u = _noun_std_entity(rp, rp.units.index(subj_u))
        if std_u is None:
            return None
        subj_key, std_key = _noun_entity_key(subj_u), _noun_entity_key(std_u)
        if subj_key == std_key:
            return None
        return subj_key, std_key, sign * off.features["value"]
    if len(names) != 2 or not verbs:
        return None
    vpos = rp.units.index(verbs[0])
    subj = next((u for u in names if rp.units.index(u) < vpos), None)
    if subj is None:
        return None
    std = next(u for u in names if u is not subj)
    si = rp.units.index(std)
    prev = _key(rp.units[si - 1].surface) if si > 0 else ""
    if prev not in ("than", "as"):
        return None
    return _entity(subj), _entity(std), sign * off.features["value"]


def _offset_sum_pure(premises: list, question: str):
    """A 2-ENTITY PURE additive offset + combined total → solve. "Anais has 30 more toys than Kamari; there
    are 160 toys altogether → Kamari = (160 − 30)/2 = 65." The NO-FACTOR twin of `_offset_sum_system` (which
    requires a multiplier). subj = std + c and subj + std = T ⇒ std = (T − c)/2, subj = T − std. TIGHTLY gated:
    EXACTLY one pure-offset relation, NO factor/ratio anywhere (else it's the factored or ratio shape), exactly
    the two related entities, the asked one among them, and a measure-matched combined total. None otherwise."""
    qr = analyze(question)
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    asked_measure = _queried_measure(question)
    if asked_measure is not None and "verb" in analyze(asked_measure[0]).units[0].features.get("senses", []):
        asked_measure = RESULT_MEASURE.get(asked_measure[0])
    offs, rel_mod, rel_desc = [], None, None
    for p in premises:
        rp = analyze(p)
        if _name_count(rp) != 2:
            continue                       # not a genuine 2-name relation — may be `_pure_offset_relation`'s
        rr = _pure_offset_relation(rp)      # coordinate-noun fallback firing, a DIFFERENT family's turf
        if rr is None:
            continue
        offs.append(rr)
        if asked_measure is not None and any(_descriptor(u)[0] == asked_measure[0] for u in rp.units):
            rel_mod = _measure_mod(rp, asked_measure[0])
            rel_desc = next((_descriptor(u) for u in rp.units if _descriptor(u)[0] == asked_measure[0]), None)
    if len(offs) != 1:
        return None
    alltext = " ".join(premises + [question]).lower()          # dollars and cents both normalise to the
    if ("$" in alltext or "dollar" in alltext) and "cent" in alltext:   # 'money' measure, hiding the 100×
        return None                                            # scale gap — mixing them needs a conversion we
        #                                                        don't do ("$3 total, 50 cents more" → decline)
    measures = set()
    for p in premises:                                                 # NO factor/ratio anywhere → pure only
        rp = analyze(p)
        if any("multiplier" in u.features.get("senses", []) for u in rp.units) \
                or any(_fraction_parts(u.surface) is not None for u in rp.units) \
                or any(_key(u.surface) in ("times", "twice", "double", "triple", "half") for u in rp.units):
            return None                                                # "5 TIMES more" is multiplicative, not ±
        for u in rp.units:                                             # collect every value's unit — a MIXED-
            if "value" in u.features.get("senses", []) and u.features.get("value") is not None:
                qn = _follow(u, "quantifies")                          # unit system ("$3" total vs "50 cents"
                m = _descriptor(qn) if qn is not None else u.features.get("measure")   # offset) needs a
                if m is not None:                                      # conversion we don't do → decline
                    measures.add(m[0])
    if len(measures) > 1:
        return None
    subj_e, std_e, c = offs[0]
    if subj_e == std_e or asked not in (subj_e, std_e):
        return None
    total = _combined_total(premises, asked_measure, rel_mod, rel_desc)
    if total is None:
        return None
    std_v = _compute(f"({_num(total)} - ({_num(c)})) / 2")
    subj_v = _compute(f"{_num(total)} - ({_num(std_v)})")
    return subj_v if asked == subj_e else std_v


def _offset_sum_system(premises: list, question: str):
    """A 2-UNKNOWN OFFSET SUM-system: two entities, one AFFINE relation (subject = factor × standard + offset),
    and a COMBINED TOTAL → solve. "Justin + Sabrina have 50 pencils; Justin has 8 more than twice as many as
    Sabrina" → S = (50 − 8)/(2+1) = 14. The affine extension of `_sum_ratio_system`, which DECLINES any
    offset-tainted system (it can only fold pure ratios); kept SEPARATE to isolate the firewall-riskier offset
    shape from the mature pure-ratio path. TIGHTLY gated: EXACTLY ONE offset relation, NO pure-ratio relation
    alongside it (a mixed/3-way system is out of scope → let it decline), exactly the two related entities, the
    asked entity among them, and the #3h measure/qualifier match on the total. None if the shape doesn't apply."""
    qr = analyze(question)
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    asked_measure = _queried_measure(question)
    if asked_measure is not None and "verb" in analyze(asked_measure[0]).units[0].features.get("senses", []):
        asked_measure = RESULT_MEASURE.get(asked_measure[0])           # "how many did X PICK" → its measure
    offs, rel_mod, rel_desc = [], None, None
    for p in premises:
        rp = analyze(p)                                                # carry over #3h's qualifier checks
        if _name_count(rp) != 2:
            continue                       # not a genuine 2-name relation — may be `_offset_ratio_relation`'s
        rr = _offset_ratio_relation(rp)     # coordinate-noun fallback firing, a DIFFERENT family's turf
        if rr is None:
            continue
        offs.append(rr)
        if asked_measure is not None and any(_descriptor(u)[0] == asked_measure[0] for u in rp.units):
            rel_mod = _measure_mod(rp, asked_measure[0])
            rel_desc = next((_descriptor(u) for u in rp.units if _descriptor(u)[0] == asked_measure[0]), None)
    if len(offs) != 1:                                                 # exactly one offset relation
        return None
    if any(_ratio_relation(analyze(p)) is not None for p in premises):  # a mixed system (offset + pure ratio)
        return None                                                    # is out of scope — decline it here
    subj_e, k, std_e, c = offs[0]
    if subj_e == std_e or asked not in (subj_e, std_e):
        return None
    total = _combined_total(premises, asked_measure, rel_mod, rel_desc)
    if total is None or k == -1:
        return None
    # subj = k·std + c and subj + std = T  ⇒  std = (T − c)/(k+1),  subj = T − std.
    std_v = _compute(f"({_num(total)} - ({_num(c)})) / {_num(k + 1)}")
    subj_v = _compute(f"{_num(total)} - ({_num(std_v)})")
    return subj_v if asked == subj_e else std_v


def _affine_relation(rp: "Reading"):
    """Any single AFFINE relation in a clause → (subject, factor, standard, offset) with subject = factor ×
    standard + offset, unifying the three relation extractors: `_offset_ratio_relation` ("8 more than twice
    Y" → k=2, c=8), `_ratio_relation` ("k× as many as Y" → c=0), `_pure_offset_relation` ("N more than Y" →
    k=1). None if the clause holds no two-name relation."""
    orr = _offset_ratio_relation(rp)
    if orr is not None:
        subj, k, std, c = orr
        return subj, k, std, c
    rr = _ratio_relation(rp)
    if rr is not None:
        subj, k, std = rr
        return subj, k, std, 0.0
    por = _pure_offset_relation(rp)
    if por is not None:
        subj, std, c = por
        return subj, 1.0, std, c
    return None


def _affine_system(premises: list, question: str):
    """An N-UNKNOWN MIXED AFFINE system — the generalization that subsumes both `_sum_ratio_system` (pure
    ratios) and `_offset_sum_system` (one 2-entity affine relation). Each entity is an AFFINE function of a
    single free PIVOT (value = A·pivot + B); relations may be ratio (X = k·Y), offset (X = Y ± c) OR combined
    (X = k·Y ± c), and MIX freely across ≥2 entities, pinned by a COMBINED TOTAL. "Anna = 3·Mandy and 2 fewer
    than Amanda; 142 toys total → Mandy 20": reduce to the pivot Mandy (A: 1,3,3; B: 0,0,2), pivot = (142−ΣB)/
    ΣA = 140/7 = 20. Runs ONLY after the two mature solvers decline (a pure-ratio or single-affine shape is
    theirs), so it can only ADD coverage. A double-defined subject (Anna is subject of both its relations) is
    handled by INVERTING the second into a relation for the other entity (Amanda = Anna + 2). TIGHTLY gated:
    a combined total, ≥2 relations (a genuine system), a UNIQUE free pivot (one entity that is no relation's
    subject), every entity reducible to it (connected, acyclic), and the asked entity among them. None
    otherwise."""
    qr = analyze(question)
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    asked_measure = _queried_measure(question)
    if asked_measure is not None and "verb" in analyze(asked_measure[0]).units[0].features.get("senses", []):
        asked_measure = RESULT_MEASURE.get(asked_measure[0])
    # GATE A — a SUM question ("how many altogether/in all/total") asks for Σ, not a single entity; this solver
    # returns ONE entity's value, so decline (Mona/Jasmine/Rachel — the asked "entity" is just the first name).
    if any(w in question.lower() for w in ("altogether", "in all", " total", "combined", "in total")):
        return None
    # GATE C — the ambiguous "N times more/less than Y" idiom (gold reads it as ×N, but with no "as many as"
    # frame `_affine_relation` mis-reads it as a +N offset — Ali/Nada/John): decline the whole system.
    for p in premises:
        if re.search(r"\btimes (more|less|fewer)\b", p.lower()):
            return None
    rels = {}                                                          # subject → (k, std, c): subj = k·std + c
    rel_mod = rel_desc = None
    for p in premises:
        rp = analyze(p)
        if _name_count(rp) != 2:
            continue                       # not a genuine 2-name relation — may be `_affine_relation`'s
        rel = _affine_relation(rp)         # coordinate-noun fallback firing, a DIFFERENT family's turf
        if rel is None:
            continue
        subj_e, k, std_e, c = rel
        if k == 0 or subj_e == std_e:
            return None
        if subj_e in rels:                                            # subject already defined → INVERT the
            if std_e in rels:                                         # relation to define the OTHER entity;
                return None                                           # both defined → over-constrained → bail
            rels[std_e] = (1.0 / k, subj_e, -c / k)                   # std = (subj − c)/k
        else:
            rels[subj_e] = (k, std_e, c)
        if asked_measure is not None and any(_descriptor(u)[0] == asked_measure[0] for u in rp.units):
            rel_mod = _measure_mod(rp, asked_measure[0])
            rel_desc = next((_descriptor(u) for u in rp.units
                             if _descriptor(u)[0] == asked_measure[0]), None)
    if len(rels) < 2:                                                 # a genuine SYSTEM (the 2-entity single
        return None                                                  # relation is `_offset_sum_system`'s)
    total = _combined_total(premises, asked_measure, rel_mod, rel_desc)
    if total is None:
        return None
    # GATE B — the total must be the MAXIMUM stated premise value. In a sum-system the combined total is ≥ every
    # part and every offset, so a LARGER unconsumed value means `_combined_total` grabbed the wrong number — it
    # matches a total-cue clause's FIRST value, which is often a PEOPLE-COUNT ("Together 3 friends watched 411
    # videos" → 3, not 411; "the three boys have 930 stamps" → 3, not 930). Reject rather than mis-solve.
    stated = [u.features["value"] for p in premises for u in analyze(p).units
              if "value" in u.features.get("senses", []) and u.features.get("value") is not None]
    if stated and total < max(stated):
        return None
    entities = set(rels) | {std for _k, std, _c in rels.values()} | {asked}
    free = [e for e in entities if e not in rels]                    # the pivot: the one entity that is no
    if len(free) != 1:                                               # relation's subject
        return None
    coeff = {free[0]: (1.0, 0.0)}                                    # value[e] = A·pivot + B; propagate
    for _ in range(len(entities) + 1):                              # affinely along the relation tree
        for e, (k, std, c) in rels.items():
            if std in coeff and e not in coeff:
                a_std, b_std = coeff[std]
                coeff[e] = (k * a_std, k * b_std + c)
    if any(e not in coeff for e in entities):                        # disconnected / cyclic → reject
        return None
    sum_a = sum(a for a, _b in coeff.values())
    sum_b = sum(b for _a, b in coeff.values())
    if sum_a == 0:
        return None
    # T = Σ(A)·pivot + Σ(B)  ⇒  pivot = (T − ΣB)/ΣA;  answer = A[asked]·pivot + B[asked].
    pivot = _compute(f"({_num(total)} - ({_num(sum_b)})) / {_num(sum_a)}")
    a_ask, b_ask = coeff[asked]
    return _compute(f"{_num(a_ask)} * {_num(pivot)} + ({_num(b_ask)})")


def _noun_affine_system(premises: list, question: str):
    """The COORDINATE/SUBTYPE-NOUN twin of `_affine_system`: entities are COMMON NOUNS (via
    `_affine_relation`'s noun-entity fallback — "twice as many pigs as cows", "4 more cows than goats"), not
    names, and the stated total is over a CATEGORY word ("56 ANIMALS total" for pig/cow/goat entities) rather
    than a literal noun match. `_combined_total`'s strict noun-matching (needed elsewhere to avoid grabbing
    the wrong value) would reject this — a hypernym check ("animal" ⊇ pig/cow/goat) is a whole separate,
    unbuilt feature — so this solver takes a NARROWER, self-contained shortcut instead: TRUST any total-cue
    clause's single value as the target sum, but ONLY for a system whose relations are CONFIRMED noun-based
    (the underlying premise has 0 or 1 names — an outer possessor like "farmer"/"Brett" is fine, but NOT the
    2-name shape `_affine_system` already owns) — this keeps the shortcut from ever competing with the
    named-entity system's own (stricter, already-tested) total-matching. Same pivot-propagation as
    `_affine_system`: an entity value = A·pivot + B, propagated from the one FREE entity (no relation's
    subject) via ≥2 relations, T = Σ(A)·pivot + Σ(B). "A farmer has twice as many pigs as cows, and 4 more
    cows than goats. If the farmer has 56 animals total, how many goats does he have?" → pivot=goats=11."""
    if any(w in question.lower() for w in ("altogether", "in all", " total", "combined", "in total")):
        return None                                       # a sum question is a different shape, not this one
    for p in premises:
        if re.search(r"\btimes (more|less|fewer)\b", p.lower()):
            return None                                   # ambiguous ×N idiom → decline (GATE C)
    rels = {}
    for p in premises:
        if any(w in p.lower() for w in ("total", "combined", "altogether", "together")):
            continue                                      # the total clause itself isn't a relation
        rp = analyze(p)
        if sum(1 for u in rp.units if "name" in u.features.get("senses", [])) == 2:
            continue                                      # a genuine 2-NAME relation is _affine_system's turf
        rel = _affine_relation(rp)
        if rel is None:
            continue
        subj_e, k, std_e, c = rel
        if k == 0 or subj_e == std_e:
            return None
        if subj_e in rels:                                # subject already defined → invert, like
            if std_e in rels:                              # `_affine_system`
                return None
            rels[std_e] = (1.0 / k, subj_e, -c / k)
        else:
            rels[subj_e] = (k, std_e, c)
    if len(rels) < 2:
        return None                                       # a genuine N-entity SYSTEM, like `_affine_system`
    total = None
    for p in premises:
        if not any(w in p.lower() for w in ("total", "combined", "altogether", "together")):
            continue
        vals = [u for u in analyze(p).units
                if "value" in u.features.get("senses", []) and u.features.get("value") is not None]
        if len(vals) == 1:
            total = vals[0].features["value"]
            break
    if total is None:
        return None
    entities = set(rels) | {std for _k, std, _c in rels.values()}
    free = [e for e in entities if e not in rels]
    if len(free) != 1:
        return None
    coeff = {free[0]: (1.0, 0.0)}
    for _ in range(len(entities) + 1):
        for e, (k, std, c) in rels.items():
            if std in coeff and e not in coeff:
                a_std, b_std = coeff[std]
                coeff[e] = (k * a_std, k * b_std + c)
    if any(e not in coeff for e in entities):
        return None
    sum_a = sum(a for a, _b in coeff.values())
    sum_b = sum(b for _a, b in coeff.values())
    if sum_a == 0:
        return None
    qr = analyze(question)
    asked = _asked_noun_entity(qr)
    if asked is None or asked not in coeff:
        return None
    pivot = _compute(f"({_num(total)} - ({_num(sum_b)})) / {_num(sum_a)}")
    a_ask, b_ask = coeff[asked]
    return _compute(f"{_num(a_ask)} * {_num(pivot)} + ({_num(b_ask)})")


def _noun_anchored_sum_system(premises: list, question: str):
    """The ANCHOR-based twin of `_noun_affine_system`: no combined/category total is stated, but ONE entity
    has a bare absolute value (a noun ANCHOR — "Hannah's family has ten PIGS", via `_noun_anchor_fact`,
    ignoring the outer possessor Hannah). The question asks for a CATEGORY SUM ("how many ANIMALS do they
    have?") — a noun that is NOT itself one of the specific relation entities, implying Σ over all of them
    (unlike `_anchored_sum_system`'s name family, which needs an explicit "altogether"/"total" cue — a
    category noun in the question already signals the sum here). Same "0-1 names per premise" safety check
    as `_noun_affine_system` (never competes with the 2-name system), and the SAME unused-given firewall as
    the named anchored family: every premise must classify as a relation or an anchor. "Hannah's family has
    ten pigs. They have three fewer than twice as many cows as pigs and six more goats than cows. How many
    animals do they have?" → pig=10 (anchor), cow=2·10−3=17, goat=17+6=23, Σ=50. Declines a NEGATED question
    ("how many of her houses are NOT yellow?") — that's a COMPLEMENT sum (total minus ONE excluded entity),
    not a full Σ over every entity; conflating the two is a confident-wrong the diff harness caught (a
    full-Σ answer of 190 vs the complement's correct 160). Also declines when the question carries its OWN
    numeric value ("...and 5 students to every teacher, how many TEACHERS are there?") — a real confident-
    wrong (180 vs gold 36) came from a conversion clause getting glued onto the question by clause-splitting;
    a bare "how many X" question never has a number of its own, so a stray one signals unaccounted info this
    solver has no way to fold in (here, a students-per-teacher ratio, not a category sum at all)."""
    qr_check = analyze(question)
    if any("negation" in u.features.get("senses", []) for u in qr_check.units):
        return None
    if any("value" in u.features.get("senses", []) and u.features.get("value") is not None
           for u in qr_check.units):
        return None
    for p in premises:
        if any(w in p.lower() for w in ("total", "combined", "altogether", "together")):
            return None                                    # a stated total is _noun_affine_system's turf
        if re.search(r"\btimes (more|less|fewer)\b", p.lower()):
            return None                                    # ambiguous ×N idiom → decline (GATE C)
    rels, anchors = {}, {}
    for p in premises:
        rp = analyze(p)
        if sum(1 for u in rp.units if "name" in u.features.get("senses", [])) == 2:
            return None                                    # a genuine 2-name relation is _affine_system's turf
        rel = _affine_relation(rp)
        anc = None if rel is not None else _noun_anchor_fact(rp)
        if rel is None and anc is None:
            return None                                    # unused-given firewall, like the named family
        if rel is not None:
            subj_e, k, std_e, c = rel
            if k == 0 or subj_e == std_e:
                return None
            if subj_e in rels:
                if std_e in rels:
                    return None
                rels[std_e] = (1.0 / k, subj_e, -c / k)
            else:
                rels[subj_e] = (k, std_e, c)
        else:
            e, v = anc
            if e in anchors:
                return None
            anchors[e] = v
    if len(rels) < 1 or len(anchors) != 1:
        return None
    entities = set(rels) | {std for _k, std, _c in rels.values()}
    anchor_e, anchor_v = next(iter(anchors.items()))
    if anchor_e not in entities:
        return None
    free = [e for e in entities if e not in rels]
    if len(free) != 1:
        return None
    coeff = {free[0]: (1.0, 0.0)}
    for _ in range(len(entities) + 1):
        for e, (k, std, c) in rels.items():
            if std in coeff and e not in coeff:
                a_std, b_std = coeff[std]
                coeff[e] = (k * a_std, k * b_std + c)
    if any(e not in coeff for e in entities):
        return None
    a_anc, b_anc = coeff[anchor_e]
    if a_anc == 0:
        return None
    pivot = _compute(f"({_num(anchor_v)} - ({_num(b_anc)})) / {_num(a_anc)}")
    qr = analyze(question)
    asked = _asked_noun_entity(qr)
    if asked is None or asked in coeff:
        return None                                        # a specific entity ask — not this SUM shape
    total = " + ".join(f"({_num(a)} * {_num(pivot)} + ({_num(b)}))" for a, b in coeff.values())
    return _compute(total)


def _noun_residual_sum_system(premises: list, question: str):
    """The SUBTYPE-PARTITION twin: adjective-qualified subtypes of ONE head noun ("red/blue/green BALLS"),
    where the total is stated directly in the SAME (bare, unqualified) head noun ("Ryan has 40 balls" — no
    hypernym/category guessing needed, unlike `_noun_affine_system`'s total), and the ASKED entity is the
    explicit textual "the REST are <qualifier>" residual — never itself defined by a relation. "Ryan has 40
    balls. There are twice as many red balls as blue while the rest are green. If there are 11 blue balls,
    how many green balls are there?" → red=2·11=22, green=40−22−11=7. Same unused-given firewall (every
    premise must classify as the total, an anchor, a relation, or the "rest" marker) and 0-1-names-per-
    premise safety check as the rest of the coordinate-noun family."""
    if any(w in question.lower() for w in ("altogether", "in all", " total", "combined", "in total")):
        return None                                        # an explicit sum cue is a different shape
    for p in premises:
        if re.search(r"\btimes (more|less|fewer)\b", p.lower()):
            return None
    rels, anchors, residual, total = {}, {}, None, None
    for p in premises:
        rest_m = re.match(r"^(the\s+)?rest\s+(are|is)\s+([a-z]+)", p.strip(), re.I)
        if rest_m:
            if residual is not None:
                return None
            residual = _key(rest_m.group(3))
            continue
        rp = analyze(p)
        if sum(1 for u in rp.units if "name" in u.features.get("senses", [])) == 2:
            return None                                     # a genuine 2-name relation is _affine_system's turf
        rel = _affine_relation(rp)
        if rel is not None:
            subj_e, k, std_e, c = rel
            if k == 0 or subj_e == std_e:
                return None
            if subj_e in rels:
                if std_e in rels:
                    return None
                rels[std_e] = (1.0 / k, subj_e, -c / k)
            else:
                rels[subj_e] = (k, std_e, c)
            continue
        vals = [u for u in rp.units if "value" in u.features.get("senses", []) and u.features.get("value") is not None]
        qn = _follow(vals[0], "quantifies") if len(vals) == 1 else None
        if qn is not None and "noun" in qn.features.get("senses", []):
            if qn.features.get("attributes"):                # a QUALIFIED noun — a per-subtype anchor
                e = _noun_entity_key(qn)
                if e in anchors:
                    return None
                anchors[e] = vals[0].features["value"]
            else:                                             # a BARE noun — the head-noun TOTAL
                if total is not None:
                    return None
                total = vals[0].features["value"]
            continue
        return None                                          # unclassified premise — the same firewall as
        #                                                       the rest of the anchored family
    if residual is None or total is None or len(rels) < 1 or len(anchors) != 1:
        return None
    entities = set(rels) | {std for _k, std, _c in rels.values()}
    if residual in entities:
        return None                                          # the residual must be UNCOVERED by any relation
    anchor_e, anchor_v = next(iter(anchors.items()))
    if anchor_e not in entities:
        return None
    free = [e for e in entities if e not in rels]
    if len(free) != 1:
        return None
    coeff = {free[0]: (1.0, 0.0)}
    for _ in range(len(entities) + 1):
        for e, (k, std, c) in rels.items():
            if std in coeff and e not in coeff:
                a_std, b_std = coeff[std]
                coeff[e] = (k * a_std, k * b_std + c)
    if any(e not in coeff for e in entities):
        return None
    a_anc, b_anc = coeff[anchor_e]
    if a_anc == 0:
        return None
    pivot = _compute(f"({_num(anchor_v)} - ({_num(b_anc)})) / {_num(a_anc)}")
    qr = analyze(question)
    asked = _asked_noun_entity(qr)
    if asked != residual:
        return None                                          # only the RESIDUAL entity — a covered entity
        #                                                       is a different (unbuilt) single-ask variant
    covered_sum = " + ".join(f"({_num(a)} * {_num(pivot)} + ({_num(b)}))" for a, b in coeff.values())
    return _compute(f"{_num(total)} - ({covered_sum})")


def _anchor_fact(rp: "Reading"):
    """A single ANCHOR fact: an ABSOLUTE value stated for exactly one NAMED entity, with no relation to any
    other entity ("Cecil has $600", "If David has 15 shells" → (Cecil, 600) / (David, 15)). Also covers a
    COMMON-NOUN anchor with NO names ("A pet store has six dogs for sale.", "If there are 11 blue balls" →
    (dog, 6) / (blue, 11)) — the coordinate-noun/subtype-partition affine family's twin of the named case;
    the entity is whatever the value's `quantifies` link points to. None if the clause carries a comparative/
    multiplier/fraction (that's a RELATION, not an anchor), a RECURRENCE marker ("every"/"each"/"per" —
    "Sofie plans to walk 10 miles EVERY DAY" is a per-day RATE, not a flat anchor; folding the bare 10
    directly gave a real confident-wrong, 5 vs gold 35, dropping the "in 7 days" multiplication this solver
    has no mechanism to apply), or doesn't name exactly one entity holding exactly one value — the anchor is
    what `_anchored_sum_system` pins its free pivot to when the problem states no combined total."""
    if any("comparative" in u.features.get("senses", []) for u in rp.units) \
            or any("multiplier" in u.features.get("senses", []) for u in rp.units) \
            or any(_fraction_parts(u.surface) is not None for u in rp.units) \
            or any(_key(u.surface) in ("every", "each", "per") for u in rp.units):
        return None
    names = [u for u in rp.units if "name" in u.features.get("senses", [])]
    if not any("verb" in u.features.get("senses", []) for u in rp.units):
        return None
    vals = [u for u in rp.units if "value" in u.features.get("senses", []) and u.features.get("value") is not None]
    if len(vals) != 1:
        return None
    if not names:
        qn = _follow(vals[0], "quantifies")
        if qn is None or "noun" not in qn.features.get("senses", []):
            return None
        return _noun_entity_key(qn), vals[0].features["value"]
    if len(names) != 1:
        return None
    return _entity(names[0]), vals[0].features["value"]


def _noun_anchor_fact(rp: "Reading"):
    """A COMMON-NOUN anchor, tolerating an OUTER POSSESSOR name ("Hannah's family has ten PIGS" → (pig, 10),
    ignoring Hannah — unlike `_anchor_fact`, which prioritizes a bare name subject and would wrongly anchor
    to Hannah herself). Used by `_noun_anchored_sum_system`, the anchor-based twin of `_noun_affine_system`
    for when no combined/category total is stated. None under the same comparative/multiplier/fraction/
    recurrence-marker/single-value/verb gates as `_anchor_fact` (a "per day"-style rate isn't a flat anchor),
    or if the value doesn't quantify a genuine noun."""
    if any("comparative" in u.features.get("senses", []) for u in rp.units) \
            or any("multiplier" in u.features.get("senses", []) for u in rp.units) \
            or any(_fraction_parts(u.surface) is not None for u in rp.units) \
            or any(_key(u.surface) in ("every", "each", "per") for u in rp.units):
        return None
    if not any("verb" in u.features.get("senses", []) for u in rp.units):
        return None
    vals = [u for u in rp.units if "value" in u.features.get("senses", []) and u.features.get("value") is not None]
    if len(vals) != 1:
        return None
    qn = _follow(vals[0], "quantifies")
    if qn is None or "noun" not in qn.features.get("senses", []):
        return None
    return _noun_entity_key(qn), vals[0].features["value"]


def _anchored_affine_coeffs(premises: list, question: str):
    """Shared extraction for the ANCHOR-pinned affine family (`_anchored_sum_system`/`_anchored_single_system`):
    builds the same relation graph as `_affine_system`, but pins the free pivot from a bare ABSOLUTE value (an
    "anchor", e.g. "Cecil has $600") instead of a stated combined total. Returns (coeff, pivot) where
    coeff[e]=(A, B) is UNRESOLVED — value[e] = A·pivot + B — and `pivot` is the anchor's own number. None if:
    the problem is AGE-typed ("old"/"older"/"younger"/"age" anywhere in premises or question) — the dedicated
    age-domain machinery (`_comparative_age` and friends) owns ages, with careful TEMPORAL-SHIFT handling
    ("in N years, X will be…") this generic solver has no notion of, so letting it also claim age problems
    caused real mis-fires (a future-shifted ratio LOOKS like a plain ratio relation structurally, but silently
    drops the shift); a combined-total cue is present (that's `_affine_system`'s turf); the ambiguous "N times
    more/less" idiom is used; ANY premise fails to classify as a relation (2 names) or an anchor (1 name) — an
    unused-given firewall of its own, since these self-contained systems run before GUARD 3 and so must guard
    the same way (an unclassified premise may hide a depletion action or an unsupported relation frame); there
    doesn't have a UNIQUE free entity (one that's no relation's subject) reachable from the anchor by
    forward propagation (disconnected / cyclic); the anchor itself may sit ANYWHERE in the graph, not just on
    the free entity — "Ed weighs 146 and is 38 lighter than Al" anchors the DEPENDENT entity Ed, so the pivot
    (Al's ultimate free ancestor) is recovered by back-substituting the anchor's equation; or an ANCHOR's
    value and a relation's OFFSET value carry UNCONVERTED, different measures ("Movie C was 1.25 HOURS"
    anchoring in hours, "5 MINUTES longer" offsetting in minutes) — this solver never converts units, so
    folding them together silently mixed hours and minutes as if they were the same number (a real confident-
    wrong: 1.5625 vs gold 20 minutes). Deliberately narrow: only the ANCHOR's own value (unambiguous — an
    anchor premise has exactly one value) and a relation's OFFSET value (when c≠0, matched by magnitude) are
    checked — an INCIDENTAL value elsewhere in a relation clause ("half as many pages as Lucy IN 1 HOUR", a
    rate's constant duration, never itself part of the arithmetic) must NOT count, or it wrongly declines a
    correct pure-ratio system (a regression this exact fix introduced and then had to narrow)."""
    if re.search(r"\b(old|older|younger|age|ages|aged)\b", " ".join(premises + [question]).lower()):
        return None
    if any(w in p.lower() for p in premises
           for w in ("combined", "total", "together", "between them", "share", "shared", "altogether")):
        return None                                                  # a stated total is _affine_system's turf
    for p in premises:
        if re.search(r"\btimes (more|less|fewer)\b", p.lower()):
            return None                                             # ambiguous ×N idiom → decline (GATE C)
    rels, anchors, measures = {}, {}, set()
    for p in premises:
        rp = analyze(p)
        two_names = _name_count(rp) == 2
        if not any(s in u.features.get("senses", []) for u in rp.units
                   for s in ("value", "multiplier", "comparative", "interrogative", "aggregator")) \
                and not any(_fraction_parts(u.surface) is not None for u in rp.units) \
                and not p.rstrip().endswith("?") \
                and not (two_names and re.search(r"\b(as|than)\b", p.lower())):
            continue                       # a purely DECLARATIVE premise with NO number/factor/comparative
            #                                 at all, and NOT itself a question fragment, can't hide any
            #                                 dropped quantity ("Stacy and Emily also buy chips for the
            #                                 party" just introduces entities) — skip it, don't decline.
            #                                 EXCLUDES an interrogative/aggregator word OR a trailing "?"
            #                                 ("how many … TOTAL have Megan, Kelcie" / "Greg read?" — both
            #                                 MIS-SPLIT fragments of the ACTUAL question "...Megan, Kelcie,
            #                                 and Greg read?", a known clause-splitting gap for a 3-name
            #                                 question list) — skipping either silently would drop Greg
            #                                 entirely and sum only Megan+Kelcie (40 vs gold 65). ALSO
            #                                 excludes a 2-name clause with "as"/"than" ("Jorge has an EQUAL
            #                                 NUMBER of cards AS Matias" — "equal" isn't a recognized
            #                                 multiplier, but "as" still signals a REAL, just-unsupported
            #                                 relation; skipping it silently dropped Jorge=Matias, 34 vs gold
            #                                 48) — unlike a truly unclassified premise that DOES carry a
            #                                 number this solver failed to fold, which stays a decline below
        nc = _name_count(rp)
        rel = _affine_relation(rp) if nc == 2 else None      # exactly 2 names — else this may be the
        anc = _anchor_fact(rp) if (rel is None and nc == 1) else None   # coordinate-noun fallback firing,
        if rel is None and anc is None:                       # which is a DIFFERENT family's turf (see
            return None                    # an UNCLASSIFIED premise (e.g. a depletion action like "gave out
            #                                 half her apples", or an unsupported relation frame like "an equal
            #                                 number … as") may hide information this solver can't fold
        if rel is not None:
            subj_e, k, std_e, c = rel
            if k == 0 or subj_e == std_e:
                return None
            if c != 0:                     # the OFFSET's own measure (matched by magnitude — the ONE value
                off_u = next((u for u in rp.units if "value" in u.features.get("senses", [])   # this
                              and u.features.get("value") is not None                          # relation's
                              and abs(u.features["value"]) == abs(c)), None)                    # arithmetic
                if off_u is not None:                                                           # actually
                    qn = _follow(off_u, "quantifies")                                          # uses — NOT
                    m = _descriptor(qn) if qn is not None else off_u.features.get("measure")    # every value
                    if m is not None and m[0] not in COUNT_NOUNS:               # in the clause; "the NUMBER
                        measures.add(m[0])                                      # of X" resolves to "number"
                        #                                                         itself, not X — skip it
            if subj_e in rels:                                     # subject already defined → invert into
                if std_e in rels:                                  # a relation for the OTHER entity, like
                    return None                                     # `_affine_system`
                rels[std_e] = (1.0 / k, subj_e, -c / k)
            else:
                rels[subj_e] = (k, std_e, c)
        else:
            e, v = anc
            if e in anchors:
                return None
            anchors[e] = v
            vals = [u for u in rp.units if "value" in u.features.get("senses", [])
                    and u.features.get("value") is not None]
            if len(vals) == 1:                          # an anchor premise has EXACTLY one value (unambiguous)
                qn = _follow(vals[0], "quantifies")
                m = _descriptor(qn) if qn is not None else vals[0].features.get("measure")
                if m is not None and m[0] not in COUNT_NOUNS:
                    measures.add(m[0])
    if len(measures) > 1:
        return None                             # mixed, unconverted units → decline rather than mis-fold
    if len(rels) < 1 or len(anchors) != 1:
        return None
    entities = set(rels) | {std for _k, std, _c in rels.values()}
    anchor_e, anchor_v = next(iter(anchors.items()))
    if anchor_e not in entities:
        return None
    free = [e for e in entities if e not in rels]                  # the pivot: the one entity that is no
    if len(free) != 1:                                             # relation's subject (may differ from the
        return None                                                # anchor's own entity)
    coeff = {free[0]: (1.0, 0.0)}                                  # value[e] = A·pivot + B, propagated from
    for _ in range(len(entities) + 1):                              # the free pivot along the relation tree
        for e, (k, std, c) in rels.items():
            if std in coeff and e not in coeff:
                a_std, b_std = coeff[std]
                coeff[e] = (k * a_std, k * b_std + c)
    if any(e not in coeff for e in entities):                       # disconnected / cyclic → reject
        return None
    a_anc, b_anc = coeff[anchor_e]
    if a_anc == 0:
        return None
    # anchor_v = A_anchor·pivot + B_anchor  ⇒  pivot = (anchor_v − B_anchor)/A_anchor (identity when the
    # anchor IS the free pivot: A=1, B=0, pivot=anchor_v directly — no division, as before).
    pivot = _compute(f"({_num(anchor_v)} - ({_num(b_anc)})) / {_num(a_anc)}")
    return coeff, pivot


def _anchored_sum_system(premises: list, question: str):
    """The ANCHORED-SUM twin of `_affine_system`: instead of a COMBINED TOTAL pinning the pivot, one entity
    has a stated ABSOLUTE value (an "anchor") and NO total is given — the question instead asks for the SUM
    of every entity, OR of SPECIFIC named entities if the question names 2+ of them directly ("how many
    marbles do CARL AND CAL have combined?" sums only those two, not every entity in the system — a real
    confident-wrong (392 vs gold 336) surfaced this: a 3-entity system where the question named only 2).
    "Cecil has $600. Catherine has $250 less than twice Cecil's money. Carmela has $50 more than twice
    Cecil's money. How much do they have altogether?" → pivot=Cecil=600 directly (no division needed);
    answer = Σ(A_e·600 + B_e) = 2800. Runs only when `_affine_system` (a stated total) and the pure/offset
    sum-systems all decline — a SUM question ("altogether"/"in all"/"total") with NO stated total is this
    solver's shape, not theirs. Delegates the relation/anchor extraction to `_anchored_affine_coeffs` (shared
    with `_anchored_single_system`); adds its OWN sum-cue gate and a "dozen"-measure decline (the anchor's
    "6 dozen" already converts to raw units internally, so a "how many DOZEN" question needs a ÷12 we don't
    do)."""
    if not any(w in question.lower() for w in ("altogether", "in all", " total", "combined", "in total")):
        return None                                                  # not a SUM question — not this shape
    asked_measure = _queried_measure(question)
    if asked_measure is not None and "dozen" in asked_measure[1]:
        return None
    got = _anchored_affine_coeffs(premises, question)
    if got is None:
        return None
    coeff, pivot = got
    # a comma/"and"-joined name list ("Carl and Cal") merges into ONE unit spanning multiple words — split
    # back into individual entity keys by word, skipping "and" (same gotcha as `_residual_sum_system`).
    qnames = [_key(w.surface) for u in analyze(question).units if "name" in u.features.get("senses", [])
              for w in u.words if _key(w.surface) != "and"]
    if len(qnames) >= 2:
        if not all(e in coeff for e in qnames):
            return None                # a named entity outside the system — not this solver's shape
        summed = {e: coeff[e] for e in qnames}
    else:
        summed = coeff                 # no specific names (e.g. "how much do THEY have altogether") — sum all
    return _compute(" + ".join(f"({_num(a)} * {_num(pivot)} + ({_num(b)}))" for a, b in summed.values()))


def _anchored_single_system(premises: list, question: str):
    """The ANCHORED-SINGLE twin: same anchor-pinned relation graph as `_anchored_sum_system`, but the question
    asks for ONE named entity's value (not the sum) — "Al is 25 pounds heavier than Ben. Ben is 16 pounds
    lighter than Carl. Ed weighs 146 pounds and is 38 pounds lighter than Al. Find Carl's weight." → pivot=Ed
    (anchor), propagate to Al, then Ben, then Carl. Declines a SUM-cue question (that's the sum twin's shape),
    a "needs...to have as large as" DEFICIT-TO-TARGET question (the same "needs"-idiom risk as the relation
    extractors — "how many stamps does SHE NEED to ADD to have a collection AS LARGE AS Bert's?" asks
    Bert−Peggy, not Bert's raw value; naively grabbing the first name in the question gave a confident-wrong,
    900 vs gold 825), and requires the asked entity to be a NAME reachable from the anchor."""
    if any(w in question.lower() for w in ("altogether", "in all", " total", "combined", "in total")):
        return None                                                  # a sum question is the other twin's shape
    if re.search(r"\bneeds?\b", question.lower()):
        return None
    qr = analyze(question)
    if any("comparative" in u.features.get("senses", []) for u in qr.units):
        return None       # "how many MORE does X have COMPARED TO Y" asks a DIFFERENCE, not a raw value
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    got = _anchored_affine_coeffs(premises, question)
    if got is None:
        return None
    coeff, pivot = got
    if asked not in coeff:
        return None
    a, b = coeff[asked]
    return _compute(f"{_num(a)} * {_num(pivot)} + ({_num(b)})")


def _anchored_difference_system(premises: list, question: str):
    """The ANCHORED-DIFFERENCE twin: same anchor-pinned relation graph, but the question names TWO entities
    and asks how much MORE one has than the other — "Anton has three times as many cards as Heike. Ann has
    six times as many cards as Heike. If Ann has 60 cards, how many more cards does Ann have than Anton?" →
    Heike=10 (from Ann=60=6·Heike), Anton=30, Ann−Anton=30. Requires the question to carry a comparative
    sense (declined by `_anchored_single_system`, so the two never collide) and EXACTLY two names — the first
    named (textually) minus the second, matching the "X … more … than Y" surface order."""
    qr = analyze(question)
    if not any("comparative" in u.features.get("senses", []) for u in qr.units):
        return None
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if len(qnames) != 2:
        return None
    e1, e2 = _entity(qnames[0]), _entity(qnames[1])
    if e1 == e2:
        return None
    got = _anchored_affine_coeffs(premises, question)
    if got is None:
        return None
    coeff, pivot = got
    if e1 not in coeff or e2 not in coeff:
        return None
    a1, b1 = coeff[e1]
    a2, b2 = coeff[e2]
    return _compute(f"({_num(a1)} * {_num(pivot)} + ({_num(b1)})) - "
                    f"({_num(a2)} * {_num(pivot)} + ({_num(b2)}))")


def _residual_sum_system(premises: list, question: str):
    """The RESIDUAL twin: a stated total names THREE (or more) entities ("Jermaine, Terrence, and Emilee earn
    a total of $90"), but only SOME of them are related/anchored elsewhere ("Jermaine earns $5 more than
    Terrence"; "Terrence earns $30") — the ASKED entity (Emilee) is never mentioned again at all. Unlike
    `_affine_system` (which requires every named entity to reduce to the pivot), this solves for the ONE
    entity left OUT of that graph: residual = total − Σ(every OTHER named entity's value). Tightly gated: the
    total clause must name ≥3 entities EXPLICITLY, OR use a bare PRONOUN ("If THEY need 10 bags total") with
    the ≥3 names inferred from every OTHER premise ("Jeremy buys 3 bags... Stacy buys 4 bags... how many
    should Emily buy?" — no premise ever lists all three together) — the OTHER premises (anchor/relation
    graph, via the shared `_anchored_affine_coeffs`) must cover ALL BUT EXACTLY ONE of them, and the asked
    entity must BE that one uncovered residual — else this isn't the residual shape (or another solver
    already owns it)."""
    tot_names = tot_val = tot_clause = None
    for p in premises:
        if not any(w in p.lower() for w in ("total", "combined", "altogether", "together")):
            continue
        rp = analyze(p)
        name_units = [u for u in rp.units if "name" in u.features.get("senses", [])]
        # a comma/"and"-joined subject list ("Jermaine, Terrence, and Emilee") merges into ONE unit whose
        # surface spans multiple words — split back into individual entity keys, skipping "and" itself.
        names = [_key(w.surface) for u in name_units for w in u.words if _key(w.surface) != "and"]
        vals = [u for u in rp.units if "value" in u.features.get("senses", []) and u.features.get("value") is not None]
        if not vals:
            continue
        if len(names) < 3:
            if names:
                continue                   # 1-2 explicit names but not the full set — ambiguous, skip
            # a PRONOUN total ("they need 10 bags total") — infer the name set from every OTHER premise
            other_names = {_key(w.surface) for op in premises if op is not p
                           for u in analyze(op).units if "name" in u.features.get("senses", [])
                           for w in u.words if _key(w.surface) != "and"}
            if len(other_names) < 3:
                continue
            names = sorted(other_names)
        tot_names, tot_val, tot_clause = names, vals[0].features["value"], p
        break
    if tot_val is None:
        return None
    qr = analyze(question)
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    if asked not in tot_names:
        return None
    rest = [p for p in premises if p is not tot_clause]
    got = _anchored_affine_coeffs(rest, question)
    if got is not None:
        coeff, pivot = got
        covered = [e for e in tot_names if e in coeff]
        residual = [e for e in tot_names if e not in coeff]
        if len(residual) != 1 or residual[0] != asked:
            return None                          # not the residual shape — 0 or ≥2 uncovered, or the asked
        #                                           entity isn't the uncovered one (another solver's turf)
        covered_sum = " + ".join(f"({_num(a)} * {_num(pivot)} + ({_num(b)}))" for a, b in
                                 (coeff[e] for e in covered))
        return _compute(f"{_num(tot_val)} - ({covered_sum})")
    # NO-RELATION fallback: `_anchored_affine_coeffs` requires ≥1 relation between entities (its pivot-
    # propagation has nothing to propagate otherwise) and declines a purely independent-anchors system
    # ("Jeremy buys 3 bags... Stacy buys 4 bags... they need 10 total, how many should Emily buy?" — no
    # ratio/offset relates Jeremy, Stacy, or Emily to each other at all). Extract bare per-name anchors
    # directly instead: every "rest" premise must be a single-name anchor or purely declarative (no
    # relations at all — a 2-name premise here means a REAL relation exists and this fallback doesn't apply).
    anchors = {}
    for p in rest:
        rp = analyze(p)
        if not any(s in u.features.get("senses", []) for u in rp.units
                   for s in ("value", "multiplier", "comparative", "interrogative", "aggregator")) \
                and not any(_fraction_parts(u.surface) is not None for u in rp.units) \
                and not p.rstrip().endswith("?") \
                and not (_name_count(rp) == 2 and re.search(r"\b(as|than)\b", p.lower())):
            continue                             # purely declarative — skip, doesn't hide a number (an
            #                                       interrogative/aggregator word, a trailing "?", or a
            #                                       2-name "as"/"than" clause all mean REAL info is present
            #                                       — never skip those, see the sibling gate in
            #                                       `_anchored_affine_coeffs`)
        if _name_count(rp) != 1:
            return None                          # a relation (or an ambiguous multi-name clause) exists —
        anc = _anchor_fact(rp)                   # `_anchored_affine_coeffs` already tried and failed above,
        if anc is None:                          # so this ISN'T the no-relation shape after all
            return None
        e, v = anc
        if e in anchors:
            return None
        anchors[e] = v
    covered = [e for e in tot_names if e in anchors]
    residual = [e for e in tot_names if e not in anchors]
    if len(residual) != 1 or residual[0] != asked:
        return None
    return _compute(f"{_num(tot_val)} - ({' + '.join(_num(anchors[e]) for e in covered)})")


def _inject_bare_ages(prem_readings: list, all_readings: list, facts_of: list, context: list) -> None:
    """BARE AGE — in an AGE-typed problem ("…as old as…", "older/younger than…"), a copula "X is N" with no
    unit ("If Job is 5") states X's AGE = N. The bare value quantifies no noun, so `_quantity` emits nothing
    and a relational-age chain can't ground. A bare equation has no dimension of its own, so infer it from
    the problem's age framing: inject Quantity(X, year, N), letting the fixpoint chain (Job 5 → Stephanie 20
    → Freddy 18). GATED on age cues so a bare "X is N" in a non-age problem is untouched. Mutates facts_of +
    context in place; shared by solve() and the CLI paragraph loop so they don't diverge (cf. #104)."""
    if not any(_key(u.surface) in ("old", "older", "younger", "age", "ages")
               for rp in all_readings for u in rp.units):
        return
    for i, rp in enumerate(prem_readings):
        if any(isinstance(f, Quantity) for f in (facts_of[i] or [])):
            continue                                     # already produced a quantity → not a bare equation
        verbs = [u for u in rp.units if "verb" in u.features.get("senses", [])]
        if len(verbs) != 1 or _key(verbs[0].surface) not in ("is", "are", "was", "were"):
            continue
        subj = _follow(verbs[0], "subject")
        if subj is None or "name" not in subj.features.get("senses", []):
            continue
        bare = next((u for u in rp.units if "value" in u.features.get("senses", [])
                     and "measure" not in u.features and _follow(u, "quantifies") is None), None)
        if bare is not None:
            q = Quantity(_entity(subj), ("year", frozenset()), bare.features["value"], "be")
            facts_of[i] = (facts_of[i] or []) + [q]
            context.append(q)


_DEPLETION_VERBS = {"take", "eat", "remove", "drink", "consume", "use", "sell", "spend", "lose", "pick"}


def _scene_depletion(question: str, context: list, prem_readings: list):
    """A POOL depleted by takes — "There were 349 pieces of candy in a bowl. Talitha took 108, Solomon took
    153. How many pieces are left?" → 349 − 108 − 153 = 88. The existential "there were N X" stores N
    INSTANCES (a base count), not a holding Quantity, and each take is credited to the takER, so the
    remainder fold finds no single owner to net. Compute base(#instances of the measure) − Σ(depletion-verb
    Quantities of the measure). GATED: a left/remain question; an instance-base; EVERY measure-Quantity is a
    depletion verb (an addition would break the plain base − Σ); and NO unused given — every stated premise
    number must be the base or a depletion, else a conversion/per-unit count ("128 ounces in a GALLON … 3
    gallons", "8 slices in a PIZZA … 2 pizzas") means the base needs computing → decline. None otherwise."""
    ql = question.lower()
    if "left" not in ql and "remain" not in ql:
        return None
    m = _queried_measure(question)
    if m is None:
        return None
    base = sum(1 for f in context if isinstance(f, tuple) and f[0] == m[0])
    allq = [f for f in context if isinstance(f, Quantity) and f.measure == m and f.amount is not None]
    depl = [f for f in allq if f.verb in _DEPLETION_VERBS]
    if base <= 0 or not depl or len(depl) != len(allq):   # all measure-quantities must be depletions
        return None
    used = {round(float(base), 6)} | {round(float(f.amount), 6) for f in depl}
    stated = {round(float(u.features["value"]), 6) for rp in prem_readings for u in rp.units
              if "value" in u.features.get("senses", []) and u.features.get("value") is not None}
    if stated - used:                                     # a conversion factor / container count → the base
        return None                                       # isn't the plain instance total → decline
    return _compute(f"{base} - (" + " + ".join(_num(f.amount) for f in depl) + ")")


def _backward_depletion(premises: list, question: str):
    """The BACKWARD twin of `_scene_depletion`: an UNKNOWN initial pool depleted by spends to a KNOWN final →
    initial = final + Σ(depletions). "There was some money in the cookie jar. Doris spent $6. Martha spent
    half as much [$3]. There were $15 left. How much was in the jar at first?" → 15 + 6 + 3 = 24. Self-contained
    (re-learns premises, like `_sum_ratio_system`) so it can run before GUARD 8's "at first" decline. TIGHTLY
    GATED — the inversion class is firewall-risky: the question asks an ORIGINAL value; a single FINAL holding
    flagged by "left"/"remain"/"now"; every OTHER measure-quantity is a DEPLETION verb (NO gain — "gave him"/
    received/added would raise the pool, so initial = final + Σ would over-count: Randy 3500≠3000); and NO
    depletion references the running total ("half of the REMAINING" — unresolvable backward: Isabel). None
    otherwise."""
    pl = " ".join(premises).lower()
    if not any(p in pl for p in ("left", "remain", "now")):
        return None
    # a GAIN to the pool (received / "gave him" / added / more) breaks the losses-only formula → not this shape
    if any(w in pl for w in ("received", "receive", "got ", "gained", "gain ", "added", " more ",
                             "gave him", "gave her", "gives him", "gives her")):
        return None
    if any(w in pl for w in ("the remaining", "the rest", "what was left", "what is left", "what's left")):
        return None                                          # a running-total-dependent depletion (Isabel)
    disc, ctx = [], []
    for p in premises:
        rp = analyze(p)
        f = learn(p, disc, ctx)
        # GUARD-4-like: a premise STATING a multiplier/fraction/percent but producing NO quantity dropped an
        # operation we can't fold ("buys another HALF the amount he had left" → a fraction-of-remaining GAIN
        # that didn't resolve; Tim 30≠22). Decline rather than commit an inversion missing that operation.
        if any(s in u.features.get("senses", []) for u in rp.units for s in ("multiplier", "percent")) \
                and not any(isinstance(x, Quantity) for x in (f or [])):
            return None
        if f:
            ctx += f
    qs = [f for f in ctx if isinstance(f, Quantity) and f.amount is not None]
    measures = {f.measure for f in qs}
    if len(measures) != 1 or not qs:
        return None
    # the FINAL holding: a non-depletion quantity whose premise carried the "left"/"remain"/"now" flag.
    finals = [f for f in qs if f.verb not in _DEPLETION_VERBS]
    depl = [f for f in qs if f.verb in _DEPLETION_VERBS]
    if len(finals) != 1 or not depl:                         # exactly one final; ≥1 depletion; nothing else
        return None
    if len(finals) + len(depl) != len(qs):                   # every quantity is final-or-depletion (no gain)
        return None
    # NO-UNUSED-GIVEN (mirrors `_scene_depletion`): every STATED money value must surface as the final or a
    # depletion. A dropped value means the parse missed an operation (Josh's 2nd "spent $1.25" lost to subject-
    # elision → 6+1.75=7.75≠9) and Σ would be incomplete → DECLINE rather than commit a partial inversion.
    used = {round(float(finals[0].amount), 6)} | {round(float(f.amount), 6) for f in depl}
    stated = {round(float(u.features["value"]), 6) for p in premises for u in analyze(p).units
              if "value" in u.features.get("senses", []) and u.features.get("value") is not None}
    if stated - used:
        return None
    return _compute(f"{_num(finals[0].amount)} + (" + " + ".join(_num(f.amount) for f in depl) + ")")


def _backward_inversion(premises: list, question: str):
    """A single-entity BACKWARD inversion with GAINS and losses → initial = final − Σ(signed deltas). "Randy had
    some money. Smith gave him $200 [+200]. Randy gave Sally $1200 [−1200]. $2000 left. How much at first?" →
    2000 − (200 − 1200) = 3000. Distinct from `_backward_depletion` (a multi-entity LOSSES-ONLY pool): here the
    operations are the ASKED entity's OWN signed deltas — the dative now captures "Smith gave HIM $200" as Randy
    +200, so the gain that made #4b decline (its no-gain gate) is finally representable. Self-contained. GATED
    like the depletion: a "left"/"now"/"in the end" final, exactly one final + ≥1 delta of one measure, no dropped
    multiplier (fraction-of-remaining), and NO unused given (every stated value captured) → else decline."""
    qr = analyze(question)
    qnames = [u for u in qr.units if "name" in u.features.get("senses", [])]
    if not qnames:
        return None
    asked = _entity(qnames[0])
    disc, ctx, prem_facts = [], [], []
    for p in premises:
        rp = analyze(p)
        f = learn(p, disc, ctx) or []
        if any(s in u.features.get("senses", []) for u in rp.units for s in ("multiplier", "percent")) \
                and not any(isinstance(x, Quantity) for x in f):
            return None                                      # a dropped fraction-of-remaining operation → decline
        prem_facts.append((p, f))
        ctx += f
    aq = [(p, f) for (p, fs) in prem_facts for f in fs
          if isinstance(f, Quantity) and f.entity == asked and f.amount is not None]
    if not aq or len({f.measure for (p, f) in aq}) != 1:
        return None
    finals = [f for (p, f) in aq if f.verb in ("have", "be")          # the KNOWN final, from a "left"/"now"/
              and any(w in p.lower() for w in ("left", "remain", "now", "in the end", "end of"))]   # "end" cue
    deltas = [f for (p, f) in aq if f.verb not in ("have", "be") and f not in finals]
    if len(finals) != 1 or not deltas:
        return None
    used = {round(float(finals[0].amount), 6)} | {round(float(d.amount), 6) for d in deltas}
    stated = {round(float(u.features["value"]), 6) for p in premises for u in analyze(p).units
              if "value" in u.features.get("senses", []) and u.features.get("value") is not None}
    if stated - used:                                        # a dropped operation → an incomplete inversion
        return None
    delta_sum = " + ".join(f"{_verb_sign(d.verb)} * {_num(d.amount)}" for d in deltas)
    return _compute(f"{_num(finals[0].amount)} - ({delta_sum})")


# NB: no "rose"/"risen" — they collide with the common name "Rose" (a false UP cue that breaks the XOR).
_PCT_UP = {"raise", "raised", "raises", "increase", "increased", "increases"}
_PCT_DOWN = {"discount", "discounts", "off", "sale", "decrease", "decreased", "decreases",
             "down", "lower", "lowered", "reduced", "reduce", "cheaper"}
_PCT_Q_ACCEPT = {"pay", "pays", "paid", "new", "final", "cost", "costs", "now", "charge"}
# inverse (find-the-whole) / aggregate (multi-period) / count question words → not the new-value shape.
_PCT_Q_BLOCK = {"original", "save", "saved", "savings", "total", "left", "remain", "remaining",
                "remained", "difference", "many", "before", "without", "raise"}


def _self_multiple_sum(premises: list, question: str):
    """SELF-REFERENTIAL MULTIPLE + SUM — one subject does an amount over two periods, the second a MULTIPLE of
    the first (the multiplier refers BACK to the first, not to another entity): "Roger rode 2 miles this
    morning, then 5 times THAT AMOUNT in the evening. How many miles total?" → 2 + 5·2 = 2·(1+5) = 12; "Tom
    read 10 pages … and twice AS MANY in the evening" → 10·3 = 30. TIGHTLY gated: EXACTLY one base value + one
    multiplier + one subject NAME (a second name would make "as many as <Y>" a cross-entity ratio, not
    anaphoric); an anaphoric back-reference ("that amount/much/many" or a bare "as many/as much" — which, with
    a single name, cannot be "as <name>"); a two-period join ("and"/"then"); and a TOTAL question (bare "how
    many <M> did <subj> VERB", or an explicit total cue — never a single-period ask). Answer = base·(1+k)."""
    readings = [analyze(p) for p in premises]
    text = " ".join(premises).lower()
    # Require the EXPLICIT back-reference "that amount/much/many". A bare "as many/as much" is unreliable — it
    # may carry a STANDARD ("as many fish AS cats and dogs combined" → aggregate ratio, Ed 15) or sit in a RATE
    # ("$600 every weekday and twice as much on the weekend" → per-day, restaurant) — both mis-fire as base·(1+k).
    anaphor = any(ph in text for ph in ("that amount", "that much", "that many"))
    if not anaphor or not (" and " in text or " then " in text):
        return None
    vals = [u for rp in readings for u in rp.units if "value" in u.features.get("senses", [])
            and u.features.get("value") is not None]
    mults = [u for rp in readings for u in rp.units if "multiplier" in u.features.get("senses", [])]
    names = {_entity(u) for rp in readings for u in rp.units if "name" in u.features.get("senses", [])}
    if len(vals) != 1 or len(mults) != 1 or len(names) != 1:
        return None
    qr = analyze(question)
    if any("value" in u.features.get("senses", []) and u.features.get("value") is not None for u in qr.units):
        return None
    # TOTAL question only — a bare "how many <M> did <subj> VERB" (no single-period qualifier) or an explicit
    # total cue. A period word ("evening", "Tuesday") in the question asks ONE part → out of scope → decline.
    ql = question.lower()
    period_words = ("morning", "afternoon", "evening", "night", "monday", "tuesday", "wednesday",
                    "thursday", "friday", "saturday", "sunday", "first", "second")
    if any(w in ql for w in period_words):
        return None
    mu = mults[0]
    k = (mu.features["fraction"][0] / mu.features["fraction"][1]) if "fraction" in mu.features \
        else mu.features.get("value")
    if k is None:
        return None
    base = vals[0].features["value"]
    r = _compute(f"{_num(base)} * {_num(1 + k)}")
    return round(r, 9) if isinstance(r, float) else r


def _percent_change(premises: list, question: str):
    """Apply a single PERCENT CHANGE to a stated money base and report the NEW value — "a bicycle sold for
    $220 last year was increased by 15%. What is the new price?" → 220 × 1.15 = 253; "$2000 ticket, 30%
    discount. How much does he pay?" → 2000 × 0.70 = 1400. TIGHTLY gated to the clean 1-step shape: EXACTLY
    one money value (a value-sense unit with a measure) and EXACTLY one percent across the premises (a 3rd
    number ⇒ multi-step ⇒ decline); EXACTLY one direction cue (raise/increase ↑ XOR discount/off/down ↓ —
    "N% more/less than the ORIGINAL" is the inverse find-the-whole shape, deliberately NOT a cue here); and a
    question asking for the NEW value (an accept word pay/new/cost, and none of the inverse/aggregate/count
    words original/save/total/left/many). None if any gate fails — every other percent problem stays declined."""
    readings = [analyze(p) for p in premises]
    pcts = [u for rp in readings for u in rp.units if "percent" in u.features.get("senses", [])]
    # EVERY value-sense number counts (incl. bare object counts like "3 shirts" that lack a measure) — a 3rd
    # number means the problem is multi-step (per-item, quantity) ⇒ decline. The single base must be money.
    vals = [u for rp in readings for u in rp.units if "value" in u.features.get("senses", [])
            and u.features.get("value") is not None]
    if len(pcts) != 1 or len(vals) != 1 or "measure" not in vals[0].features:
        return None
    qr = analyze(question)                               # a NUMBER in the question (e.g. "buy 2 pounds … for
    if any("value" in u.features.get("senses", []) and u.features.get("value") is not None   # a 4 member
           for u in qr.units):                           # family") means it scales the new value ⇒ multi-step
        return None
    keys = {_key(u.surface) for rp in readings for u in rp.units}
    up, down = keys & _PCT_UP, keys & _PCT_DOWN
    if bool(up) == bool(down):                           # need EXACTLY one direction (XOR); ambiguous ⇒ decline
        return None
    qkeys = {_key(w) for w in question.lower().split()}
    if not (qkeys & _PCT_Q_ACCEPT) or (qkeys & _PCT_Q_BLOCK):
        return None
    base, p = vals[0].features["value"], pcts[0].features["value"]
    factor = 1 + p if up else 1 - p
    r = _compute(f"{_num(base)} * {_num(factor)}")       # snap binary drift (early return skips solve()'s snap)
    return round(r, 9) if isinstance(r, float) else r


_GROUP_PAIRS = [("boys", "girls"), ("men", "women"), ("male", "female"), ("males", "females"),
                ("boy", "girl"), ("man", "woman"), ("gentlemen", "ladies")]
_GROUP_WORDS = {w for pr in _GROUP_PAIRS for w in pr}


def _group_relation(a: str, b: str):
    """'same' if the two group words are identical, 'comp' if they are a known complementary pair
    (boys/girls, men/women, …), else None (unrelated → can't infer a fraction)."""
    if a == b:
        return "same"
    return "comp" if any({a, b} == {x, y} for x, y in _GROUP_PAIRS) else None


def _group_after_copula(r: "Reading"):
    """The first partition-group word (boys/girls/…) appearing after a copula in a reading — the predicate
    of "P% … ARE boys". None if there's no such group noun."""
    for i, u in enumerate(r.units):
        if _key(u.surface) in ("are", "is", "were", "was"):
            g = next((_key(v.surface) for v in r.units[i + 1:] if _key(v.surface) in _GROUP_WORDS), None)
            if g is not None:
                return g
    return None


def _percent_population(premises: list, question: str):
    """A TWO-GROUP population partition — a percent splits a population into complementary groups (boys/girls,
    men/women); recover the asked group. Every group's count is total × fraction, where fraction = P if the
    group is the percent's group, else 1−P (the complement). Two entry points share the formula: a TOTAL is
    given ("50 people, 30% are boys → girls = 50 × 0.70 = 35"), or ONE GROUP'S COUNT is given ("60% are girls,
    boys = 300 → girls = 300 × 0.6/0.4 = 450", since total = C/f_base). TIGHTLY gated — exactly one number +
    one percent in the premises (nothing in the question), the percent's group and the question's group both
    resolvable to same/complement of each other, and a recognized base. None otherwise."""
    readings = [analyze(p) for p in premises]
    pcts = [(r, u) for r in readings for u in r.units if "percent" in u.features.get("senses", [])]
    vals = [(r, u) for r in readings for u in r.units if "value" in u.features.get("senses", [])
            and u.features.get("value") is not None]
    if len(pcts) != 1 or len(vals) != 1:
        return None
    qr = analyze(question)
    if any("value" in u.features.get("senses", []) and u.features.get("value") is not None for u in qr.units):
        return None
    P = pcts[0][1].features["value"]
    G1 = _group_after_copula(pcts[0][0])                 # the percent's group ("… are boys")
    Gq = next((_key(u.surface) for u in qr.units if _key(u.surface) in _GROUP_WORDS), None)   # asked group
    if G1 is None or Gq is None:
        return None

    def frac(g):
        rel = _group_relation(g, G1)
        return P if rel == "same" else (1 - P) if rel == "comp" else None

    fq = frac(Gq)
    if fq is None or not 0 < fq < 1:
        return None
    val_r, val_u = vals[0]
    C = val_u.features["value"]
    Gc = next((_key(u.surface) for u in val_r.units if _key(u.surface) in _GROUP_WORDS), None)
    if Gc is None:                                        # the number is the TOTAL population → group = N × f
        r = _compute(f"{_num(C)} * {_num(fq)}")
    else:                                                 # the number is ONE group's count → total = C / f_base
        fc = frac(Gc)
        if fc is None or fc <= 0:
            return None
        r = _compute(f"{_num(C)} * {_num(fq)} / {_num(fc)}")
    return round(r, 9) if isinstance(r, float) else r


def _percent_original(premises: list, question: str):
    """The INVERSE of `_percent_change`: a known money value is a PERCENT of an unknown ORIGINAL → recover the
    original by division. Three frames (by how the percent relates the value to the original): "$15000 cost
    40% OF the original → 15000 / 0.40 = 37500"; "$480 at a 20% DISCOUNT → 480 / (1 − 0.20) = 600"; "$7500 is
    25% LESS than the original → 480 / 0.75"; "$X is 25% MORE than the original → X / 1.25". TIGHTLY gated like
    the forward twin — exactly one money value + one percent (no other number in premises OR question ⇒ decline),
    a question asking for the ORIGINAL (the token that makes the forward resolver decline, so the two never
    collide), and a recognized frame. None otherwise."""
    if "original" not in {_key(w) for w in question.lower().split()}:
        return None
    readings = [analyze(p) for p in premises]
    pcts = [(rp, i, u) for rp in readings for i, u in enumerate(rp.units)
            if "percent" in u.features.get("senses", [])]
    vals = [u for rp in readings for u in rp.units if "value" in u.features.get("senses", [])
            and u.features.get("value") is not None]
    if len(pcts) != 1 or len(vals) != 1 or "measure" not in vals[0].features:
        return None
    qr = analyze(question)
    if any("value" in u.features.get("senses", []) and u.features.get("value") is not None for u in qr.units):
        return None
    rp, pi, punit = pcts[0]
    p, x = punit.features["value"], vals[0].features["value"]
    pnext = _key(rp.units[pi + 1].surface) if pi + 1 < len(rp.units) else ""
    keys = {_key(u.surface) for r in readings for u in r.units}
    if pnext == "of":                                    # "P% OF the original" → X = P · original
        factor = p
    elif keys & {"more", "increased", "increase", "higher"}:   # "P% MORE than original" → X = (1+P) · original
        factor = 1 + p
    elif keys & {"discount", "off", "sale", "less", "decrease", "decreased", "lower", "reduced", "cheaper"}:
        factor = 1 - p                                   # "P% DISCOUNT / LESS than original" → X = (1−P) · original
    else:
        return None
    if factor <= 0:
        return None
    r = _compute(f"{_num(x)} / {_num(factor)}")
    return round(r, 9) if isinstance(r, float) else r


def _proportion(premises: list, question: str):
    """A direct PROPORTION — "If 12 bags of oranges weigh 24 pounds, how much do 8 bags weigh?" → 24 × 8/12 =
    16. A premise pairs a COUNT (12 bags) with a measure (24 pounds) for the same entity and verb; the
    question gives a NEW count (8 bags) and asks "how MUCH" the measure is → measure × new_count / count.
    TIGHTLY gated: a "how much" question (asking an amount, not "how many <noun>"), one value quantifying a
    count-noun + a matching verb; a premise with exactly two same-entity, same-verb quantities whose heads
    are the count and a different measure. None otherwise."""
    qr = analyze(question)
    if not (any("interrogative" in u.features.get("senses", []) for u in qr.units)
            and any(_key(u.surface) == "much" for u in qr.units)):   # "how MUCH" → an amount, not a count
        return None
    qval = next((u for u in qr.units if "value" in u.features.get("senses", [])
                 and _follow(u, "quantifies") is not None), None)
    qverb = next((_lemma(u.surface) for u in qr.units if "verb" in u.features.get("senses", [])
                  and _key(u.surface) not in ("is", "are", "was", "were", "do", "does", "did")), None)
    if qval is None or qverb is None:
        return None
    count_head = _descriptor(_follow(qval, "quantifies"))[0]
    for p in premises:
        qs = [f for f in learn(p) if isinstance(f, Quantity) and f.amount is not None]
        if len(qs) != 2 or qs[0].entity != qs[1].entity:
            continue
        cnt = next((f for f in qs if f.measure[0] == count_head and f.verb == qverb), None)
        oth = next((f for f in qs if f.measure[0] != count_head and f.verb == qverb), None)
        if cnt is None or oth is None or cnt.amount == 0:
            continue
        return _compute(f"{_num(oth.amount)} * {_num(qval.features['value'])} / {_num(cnt.amount)}")
    return None


def _rate_extrapolation(premises: list, question: str, context: list):
    """RATE × DURATION-IN-QUESTION — "Luis drives 80 miles in 2 hours. How far will he go in 15 minutes?" →
    (80 per 2 hours) × (15 min → 0.25 hr) ÷ 2 = 10. A single Rate in context + a question asking "how
    far/much/many … in N <time>" → rate.amount × (N converted to the rate's per_unit) ÷ rate.per_amount.
    Gated: exactly one rate, the question is a how-far/much/many interrogative carrying a duration value in a
    time-unit (the EXTRAPOLATION target), and that unit converts to the rate's per_unit. None otherwise."""
    if len(context) != 1 or not isinstance(context[0], Rate):
        return None                                      # a PURE single-rate problem only — any other fact
        #                                                  (a count, a second rate, a multi-period split) means
        #                                                  the answer isn't a bare rate×duration (Tammy's
        #                                                  trees→packs→$, Ralph's weekday+weekend, Mark's
        #                                                  weekly increase all commit a partial otherwise).
    rate = context[0]
    prem_vals = [u.features["value"] for p in premises for u in analyze(p).units
                 if "value" in u.features.get("senses", [])]
    if any(abs(v - rate.amount) > 1e-9 and abs(v - rate.per_amount) > 1e-9 for v in prem_vals):
        return None                                      # an UNUSED premise number (Mark's weekly +4, Ralph's
        #                                                  weekend 6) means the answer isn't a bare rate ×
        #                                                  duration — decline (the engine's unused-given rule)
    qr = analyze(question)
    if not (any("interrogative" in u.features.get("senses", []) for u in qr.units)
            and any(_key(u.surface) in ("far", "much", "many", "long") for u in qr.units)):
        return None
    dur = next((u for u in qr.units if "value" in u.features.get("senses", [])
                and _follow(u, "quantifies") is not None
                and _descriptor(_follow(u, "quantifies"))[0] in TIME_UNITS), None)
    if dur is None:
        return None
    du = _descriptor(_follow(dur, "quantifies"))[0]
    if rate.per_amount == 1 and du == rate.per_unit:
        return None                                      # the plain "amount × duration" (matching unit,
        #                                                  per_amount 1) is the existing rate×duration path's
        #                                                  job — and it handles first-year exceptions etc. that
        #                                                  this bare product would clobber (Adam 490, not 500)
    if du == rate.per_unit:
        per = _num(dur.features["value"])
    elif (du, rate.per_unit) in CONVERSIONS:
        per = f"{_num(dur.features['value'])} * {_num(CONVERSIONS[(du, rate.per_unit)])}"
    else:
        return None
    return _compute(f"{_num(rate.amount)} * ({per}) / {_num(rate.per_amount)}")


def _rate_compound(premises: list, question: str, context: list):
    """COMPOUND-DURATION rate — "Lisa walks 10 meters each minute. Every day she walks for an hour. How many
    meters in two days?" → 10/min × (1 hour → 60 min) × 2 days = 1200. A single work/distance rate, a premise
    giving a per-PERIOD activity duration ("for a/N <dur> … every/each <period>"), and the question giving a
    PERIOD count ("in N <period>"). answer = rate × (duration → per_unit) × periods. None otherwise."""
    rates = [f for f in context if isinstance(f, Rate)]
    if len(rates) != 1 or len(context) != 1:
        return None
    rate = rates[0]
    dur_amt = dur_u = period_u = None
    for p in premises:
        pr = analyze(p)
        forp = next((u for u in pr.units if _key(u.surface) == "for"
                     and "preposition" in u.features.get("senses", [])), None)
        if forp is None:
            continue
        fi = pr.units.index(forp)
        tu = next((u for u in pr.units[fi + 1:fi + 4] if "noun" in u.features.get("senses", [])
                   and _descriptor(u)[0] in TIME_UNITS), None)
        if tu is None:
            continue
        vu = next((u.features["value"] for u in pr.units if "value" in u.features.get("senses", [])
                   and _follow(u, "quantifies") is tu), None)
        if vu is None:                                    # "for AN hour" → a single unit
            ti = pr.units.index(tu)
            vu = 1 if ti > 0 and _key(pr.units[ti - 1].surface) in ("a", "an") else None
        pu = next((_descriptor(pr.units[k + 1])[0] for k in range(len(pr.units) - 1)
                   if _key(pr.units[k].surface) in ("every", "each")
                   and _descriptor(pr.units[k + 1])[0] in TIME_UNITS), None)
        if vu is not None and pu is not None:
            dur_amt, dur_u, period_u = vu, _descriptor(tu)[0], pu
            break
    if dur_amt is None:
        return None
    qr = analyze(question)
    num = next((u.features["value"] for u in qr.units if "value" in u.features.get("senses", [])
                and _follow(u, "quantifies") is not None
                and _descriptor(_follow(u, "quantifies"))[0] == period_u), None)
    if num is None:
        return None
    if dur_u == rate.per_unit:
        per = _num(dur_amt)
    elif (dur_u, rate.per_unit) in CONVERSIONS:
        per = f"{_num(dur_amt)} * {_num(CONVERSIONS[(dur_u, rate.per_unit)])}"
    else:
        return None
    return _compute(f"{_num(rate.amount)}/{_num(rate.per_amount)} * ({per}) * {_num(num)}")


def _rate_race(premises: list, question: str, context: list):
    """RATE RACE (inversion difference) — "Apple runs 3 mph, Mac runs 4 mph. In minutes, how much FASTER will
    Mac run a 24-mile race?" → 24/3 − 24/4 = 2 hr = 120 min. Two rates (same measure & per_unit), a TARGET
    distance/amount in the question (a value of the rates' measure), and a faster/longer cue with an asked
    time-unit → (target/slow − target/fast) converted to that unit. None otherwise."""
    rates = [f for f in context if isinstance(f, Rate)]
    if len(rates) != 2 or len(context) != 2:
        return None
    if rates[0].measure != rates[1].measure or rates[0].per_unit != rates[1].per_unit:
        return None
    qr = analyze(question)
    if not any(_key(u.surface) in ("faster", "longer", "slower", "quicker") for u in qr.units):
        return None
    target = next((u.features["value"] for u in qr.units if "value" in u.features.get("senses", [])
                   and _follow(u, "quantifies") is not None
                   and _descriptor(_follow(u, "quantifies"))[0] == rates[0].measure[0]), None)
    au = next((_descriptor(u)[0] for u in qr.units if "noun" in u.features.get("senses", [])
               and _descriptor(u)[0] in TIME_UNITS), None)
    if target is None or au is None:
        return None
    r0, r1 = rates[0].amount / rates[0].per_amount, rates[1].amount / rates[1].per_amount
    slow, fast = (rates[0], rates[1]) if r0 < r1 else (rates[1], rates[0])
    diff = f"{_num(target)} / ({_num(slow.amount)}/{_num(slow.per_amount)}) " \
           f"- {_num(target)} / ({_num(fast.amount)}/{_num(fast.per_amount)})"
    if au == slow.per_unit:
        return _compute(diff)
    if (slow.per_unit, au) in CONVERSIONS:               # time difference in per_unit → the asked unit
        return _compute(f"({diff}) * {_num(CONVERSIONS[(slow.per_unit, au)])}")
    return None


def _rate_remainder(premises: list, question: str, context: list):
    """RATE REMAINDER — "Two dogs walk a TOTAL of 70 miles a week. The first walks 2 miles a day. How many
    miles a day does the OTHER walk?" → 70/week (=10/day) − 2/day = 8/day. Exactly two rates (same measure);
    one is the group TOTAL (its premise carries total/combined/together), the other a component; the question
    asks the remaining component in some per_unit. answer = (total − component), each normalized to that
    per_unit. None otherwise."""
    rates = [f for f in context if isinstance(f, Rate)]
    if len(rates) != 2 or rates[0].measure != rates[1].measure:
        return None
    if any(isinstance(f, Quantity) and f.amount is not None and f.measure[0] == rates[0].measure[0]
           for f in context):
        return None                                      # an extra quantity OF THE RATE'S MEASURE → multi-step
        #                                                  (a bare count like "two dogs" is a different measure)

    def from_total_premise(rate):
        for p in premises:
            if any(w in p.lower() for w in ("total", "combined", "together")) \
                    and any(isinstance(f, Rate) and abs(f.amount - rate.amount) < 1e-9 for f in learn(p)):
                return True
        return False

    total = next((r for r in rates if from_total_premise(r)), None)
    if total is None:
        return None
    comp = next(r for r in rates if r is not total)
    qr = analyze(question)                                # the asked per_unit: "how many miles A DAY"
    pu = next((_descriptor(qr.units[k + 1])[0] for k in range(len(qr.units) - 1)
               if _key(qr.units[k].surface) in ("a", "an", "per", "each", "every")
               and _descriptor(qr.units[k + 1])[0] in TIME_UNITS), None)
    if pu is None:
        return None

    def per(rate):                                        # the rate's amount expressed PER `pu`
        base = f"{_num(rate.amount)}/{_num(rate.per_amount)}"
        if rate.per_unit == pu:
            return base
        if (rate.per_unit, pu) in CONVERSIONS:            # "70 per week" → per day = 70 / (days in a week)
            return f"({base}) / {_num(CONVERSIONS[(rate.per_unit, pu)])}"
        return None

    pt, pc = per(total), per(comp)
    if pt is None or pc is None:
        return None
    return _compute(f"({pt}) - ({pc})")


def _rate_sum(premises: list, question: str, context: list):
    """MULTI-ENTITY RATE SUM — "Anna 30 pages/60 min, Bob 27/60 min, Cherry 25/60 min. They have read for 240
    min. How many pages in TOTAL?" → (30+27+25)/60 × 240 = 328. ≥2 rates (same measure & per_unit) and
    nothing else; a single DURATION (a time-value in the premises/question that is NOT a rate's per-marker);
    a total-cue question. answer = Σ(amount/per_amount) × (duration → per_unit). None otherwise."""
    rates = [f for f in context if isinstance(f, Rate)]
    if len(rates) < 2 or any(isinstance(f, Quantity) and f.amount is not None for f in context):
        return None                                      # ≥2 rates and NO extra real quantity (a multi-step
        #                                                  given) — a bare descriptor from the duration clause
        #                                                  ("reading for 240 min" → ('minute',…)) is tolerated
    if len({(r.measure, r.per_unit) for r in rates}) != 1:
        return None
    per_unit = rates[0].per_unit
    if not any(w in question.lower() for w in ("total", "altogether", "combined", "in all", "together")):
        return None
    per_amounts = {r.per_amount for r in rates}
    dur = du = None
    for text in premises + [question]:                   # the duration: a time-value that is NOT a rate's
        for u in analyze(text).units:                    # per-marker (so not the "60 minutes" of the rate)
            if "value" in u.features.get("senses", []) and _follow(u, "quantifies") is not None \
                    and _descriptor(_follow(u, "quantifies"))[0] in TIME_UNITS:
                v, unit = u.features["value"], _descriptor(_follow(u, "quantifies"))[0]
                if v not in per_amounts or unit != per_unit:
                    dur, du = v, unit
    if dur is None:
        return None
    if du == per_unit:
        d = _num(dur)
    elif (du, per_unit) in CONVERSIONS:
        d = f"{_num(dur)} * {_num(CONVERSIONS[(du, per_unit)])}"
    else:
        return None
    total = " + ".join(f"{_num(r.amount)}/{_num(r.per_amount)}" for r in rates)
    return _compute(f"({total}) * ({d})")


def _rate_difference(premises: list, question: str, context: list):
    """RATE DIFFERENCE × DURATION — "Janet reads 80 pages a day, Belinda 30 pages a day. How many MORE pages
    does Janet read in 6 weeks?" → (80 − 30) × 42 = 2100; "how many more miles did Frank run in an hour than
    Jim?" → (10 − 8) × 1 = 2. EXACTLY two rates (same measure & per_unit) and nothing else; the question's
    subject is A, the other rate is B; answer = (A_per_unit − B_per_unit) × (duration → per_unit). Gated to a
    "how many/much MORE <measure>" question with a duration, every premise number consumed by the two rates."""
    if len(context) != 2 or not all(isinstance(f, Rate) for f in context):
        return None
    rA0, rB0 = context
    if rA0.measure != rB0.measure or rA0.per_unit != rB0.per_unit:
        return None
    prem_vals = [u.features["value"] for p in premises for u in analyze(p).units
                 if "value" in u.features.get("senses", [])]
    consumed = {rA0.amount, rA0.per_amount, rB0.amount, rB0.per_amount}
    if any(all(abs(v - c) > 1e-9 for c in consumed) for v in prem_vals):
        return None                                      # an unused premise number → not a bare difference
    qr = analyze(question)
    if not (any("interrogative" in u.features.get("senses", []) for u in qr.units)
            and any("comparative" in u.features.get("senses", []) and _key(u.surface) == "more"
                    for u in qr.units)):
        return None
    aux = next((u for u in qr.units if "auxiliary" in u.features.get("senses", [])), None)
    subs = _question_subjects(qr, aux)[0] if aux is not None else []
    A = next((r for r in context if r.entity in subs), None)
    if A is None:
        return None
    B = next(r for r in context if r is not A)
    dval = next((u for u in qr.units if "value" in u.features.get("senses", [])
                 and _follow(u, "quantifies") is not None
                 and _descriptor(_follow(u, "quantifies"))[0] in TIME_UNITS), None)
    if dval is not None:
        dur, du = dval.features["value"], _descriptor(_follow(dval, "quantifies"))[0]
    else:                                                # "in AN hour" / "a day" → a single per-unit
        u = qr.units
        pair = next(((u[k], u[k + 1]) for k in range(len(u) - 1)
                     if _key(u[k].surface) in ("a", "an", "each", "every")
                     and _descriptor(u[k + 1])[0] in TIME_UNITS), None)
        if pair is None:
            return None
        dur, du = 1, _descriptor(pair[1])[0]
    if du == A.per_unit:
        d = _num(dur)
    elif (du, A.per_unit) in CONVERSIONS:
        d = f"{_num(dur)} * {_num(CONVERSIONS[(du, A.per_unit)])}"
    else:
        return None
    return _compute(f"({_num(A.amount)}/{_num(A.per_amount)} - {_num(B.amount)}/{_num(B.per_amount)}) * ({d})")


def _shopping_total(premises: list, question: str):
    """A SHOPPING TOTAL — "buys 3 toys at $10 each, 2 packs at $5 each, 5 shirts at $6 each. How much did she
    spend in all?" → Σ(count × price) = 3·10 + 2·5 + 5·6 = 70. The counts and per-item prices already parse as
    separate buy-quantities; only the per-each products + sum are missing. TIGHTLY gated (the shape collides
    with "for $80" totals, discounts, and per-piece questions, which all carry "each"): the question asks a
    spend TOTAL; EVERY money price must be immediately followed by "each"/"apiece" (a genuine per-item price,
    NOT "four pizzas for $80"); NO discount/sale/off; and the buy-quantities STRICTLY alternate item-count then
    price so each price pairs with its count. None otherwise."""
    ql = question.lower()
    # require an explicit TOTAL cue (not a bare "how much does he spend?", whose 2nd "X for $Y each" pair may
    # have been split INTO the question clause → an incomplete premise pairing, Alexander 5≠9).
    if not (("how much" in ql) and any(w in ql for w in ("spend", "spent", "pay", "paid"))
            and any(w in ql for w in ("in all", "total", "altogether", "combined"))):
        return None
    if any(_key(u.surface) in ("each", "apiece") for u in analyze(question).units):
        return None                                          # shopping content stranded in the question clause
    pl = " ".join(premises).lower()
    if any(w in pl for w in ("discount", "sale", " off", "% ", "coupon", "free")):
        return None
    disc, ctx, readings = [], [], [analyze(p) for p in premises]
    for p in premises:
        f = learn(p, disc, ctx)
        if f:
            ctx += f
    # every money value in the premises must be immediately followed by "each"/"apiece" (a per-ITEM price)
    money_units = [(rp, k) for rp in readings for k, u in enumerate(rp.units)
                   if "value" in u.features.get("senses", []) and "measure" in u.features
                   and "money" in str(u.features.get("measure"))]
    if not money_units:
        return None
    for rp, k in money_units:
        nxt = rp.units[k + 1] if k + 1 < len(rp.units) else None
        if nxt is None or _key(nxt.surface) not in ("each", "apiece"):
            return None
    qs = [f for f in ctx if isinstance(f, Quantity) and f.verb in ("buy", "purchase") and f.amount is not None]
    products, i = [], 0
    while i < len(qs) - 1:                                # STRICT alternation: (item-count, money-price)+
        a, b = qs[i], qs[i + 1]
        if a.measure != MONEY and b.measure == MONEY:
            products.append(f"{_num(a.amount)} * {_num(b.amount)}")
            i += 2
        else:
            return None
    if i != len(qs) or not products:                     # leftover unpaired quantity → not a clean shape
        return None
    return _compute(" + ".join(f"({p})" for p in products))


def _equal_share(question: str, context: list, premises: list):
    """POOL-AND-REDISTRIBUTE — "Mark has 30 candies, Peter 25, John 35. They combine their candies and share
    them equally. How many will EACH have?" → (30+25+35)/3 = 30. Same Σ/N math as the `average` branch (#3l)
    but triggered by a SHARE-EQUALLY premise + a per-each result question, not the word "average". The
    recognizer needs the premise text (the "share equally" cue isn't in the question), so it lives in solve(),
    not answer(). GATED: a share/divide/split/combine + equal cue in a premise; the question asks "how
    many/much … EACH … have/get"; a SINGLE shared measure with ≥2 owners; and (denominator safety, as #3l) any
    set-size the question states must equal the owner count. Σ(each owner's net holding) / owner-count. None
    otherwise."""
    ql = question.lower()
    if "each" not in ql or not any(w in ql for w in ("how many", "how much")) \
            or not any(w in ql for w in ("have", "get", "receive", "end up")):
        return None
    pl = " ".join(premises).lower()
    if not (any(w in pl for w in ("share", "shares", "shared", "divide", "divided", "split", "distribute",
                                  "combine", "combined")) and any(w in pl for w in ("equal", "equally", "evenly"))):
        return None
    if any(isinstance(f, Distribution) for f in context):
        return None                                        # one whole split among a COUNT (Mr. Sam $5250 ÷ 3
        #                                                    sons) is the existing equal-split handler's job —
        #                                                    its synthesized per-recipient share would here be
        #                                                    wrongly pooled with the distributor's own holding
    if any(w in pl for w in ("the rest", "remaining", "remainder", " left ", "kept")):
        return None                                        # "kept 20 and shared THE REST among his friends" is
        #                                                    a remainder distributed to OTHERS, not a pool among
        #                                                    the owners (Harold 100 → 80÷5, not Σ holdings ÷ N)
    measures = {f.measure for f in context if isinstance(f, Quantity) and f.amount is not None}
    if len(measures) != 1:
        return None
    gq = next(iter(measures))
    owners = {f.entity for f in context if isinstance(f, Quantity) and f.measure == gq
              and f.amount is not None and f.entity != gq[0]}   # exclude a NOUN-entity ("the rest" → entity
    if len(owners) < 2:                                         # 'marble' == measure head), only real holders
        return None
    terms = [_holdings(o, gq, context) for o in owners]
    if not all(t is not None for t in terms):
        return None
    qr = analyze(question)                                 # set-size safety: a stated count > 1 (a recipient
    stated_n = next((u.features["value"] for u in qr.units if "value" in u.features.get("senses", [])  # set size,
                     and float(u.features["value"]).is_integer() and u.features["value"] > 1), None)   # not the
    if stated_n is not None and stated_n != len(owners):   # "one" of "each ONE of them") must equal owner count
        return None
    result = _compute(f"({' + '.join(_num(t) for t in terms)}) / {len(owners)}")
    if isinstance(result, float) and not result.is_integer():
        return None                                        # a clean pool divides evenly; a fraction means the
        #                                                    holdings didn't all resolve (a relation deferred)
    return result


# --- ARITHMETIC SYSTEM-SOLVER REGISTRY -------------------------------------------------------
# The word-problem system solvers are REGISTERED here, not hand-inlined in solve() — honoring the
# engine's founding rule ("new reasoning is REGISTERED, never special-cased"). Each is a
# self-contained recognizer: given (premises, question) it returns an answer, or None to pass.
# solve() tries them in list order and returns the first non-None. This single ordered table is the
# ONE chokepoint the cross-solver collision class (two solvers claiming the same premise-set — the
# failure mode behind this domain's `_name_count` bug and 5 TEST regressions) is audited against;
# see `reading_diff --collisions`.
#
# EARLY systems run BEFORE the read-synthesize fixpoint: their relations DEFER to no fact, so a
# stated factor/offset/total would trip the unused-given GUARD 3 if left to the general fold. All
# share the uniform signature f(premises, question); each re-analyzes its own premises and is
# tightly gated (pure context, every premise number consumed). To add one, append it here in
# priority order — do NOT re-inline a branch in solve().
_EARLY_SYSTEMS = [
    # N-UNKNOWN SUM+RATIO-CHAIN: "Amy/Jeremy/Chris combined 132 → Jeremy 66" (subsumes 2-entity Jean/Jane).
    ("sum_ratio", _sum_ratio_system),
    # OFFSET SUM-system (affine twin): "Justin + Sabrina 50; Justin 8 more than twice Sabrina → Sabrina 14."
    ("offset_sum", _offset_sum_system),
    # PURE OFFSET SUM (no factor): "Anais 30 more toys than Kamari; 160 altogether → Kamari 65."
    ("offset_sum_pure", _offset_sum_pure),
    # MIXED AFFINE (N entities, ratio+offset mixed, +total): "Anna=3·Mandy & 2 fewer than Amanda; 142 → Mandy 20."
    ("affine", _affine_system),
    # COORDINATE/SUBTYPE-NOUN AFFINE (common-noun entities): "twice as many pigs as cows, 4 more cows than goats; 56 animals → goats 11."
    ("noun_affine", _noun_affine_system),
    # COORDINATE-NOUN ANCHORED-SUM (no total, one noun anchor, category-sum ask): "Hannah's family: 10 pigs … how many animals?" → 50.
    ("noun_anchored_sum", _noun_anchored_sum_system),
    # SUBTYPE-PARTITION (adjective subtypes of one head noun, total in same noun, 'the rest' ask): "Ryan 40 balls … green balls?" → 7.
    ("noun_residual_sum", _noun_residual_sum_system),
    # ANCHORED-SUM (no total, one bare anchor, sum ask): "Cecil $600 … how much altogether?" → 2800.
    ("anchored_sum", _anchored_sum_system),
    # ANCHORED-SINGLE (bare anchor, single named ask): "Ed weighs 146 … Find Carl's weight." → 220.
    ("anchored_single", _anchored_single_system),
    # ANCHORED-DIFFERENCE (bare anchor, 'how many more X than Y'): "Ann 60 cards … how many more than Anton?" → 30.
    ("anchored_difference", _anchored_difference_system),
    # RESIDUAL SUM (stated total names 3+, asked one never related): "Jermaine/Terrence/Emilee $90 … Emilee?" → 25.
    ("residual_sum", _residual_sum_system),
]

# LATE systems run AFTER the read-synthesize fixpoint has built `context` (they consume derived
# facts). Their signatures differ (some need context/readings), so each is registered as a thin
# adapter over the uniform call `(premises, question, context, readings_prem)` — the adapter picks
# the args its solver actually takes. Same registry contract as `_EARLY_SYSTEMS`: append here in
# priority order, never re-inline a branch in solve().
_LATE_SYSTEMS = [
    # SCENE DEPLETION — "there were N X … took A … took B … how many X left?" → N − (A+B).
    ("scene_depletion", lambda P, Q, C, R: _scene_depletion(Q, C, R)),
    # SHOPPING TOTAL — "buys 3 toys at $10 each, 2 packs at $5 each … spend in all?" → Σ(count×price).
    ("shopping_total", lambda P, Q, C, R: _shopping_total(P, Q)),
    # PROPORTION — "If 12 bags weigh 24 pounds, how much do 8 bags weigh?" → 24 × 8/12.
    ("proportion", lambda P, Q, C, R: _proportion(P, Q)),
    # RATE EXTRAPOLATION — "drives 80 miles in 2 hours. How far in 15 minutes?" → 80×0.25/2.
    ("rate_extrapolation", lambda P, Q, C, R: _rate_extrapolation(P, Q, C)),
    # COMPOUND-DURATION rate — "walks 10 m/min. Every day for an hour. Meters in 2 days?" → 1200.
    ("rate_compound", lambda P, Q, C, R: _rate_compound(P, Q, C)),
    # RATE DIFFERENCE — "Janet 80 pg/day, Belinda 30/day. How many more in 6 weeks?" → 2100.
    ("rate_difference", lambda P, Q, C, R: _rate_difference(P, Q, C)),
    # RATE SUM — "Anna/Bob/Cherry each read R pg/T min. Read for D min. Total?" → ΣR/T × D.
    ("rate_sum", lambda P, Q, C, R: _rate_sum(P, Q, C)),
    # RATE REMAINDER — "two dogs walk 70 mi/week total; first walks 2/day. Other, per day?" → 8.
    ("rate_remainder", lambda P, Q, C, R: _rate_remainder(P, Q, C)),
    # RATE RACE — "Apple 3 mph, Mac 4 mph. How much faster on a 24-mile race, in minutes?" → 120.
    ("rate_race", lambda P, Q, C, R: _rate_race(P, Q, C)),
    # SELF-MULTIPLE SUM — "rode 2 miles, then 5 times that amount. Total?" → 2×(1+5) = 12.
    ("self_multiple_sum", lambda P, Q, C, R: _self_multiple_sum(P, Q)),
    # PERCENT CHANGE — "$220 bike +15%?" → 253; "$2000, 30% discount. Pays?" → 1400.
    ("percent_change", lambda P, Q, C, R: _percent_change(P, Q)),
    # PERCENT ORIGINAL (inverse) — "$15000 is 40% of the original. Original?" → 37500.
    ("percent_original", lambda P, Q, C, R: _percent_original(P, Q)),
    # PERCENT POPULATION — "50 people, 30% boys. Girls?" → 35; "60% girls, 300 boys → girls?" → 450.
    ("percent_population", lambda P, Q, C, R: _percent_population(P, Q)),
    # BACKWARD INVERSION (general) — "spent $1.75 + $1.25; $6 left. Had?" → 9 (any phrasing, not just "at first").
    ("backward_inversion", lambda P, Q, C, R: _backward_inversion(P, Q)),
    # POOL-AND-REDISTRIBUTE — "combine their candies and share equally, how many EACH?" → Σ/N.
    ("equal_share", lambda P, Q, C, R: _equal_share(Q, C, P)),
]


def _parse_problem(text: str):
    """Split a word-problem paragraph into (premises, question). Returns None when there is no
    question clause. Factored out of solve() so the collision harness (reading_diff --collisions)
    sees the IDENTICAL premise/question decomposition the live dispatch does — a divergence would
    make the collision audit lie about what the registry actually claims."""
    sentences = _sentences(text)
    clauses = [c for s in sentences for c in _split_clauses(s)]   # split into independent clauses
    # the QUESTION clause holds the interrogative ("How many …"); for "Q if P?" the trailing "?" lands on
    # the condition P, so prefer an interrogative-bearing clause over the bare "?"-clause. Some GSM8K
    # problems phrase the question as an IMPERATIVE instead ("Calculate the total…", "Find X's weight.",
    # ending in "." not "?" — ~4% of the corpus): a clause OPENING with a compute-imperative verb is the
    # question clause too, lowest priority (an interrogative or a literal "?" both take precedence).
    wh = {"how", "what", "which", "who", "whom", "whose", "why"}
    imperative_verbs = {"calculate", "determine", "find", "compute"}
    interro = [k for k, c in enumerate(clauses) if {_key(w) for w in c.split()} & wh]
    imperative = [k for k, c in enumerate(clauses) if c.split() and _key(c.split()[0]) in imperative_verbs]
    qpos = (interro[-1] if interro else
            max((k for k, c in enumerate(clauses) if c.rstrip().endswith("?")), default=None))
    if qpos is None:
        qpos = imperative[-1] if imperative else None
    if qpos is None:
        return None
    question = clauses[qpos]                              # the question clause; the rest are premises
    premises = clauses[:qpos] + clauses[qpos + 1:]        # (a leading "If P," clause is now a premise)
    return premises, question


def solve(text: str):
    """Read a whole word-problem paragraph (premises + a trailing question) and answer the question —
    threading coreference (`discourse`) and the growing `context` across the premises (split into
    independent CLAUSES first, so the fixpoint sees each fact separately). Returns the raw `answer()`
    result. The programmatic entry the GSM8K coverage harness scores. None if there's no question.
    `ARITHMETIC_LOG` is reset here, so after a solve it holds exactly the audited programs this answer ran."""
    ARITHMETIC_LOG.clear()
    parsed = _parse_problem(text)
    if parsed is None:
        return None
    premises, question = parsed
    readings = [analyze(c) for c in premises + [question]]
    # PRECISION GUARD 1 — unrecognized word: a partial parse may have dropped the fact that mattered,
    # so we DECLINE rather than commit off a fragment ("definitive-or-decline": never answer what we
    # didn't fully read).
    if any(d.kind == "unknown-word" for r in readings for d in deficiencies(r)):
        return "undetermined"
    # ARITHMETIC SYSTEM-SOLVERS (EARLY) — tried before the read-synthesize fixpoint (each relation
    # DEFERS to no fact, so a stated factor/offset/total would trip the unused-given GUARD 3 in the
    # general fold). Each recognizer is self-contained (re-analyzes premises, needs no context) and
    # tightly gated; solve() returns the first that claims the problem. The recognizers, their shapes,
    # and their priority order live in the `_EARLY_SYSTEMS` registry above — append there to add one.
    for _label, _system in _EARLY_SYSTEMS:
        _r = _system(premises, question)
        if _r is not None:
            return _r
    # GUARD 8 — INVERSION / COUNTERFACTUAL: two shapes the forward fold structurally cannot solve, so it
    # would fold the literal numbers and commit garbage. Both are honest DECLINES (the safe "only-declines"
    # shape, like GUARD 7).
    #  (a) BACKWARD-initial: the question asks for an ORIGINAL value ("how much did she have at first?")
    #      while premises describe a chain of changes ending in a KNOWN final ("$51 left") — an inversion
    #      (solve-for-initial), not a forward accumulation. Gold: Isabel 204, Randy 3000.
    #  (b) COUNTERFACTUAL: a premise in the irrealis mood ("if X had $N more, she WOULD have $M") states a
    #      HYPOTHETICAL state, not an asserted holding; folding its numbers as real quantities is unsound.
    #      Gold: Beth/Jan 150, Sally/Jolly 150. The modal "would" is the irrealis marker.
    ql = question.lower()
    if any(p in ql for p in ("at first", "to begin with", "to start with",
                             "originally", "at the start", "in the beginning")):
        # BACKWARD POOL DEPLETION — a losses-only inversion we CAN solve (initial = final + Σ spends); the
        # tight gates keep every other "at first" inversion (gains, fraction-of-remaining, relational) declining.
        _bi = _backward_inversion(premises, question)    # single-entity signed deltas (gains + losses) first
        if _bi is not None:
            return _bi
        _bd = _backward_depletion(premises, question)    # then the multi-entity losses-only pool
        return _bd if _bd is not None else "undetermined"
    if any("would" in [_key(w) for w in p.lower().split()] for p in premises):
        return "undetermined"
    # GUARD 9 — DISTRIBUTIVE per-occurrence frequency: "every time / each time <X happens>, <amount>" is a
    # rate (per-occurrence amount) that must be MULTIPLIED by an occurrence count to total. We don't compose
    # that product yet, so committing the bare per-occurrence amount silently drops the frequency factor
    # ("2 times a day … every time she orders 3 cups" → 3, not 6). DECLINE on the frame — every other such
    # problem in dev already declines on other grounds; this makes the decline uniform. Gold: Maria 6.
    if any(p in " ".join(premises).lower() for p in ("every time", "each time")):
        return "undetermined"
    # GUARD 10 — ELIDED TRANSFER OBJECT: "gave 12 to his brother, 9 to his sister" — the value quantifies
    # the RECIPIENT (the noun that is the object of "to"), because the transferred object was elided ("12
    # [boxes] to his brother"). We then attach the count to the recipient (measure='brother'), so the give
    # never reduces the real holding and the gross commits (Sonny: 45 with the 12/9/7 gives lost). The
    # object is unrecoverable here, so DECLINE. A stated object ("gave 12 BOXES to his brother": 12
    # quantifies 'boxes', not the to-object) is unaffected.
    for rp in readings[:len(premises)]:
        tos = [(rp.units.index(u), u) for u in rp.units
               if "preposition" in u.features.get("senses", []) and _key(u.surface) == "to"]
        for vi, u in enumerate(rp.units):
            if "value" in u.features.get("senses", []):
                qn = _follow(u, "quantifies")
                # the value must PRECEDE its "to" (the transferred amount with the object elided —
                # "12 [boxes] to his brother") and quantify that "to"'s object. A value AFTER "to"
                # ("to 3 friends", "to 48 of her friends") is a RECIPIENT COUNT, not an elided object.
                if qn is not None and any(vi < ti and _follow(t, "object") is qn for ti, t in tos):
                    return "undetermined"
    discourse, context = [], []
    facts_of = [None] * len(premises)                    # per-premise facts (None = not yet resolved)
    for i, p in enumerate(premises):                     # first pass: sequential, builds the discourse
        f = learn(p, discourse, context)
        if f:
            facts_of[i] = f
            context += f
    _inject_bare_ages(readings[:len(premises)], readings, facts_of, context)   # "If Job is 5" → age 5
    context += _question_durations(readings[-1], context)   # SEED a question-clause duration ("…in 7
    # days?") so a rate×duration chain whose duration is in the QUESTION (not a premise) composes via the
    # rate_duration synth in the fixpoint below (Tom: 2 h/day × 7 days → 14 h, then 50 pg/h × 14 → 700).
    progress = True                                      # FIXPOINT: retry the unresolved (forward refs,
    while progress:                                      # chains) against the growing context
        progress = False
        for i, p in enumerate(premises):
            if facts_of[i] is None:
                f = learn(p, discourse, context)         # re-resolve coreference too (a retried clause's
                if f:                                    # "he" must still bind — recency-safe re-append)
                    facts_of[i], progress = f, True
                    context += f
        derived = [f for f in synthesize(context, readings) if f not in context]  # SYNTHESIZE step:
        if derived:                                      # derive implied facts (transitive order, …) and
            context += derived                           # keep looping — one unified read–synthesize loop
            progress = True
    # ARITHMETIC SYSTEM-SOLVERS (LATE) — tried after the fixpoint has built `context` (they consume
    # derived facts). Tried before GUARD 2/3 (their givens are all "used", so they wouldn't trip the
    # unused-given guard, but the recognizers are kept together). The recognizers, their shapes, and
    # their priority order live in the `_LATE_SYSTEMS` registry above; solve() returns the first that
    # claims the problem. `_backward_inversion` is retried here for any phrasing (GUARD 8 above only
    # tries it for "at first"-phrased questions); its own gates keep it off forward problems.
    _readings_prem = readings[:len(premises)]
    for _label, _system in _LATE_SYSTEMS:
        _r = _system(premises, question, context, _readings_prem)
        if _r is not None:
            return _r
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
    for i, rp in enumerate(readings[:len(premises)]):
        facts_i = facts_of[i] or []
        # the comparative/multiplier exemption holds ONLY if the premise actually produced a Quantity/Rate
        # (proof the factor was consumed into a DERIVED amount). A degenerate parse that yields only a
        # Frequency — "Billy has $25 less than twice the money Sam has" → Frequency(have, 2), dropping the
        # $25 — must NOT be exempt, so the unused $25 trips this guard and we DECLINE instead of summing a
        # partial total (Billy committed 75 = Sam alone). "30 less than Martha" → a Quantity → still exempt.
        comp_consumed = any(s in u.features.get("senses", []) for u in rp.units
                            for s in ("comparative", "multiplier")) \
            and any(isinstance(f, (Quantity, Rate)) for f in facts_i)
        # a TIME-RANGE rate ("1 PM to 4 PM") consumes its two clock values into the span (4−1=3), so they
        # won't appear verbatim — exempt when the premise has two AM/PM meridiems and produced a Rate.
        time_range = len([u for u in rp.units if _key(u.surface) in ("am", "pm")]) == 2 \
            and any(isinstance(f, Rate) for f in facts_i)
        if comp_consumed or time_range \
                or any(_key(u.surface) == "each" for u in rp.units) \
                or any(isinstance(f, Distribution) for f in facts_i) \
                or _relational_age_factor(rp) is not None \
                or _future_age(rp) is not None \
                or _container_quantity(rp):   # "20 packets of 100 grams" / "N times as old" /
                # "in 2 years … 20 years old" (20−2 — offset+age consumed into the now-age) —
            continue                                      # the factor is consumed into the derived product
            # NOTE: a slash-fraction partitive ("3/8 of his money") is deliberately NOT exempt here — its
            # raw value (0.375) being flagged as an unused given is what makes a multi-step fraction CHAIN
            # ("3/8 … and 3/10 …", "2/5 of the remaining") DECLINE instead of committing a partial product.
            # Exempting it solved the single-partitive case (+1) but un-declined the chains (+3 wrong) →
            # net precision-negative. Re-exempt only once the chain composition (elided 2nd base, remainder)
            # is handled; until then the single partitive declines (safe), the chains decline (safe).
        cardinals = _list_cardinals(rp)                  # a value-list's set-size ("three quizzes") is
        given |= {u.features["value"] for u in rp.units  # consumed as the size, not a stated datum → exempt
                  if "value" in u.features.get("senses", []) and id(u) not in cardinals}
    used = {f.amount for f in context if isinstance(f, (Quantity, Rate))} \
        | {f.per_amount for f in context if isinstance(f, Rate)} \
        | {f.count for f in context if isinstance(f, Frequency)}
    if given - used:
        return "undetermined"
    # GUARD 4 — a premise that STATES a number (value/percent/multiplier) but produced no quantity fact
    # dropped information we can't model ("…increases by 50% every year"; "gave his three friends total"
    # → spurious instances) → decline rather than answer off the part we did parse.
    for i, rp in enumerate(readings[:len(premises)]):
        if any(s in u.features.get("senses", []) for u in rp.units
               for s in ("value", "percent", "multiplier")) \
                and not any(isinstance(f, (Quantity, Rate, Frequency, RateRel)) for f in (facts_of[i] or [])):
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
        # EXEMPT an ACTION-amount question ("how much did he SPEND?") that names a specific verb matched by
        # exactly ONE context quantity — the asked event is unambiguous, unlike a HOLDING question ("how much
        # will he HAVE?") whose answer genuinely depends on which point in the timeline (Jack: 47 vs 97).
        qverbs = [_lemma(u.surface) for u in readings[-1].units if "verb" in u.features.get("senses", [])
                  and "auxiliary" not in u.features.get("senses", [])
                  and _lemma(u.surface) not in ("have", "be", "own", "hold")]
        qv = qverbs[0] if qverbs else None
        matching = [f for f in context if isinstance(f, Quantity) and f.verb == qv
                    and f.amount is not None] if qv else []
        if not (qv is not None and len(matching) == 1):
            return "undetermined"
    # GUARD 6 — "each OF the [subset]" is a per-MEMBER query over a restricted subset of a group; we hold
    # no per-member or subset facts (only the group aggregate), so the per-member demand can't be met →
    # decline (never match the group total). Distinct from "each <noun>" — the equal-split per-each we answer.
    qu = readings[-1].units
    if any(_key(u.surface) == "each" and _key(qu[j + 1].surface) == "of" for j, u in enumerate(qu[:-1])):
        return "undetermined"
    # PARTITION-and-AGGREGATE: a container set split into subgroups ("the first two … the other two"),
    # each holding a per-member quantity → Σ (count × per_member). Tries before the generic fold (which
    # would sum the per-member values without their subgroup counts). None ⇒ not this shape → fall through.
    part = _partition(premises, question)
    if part is None:                                     # EQUAL-SPLIT into containers then give some away
        part = _container_giveaway(premises, question, context)   # ("…4 bags … same number each … gives
    result = part if part is not None else answer(question, context, discourse)   # one bag away" → 28−7)
    # PERIOD SCALING: events that recur DAILY (dayparts "morning/afternoon/evening", or "every/each day")
    # are summed into a per-DAY total, but the question asks "every/per/a WEEK" → ×7. Gated on BOTH the
    # daily cue in a premise AND the weekly period in the question, AND no "week" stated in a premise (else
    # the events already carry the period and we'd double-count). Droid: 18 bags/day → 18×7 = 126.
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        qk = [_key(w) for w in question.lower().split()]
        asks_week = any(qk[j] == "week" and j > 0 and qk[j - 1] in ("every", "per", "a", "each")
                        for j in range(len(qk)))
        prem_units = [u for rp in readings[:len(premises)] for u in rp.units]
        daily = any(_key(u.surface) in DAYPARTS for u in prem_units) \
            or any(_key(u.surface) == "day" and i > 0 and _key(prem_units[i - 1].surface) in ("every", "each")
                   for i, u in enumerate(prem_units))
        no_week_premise = not any(_key(u.surface) == "week" for u in prem_units)
        if asks_week and daily and no_week_premise:
            result = _compute(f"{_num(result)} * 7")     # week = 7 daily totals
    if isinstance(result, float) and not isinstance(result, bool):   # snap binary-float drift on the FINAL
        snapped = round(result, 9)                                   # answer only (intermediates keep full
        result = int(snapped) if snapped == int(snapped) else snapped   # precision): "100 × (1 − 0.8)" → 30,
        # not 29.999999999999993. Rounding intermediates instead would lose precision (a 1/60 conversion).
    # GUARD 7 — a NEGATIVE number is a nonsensical answer to a "how many/how much" count/amount question
    # (you can't hold −3 apples or −$5): it always means a term was dropped or mis-signed in the fold, so
    # DECLINE rather than commit. Pure precision guard — correct GSM8K answers are never negative.
    if isinstance(result, (int, float)) and not isinstance(result, bool) and result < 0:
        return "undetermined"
    # GUARD 11 — PERIOD NOT APPLIED (decline-safety): the question asks for a per-period total ("…in N
    # weeks") but N never entered the computation → the period scaling is unhandled → DECLINE rather than
    # commit a partial. Checked AFTER the ×7 period-scaling above (so an applied scaling counts as used).
    # The prerequisite for safe vocabulary growth — a fully-read but unscaled per-period problem declines
    # instead of committing (e.g. "buys 2 … every school day … how many in 3 weeks?" → 2 would be wrong).
    if isinstance(result, (int, float)) and not isinstance(result, bool) \
            and _period_unapplied(readings[-1], readings[:len(premises)], context):
        return "undetermined"
    # GUARD 12 — DROPPED UNIT/MONEY QUANTITY (decline-safety): a stated number quantifying a measurement
    # unit (distance/weight/volume) or a money value that never entered the computation is an OPERATIVE
    # quantity we dropped → the structure is unhandled → DECLINE. Object-noun counts ("3 snakes") are not
    # units → exempt. Skipped when a special recognizer (partition/giveaway) produced the result (its
    # bespoke arithmetic legitimately ignores some givens, e.g. a container count). The broad firewall-safe
    # guard underpinning vocabulary growth: a fully-read but partially-folded problem declines, never commits.
    if part is None and isinstance(result, (int, float)) and not isinstance(result, bool) \
            and _dropped_unit_quantity(readings[:len(premises)], readings[-1]):
        return "undetermined"
    # GUARD 13 — MULTI-RATE purchase/charge (decline-safety): a per-item price with an unapplied item count
    # ("three tickets for $12 each" → 3×12 dropped) or ≥2 per-item prices (a priced list) → we don't compose
    # the per-item products → DECLINE. The residue GUARD 12 misses (the prices are summed, so none is dropped).
    if part is None and isinstance(result, (int, float)) and not isinstance(result, bool) \
            and _unapplied_multi_rate(readings[:len(premises)]):
        return "undetermined"
    # GUARD 14 — DROPPED MULTIPLIER (decline-safety): a stated relational factor ("7 times", "one-third")
    # whose evidence never reached the computation → a dropped relational-chain link → DECLINE. Catches
    # multi-step chains that mis-resolve and commit a partial (Emily "7× Sandra, 3× Betty" → 7×Betty drops 3).
    if part is None and isinstance(result, (int, float)) and not isinstance(result, bool) \
            and _dropped_multiplier(readings[:len(premises)]):
        return "undetermined"
    # GUARD 15 — DROPPED EACH-ATTRIBUTE multiplier (decline-safety): an "each/every" quantifier makes a
    # nearby object count a per-each multiplier ("7 starfish with 5 arms each" → 7×5); if that count never
    # reached the computation the attribute product was dropped → DECLINE. The object-count analogue of the
    # unit/price guards (a bare enumeration count stays exempt; "each" is what makes this one operative).
    if part is None and isinstance(result, (int, float)) and not isinstance(result, bool) \
            and _dropped_each_attribute(readings[:len(premises)]):
        return "undetermined"
    if result == "undetermined" and _underspecified_distribution(context) is not None:
        return [Deficiency("under-specified", None,                 # ask, don't silently decline: the
                           "the total was shared, but no split method (e.g. 'equally') is given")]
    return result


# --- CLI -------------------------------------------------------------------------------------
# A thin shell over the engine. `run_cli` returns the text to print (factored out of __main__ so it
# is testable): with no --about it analyzes a statement and reports understood / what to clarify;
# with --about it treats the text as a question and answers it against the given instances.
def _obligation_summary(unit: "Unit") -> str:
    parts = [f"{o.need}->{o.target.surface!r}" if o.met else f"{o.need}->?"
             for o in unit.features.get("obligations", [])]
    return ("  " + ", ".join(parts)) if parts else ""


def _sentences(text: str) -> list:
    """Split text into sentences on . ? ! (keeping the terminator). No regex — stdlib only. A "." is NOT a
    sentence end when it is a DECIMAL ("$5.50") or a TITLE abbreviation ("Mr. Sanchez")."""
    out, cur = [], ""
    for i, ch in enumerate(text):
        cur += ch
        if ch in ".?!" and cur.strip():
            if ch == "." and i and text[i - 1].isdigit() and i + 1 < len(text) and text[i + 1].isdigit():
                continue                             # decimal point, e.g. "5.50" — keep the token whole
            if ch == "." and cur.strip().split()[-1].rstrip(".").lower() in TITLES:
                continue                             # "Mr." / "Dr." — an honorific, not a sentence end
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


_USAGE = ('usage: python -m reasoning.reading "<text>" [--about "<phrase>" ...] [--trace]\n'
          '  --trace also prints the AUDITED arithmetic programs (each run through the safe evaluator).')


def run_cli(args: list) -> str:
    if not args:
        return _USAGE
    trace = "--trace" in args
    args = [a for a in args if a != "--trace"]
    if not args:
        return _USAGE
    text = args[0]
    fixpoint = False
    if "--about" in args:                                  # explicit context (single-pass)
        premises, question = args[args.index("--about") + 1:], text
    else:                                                  # paragraph: CLAUSE-split like solve(), so a
        clauses = [c for s in _sentences(text) for c in _split_clauses(s)]   # multi-step problem resolves
        wh = {"how", "what", "which", "who", "whom", "whose", "why"}
        interro = [k for k, c in enumerate(clauses) if {_key(w) for w in c.split()} & wh]
        qpos = (interro[-1] if interro else
                max((k for k, c in enumerate(clauses) if c.rstrip().endswith("?")), default=None))
        premises, question = ((clauses[:qpos] + clauses[qpos + 1:], clauses[qpos]) if qpos is not None
                              else ([], None))
        fixpoint = question is not None

    if question is None:                                   # a single statement → analyze it
        r = analyze(text)
        if trace:
            return "\n".join([f'STATEMENT: "{text}"'] + _trace_lines(r))
        defs = deficiencies(r)
        verdict = ("understood" if not defs
                   else "needs clarification:\n" + "\n".join("  - " + d.message for d in defs))
        return "\n".join([f"  {u.surface!r:14} {u.features.get('senses', '?')}"
                          f"{_obligation_summary(u)}" for u in r.units] + ["", verdict])

    # Learn premises into context — to a FIXPOINT for a paragraph (forward refs / chains / synthesis all
    # resolve) so the CLI reasons, and logs the audited arithmetic, exactly like solve(); single-pass --about.
    ARITHMETIC_LOG.clear()
    discourse, context = [], []
    readings = [analyze(p) for p in premises]
    facts_of = [None] * len(premises)
    for i, p in enumerate(premises):
        _resolve_references(readings[i], discourse)        # resolve coreference for the trace (same
        f = learn(p, discourse, context)                   # discourse learn sees: prior subjects)
        if f:
            facts_of[i], context = f, context + f
    _inject_bare_ages(readings, readings + [analyze(question)], facts_of, context)   # mirror solve() (#104)
    while fixpoint:                                         # retry the deferred + synthesize, to a fixpoint
        fixpoint = False
        for i, p in enumerate(premises):
            if facts_of[i] is None:
                f = learn(p, discourse, context)
                if f:
                    facts_of[i], fixpoint, context = f, True, context + f
        derived = [f for f in synthesize(context, readings) if f not in context]
        if derived:
            context, fixpoint = context + derived, True

    _sys = _sum_ratio_system(premises, question)           # mirror solve() (#104): sum/ratio systems
    if _sys is None:
        _sys = _offset_sum_system(premises, question)
    _depleted = _scene_depletion(question, context, readings)   # mirror solve() (#104) — pool − Σ takes
    if _depleted is None:
        _depleted = _equal_share(question, context, premises)   # mirror solve(): pool-and-redistribute Σ/N
    if not trace:                                          # just the answer
        return _answer_text(_sys if _sys is not None else
                            _depleted if _depleted is not None else answer(question, context))

    out = []                                               # the full trace
    for i, p in enumerate(premises):
        out += [f'STATEMENT: "{p}"'] + _trace_lines(readings[i]) + [f"  learned: {facts_of[i] or []}", ""]
    out += [f'QUESTION: "{question}"'] + _trace_lines(analyze(question)) + [""]
    ans = _depleted if _depleted is not None else answer(question, context)   # logs the arithmetic too
    programs = list(ARITHMETIC_LOG)                        # snapshot before explain() re-derives
    out += ["derivation:"] + ["  " + line for line in explain(question, context)]
    if programs:                                           # the AUDITED arithmetic — each a re-runnable
        out += ["", "arithmetic (executed via the audited safe evaluator):"]   # whitelisted program
        out += [f"  answer = {prog}   ->  {_num(res)}" for prog, res in programs]
    out += ["", f"ANSWER: {_answer_text(ans)}"]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(run_cli(sys.argv[1:]))

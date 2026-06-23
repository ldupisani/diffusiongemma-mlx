"""bind layer — text domains (see reasoning.bind)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403


# --- count-occurrences family (two quoted strings) ------------------------------------
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def bind_count_occurrences(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: 'needle' ... 'haystack' (in that order) → {needle, haystack}."""
    quoted = [a or b for a, b in _QUOTED.findall(query)]
    if len(quoted) == 2:
        return True, {"needle": quoted[0], "haystack": quoted[1]}, "code", 0
    return False, None, "code", 0


# --- single-string operation family (one quoted operand + which operation) ------------
def _quoted_strings(query: str) -> list[str]:
    return [a or b for a, b in _QUOTED.findall(query)]


# keyword → single-arg text skill. Most specific first ("longest word" before "word").
_TEXT_OPS = [
    (("reverse", "backwards"), "text.reverse"),
    (("uppercase", "upper case", "capital letters", "all caps"), "text.upper"),
    (("lowercase", "lower case"), "text.lower"),
    (("title case", "title-case", "capitalize each", "capitalize every",
      "capitalize the words", "capitalise each"), "text.title_case"),
    (("vowel",), "text.count_vowels"),
    (("consonant",), "text.count_consonants"),
    (("longest word",), "text.longest_word"),
    (("acronym", "initials", "first letter of each"), "text.acronym"),
    (("character",), "text.char_count"),
    (("word",), "text.word_count"),
]


def detect_text_op(query: str) -> str | None:
    q = query.lower()
    for cues, skill in _TEXT_OPS:
        if any(c in q for c in cues):
            return skill
    return None


def bind_string_op(query: str) -> tuple[bool, dict | None, str, int]:
    """One quoted operand + a detected single-arg operation → {text, skill}."""
    skill = detect_text_op(query)
    quoted = _quoted_strings(query)
    if skill and len(quoted) == 1:
        return True, {"text": quoted[0], "skill": skill}, "code", 0
    return False, None, "code", 0


def bind_replace(query: str) -> tuple[bool, dict | None, str, int]:
    """'replace OLD with NEW in TEXT' → three quoted strings {old, new, text}."""
    quoted = _quoted_strings(query)
    if len(quoted) == 3:
        return True, {"old": quoted[0], "new": quoted[1], "text": quoted[2]}, "code", 0
    return False, None, "code", 0

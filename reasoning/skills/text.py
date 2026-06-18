"""Text / collection skills — counting, sorting, set/sequence operations.

Pure functions over strings and lists; the mechanical building blocks for query types
like "how many times does X appear", "sort these", "how many unique items".
"""

from __future__ import annotations

from .base import register


@register("text.count_occurrences", "count non-overlapping occurrences of needle in haystack",
          [(("banana", "a"), 3), (("hello world", "o"), 2)])
def count_occurrences(haystack: str, needle: str) -> int:
    return haystack.count(needle)


@register("text.word_count", "count whitespace-separated words in a string",
          [(("the quick brown fox",), 4), (("",), 0)])
def word_count(s: str) -> int:
    return len(s.split())


@register("text.sort_items", "sort a list (ascending; reverse=True for descending)",
          [(([3, 1, 2],), [1, 2, 3]), ((["b", "a", "c"],), ["a", "b", "c"])])
def sort_items(items: list, reverse: bool = False) -> list:
    return sorted(items, reverse=reverse)


@register("text.unique", "the distinct items of a list, order preserved",
          [(([1, 1, 2, 3, 3],), [1, 2, 3]), ((["a", "a", "b"],), ["a", "b"])])
def unique(items: list) -> list:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


@register("text.length", "the number of items in a list (count, how many)",
          [(([1, 2, 3],), 3), (([],), 0)])
def length(items: list) -> int:
    return len(items)


@register("text.join", "join list items into a string with a separator",
          [((["a", "b", "c"],), "a, b, c")])
def join(items: list, sep: str = ", ") -> str:
    return sep.join(str(x) for x in items)


# --- single-string operations ---------------------------------------------------------
@register("text.reverse", "the string reversed",
          [(("hello world",), "dlrow olleh"), (("abc",), "cba")])
def reverse(s: str) -> str:
    return s[::-1]


@register("text.upper", "the string in UPPERCASE",
          [(("hello world",), "HELLO WORLD")])
def upper(s: str) -> str:
    return s.upper()


@register("text.lower", "the string in lowercase",
          [(("HELLO",), "hello")])
def lower(s: str) -> str:
    return s.lower()


@register("text.title_case", "the string with each word capitalized (title case)",
          [(("the lazy dog",), "The Lazy Dog")])
def title_case(s: str) -> str:
    return s.title()


@register("text.char_count", "the number of characters in a string (including spaces)",
          [(("hello world",), 11), (("",), 0)])
def char_count(s: str) -> int:
    return len(s)


@register("text.count_vowels", "the number of vowels (a, e, i, o, u) in a string",
          [(("encyclopedia",), 5), (("sky",), 0)])
def count_vowels(s: str) -> int:
    return sum(1 for c in s.lower() if c in "aeiou")


@register("text.count_consonants", "the number of consonant letters in a string",
          [(("python",), 5), (("aeiou",), 0)])
def count_consonants(s: str) -> int:
    return sum(1 for c in s.lower() if c.isalpha() and c not in "aeiou")


@register("text.is_palindrome", "whether a string reads the same forwards and backwards "
          "(ignoring case and non-alphanumerics)",
          [(("racecar",), True), (("hello",), False),
           (("A man a plan a canal Panama",), True)])
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]


@register("text.longest_word", "the longest word in a string (first on ties)",
          [(("the quick brown fox",), "quick")])
def longest_word(s: str) -> str:
    return max(s.split(), key=len)


@register("text.acronym", "the acronym formed from the first letter of each word",
          [(("North Atlantic Treaty Organization",), "NATO"), (("as soon as possible",), "ASAP")])
def acronym(s: str) -> str:
    return "".join(w[0].upper() for w in s.split())


@register("text.replace", "a string with every occurrence of old replaced by new",
          [(("the cat sat on the mat", "cat", "dog"), "the dog sat on the mat")])
def replace(s: str, old: str, new: str) -> str:
    return s.replace(old, new)

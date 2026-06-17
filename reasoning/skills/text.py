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

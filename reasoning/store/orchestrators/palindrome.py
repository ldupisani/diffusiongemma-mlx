"""Whether a string is a palindrome — coded, zero model calls.

Custom run() (like leap_year) so the boolean becomes a natural Yes/No answer.
"""

from reasoning import skills
from reasoning.bind import _quoted_strings
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "palindrome",
    "query_type": "palindrome",
    "description": "whether a string reads the same forwards and backwards",
    "example_queries": [
        "Is 'racecar' a palindrome?",
        "Is 'hello' a palindrome?",
    ],
    "keywords": ["palindrome"],
    "model_free": True,
}


def precondition(query):
    return "palindrome" in query.lower()


def run(query):
    quoted = _quoted_strings(query)
    if len(quoted) != 1:
        return explain("need exactly one quoted string to test for a palindrome")
    s = quoted[0]
    result = skills.call("text.is_palindrome", s)
    answer = f"Yes, '{s}' is a palindrome." if result else f"No, '{s}' is not a palindrome."
    return PatternResult(
        answer=answer,
        trace=[{"step": "is_palindrome", "via": "code", "skill": "text.is_palindrome",
                "output": result},
               {"step": "summary", "model_calls": 0}],
    )

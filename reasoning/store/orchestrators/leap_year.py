"""Whether a year is a leap year — coded, zero model calls.

Custom run() (not run_guarded) so the boolean becomes a natural "Yes/No" sentence while
still reporting a model-free trace.
"""

from reasoning import skills
from reasoning.bind import bind_year
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "leap_year",
    "query_type": "leap_year",
    "description": "whether a given year is a leap year",
    "example_queries": [
        "Is 2024 a leap year?",
        "Was 1900 a leap year?",
    ],
    "keywords": ["leap", "year"],
    "model_free": True,
}


def precondition(query):
    return "leap year" in query.lower()


def run(query):
    ok, data, via, _ = bind_year(query)
    if not ok:
        return explain("could not find a 4-digit year")
    leap = skills.call("date.is_leap_year", data["year"])
    year = data["year"]
    answer = (f"Yes, {year} is a leap year." if leap
              else f"No, {year} is not a leap year.")
    return PatternResult(
        answer=answer,
        trace=[{"step": "bind", "via": via, "model_calls": 0},
               {"step": "is_leap_year", "via": "code", "skill": "date.is_leap_year",
                "output": leap},
               {"step": "summary", "model_calls": 0}],
    )
